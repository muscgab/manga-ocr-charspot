"""NVIDIA GPU backend (PyTorch). Everything after image decoding runs on the GPU:
  - preprocessing: Pillow's BICUBIC resize re-implemented with Pillow's fixed-point arithmetic + median-border canvas (gpu_preprocess.py),
    the same pixels as core.preprocess;
  - dense network (fp32 by default; fp16 optional) on batches of crops zero-padded to a common canvas (heights and widths rounded up on
    a ladder of cell counts); the network is masked so each crop's outputs equal its single-crop outputs;
  - box decoding, greedy NMS (exact, batched), furigana removal, tail (classification + reading-order scores) and the greedy reading-order
    chain, batched over the crops of a canvas.
Only the final boxes / class ids / order come back to the CPU, once per batch. Exception: dense batches of up to CPU_DECODE_MAX crops
copy their small stride-8 maps to the CPU and decode them there with core.detect (faster for a few crops); tail and chain stay on the GPU.
Optional CUDA graphs (default on GPUs with >= 4 GB; smaller GPUs also get smaller batches) for a fixed set of small shapes: the dense network for the 8 most common canvases of
the development set at batch 1 / 2 / 4 / 8, and tail + chain for 1 / 2 / 4 / 8 rows x 8-128 characters (padded to powers of two). Any
other shape runs as plain PyTorch with the same results; no graph is captured for it."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .. import core
from ..gpu_preprocess import GpuPreprocess
from ..model import build

LADDER = (1, 2, 3, 4, 6, 8, 11, 14, 20, 28, 40, 56, 80, 112, 160, 196, 256)   # canvas side lengths in cells
GRAPH_CANVASES = ((28, 11), (40, 8), (20, 14), (28, 14), (56, 6), (20, 20), (11, 28), (40, 6))   # cells; 96.8% of Read v1.2 crops
GRAPH_BATCHES = (1, 2, 4, 8)
GRAPH_CHARS = (8, 16, 32, 64, 128)


def canvas(h_px: int, w_px: int) -> tuple[int, int]:
    up = lambda n: core.CELL * next(v for v in LADDER if v >= n // core.CELL)
    return up(h_px), up(w_px)


def nms_keep(bb, cnt):
    """Exact greedy NMS of core.nms, batched: bb [B, K, 4] sorted by score (descending) per row, cnt [B] boxes per row -> keep [B, K].
    keep_j = no kept i < j suppresses j; iterated to its fixed point, which is the greedy result."""
    B, K = bb.shape[:2]
    area = (bb[..., 2] - bb[..., 0]).clamp(min=0) * (bb[..., 3] - bb[..., 1]).clamp(min=0)
    cx, cy = (bb[..., 0] + bb[..., 2]) / 2, (bb[..., 1] + bb[..., 3]) / 2
    iw = (torch.minimum(bb[:, :, None, 2], bb[:, None, :, 2]) - torch.maximum(bb[:, :, None, 0], bb[:, None, :, 0])).clamp(min=0)
    ih = (torch.minimum(bb[:, :, None, 3], bb[:, None, :, 3]) - torch.maximum(bb[:, :, None, 1], bb[:, None, :, 1])).clamp(min=0)
    inter = iw * ih
    hx, hy = (bb[..., 2] - bb[..., 0]) * 0.25, (bb[..., 3] - bb[..., 1]) * 0.25
    inside = ((cx[:, None, :] - cx[:, :, None]).abs() <= hx[:, :, None]) & ((cy[:, None, :] - cy[:, :, None]).abs() <= hy[:, :, None])
    sup = (~(inter / (area[:, :, None] + area[:, None, :] - inter + 1e-6) < core.NMS_IOU)) | inside          # sup[b, i, j]: kept i removes j
    sup = (sup & torch.ones(K, K, dtype=torch.bool, device=bb.device).triu(1)).float()
    valid = torch.arange(K, device=bb.device)[None] < cnt[:, None]; keep = valid.clone()
    for _ in range(K):
        new = valid & ~(torch.bmm(keep.float()[:, None, :], sup)[:, 0] > 0)
        if torch.equal(new, keep):
            break
        keep = new
    return keep


def chain(pair, st, en, n):
    """core.chain batched: pair [B, N, N], st / en [B, N], n [B] characters per row -> order [B, N] (positions >= n unused)."""
    B, N = st.shape
    ar = torch.arange(N, device=st.device); used = ar[None] >= n[:, None]                # padding counts as used
    ninf = st.new_full((), float("-inf")); rows = torch.arange(B, device=st.device)                 # no host copies (graph capture)
    seq = torch.zeros(B, N, dtype=torch.long, device=st.device)
    cur = torch.where(used, ninf, st).argmax(1); seq[:, 0] = cur; used = used.clone().scatter_(1, cur[:, None], True)
    for k in range(1, N):
        sc = torch.where(used, ninf, pair[rows, cur]); j = sc.argmax(1)
        restart = en[rows, cur] > sc[rows, j]
        j = torch.where(restart, torch.where(used, ninf, st).argmax(1), j)
        live = k < n
        cur = torch.where(live, j, cur); seq[:, k] = torch.where(live, j, 0); used[rows, j] = used[rows, j] | live   # no host sync
    return seq


class CudaBackend:
    def __init__(self, model_dir: Path, precision: str = "fp32", device: str = "cuda", max_batch: int | None = None,
                 tail_budget: int | None = None, nms_budget: int | None = None, graphs: bool | None = None):
        """precision of the dense network: "fp32" (the same outputs as the ONNX model) or "fp16" (faster; a few texts differ).
        max_batch: crops per dense call; tail_budget: rows x characters^2 per tail call (bounds the [rows, n, n, d] pair tensor);
        nms_budget: rows x candidates^2 per NMS call; graphs: CUDA graphs for the small shapes above (they hold ~0.5-1 GB).
        None = by GPU memory: >= 4 GB: 64, 2^18, 2^23, graphs on; smaller GPUs: 8, 2^17, 2^21, graphs off."""
        assert precision in ("fp32", "fp16"), precision
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        w = load_file(str(model_dir / cfg["weights"]))
        n_out = w["tail.cls.weight"].shape[0]
        self.dev, self.precision = torch.device(device), precision
        self.dt = torch.float16 if precision == "fp16" else torch.float32
        self.dense, self.tail = (m.eval() for m in build(cfg, n_out))
        self.dense.load_state_dict({k[6:]: v for k, v in w.items() if k.startswith("dense.")})
        self.tail.load_state_dict({k[5:]: v for k, v in w.items() if k.startswith("tail.")})
        self.dense.to(self.dev).to(memory_format=torch.channels_last); self.tail.to(self.dev)
        if precision == "fp16":
            self.dense.half()
        self.out_str = core.load_vocab(model_dir / cfg["vocab"], cfg["n_cls"])
        big = torch.cuda.get_device_properties(self.dev).total_memory >= 4 << 30
        pick = lambda v, a, b: v if v is not None else (a if big else b)
        self.max_batch, self.tail_budget, self.nms_budget = pick(max_batch, 64, 8), pick(tail_budget, 1 << 18, 1 << 17), pick(nms_budget, 1 << 23, 1 << 21)
        graphs = pick(graphs, True, False)
        self.prep = GpuPreprocess(core.BUDGET, self.dev)
        self.graphs, self._g = graphs, {}
        self._pool = torch.cuda.graph_pool_handle() if graphs else None   # one pool: each graph's outputs are consumed before the next replay

    # ---- CUDA graphs (fallback: plain PyTorch) ----------------------------------------------------------------------------
    def _run(self, key, fn, inputs):
        g = self._g.get(key)
        if g is None:
            st = [t.clone() for t in inputs]
            s = torch.cuda.Stream(self.dev); s.wait_stream(torch.cuda.current_stream(self.dev))
            with torch.cuda.stream(s):
                for _ in range(2):
                    fn(*st)
            torch.cuda.current_stream(self.dev).wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                so = fn(*st)
            g = self._g[key] = (graph, st, so)
        for d, t in zip(g[1], inputs):
            d.copy_(t)
        g[0].replay()
        return g[2]

    def _dense_fn(self, x, hw):
        return tuple(z.float() for z in self.dense(x.to(self.dt).contiguous(memory_format=torch.channels_last), hw))

    def _dense(self, x, hw):
        B, _, H, W = x.shape
        if not (self.graphs and B <= GRAPH_BATCHES[-1] and (H // core.CELL, W // core.CELL) in GRAPH_CANVASES):
            return self._dense_fn(x, hw)
        b = next(v for v in GRAPH_BATCHES if v >= B)
        if b > B:                                                             # padding crops: one cell, ignored afterwards
            x = torch.cat((x, x.new_zeros(b - B, *x.shape[1:]))); hw = torch.cat((hw, hw.new_full((b - B, 2), core.CELL)))
        return tuple(z[:B] for z in self._run(("dense", b, H, W), self._dense_fn, (x, hw)))

    def _tail_fn(self, e, geo, mask, n):
        ids, pair, st, en = self.tail(e, geo, mask)
        return ids, chain(pair, st, en, n)

    def _tail(self, e, geo, mask, n):
        R, N = mask.shape
        if not (self.graphs and R <= GRAPH_BATCHES[-1] and N <= GRAPH_CHARS[-1]):
            return self._tail_fn(e, geo, mask, n)
        r, c = next(v for v in GRAPH_BATCHES if v >= R), next(v for v in GRAPH_CHARS if v >= N)
        e2 = e.new_zeros(r, c, e.shape[-1]); e2[:R, :N] = e; g2 = geo.new_ones(r, c, 4); g2[:R, :N] = geo   # padded rows / characters:
        m2 = mask.new_zeros(r, c); m2[:R, :N] = mask; n2 = n.new_zeros(r); n2[:R] = n                       # masked out, n = 0
        e, geo, mask, n = e2, g2, m2, n2
        ids, seq = self._run(("tail", r, c), self._tail_fn, (e, geo, mask, n))
        return ids[:R, :N], seq[:R, :N]

    @torch.inference_mode()
    def warmup(self):
        """CUDA / cuDNN initialisation; with graphs, captures all of them (dense: 8 canvases x 4 batch sizes, tail: 4 x 5)."""
        self.read_many([np.full((64, 32, 3), 255, np.uint8)])
        if self.graphs:
            for b in GRAPH_BATCHES:
                for h, w in GRAPH_CANVASES:
                    self._dense(torch.zeros(b, 3, h * core.CELL, w * core.CELL, dtype=torch.uint8, device=self.dev),
                                torch.full((b, 2), core.CELL, dtype=torch.long, device=self.dev))
                for c in GRAPH_CHARS:
                    self._tail(torch.zeros(b, c, self.tail.emb_dim, device=self.dev), torch.ones(b, c, 4, device=self.dev),
                               torch.ones(b, c, dtype=torch.bool, device=self.dev), torch.full((b,), c, dtype=torch.long, device=self.dev))
        torch.cuda.synchronize(self.dev)

    def read_many(self, imgs: list) -> list[dict]:
        return self.infer([np.asarray(core.to_rgb(im), np.uint8) for im in imgs])

    MERGE_OVERHEAD = 1024     # cost of one extra dense call, in padded cells (~3-6 ms): small groups merge into a larger canvas
    ROWS_PER_TAIL = 512       # rows per tail call at most
    CPU_DECODE_MAX = 8        # dense batches of up to this many crops are decoded on the CPU (core.detect): for a few crops the
                              # many small GPU kernels and host syncs of the batched decoder cost more (~4 ms vs ~0.3 ms per crop)

    def batches(self, shapes: list) -> list[tuple[list[int], tuple[int, int]]]:
        """Crops grouped by canvas; small groups merged into a common canvas when that saves dense calls (with graphs, only into the
        graph canvases), then cut into batches of at most max_batch."""
        groups = defaultdict(list)
        for i, (h, w) in enumerate(shapes):
            groups[canvas(h, w)].append(i)
        allowed = {(h * core.CELL, w * core.CELL) for h, w in GRAPH_CANVASES} if self.graphs else None
        cost = lambda hw, n: self.MERGE_OVERHEAD + n * hw[0] * hw[1] / core.CELL ** 2
        G = [[hw, idx] for hw, idx in groups.items()]
        while True:
            best = None
            for a in range(len(G)):
                for b in range(a + 1, len(G)):
                    (ha, ia), (hb, ib) = G[a], G[b]; m = (max(ha[0], hb[0]), max(ha[1], hb[1]))
                    if (allowed is not None and m not in allowed) or len(ia) + len(ib) > self.max_batch:
                        continue
                    gain = cost(ha, len(ia)) + cost(hb, len(ib)) - cost(m, len(ia) + len(ib))
                    if gain > 0 and (best is None or gain > best[0]):
                        best = (gain, a, b, m)
            if best is None:
                break
            _, a, b, m = best; G[a] = [m, G[a][1] + G[b][1]]; del G[b]
        return [(idx[k:k + self.max_batch], hw) for hw, idx in G for k in range(0, len(idx), self.max_batch)]

    @torch.inference_mode()
    def infer(self, arrs: list) -> list[dict]:
        """arrs: uint8 HWC RGB arrays -> results in input order. Dense + decoding per canvas batch; tail + chain once over all crops of
        the call (rows of similar length together)."""
        out: list[dict] = [{"text": "", "chars": []} for _ in arrs]
        crops = [self.prep(a) for a in arrs]                                   # (uint8 [3, H, W] on the GPU, (fx, fy))
        D, rows = [], []                                                       # per batch: tensors; rows: (n, batch, row, crop index)
        for idx, (H, W) in self.batches([x.shape[1:] for x, _ in crops]):
            d = self._decode([crops[i][0] for i in idx], H, W); D.append(d)
            rows += [(n, len(D) - 1, r, idx[r]) for r, n in enumerate(d["nk"]) if n > 0]
        host = [{k: d[k].cpu().numpy() for k in ("KB", "KS")} for d in D]
        rows.sort(key=lambda t: -t[0])
        k = 0
        while k < len(rows):
            nc = rows[k][0]; j = k + 1
            while j < len(rows) and j - k < self.ROWS_PER_TAIL and (j - k + 1) * nc * nc <= self.tail_budget:
                j += 1
            part = rows[k:k + max(1, j - k)]; k += len(part); R = len(part)
            e = torch.zeros(R, nc, self.tail.emb_dim, device=self.dev); geo = torch.ones(R, nc, 4, device=self.dev)
            mask = torch.zeros(R, nc, dtype=torch.bool, device=self.dev)
            by_b = defaultdict(list)
            for p, (_, bi, r, _) in enumerate(part):
                by_b[bi].append((p, r))
            for bi, pr in by_b.items():
                P = torch.tensor([p for p, _ in pr], device=self.dev); Rr = torch.tensor([r for _, r in pr], device=self.dev); m = min(nc, D[bi]["E"].shape[1])
                e[P, :m] = D[bi]["E"][Rr, :m]; geo[P, :m] = D[bi]["geo"][Rr, :m]; mask[P, :m] = D[bi]["mask"][Rr, :m]
            nv = torch.tensor([n for n, _, _, _ in part], device=self.dev)
            ids, seq = self._tail(e, geo, mask, nv)
            ids, seq = ids.cpu().numpy(), seq.cpu().numpy()
            for p, (n, bi, r, i) in enumerate(part):
                out[i] = core.result(seq[p, :n].tolist(), ids[p, :n], host[bi]["KB"][r, :n], host[bi]["KS"][r, :n], crops[i][1], self.out_str)
        return out

    def _decode(self, xs, H, W):
        """dense + decoding of one canvas batch -> per-batch tensors on the GPU and kept counts (CPU)."""
        dev, B = self.dev, len(xs)
        x = torch.zeros(B, 3, H, W, dtype=torch.uint8, device=dev); hw = torch.empty(B, 2, dtype=torch.long)
        for r, a in enumerate(xs):
            x[r, :, :a.shape[1], :a.shape[2]] = a; hw[r, 0], hw[r, 1] = a.shape[1], a.shape[2]
        hw_c = hw; hw = hw.to(dev)
        heat, box, ruby, emb = self._dense(x, hw)
        if B <= self.CPU_DECODE_MAX:
            return self._decode_cpu(heat, box, ruby, emb, hw_c)
        # ---- candidates (inside each crop), sorted by score per row; centres in float64 as numpy promotes int64 + float32 ----
        h8, w8 = heat.shape[-2:]
        inside = (torch.arange(h8, device=dev)[None, :, None] < (hw[:, 0] // 8)[:, None, None]) & \
                 (torch.arange(w8, device=dev)[None, None, :] < (hw[:, 1] // 8)[:, None, None])
        cand = ((heat[:, 0] > core.HEAT_THR) & inside).nonzero(); bi, ci, cj = cand[:, 0], cand[:, 1], cand[:, 2]
        bx = box[bi, :, ci, cj]; sc = heat[bi, 0, ci, cj]; rv = ruby[bi, 0, ci, cj]
        cx, cy = (cj.double() + bx[:, 0].double()) * 8, (ci.double() + bx[:, 1].double()) * 8
        w2, h2 = (torch.exp(bx[:, 2]) * 8 / 2).double(), (torch.exp(bx[:, 3]) * 8 / 2).double()
        bb = torch.stack((cx - w2, cy - h2, cx + w2, cy + h2), 1).float()
        o1 = torch.sort(-sc, stable=True).indices; order = o1[torch.sort(bi[o1], stable=True).indices]   # row, then score desc (stable)
        bi_s = bi[order]; cnt = torch.bincount(bi_s, minlength=B); pos = torch.arange(len(bi_s), device=dev) - (torch.cumsum(cnt, 0) - cnt)[bi_s]
        cnt_c = cnt.cpu(); K = max(1, int(cnt_c.max()) if B else 1)
        BB = torch.zeros(B, K, 4, device=dev); BB[bi_s, pos] = bb[order]
        SC = torch.zeros(B, K, device=dev); SC[bi_s, pos] = sc[order]; RV = torch.ones(B, K, device=dev); RV[bi_s, pos] = rv[order]
        keep = torch.zeros(B, K, dtype=torch.bool, device=dev); rows_k = torch.argsort(cnt_c, descending=True); r0 = 0
        while r0 < B:
            kc = max(1, int(cnt_c[rows_k[r0]])); bc = max(1, min(B - r0, self.nms_budget // (kc * kc))); rr = rows_k[r0:r0 + bc].to(dev)
            keep[rr, :kc] = nms_keep(BB[rr, :kc], cnt[rr]); r0 += bc
        keep &= RV < core.RUBY_THR
        # ---- kept boxes, geometry (median long side, lower median as core) ----
        nk = keep.sum(1); nk_c = nk.cpu(); N = max(1, int(nk_c.max()))
        kpos = torch.cumsum(keep.long(), 1) - 1; b_i, k_i = keep.nonzero(as_tuple=True)
        KB = torch.zeros(B, N, 4, device=dev); KB[b_i, kpos[b_i, k_i]] = BB[b_i, k_i]
        KS = torch.zeros(B, N, device=dev); KS[b_i, kpos[b_i, k_i]] = SC[b_i, k_i]
        mask = torch.arange(N, device=dev)[None] < nk[:, None]
        sz = torch.maximum(KB[..., 2] - KB[..., 0], KB[..., 3] - KB[..., 1]); sz = torch.where(mask, sz, torch.full_like(sz, float("inf")))
        u = torch.sort(sz, 1).values.gather(1, ((nk - 1).clamp(min=0) // 2)[:, None]); u = torch.where(nk[:, None] > 0, u, torch.ones_like(u))
        geo = torch.stack(((KB[..., 0] + KB[..., 2]) / 2 / u, (KB[..., 1] + KB[..., 3]) / 2 / u, (KB[..., 2] - KB[..., 0]).clamp(min=1) / u,
                           (KB[..., 3] - KB[..., 1]).clamp(min=1) / u), -1)
        geo = torch.where(mask[..., None], geo, torch.ones_like(geo))
        # box features now: graph replays may reuse the dense outputs' memory
        return {"KB": KB, "KS": KS, "geo": geo, "mask": mask, "E": self.tail.pooled(emb, KB), "nk": nk_c.tolist()}

    def _decode_cpu(self, heat, box, ruby, emb, hw):
        """Decoding of a small batch on the CPU with core.detect (the reference decoder of the ONNX model); same tensors out."""
        dev, B = self.dev, heat.shape[0]
        maps = torch.cat((heat, box, ruby), 1).cpu().numpy()                     # [B, 6, h8, w8]
        dets = []
        for r in range(B):
            h8, w8 = int(hw[r, 0]) // 8, int(hw[r, 1]) // 8; m = maps[r, :, :h8, :w8]
            dets.append(core.detect(m[0:1], m[1:5], m[5:6]))
        nk = [0 if d is None else len(d[0]) for d in dets]; N = max(1, max(nk))
        KB = np.zeros((B, N, 4), np.float32); KS = np.zeros((B, N), np.float32); geo = np.ones((B, N, 4), np.float32)
        for r, d in enumerate(dets):
            if d is not None:
                KB[r, :nk[r]], KS[r, :nk[r]], geo[r, :nk[r]] = d
        KB, KS, geo = (torch.from_numpy(a).to(dev) for a in (KB, KS, geo))
        mask = torch.arange(N, device=dev)[None] < torch.tensor(nk, device=dev)[:, None]
        return {"KB": KB, "KS": KS, "geo": geo, "mask": mask, "E": self.tail.pooled(emb, KB), "nk": nk}
