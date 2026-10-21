"""NVIDIA GPU backend (PyTorch). Each crop is zero-padded to a canvas (height and width rounded up on a fixed ladder of 16-px cell
counts, so few distinct sizes occur); crops of one call that share a canvas run together in batches of up to 64, the batch padded
to a power of two. The dense network (fp16, channels-last) is masked so that each crop's outputs equal its single-crop outputs.
Boxes are decoded on the CPU with the same code as the ONNX backend; the tail (classification + reading-order scores, fp32)
runs once per batch on the padded set of detected characters. Small batches run through CUDA graphs, captured the first time a
(batch, canvas) or (batch, characters) size is seen, or at start with warmup()."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .. import core
from ..model import Dense, Tail

LADDER = (1, 2, 3, 4, 6, 8, 11, 14, 20, 28, 40, 56, 80, 112, 160, 196)   # canvas side lengths in cells
BATCHES = (1, 2, 4, 8, 16, 32, 64)                                         # padded batch sizes
GRAPH_BATCH, GRAPH_CHARS, GRAPH_MAX = 8, 128, 160   # CUDA graphs for batches <= 8 and tails <= 128 characters, at most 160 graphs
                                                     # (each graph holds ~15 MB of device memory); the rest runs eagerly
COMMON = ((28, 8), (20, 11), (28, 11), (40, 6), (20, 14), (8, 28), (11, 20), (14, 20), (11, 28), (6, 40), (14, 14), (56, 4),
          (40, 4), (4, 56), (28, 6), (3, 80), (80, 3))   # canvases (cells) of the development set (Read v1.2), most frequent first


def canvas(h_px: int, w_px: int) -> tuple[int, int]:
    up = lambda n: core.CELL * next(v for v in LADDER if v >= n // core.CELL)
    return up(h_px), up(w_px)


class CudaBackend:
    def __init__(self, model_dir: Path, fp16: bool = True, device: str = "cuda", graphs: bool = True, cudnn_benchmark: bool = False):
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        w = load_file(str(model_dir / cfg["weights"]))
        n_out = w["tail.cls.weight"].shape[0]
        self.dev, self.fp16 = torch.device(device), fp16
        self.dt = torch.float16 if fp16 else torch.float32
        self.dense = Dense().eval(); self.dense.load_state_dict({k[6:]: v for k, v in w.items() if k.startswith("dense.")})
        self.tail = Tail(n_out).eval(); self.tail.load_state_dict({k[5:]: v for k, v in w.items() if k.startswith("tail.")})
        self.dense.to(self.dev).to(memory_format=torch.channels_last); self.tail.to(self.dev)
        if fp16:
            self.dense.half()
        self.out_str = core.load_vocab(model_dir / cfg["vocab"], cfg["n_cls"])
        torch.backends.cudnn.benchmark = cudnn_benchmark        # per-shape autotuning costs time and memory
        self.graphs, self._g = graphs, {}
        self._pool = torch.cuda.graph_pool_handle() if graphs else None   # one memory pool for all graphs: each graph's outputs are
                                                                          # read before the next replay, so captures may share memory

    # ---- CUDA graphs -------------------------------------------------------------------------------------------------------
    def _run(self, key, fn, inputs):
        g = self._g.get(key)
        if g is None:
            if not (self.graphs and inputs[0].shape[0] <= GRAPH_BATCH and len(self._g) < GRAPH_MAX
                    and (key[0] == "dense" or key[2] <= GRAPH_CHARS)):
                return fn(*inputs)
            st = [t.clone() for t in inputs]
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    fn(*st)
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                so = fn(*st)
            g = self._g[key] = (graph, st, so)
        for d, t in zip(g[1], inputs):
            d.copy_(t)
        g[0].replay()
        return g[2]

    def _dense(self, x, valid):
        return self._run(("dense", *x.shape), self.dense, (x, valid))

    def _tail_packed(self, e, geo, mask):
        """-> [B, N, 3 + N] float: class id, start, end, then the pair scores of the row."""
        ids, pair, st, en = self.tail(e, geo, mask)
        return torch.cat((ids.float()[..., None], st[..., None], en[..., None], pair), -1)

    @torch.inference_mode()
    def warmup(self, canvases=COMMON):
        """Capture the graphs and prepare cuDNN for the given canvases (cells; default: the 17 of the development set)."""
        for b in BATCHES:
            for h, w in canvases:
                x = torch.zeros(b, 3, h * core.CELL, w * core.CELL, device=self.dev, dtype=self.dt).contiguous(memory_format=torch.channels_last)
                self._dense(x, torch.full((b, 2), core.CELL, device=self.dev))
            for n in (8, 16, 32, 64, 128) if b <= GRAPH_BATCH else ():
                self._run(("tail", b, n), self._tail_packed, (torch.zeros(b, n, 64, device=self.dev), torch.ones(b, n, 4, device=self.dev),
                                                              torch.zeros(b, n, dtype=torch.bool, device=self.dev)))
        torch.cuda.synchronize()

    # ---- inference ---------------------------------------------------------------------------------------------------------
    def read_many(self, imgs: list) -> list[dict]:
        return self.infer(self.prepare(imgs))

    @staticmethod
    def prepare(imgs: list) -> list:
        """CPU part (can run in worker processes): list of (uint8 input array, pixel factors)."""
        return [core.preprocess(core.to_rgb(im)) for im in imgs]

    MERGE_OVERHEAD = 2048     # cost of one extra dense call, in padded cells (~3 ms): small groups merge into a larger warm canvas
    TAIL_BUDGET = 1 << 20     # rows x characters^2 per tail call (bounds the [rows, n, n, 64] pair tensor)

    @classmethod
    def batches(cls, shapes: list) -> list[tuple[list[int], tuple[int, int]]]:
        """Crops grouped by canvas; small groups merged into a common canvas when that saves dense calls (only into canvases of
        COMMON, which warmup() prepares), then cut into batches of at most 64."""
        groups = defaultdict(list)
        for i, (h, w) in enumerate(shapes):
            groups[canvas(h, w)].append(i)
        warm = {(h * core.CELL, w * core.CELL) for h, w in COMMON}
        cost = lambda hw, n: cls.MERGE_OVERHEAD + n * hw[0] * hw[1] / core.CELL ** 2
        G = [[hw, idx] for hw, idx in groups.items()]
        while True:
            best = None
            for a in range(len(G)):
                for b in range(a + 1, len(G)):
                    (ha, ia), (hb, ib) = G[a], G[b]; m = (max(ha[0], hb[0]), max(ha[1], hb[1]))
                    if m not in warm or len(ia) + len(ib) > BATCHES[-1]:
                        continue
                    gain = cost(ha, len(ia)) + cost(hb, len(ib)) - cost(m, len(ia) + len(ib))
                    if gain > 0 and (best is None or gain > best[0]):
                        best = (gain, a, b, m)
            if best is None:
                break
            _, a, b, m = best; G[a] = [m, G[a][1] + G[b][1]]; del G[b]
        return [(idx[k:k + BATCHES[-1]], hw) for hw, idx in G for k in range(0, len(idx), BATCHES[-1])]

    @torch.inference_mode()
    def infer(self, pre: list) -> list[dict]:
        out: list[dict | None] = [None] * len(pre)
        rows = []                                                    # (crop index, detections, pooled embeddings [m, 64])
        for idx, (H, W) in self.batches([x.shape[1:] for x, _ in pre]):
            nb = next(b for b in BATCHES if b >= len(idx))
            xb = np.zeros((nb, 3, H, W), np.uint8); vb = np.full((nb, 2), core.CELL, np.int64)
            for r, i in enumerate(idx):
                a = pre[i][0]; xb[r, :, :a.shape[1], :a.shape[2]] = a; vb[r] = a.shape[1:]
            x = torch.from_numpy(xb).to(self.dev).to(self.dt).contiguous(memory_format=torch.channels_last)
            heat, box, ruby, emb = self._dense(x, torch.from_numpy(vb).to(self.dev))
            maps = torch.cat((heat, box, ruby), 1)[:len(idx)].cpu().float().numpy()   # [B, 6, h, w]
            dets = []
            for r, i in enumerate(idx):
                h8, w8 = pre[i][0].shape[1] // 8, pre[i][0].shape[2] // 8; m = maps[r, :, :h8, :w8]
                dets.append(core.detect(m[0:1], m[1:5], m[5:6]))
            live = [r for r, d in enumerate(dets) if d is not None]
            for r, d in enumerate(dets):
                if d is None:
                    out[idx[r]] = {"text": "", "chars": []}
            if not live:
                continue
            n = max(len(dets[r][0]) for r in live); kb = np.zeros((len(live), n, 4), np.float32)
            for j, r in enumerate(live):
                kb[j, :len(dets[r][0])] = dets[r][0]
            e = self.tail.pooled(emb[live].float(), torch.from_numpy(kb).to(self.dev))     # before the next dense replay
            rows += [(idx[r], dets[r], e[j, :len(dets[r][0])]) for j, r in enumerate(live)]
        rows.sort(key=lambda t: len(t[1][0]))
        k = 0
        while k < len(rows):                                         # tail: rows of similar length together, bounded size
            j = k + 1
            while j < len(rows) and j - k < 512 and (j - k + 1) * len(rows[j][1][0]) ** 2 <= self.TAIL_BUDGET:
                j += 1
            part = rows[k:j]; k = j
            R, n = len(part), len(part[-1][1][0])
            if self.graphs and R <= GRAPH_BATCH and n <= GRAPH_CHARS:
                R, n = next(b for b in BATCHES if b >= R), max(8, 1 << (n - 1).bit_length())   # few tail graphs
            geo = np.ones((R, n, 4), np.float32); mask = np.zeros((R, n), bool)
            for r, (_, d, _) in enumerate(part):
                m = len(d[0]); geo[r, :m] = d[2]; mask[r, :m] = True
            E = torch.nn.utils.rnn.pad_sequence([e for _, _, e in part], batch_first=True)
            E = torch.nn.functional.pad(E, (0, 0, 0, n - E.shape[1], 0, R - E.shape[0]))
            packed = self._run(("tail", R, n), self._tail_packed, (E, torch.from_numpy(geo).to(self.dev), torch.from_numpy(mask).to(self.dev)))
            packed = packed[:len(part)].cpu().numpy()
            ids, st, en, pair = packed[..., 0].astype(np.int64), packed[..., 1], packed[..., 2], packed[..., 3:]
            for r, (i, d, _) in enumerate(part):
                m = len(d[0]); kb_k, sc_k, _ = d
                seq = core.chain(pair[r, :m, :m], st[r, :m], en[r, :m])
                out[i] = core.result(seq, ids[r, :m], kb_k, sc_k, pre[i][1], self.out_str)
        return out
