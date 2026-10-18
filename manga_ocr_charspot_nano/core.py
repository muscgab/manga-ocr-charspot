"""Backend-independent parts: preprocessing, box decoding / NMS, reading-order chain (numpy)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

CELL = 16          # input cell size in pixels; the dense heads run at stride 8
BUDGET = 196       # at most 196 cells per crop
HEAT_THR = 0.3     # character-centre heatmap threshold
NMS_IOU = 0.2      # box NMS IoU threshold (chosen on the development set)
RUBY_THR = 0.5     # boxes flagged as furigana above this are dropped
COMBOS = [a + b for a in "!?" for b in "!?"] + [a + b + c for a in "!?" for b in "!?" for c in "!?"]   # tate-chu-yoko !? glyphs


def load_vocab(path: Path, n_cls: int) -> list[str]:
    chars = json.loads(Path(path).read_text(encoding="utf-8"))["characters"]
    return chars + [""] * (n_cls - len(chars)) + COMBOS


def to_rgb(img) -> Image.Image:
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return img.convert("RGB")


def pad_grid(height: int, width: int, budget: int = BUDGET) -> tuple[float, int, int]:
    """Largest scale whose 14-px grid fits in `budget` cells (same search as manga-ocr-nar-small); cells are 16 px here."""
    lo, hi = 1e-4, 1000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if math.ceil(height * mid / 14) * math.ceil(width * mid / 14) <= budget:
            lo = mid
        else:
            hi = mid
    return lo * CELL / 14, math.ceil(height * lo / 14), math.ceil(width * lo / 14)


def preprocess(img: Image.Image) -> tuple[np.ndarray, tuple[float, float]]:
    """-> uint8 [3, H, W] (H and W multiples of 16), and (x, y) factors from input pixels back to crop pixels."""
    s, gh, gw = pad_grid(img.height, img.width)
    rw, rh = max(1, min(gw * CELL, round(img.width * s))), max(1, min(gh * CELL, round(img.height * s)))
    small = np.asarray(img.resize((rw, rh), Image.Resampling.BICUBIC), np.uint8)
    border = np.concatenate([small[0], small[-1], small[:, 0], small[:, -1]])
    arr = np.empty((gh * CELL, gw * CELL, 3), np.uint8)
    arr[:] = np.median(border, axis=0).astype(np.uint8)
    arr[:rh, :rw] = small
    return np.ascontiguousarray(arr.transpose(2, 0, 1)), (img.width / rw, img.height / rh)


def nms(b: np.ndarray, s: np.ndarray, thr: float = NMS_IOU) -> list[int]:
    """Greedy NMS by score; also drops boxes whose centre lies in the central half of a kept box. The pairwise suppression
    matrix is built at once; the greedy pass over it keeps exactly the boxes of the one-at-a-time version."""
    order = np.argsort(-s, kind="stable"); b = b[order]
    area = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    inter = (np.clip(np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]), 0, None)
             * np.clip(np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]), 0, None))
    hx, hy = (b[:, 2] - b[:, 0]) * 0.25, (b[:, 3] - b[:, 1]) * 0.25
    inside = (np.abs(cx[None, :] - cx[:, None]) <= hx[:, None]) & (np.abs(cy[None, :] - cy[:, None]) <= hy[:, None])
    sup = ~(inter / (area[:, None] + area[None, :] - inter + 1e-6) < thr) | inside       # sup[i, j]: kept i removes j
    alive = np.ones(len(b), bool); keep = []
    for i in range(len(b)):
        if alive[i]:
            keep.append(int(order[i])); alive &= ~sup[i]
    return keep


def detect(heat: np.ndarray, box: np.ndarray, ruby: np.ndarray):
    """Dense outputs of one crop ([1,h,w], [4,h,w], [1,h,w]) -> kept boxes [n,4] (input px), scores [n], geometry [n,4] for the tail."""
    hm = heat[0]; ci, cj = np.nonzero(hm > HEAT_THR)
    if not len(ci):
        return None
    bx = box[:, ci, cj]
    cx, cy = (cj + bx[0]) * 8, (ci + bx[1]) * 8; w, h = np.exp(bx[2]) * 8, np.exp(bx[3]) * 8
    b = np.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), 1).astype(np.float32)
    keep = [k for k in nms(b, hm[ci, cj]) if ruby[0, ci[k], cj[k]] < RUBY_THR]
    if not keep:
        return None
    kb = b[keep]; sc = hm[ci, cj][keep]
    sz = np.maximum(kb[:, 2] - kb[:, 0], kb[:, 3] - kb[:, 1]); u = float(np.sort(sz)[(len(sz) - 1) // 2])
    geo = np.stack(((kb[:, 0] + kb[:, 2]) / 2 / u, (kb[:, 1] + kb[:, 3]) / 2 / u,
                    np.clip(kb[:, 2] - kb[:, 0], 1, None) / u, np.clip(kb[:, 3] - kb[:, 1], 1, None) / u), 1).astype(np.float32)
    return kb, sc, geo


def chain(pair: np.ndarray, st: np.ndarray, en: np.ndarray) -> list[int]:
    """Greedy reading order: start at the best start score, follow the best unused successor, restart when 'end' wins."""
    n = len(st); used = np.zeros(n, bool); cur = int(st.argmax()); seq = [cur]; used[cur] = True
    while not used.all():
        sc = np.where(used, -np.inf, pair[cur]); j = int(np.argmax(sc))
        if en[cur] > sc[j]:
            rest = np.where(~used)[0]; j = int(rest[np.argmax(st[rest])])
        seq.append(j); used[j] = True; cur = j
    return seq


def result(seq, ids, kb, sc, fx_fy, out_str) -> dict:
    fx, fy = fx_fy; chars = []
    for k in seq:
        x0, y0, x1, y1 = (float(v) for v in kb[k])
        chars.append({"char": out_str[int(ids[k])], "box": [round(x0 * fx, 1), round(y0 * fy, 1), round(x1 * fx, 1), round(y1 * fy, 1)],
                      "score": round(float(sc[k]), 4)})
    return {"text": "".join(c["char"] for c in chars), "chars": chars}
