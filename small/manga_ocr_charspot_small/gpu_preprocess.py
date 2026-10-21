"""GPU preprocessing for the CUDA backend: the same pixels as core.preprocess (Pillow BICUBIC re-implemented with Pillow's 8-bit fixed-point
arithmetic and int32 accumulators, horizontal pass first, then the median-border canvas). Only the scalar geometry and the small
resize coefficient tables are computed on the CPU. Adapted from manga-ocr-nar-small's gpu_preprocess.py."""
from __future__ import annotations

import math

import numpy as np
import torch

from . import core

PRECISION_BITS = 22


def _bicubic(x):
    a = -0.5
    x = np.abs(x)
    return np.where(x < 1.0, ((a + 2.0) * x - (a + 3.0)) * x * x + 1,
                    np.where(x < 2.0, (((x - 5) * x + 8) * x - 4) * a, 0.0))


def pil_coeffs(in_size: int, out_size: int):
    """-> xmin [out] int64, kk [out, ksize] int64 (fixed point), ksize; exactly Pillow's precompute_coeffs + normalize_coeffs_8bpc."""
    scale = float(in_size) / out_size
    filterscale = max(scale, 1.0)
    support = 2.0 * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    xx = np.arange(out_size, dtype=np.float64)
    center = (xx + 0.5) * scale
    ss = 1.0 / filterscale
    xmin = np.maximum(np.trunc(center - support + 0.5).astype(np.int64), 0)          # C (int) cast truncates toward zero
    xmax = np.minimum(np.trunc(center + support + 0.5).astype(np.int64), in_size) - xmin
    x = np.arange(ksize, dtype=np.float64)[None, :]
    w = _bicubic((x + xmin[:, None] - center[:, None] + 0.5) * ss)
    w = np.where(x < xmax[:, None], w, 0.0)
    ww = np.zeros(out_size)                       # Pillow sums the weights sequentially; numpy's pairwise sum can differ in the last bit
    for j in range(ksize):
        ww = ww + w[:, j]
    w = np.where(ww[:, None] != 0.0, w / np.where(ww == 0.0, 1.0, ww)[:, None], w)
    k = np.where(w < 0, np.trunc(-0.5 + w * (1 << PRECISION_BITS)), np.trunc(0.5 + w * (1 << PRECISION_BITS))).astype(np.int64)
    return xmin, k, ksize


class GpuPreprocess:
    CHUNK = 1 << 24                                       # gathered int32 elements per resize chunk (64 MB)

    def __init__(self, budget: int = core.BUDGET, device="cuda"):
        self.budget, self.dev = budget, torch.device(device)
        self._coef = {}

    def _pass(self, x: torch.Tensor, in_size: int, out_size: int, dim: int) -> torch.Tensor:
        """one resample pass along dim (0 = rows / vertical, 1 = columns / horizontal) of a uint8 HWC tensor"""
        key = (in_size, out_size)
        if key not in self._coef:
            xmin, k, ksize = pil_coeffs(in_size, out_size)
            src = np.minimum(xmin[:, None] + np.arange(ksize)[None, :], in_size - 1)                      # dead taps have k == 0
            self._coef[key] = (torch.from_numpy(src).to(self.dev), torch.from_numpy(k.astype(np.int32)).to(self.dev))
            if len(self._coef) > 4096:
                self._coef.pop(next(iter(self._coef)))
        src, k = self._coef[key]
        xi = x.to(torch.int32)
        if dim == 1:                                                          # [H, W, 3] -> [H, out, 3], chunked over rows
            step = max(1, self.CHUNK // (src.numel() * 3))
            acc = torch.cat([(xi[r:r + step][:, src] * k[None, :, :, None]).sum(2) for r in range(0, xi.shape[0], step)], 0)
        else:                                                                 # [H, W, 3] -> [out, W, 3], chunked over output rows
            step = max(1, self.CHUNK // (src.shape[1] * xi.shape[1] * 3))
            acc = torch.cat([(xi[src[o:o + step]] * k[o:o + step, :, None, None]).sum(1) for o in range(0, src.shape[0], step)], 0)
        acc = acc + (1 << (PRECISION_BITS - 1))
        return torch.clamp(acc >> PRECISION_BITS, 0, 255).to(torch.uint8)

    @torch.inference_mode()
    def __call__(self, arr: np.ndarray):
        """uint8 HWC RGB array -> (uint8 [3, H, W] on the GPU, H and W multiples of 16; (x, y) factors from crop px back to input px)."""
        arr = np.ascontiguousarray(arr)
        x = torch.from_numpy(arr if arr.flags.writeable else arr.copy()).to(self.dev, non_blocking=True)   # arrays from PIL are read-only
        H, W = int(x.shape[0]), int(x.shape[1])
        s, gh, gw = core.pad_grid(H, W, self.budget)
        rw, rh = max(1, min(gw * core.CELL, round(W * s))), max(1, min(gh * core.CELL, round(H * s)))
        small = x
        if rw != W:
            small = self._pass(small, W, rw, 1)
        if rh != H:
            small = self._pass(small, H, rh, 0)
        border = torch.cat([small[0], small[-1], small[:, 0], small[:, -1]], 0)        # [n, 3], as core.preprocess
        srt = border.sort(0).values; n = srt.shape[0]
        med = srt[n // 2] if n % 2 else ((srt[n // 2 - 1].double() + srt[n // 2].double()) / 2).to(torch.uint8)
        out = med.view(3, 1, 1).expand(3, gh * core.CELL, gw * core.CELL).clone()
        out[:, :rh, :rw] = small.permute(2, 0, 1)
        return out, (W / rw, H / rh)
