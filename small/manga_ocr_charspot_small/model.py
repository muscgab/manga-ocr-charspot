"""PyTorch definition (inference): dense network (CNN backbone + attention blocks with QK-Norm at stride 16 -> stride-8 heads) and the
tail (character classification + reading-order scores over the detected characters). Batch norms of the trained network are folded
into the convolutions. Widths and depths come from config.json ("arch")."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def dw(c, stride=1):
    return nn.Conv2d(c, c, 3, stride, 1, groups=c)


class SE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a, self.b = nn.Conv2d(c, c // 4, 1), nn.Conv2d(c // 4, c, 1)

    def forward(self, x):
        return x * F.hardsigmoid(self.b(F.relu(self.a(x.mean((2, 3), keepdim=True)))))


class Block(nn.Module):
    def __init__(self, c, se=False):
        super().__init__()
        self.dw, self.se = dw(c), SE(c) if se else None
        self.pw1, self.pw2 = nn.Conv2d(c, 2 * c, 1), nn.Conv2d(2 * c, c, 1)

    def forward(self, x):
        y = self.dw(x)
        if self.se is not None:
            y = self.se(y)
        return y + self.pw2(F.hardswish(self.pw1(y)))


class Down(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.dw, self.pw1, self.pw2 = dw(ci, 2), nn.Conv2d(ci, co, 1), nn.Conv2d(co, co, 1)

    def forward(self, x):
        return self.pw2(F.hardswish(self.pw1(self.dw(x))))


class Stem(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(3, c // 2, 3, 2, 1)
        self.b1, self.b2 = nn.Conv2d(c // 2, c // 4, 2), nn.Conv2d(c // 4, c // 2, 2)
        self.c2, self.c3 = nn.Conv2d(c, c // 2, 3, 2, 1), nn.Conv2d(c // 2, c, 1)

    def forward(self, x):
        x = F.relu(self.c1(x))
        b = F.relu(self.b2(F.pad(F.relu(self.b1(F.pad(x, (0, 1, 0, 1)))), (0, 1, 0, 1))))
        m = F.max_pool2d(F.pad(x, (0, 1, 0, 1)), 2, 1)
        return F.relu(self.c3(F.relu(self.c2(torch.cat((m, b), 1)))))


def stage(ci, co, n, down, se_first):
    mods = [Down(ci, co)] if down else []
    for k in range(n - len(mods)):
        mods.append(Block(co, se=(k % 2 == 0) and se_first))
    return nn.Sequential(*mods)


class Backbone(nn.Module):
    def __init__(self, c, n):
        super().__init__()
        self.stem = Stem(c[0])
        self.s1, self.s2, self.s3 = stage(c[0], c[0], n[0], False, True), stage(c[0], c[1], n[1], True, False), stage(c[1], c[2], n[2], True, True)


class Attn(nn.Module):
    """Pre-norm attention + MLP block; q and k are layer-normed per head (QK-Norm)."""

    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj, self.fc1, self.fc2 = nn.Linear(d, 3 * d), nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(2 * d, d)
        self.qn, self.kn = nn.LayerNorm(d // heads), nn.LayerNorm(d // heads)

    def forward(self, t, key_mask=None):
        B, N, D = t.shape; hd = D // self.h
        q, k, v = self.qkv(self.n1(t)).view(B, N, 3, self.h, hd).permute(2, 0, 3, 1, 4).unbind(0)
        a = (self.qn(q) * hd ** -0.5) @ self.kn(k).transpose(-1, -2)
        if key_mask is not None:
            a = a.masked_fill(~key_mask[:, None, None, :], -1e4)
        a = a.softmax(-1) @ v
        t = t + self.proj(a.transpose(1, 2).reshape(B, N, D))
        return t + self.fc2(F.gelu(self.fc1(self.n2(t))))


class Dense(nn.Module):
    def __init__(self, c=(108, 216, 432), n=(2, 3, 5), att=2, heads=6, emb=144):
        super().__init__()
        c1, c2, c3 = c
        self.bb = Backbone(c, n)
        self.cpe, self.att = dw(c3), Attn(c3, heads)                      # first attention block (names as in training)
        self.extra = nn.ModuleList(nn.ModuleList([dw(c3), Attn(c3, heads)]) for _ in range(att - 1))
        self.lat = nn.Conv2d(c3, c2, 1)
        self.mix_dw, self.mix_pw = dw(c2), nn.Conv2d(c2, c2, 1)
        self.heat, self.box, self.ruby, self.emb = nn.Conv2d(c2, 1, 1), nn.Conv2d(c2, 4, 1), nn.Conv2d(c2, 1, 1), nn.Conv2d(c2, emb, 1)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255)

    def attn_blocks(self):
        return [(self.cpe, self.att)] + [(e[0], e[1]) for e in self.extra]

    def forward(self, img, valid=None):
        """img [B, 3, H, W], raw 0..255. valid [B, 2] (height, width in px, multiples of 16) marks crops of different sizes
        zero-padded into one batch; every layer then sees exactly what it sees on the crop alone (zeros outside, SE means and
        attention over the crop only), so the outputs inside each crop equal its single-crop outputs."""
        if valid is not None:
            return self._masked(img, valid)
        x = (img - self.mean) / self.std
        f8 = self.bb.s2(self.bb.s1(self.bb.stem(x)))
        f16 = self.bb.s3(f8)
        B, D, h, w = f16.shape
        for cpe, att in self.attn_blocks():
            f16 = f16 + cpe(f16)
            f16 = att(f16.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, D, h, w)
        p8 = f8 + F.interpolate(self.lat(f16), scale_factor=2.0, mode="nearest")
        p8 = F.hardswish(self.mix_pw(self.mix_dw(p8)))
        return torch.sigmoid(self.heat(p8)), self.box(p8), torch.sigmoid(self.ruby(p8)), self.emb(p8)

    def _masked(self, img, valid):
        H, W = img.shape[-2:]
        M = {}
        for st in (1, 2, 4, 8, 16):
            ys = torch.arange(H // st, device=img.device); xs = torch.arange(W // st, device=img.device)
            M[st] = ((ys[None, :, None] < valid[:, 0, None, None] // st) & (xs[None, None, :] < valid[:, 1, None, None] // st))[:, None].to(img.dtype)

        def se(m, y, k):
            g = (y * M[k]).sum((2, 3), keepdim=True) / M[k].sum((2, 3), keepdim=True)
            return y * F.hardsigmoid(m.b(F.relu(m.a(g))))

        def run(mods, x, k):
            for m in mods:
                if isinstance(m, Down):
                    k *= 2; x = m.pw2(F.hardswish(m.pw1(m.dw(x)))) * M[k]
                else:
                    y = m.dw(x)
                    if m.se is not None:
                        y = se(m.se, y, k)
                    x = (y + m.pw2(F.hardswish(m.pw1(y)))) * M[k]
            return x, k

        st_ = self.bb.stem
        x = (img - self.mean) / self.std * M[1]
        x = F.relu(st_.c1(x)) * M[2]
        b = F.relu(st_.b1(F.pad(x, (0, 1, 0, 1)))) * M[2]
        b = F.relu(st_.b2(F.pad(b, (0, 1, 0, 1))))
        m = F.max_pool2d(F.pad(x, (0, 1, 0, 1)), 2, 1)
        x = F.relu(st_.c3(F.relu(st_.c2(torch.cat((m, b), 1) * M[2])))) * M[4]
        x, _ = run(self.bb.s1, x, 4)
        f8, _ = run(self.bb.s2, x, 4)
        f16, _ = run(self.bb.s3, f8, 8)
        B, D, h, w = f16.shape
        km = M[16].flatten(1) > 0
        for cpe, att in self.attn_blocks():
            f16 = (f16 + cpe(f16)) * M[16]
            f16 = att(f16.flatten(2).transpose(1, 2), km).transpose(1, 2).reshape(B, D, h, w) * M[16]
        p8 = (f8 + F.interpolate(self.lat(f16), scale_factor=2.0, mode="nearest")) * M[8]
        p8 = F.hardswish(self.mix_pw(self.mix_dw(p8)))
        return torch.sigmoid(self.heat(p8)), self.box(p8), torch.sigmoid(self.ruby(p8)), self.emb(p8) * M[8]


class Tail(nn.Module):
    def __init__(self, n_out: int, emb: int = 144, d: int = 192, layers: int = 4, heads: int = 12):
        super().__init__()
        self.emb_dim = emb
        self.pool = nn.Linear(4 * emb, emb)
        self.inp = nn.Linear(emb + 4, d)
        self.blocks = nn.ModuleList(Attn(d, heads) for _ in range(layers))
        self.cls = nn.Linear(emb + d, n_out)
        self.pa, self.pb, self.pc = nn.Linear(d, d), nn.Linear(d, d, bias=False), nn.Linear(4, d, bias=False)
        self.pout = nn.Linear(d, 1)
        self.start, self.end = nn.Linear(d, 1), nn.Linear(d, 1)
        self.register_buffer("qx", torch.tensor([0.25, 0.75, 0.25, 0.75]), persistent=False)   # sample points inside each box
        self.register_buffer("qy", torch.tensor([0.25, 0.25, 0.75, 0.75]), persistent=False)

    def pooled(self, emb, boxes):
        """emb [B, E, H8, W8], boxes [B, N, 4] (input px) -> [B, N, E]: bilinear samples at the 4 quarter points of each box."""
        B, N = boxes.shape[:2]; H8, W8 = emb.shape[-2:]
        px = boxes[..., 0:1] + self.qx * (boxes[..., 2:3] - boxes[..., 0:1]); py = boxes[..., 1:2] + self.qy * (boxes[..., 3:4] - boxes[..., 1:2])
        grid = torch.stack((px / (W8 * 8) * 2 - 1, py / (H8 * 8) * 2 - 1), -1)
        s = F.grid_sample(emb, grid, mode="bilinear", align_corners=False)            # [B, E, N, 4]
        return self.pool(s.permute(0, 2, 3, 1).reshape(B, N, -1))

    def forward(self, e, geo, mask):
        """Padded batch: e [B, N, E], geo [B, N, 4], mask [B, N] -> class ids [B, N], pair [B, N, N], start [B, N], end [B, N]."""
        h = self.inp(torch.cat((e, geo), -1))
        for blk in self.blocks:
            h = blk(h, mask)
        ids = self.cls(torch.cat((e, h), -1)).argmax(-1)
        dd = torch.cat((geo[:, None, :, :2] - geo[:, :, None, :2], torch.log(geo[:, None, :, 2:] / geo[:, :, None, 2:])), -1)
        pair = self.pout(F.relu(self.pa(h)[:, :, None] + self.pb(h)[:, None] + self.pc(dd)))[..., 0]
        return ids, pair, self.start(h)[..., 0], self.end(h)[..., 0]


def build(cfg: dict, n_out: int):
    a = cfg["arch"]
    return (Dense(tuple(a["c"]), tuple(a["n"]), a["att"], a["heads"], a["emb"]),
            Tail(n_out, a["emb"], a["d"], a["layers"], a["theads"]))
