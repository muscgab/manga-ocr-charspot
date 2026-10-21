"""PyTorch definition (inference): dense network (CNN backbone + one attention block -> stride-8 heads) and the tail
(character classification + reading-order scores over the detected characters)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
EMB, ORD = 64, 64


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
    def __init__(self):
        super().__init__()
        self.stem = Stem(48)
        self.s1, self.s2, self.s3 = stage(48, 48, 2, False, True), stage(48, 96, 3, True, False), stage(96, 192, 5, True, True)


class Attn(nn.Module):
    def __init__(self, d=192, heads=3):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj, self.fc1, self.fc2 = nn.Linear(d, 3 * d), nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(2 * d, d)

    def forward(self, t, key_mask=None):
        B, N, D = t.shape; hd = D // self.h
        q, k, v = self.qkv(self.n1(t)).view(B, N, 3, self.h, hd).permute(2, 0, 3, 1, 4).unbind(0)
        a = (q * hd ** -0.5) @ k.transpose(-1, -2)
        if key_mask is not None:
            a = a.masked_fill(~key_mask[:, None, None, :], -1e4)
        a = a.softmax(-1) @ v
        t = t + self.proj(a.transpose(1, 2).reshape(B, N, D))
        return t + self.fc2(F.gelu(self.fc1(self.n2(t))))


class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = Backbone()
        self.cpe, self.att = dw(192), Attn(192, 3)
        self.lat = nn.Conv2d(192, 96, 1)
        self.mix_dw, self.mix_pw = dw(96), nn.Conv2d(96, 96, 1)
        self.heat, self.box, self.ruby, self.emb = nn.Conv2d(96, 1, 1), nn.Conv2d(96, 4, 1), nn.Conv2d(96, 1, 1), nn.Conv2d(96, EMB, 1)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255)

    def forward(self, img, valid=None):
        """img [B, 3, H, W], raw 0..255. valid [B, 2] (height, width in px, multiples of 16) marks crops of different sizes
        zero-padded into one batch; every layer then sees exactly what it sees on the crop alone (zeros outside, SE means and
        attention over the crop only), so the outputs inside each crop equal its single-crop outputs."""
        if valid is not None:
            return self._masked(img, valid)
        x = (img - self.mean) / self.std
        f8 = self.bb.s2(self.bb.s1(self.bb.stem(x)))
        f16 = self.bb.s3(f8)
        f16 = f16 + self.cpe(f16)
        B, D, h, w = f16.shape
        f16 = self.att(f16.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, D, h, w)
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
        f16 = (f16 + self.cpe(f16)) * M[16]
        B, D, h, w = f16.shape
        f16 = self.att(f16.flatten(2).transpose(1, 2), M[16].flatten(1) > 0).transpose(1, 2).reshape(B, D, h, w) * M[16]
        p8 = (f8 + F.interpolate(self.lat(f16), scale_factor=2.0, mode="nearest")) * M[8]
        p8 = F.hardswish(self.mix_pw(self.mix_dw(p8)))
        return torch.sigmoid(self.heat(p8)), self.box(p8), torch.sigmoid(self.ruby(p8)), self.emb(p8) * M[8]


class OrderBlock(nn.Module):
    def __init__(self, d=ORD, heads=4):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 2 * d), nn.Linear(2 * d, d)


class Tail(nn.Module):
    def __init__(self, n_out: int, layers: int = 2):
        super().__init__()
        self.pool = nn.Linear(4 * EMB, EMB)
        self.inp = nn.Linear(EMB + 4, ORD)
        self.blocks = nn.ModuleList(OrderBlock() for _ in range(layers))
        self.cproj = nn.Linear(EMB + ORD, EMB)
        self.cls = nn.Linear(EMB, n_out)
        self.pa, self.pb, self.pc = nn.Linear(ORD, ORD), nn.Linear(ORD, ORD, bias=False), nn.Linear(4, ORD, bias=False)
        self.pout = nn.Linear(ORD, 1)
        self.start, self.end = nn.Linear(ORD, 1), nn.Linear(ORD, 1)
        self.register_buffer("qx", torch.tensor([0.25, 0.75, 0.25, 0.75]), persistent=False)   # sample points inside each box
        self.register_buffer("qy", torch.tensor([0.25, 0.25, 0.75, 0.75]), persistent=False)

    def pooled(self, emb, boxes):
        """emb [B, 64, H8, W8], boxes [B, N, 4] (input px) -> [B, N, 64]: bilinear samples at the 4 quarter points of each box."""
        B, N = boxes.shape[:2]; H8, W8 = emb.shape[-2:]
        px = boxes[..., 0:1] + self.qx * (boxes[..., 2:3] - boxes[..., 0:1]); py = boxes[..., 1:2] + self.qy * (boxes[..., 3:4] - boxes[..., 1:2])
        grid = torch.stack((px / (W8 * 8) * 2 - 1, py / (H8 * 8) * 2 - 1), -1)
        s = F.grid_sample(emb, grid, mode="bilinear", align_corners=False)            # [B, 64, N, 4]
        return self.pool(s.permute(0, 2, 3, 1).reshape(B, N, -1))

    def forward(self, e, geo, mask):
        """Padded batch: e [B, N, 64], geo [B, N, 4], mask [B, N] -> class ids [B, N], pair [B, N, N], start [B, N], end [B, N]."""
        h = self.inp(torch.cat((e, geo), -1)); B, N, d = h.shape
        for blk in self.blocks:
            q, k, v = blk.qkv(blk.n1(h)).view(B, N, 3, blk.h, d // blk.h).permute(2, 0, 3, 1, 4).unbind(0)
            a = ((q * (d // blk.h) ** -0.5) @ k.transpose(-1, -2)).masked_fill(~mask[:, None, None, :], -1e4).softmax(-1) @ v
            h = h + blk.proj(a.transpose(1, 2).reshape(B, N, d))
            h = h + blk.fc2(F.gelu(blk.fc1(blk.n2(h))))
        ids = self.cls(self.cproj(torch.cat((e, h), -1))).argmax(-1)
        dd = torch.cat((geo[:, None, :, :2] - geo[:, :, None, :2], torch.log(geo[:, None, :, 2:] / geo[:, :, None, 2:])), -1)
        pair = self.pout(F.relu(self.pa(h)[:, :, None] + self.pb(h)[:, None] + self.pc(dd)))[..., 0]
        return ids, pair, self.start(h)[..., 0], self.end(h)[..., 0]
