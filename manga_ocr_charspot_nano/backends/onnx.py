"""CPU backend: ONNX Runtime fp32, one crop at a time (dense graph -> numpy decoding -> tail graph -> reading-order chain)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .. import core


def _session(path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class OnnxBackend:
    def __init__(self, model_dir: Path, threads: int | None = None):
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        threads = threads or min(8, os.cpu_count() or 1)
        self.dense = _session(model_dir / cfg["onnx"]["dense"], threads)
        self.tail = _session(model_dir / cfg["onnx"]["tail"], 1)     # the tail is tiny; a second multi-thread pool only competes for cores
        self.out_str = core.load_vocab(model_dir / cfg["vocab"], cfg["n_cls"])

    def read_one(self, img) -> dict:
        x, f = core.preprocess(core.to_rgb(img))
        heat, box, ruby, emb = self.dense.run(None, {"img": x[None].astype(np.float32)})
        d = core.detect(heat[0], box[0], ruby[0])
        if d is None:
            return {"text": "", "chars": []}
        kb, sc, geo = d
        ids, pair, st, en = self.tail.run(None, {"emb": emb, "kb": kb, "geo": geo})
        return core.result(core.chain(pair, st, en), ids, kb, sc, f, self.out_str)

    def read_many(self, imgs: list) -> list[dict]:
        return [self.read_one(im) for im in imgs]
