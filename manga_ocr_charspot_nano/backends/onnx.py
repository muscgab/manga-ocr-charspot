"""CPU backend: ONNX Runtime fp32, one crop at a time. model.onnx holds the whole model after preprocessing (dense network, box
decoding and NMS, furigana removal, tail, reading-order chain), so one session call per crop."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .. import core


class OnnxBackend:
    def __init__(self, model_dir: Path, threads: int | None = None):
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        so = ort.SessionOptions(); so.intra_op_num_threads = threads or min(8, os.cpu_count() or 1); so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(model_dir / cfg["onnx"]), so, providers=["CPUExecutionProvider"])
        self.out_str = core.load_vocab(model_dir / cfg["vocab"], cfg["n_cls"])

    def read_one(self, img) -> dict:
        x, f = core.preprocess(core.to_rgb(img))
        boxes, scores, ids, order = self.sess.run(None, {"img": x[None].astype(np.float32)})
        if not len(order):
            return {"text": "", "chars": []}
        return core.result(order.tolist(), ids, boxes, scores, f, self.out_str)

    def read_many(self, imgs: list) -> list[dict]:
        return [self.read_one(im) for im in imgs]
