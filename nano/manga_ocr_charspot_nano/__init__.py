"""manga-ocr-charspot-nano: Japanese manga text-box OCR by character detection + learned reading order (1.7M parameters, no decoder).

    from manga_ocr_charspot_nano import MangaOcrCharspotNano
    ocr = MangaOcrCharspotNano()                  # backend "auto" | "onnx" (CPU) | "cuda"
    ocr("crop.png")                       # one crop -> str
    ocr([crop1, crop2, ...])              # several crops -> list[str]
    ocr.read("crop.png")                  # -> {"text": str, "chars": [{"char", "box": [x0, y0, x1, y1], "score"}, ...]} in reading order

Crops may be file paths, PIL images or uint8 HWC RGB arrays. Boxes are in pixels of the given crop."""
from __future__ import annotations

import os
from pathlib import Path

__version__ = "1.1.1"
__all__ = ["MangaOcrCharspotNano"]
DEFAULT_DIR = Path(os.environ.get("MANGA_OCR_CHARSPOT_NANO_DIR") or Path(__file__).resolve().parent.parent)


def _auto() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "onnx"


class MangaOcrCharspotNano:
    def __init__(self, model_dir: str | Path | None = None, backend: str = "auto", threads: int | None = None):
        model_dir = Path(model_dir or DEFAULT_DIR)
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(f"no config.json in {model_dir}: install from a checkout with `pip install -e .`, "
                                    "or pass model_dir / set MANGA_OCR_CHARSPOT_NANO_DIR to the directory holding the model files")
        self.backend = _auto() if backend == "auto" else backend
        if self.backend == "onnx":
            from .backends.onnx import OnnxBackend
            self._b = OnnxBackend(model_dir, threads)
        elif self.backend == "cuda":
            from .backends.cuda import CudaBackend
            self._b = CudaBackend(model_dir)
        else:
            raise ValueError(f"unknown backend {backend!r}")

    def warmup(self):
        """CUDA backend: capture the CUDA graphs for the common input sizes once (about 11 s); no-op for ONNX."""
        if hasattr(self._b, "warmup"):
            self._b.warmup()
        return self

    def read(self, crops):
        """One crop -> dict; a list of crops -> list of dicts."""
        if isinstance(crops, (list, tuple)):
            return self._b.read_many(list(crops))
        return self._b.read_many([crops])[0]

    def __call__(self, crops):
        r = self.read(crops)
        return [x["text"] for x in r] if isinstance(r, list) else r["text"]
