"""manga-ocr-charspot-small [--backend auto|onnx|cuda] [--model-dir DIR] [--chars] crop.png ...  -> one "path<TAB>text" line per crop;
with --chars, one JSON line per crop with the per-character boxes and scores."""
from __future__ import annotations

import argparse
import json
import sys

from . import MangaOcrCharspotSmall


def main(argv=None):
    ap = argparse.ArgumentParser(prog="manga-ocr-charspot-small")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--backend", default="auto", choices=("auto", "onnx", "cuda"))
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--chars", action="store_true")
    a = ap.parse_args(argv)
    ocr = MangaOcrCharspotSmall(a.model_dir, backend=a.backend)
    out = sys.stdout
    if hasattr(out, "reconfigure"):
        out.reconfigure(encoding="utf-8")
    for path, r in zip(a.images, ocr.read(list(a.images))):
        out.write(json.dumps({"path": path, **r}, ensure_ascii=False) + "\n" if a.chars else f"{path}\t{r['text']}\n")


if __name__ == "__main__":
    main()
