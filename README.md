# manga-ocr-charspot

English | [中文](README_zh.md)

OCR for Japanese manga text boxes by character detection. The model detects every character in a cropped text box, classifies it, and puts the characters in reading order with a small learned ordering head; there is no text decoder. Besides the text, it returns a box and a score for each character.

This repository holds the code and model cards of the two released models (weights are on Hugging Face and ModelScope):

| | [manga-ocr-charspot-small](small) | [manga-ocr-charspot-nano](nano) |
|---|---|---|
| Parameters | 11M | 1.7M |
| Read v1.3 Composite ↑ | **97.17** | 95.75 |
| Read v1.3 CER % ↓ | **1.37** | 2.14 |
| Read v1.3 EM % ↑ | **92.37** | 89.40 |
| EM printed / non-printed % ↑ | **96.49** / **79.71** | 95.59 / 70.35 |
| EM horizontal % ↑ (439 boxes) | **75.40** | 65.83 |
| CPU per crop, 8 / 4 threads ↓ | 19.5 / 27.0 ms | **7.0** / **7.9** ms |
| GPU per image / per page ↓ | 7.2 / 43.2 ms | **5.2** / **28.3** ms |
| GPU max img/s ↑ | 274 | **633** |
| GPU precision | fp32 | fp16 |
| Weights | [Hugging Face](https://huggingface.co/muscgab/manga-ocr-charspot-small) · [ModelScope](https://modelscope.cn/models/muscgab/manga-ocr-charspot-small) | [Hugging Face](https://huggingface.co/muscgab/manga-ocr-charspot-nano) · [ModelScope](https://modelscope.cn/models/muscgab/manga-ocr-charspot-nano) |
| Model card | [small/README.md](small/README.md) | [nano/README.md](nano/README.md) · [technical report](nano/TECHNICAL_REPORT.md) |

small is the more accurate model; nano is the smaller and faster one. Both share the same design, package interface and output format.

- **Accuracy**: [JMangaBench_Read v1.3.2](https://github.com/muscgab/JMangaBench_Read), 7,789 boxes, official scorer (RIN v1.2). Read v1.2 (3,142 boxes, part of v1.3) was the development set of both models. Comparisons with other models and Manga109-s results are in each model card.
- **CPU**: Xeon Gold 6148, ONNX Runtime fp32, batch 1, p50 over 1,000 Read v1.3 boxes drawn at random with seed 0 (the same boxes for every model in the model cards).
- **GPU**: RTX 2080 Ti with the Read leaderboard's speed protocol (per image = batch 1, mean over 7,789 boxes; per page = 896 batches sized like real pages, p50; max img/s = best batch size with preprocessing in 4 CPU worker processes).

## Usage

Each model is its own package, installed from its Hugging Face (or ModelScope) repository:

```bash
git clone https://huggingface.co/muscgab/manga-ocr-charspot-small   # or manga-ocr-charspot-nano
cd manga-ocr-charspot-small
pip install -e ".[cpu]"   # ONNX Runtime (CPU); ".[cuda]" for the PyTorch GPU backend
manga-ocr-charspot-small crop.png            # one "path<TAB>text" line per crop
manga-ocr-charspot-small --chars crop.png    # one JSON line per crop with per-character boxes and scores
```

```python
from manga_ocr_charspot_small import MangaOcrCharspotSmall   # or: from manga_ocr_charspot_nano import MangaOcrCharspotNano

ocr = MangaOcrCharspotSmall()           # backend "auto" (CUDA if available, else ONNX on CPU), "onnx" or "cuda"
ocr("crop.png")                         # -> str
ocr.read("crop.png")                    # -> {"text": ..., "chars": [{"char": "今", "box": [x0, y0, x1, y1], "score": 0.93}, ...]}
```

Each package also ships `model.onnx`, the whole model after preprocessing in one graph, usable from other languages and runtimes (see the model cards).

## Limitations

Both models are weaker than decoder-based OCR on rough handwriting and on horizontal text (a left–right kanji can be detected as two characters). Details and numbers are in the Limitations section of each model card.

## License

Code: Apache-2.0 ([LICENSE](LICENSE)). Weights: CC BY-NC-SA 4.0 ([LICENSE_MODEL](LICENSE_MODEL); trained on AnimeText-OCR). Data sources and attribution: [small/NOTICE.md](small/NOTICE.md), [nano/NOTICE.md](nano/NOTICE.md).
