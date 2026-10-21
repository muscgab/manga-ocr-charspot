# manga-ocr-charspot

[English](README.md) | 中文

基于逐字检测的日文漫画文本框 OCR。模型在裁好的文本框里检测每一个字、逐字分类，再用一个小的阅读顺序模块把字排成文本；没有文本解码器。除了文字，还输出每个字的框和分数。

本仓库放两个已发布模型的代码和模型卡（权重在 Hugging Face 和 ModelScope）：

| | [manga-ocr-charspot-small](small) | [manga-ocr-charspot-nano](nano) |
|---|---|---|
| 参数量 | 11M | 1.7M |
| Read v1.3 Composite ↑ | **97.17** | 95.75 |
| Read v1.3 CER % ↓ | **1.37** | 2.14 |
| Read v1.3 EM % ↑ | **92.37** | 89.40 |
| EM 印刷 / 非印刷 % ↑ | **96.49** / **79.71** | 95.59 / 70.35 |
| EM 横排 % ↑（439 框） | **75.40** | 65.83 |
| CPU 逐张，8 / 4 线程 ↓ | 19.5 / 27.0 ms | **7.0** / **7.9** ms |
| GPU 逐张 / 逐页 ↓ | 7.2 / 43.2 ms | **5.2** / **28.3** ms |
| GPU 最大吞吐（张/秒）↑ | 274 | **633** |
| GPU 精度 | fp32 | fp16 |
| 权重 | [Hugging Face](https://huggingface.co/muscgab/manga-ocr-charspot-small) · [ModelScope](https://modelscope.cn/models/muscgab/manga-ocr-charspot-small) | [Hugging Face](https://huggingface.co/muscgab/manga-ocr-charspot-nano) · [ModelScope](https://modelscope.cn/models/muscgab/manga-ocr-charspot-nano) |
| 模型卡 | [small/README_zh.md](small/README_zh.md) | [nano/README_zh.md](nano/README_zh.md) · [技术报告](nano/TECHNICAL_REPORT_zh.md) |

small 更准；nano 更小更快。两者设计、包接口和输出格式相同。

- **精度**：[JMangaBench_Read v1.3.2](https://github.com/muscgab/JMangaBench_Read)，7,789 框，官方打分脚本（RIN v1.2）。Read v1.2（3,142 框，属于 v1.3）是两个模型的开发集。与其他模型的对比和 Manga109-s 成绩见各自的模型卡。
- **CPU**：Xeon Gold 6148，ONNX Runtime fp32，batch 1，从 Read v1.3 随机抽 1,000 框（种子 0）的逐张中位数（模型卡里所有模型用的是同一批框）。
- **GPU**：RTX 2080 Ti，按 Read 排行榜的测速方法（逐张 = batch 1，7,789 框的平均；逐页 = 按真实页面框数分成的 896 批的 p50；最大吞吐 = 最佳 batch，预处理在 4 个 CPU 工作进程里做）。

## 用法

每个模型是独立的包，从各自的 Hugging Face（或 ModelScope）仓库安装：

```bash
git clone https://huggingface.co/muscgab/manga-ocr-charspot-small   # 或 manga-ocr-charspot-nano
cd manga-ocr-charspot-small
pip install -e ".[cpu]"   # ONNX Runtime（CPU）；GPU 用 ".[cuda]"（PyTorch 后端）
manga-ocr-charspot-small crop.png            # 每个裁图输出一行 "路径<TAB>文字"
manga-ocr-charspot-small --chars crop.png    # 每个裁图输出一行 JSON，含逐字的框和分数
```

```python
from manga_ocr_charspot_small import MangaOcrCharspotSmall   # 或：from manga_ocr_charspot_nano import MangaOcrCharspotNano

ocr = MangaOcrCharspotSmall()           # 后端 "auto"（有 CUDA 用 CUDA，否则用 CPU 上的 ONNX）、"onnx" 或 "cuda"
ocr("crop.png")                         # -> str
ocr.read("crop.png")                    # -> {"text": ..., "chars": [{"char": "今", "box": [x0, y0, x1, y1], "score": 0.93}, ...]}
```

每个包还附带 `model.onnx`：预处理之后的整个模型在一张图里，其他语言和运行时也能直接用（见模型卡）。

## 局限

两个模型在潦草手写和横排文字上都弱于带解码器的 OCR（横排时一个左右结构的汉字可能被检测成两个字）。具体说明和数字见各模型卡的“局限”一节。

## 许可

代码：Apache-2.0（[LICENSE](LICENSE)）。权重：CC BY-NC-SA 4.0（[LICENSE_MODEL](LICENSE_MODEL)；训练用到 AnimeText-OCR）。数据来源与署名：[small/NOTICE.md](small/NOTICE.md)、[nano/NOTICE.md](nano/NOTICE.md)。
