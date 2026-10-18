# manga-ocr-charspot-nano

[English](README.md) | 中文 · [GitHub](https://github.com/muscgab/manga-ocr-charspot) · [Hugging Face](https://huggingface.co/muscgab/manga-ocr-charspot-nano) · [ModelScope](https://modelscope.cn/models/muscgab/manga-ocr-charspot-nano) · [技术报告](TECHNICAL_REPORT_zh.md)

日文漫画文本框 OCR，1.7M 参数。模型在裁好的文本框里检测每一个字、逐字分类，再用一个小的阅读顺序模块把字排成文本。没有文本解码器。除了文字，还输出每个字的框和分数。

<img src="samples/char_boxes.png" width="100%">

模型在三个 Read v1.3 框上的输出：红框是逐字框，蓝色数字是阅读顺序。裁图来自 [JMangaBench_Read](https://github.com/muscgab/JMangaBench_Read)（CC BY-NC-SA 4.0），从模型读对的框中挑选。

## 精度

### JMangaBench_Read v1.3（同领域；未用于训练和选模型）

[JMangaBench_Read v1.3.1](https://github.com/muscgab/JMangaBench_Read)，7,789 框，官方打分脚本（RIN v1.1）。"新增" = v1.3 新加的 4,660 框；Read v1.2 是开发集（见[评测说明](#评测说明)）。

| 模型 | 参数 | 综合分 ↑ | CER % ↓ | EM % ↑ | EM 新增 % ↑ | EM 印刷 % ↑ | EM 非印刷 % ↑ |
|---|---|---|---|---|---|---|---|
| **manga-ocr-charspot-nano** | **1.7M** | 95.74 | 2.14 | 89.38 | 89.55 | **95.59** | 70.29 |
| manga-ocr-nar-small | 54M | 97.29 | 1.71 | 91.36 | 91.65 | 94.74 | 80.96 |
| manga-ocr-nar-small-distil | 30M | 96.40 | 2.29 | 89.09 | 88.95 | 93.21 | 76.41 |
| Hayai v2.5 Nova p384 | 157M | 96.64 | 2.38 | 88.18 | 88.41 | 91.93 | 76.62 |
| PaddleOCR-VL-For-Manga | 959M | 95.04 | 1.99 | 87.89 | 87.92 | 95.17 | 65.53 |
| mangaOCR | 111M | 90.27 | 4.81 | 78.51 | 78.48 | 88.34 | 48.27 |

EM：归一化后预测与标签完全一致的框所占比例；CER：总编辑距离除以标签总字数。综合分（Composite）是 Read 的排名指标：1 − 归一化编辑距离，印刷与非印刷各占一半。本模型非印刷较弱，所以综合分低于 Nova，而 EM 高于 Nova。

### Manga109-s（不同领域；未用于训练）

Manga109-s 全部 123,014 个文本标注（商业漫画扫描，87 册），用数据集自带的框裁图；标签和预测都用 RIN v1.1 归一化，EM、CER 定义同上。在 Manga109-s 上训练过的模型（Hayai、mangaOCR）不列入。baberuOCR v1.1（训练数据未公开）用 Read 排行榜的生成设置；Preview 即 [manga-ocr-nar-preview](https://huggingface.co/muscgab/manga-ocr-nar-preview)。

| 模型 | 参数 | EM % ↑ | CER % ↓ | EM 1–4 字 | 5–16 | 17–32 | > 32 |
|---|---|---|---|---|---|---|---|
| **manga-ocr-charspot-nano** | **1.7M** | 89.10 | 1.98 | 94.27 | 90.37 | 84.46 | 60.21 |
| manga-ocr-nar-small | 54M | 92.08 | 1.39 | 95.92 | 92.23 | 90.11 | 74.39 |
| baberuOCR v1.1 | 113M | 86.68 | 2.51 | 94.73 | 87.92 | 79.75 | 54.40 |
| Preview | 196M | 86.51 | 2.46 | 91.74 | 87.30 | 83.22 | 55.59 |

各长度档框数：26,336 / 65,916 / 27,580 / 3,182。

## 速度

每个裁图端到端计时（图像已解码；含预处理、网络、解码和阅读顺序），batch 1，Read v1.3 全部 7,789 框的中位数。

**CPU**（ONNX Runtime fp32，8 线程）：

| 模型 | 格式 | Xeon Gold 6148 | M1 Pro |
|---|---|---|---|
| **manga-ocr-charspot-nano** | ONNX fp32 | 8.2 ms | 6.2 ms * |
| manga-ocr-nar-small-distil | ONNX int8 | 37.3 ms | 21.0 ms |
| manga-ocr-nar-small | ONNX fp32 | 58.2 ms | — |

\* 在有负载的笔记本上测得（load average 4–13）；Xeon 测时空闲。

**NVIDIA GPU**（RTX 2080 Ti，Read 的测速口径：单张 = batch 1；按页 = 每次一页的框，取中位数；最大吞吐 = 最佳 batch，CPU 预处理放在子进程）：

| 模型 | 单张 ↓ | 按页 ↓ | 最大吞吐（张/秒）↑ |
|---|---|---|---|
| **manga-ocr-charspot-nano**（CUDA graph，fp16） | 4.7 ms | 28.6 ms | 648 |
| manga-ocr-nar-small-distil | 8.4 ms | 35.0 ms | 287 |
| manga-ocr-nar-small | 9.1 ms | 35.5 ms | 299 |
| Hayai v2.5 Nova p384 | 54.0 ms | 252.0 ms | 175 |
| mangaOCR | 84.9 ms | 242.4 ms | 42 |

manga-ocr-charspot-nano 以外的数字取自 [Read 排行榜](https://github.com/muscgab/JMangaBench_Read) 和 manga-ocr-nar-small 的模型卡（同一机器、同一口径）。manga-ocr-charspot-nano 在 `warmup()` 之后测量（启动时一次，11 s，为开发集用到的 17 种输入画布捕获 CUDA graph；之后占用显存 3.1 GB）。CUDA 后端把一次调用里的裁图补零到少数几种画布尺寸，每种画布作为一个带掩码的批来跑；网络只有 1.7M 参数，GPU 大部分时间空闲，瓶颈是每次调用里的 CPU 工作（框解码、NMS、排序）。

空闲一段时间后处理器会降频，下一框会变慢：空闲的 Xeon（调频策略 `powersave`）上，空闲 1–20 s 后为 15 ms，连续处理时约 8 ms；RTX 2080 Ti 上，空闲 1–5 s 后 8.9 ms、20 s 后 24 ms，连续处理时 4.2 ms。锁定频率可以避免。

## 用法

```bash
git clone https://huggingface.co/muscgab/manga-ocr-charspot-nano   # 或：modelscope download --model muscgab/manga-ocr-charspot-nano --local_dir manga-ocr-charspot-nano
cd manga-ocr-charspot-nano
pip install -e ".[cpu]"   # ONNX Runtime（CPU）；GPU 后端用 ".[cuda]"
manga-ocr-charspot-nano crop.png            # 每张裁图输出一行 "路径<TAB>文字"
manga-ocr-charspot-nano --chars crop.png    # 每张裁图输出一行 JSON，含逐字框和分数
```

```python
from manga_ocr_charspot_nano import MangaOcrCharspotNano

ocr = MangaOcrCharspotNano()                    # backend "auto"（有 CUDA 用 CUDA，否则 CPU 上的 ONNX）、"onnx" 或 "cuda"
ocr("crop.png")                         # -> str
ocr(["a.png", "b.png"])                 # -> 字符串列表
ocr.read("crop.png")                    # -> {"text": ..., "chars": [{"char": "今", "box": [x0, y0, x1, y1], "score": 0.93}, ...]}
ocr.warmup()                            # CUDA 后端：一次性捕获 CUDA graph（约 11 s）；CPU 上什么也不做
```

`model.onnx` 把预处理之后的整个模型（含框解码、NMS 和阅读顺序）放在一张图里，也可以在其他语言和运行时里直接使用：输入 `img` [1, 3, H, W]（预处理后的裁图，0–255 原始像素值），输出 `boxes` [N, 4]、`scores` [N]、`ids` [N] 和 `order` [N]（以下标表示的阅读顺序）；没有字的裁图得到 N = 0。

`read` 按阅读顺序返回各字；框以输入裁图的像素为单位，分数是字中心热图的峰值。縦中横的 `!?` 组合是一个条目，内容为两到三个字符。输入可以是文件路径、PIL 图像或 uint8 RGB 数组。模型文件默认从仓库目录读取；也可以用 `MangaOcrCharspotNano(model_dir=...)` 或环境变量 `MANGA_OCR_CHARSPOT_NANO_DIR` 指定。

## 原理

<p align="center"><img src="samples/pipeline.png" width="70%"></p>

从上到下：一次卷积前向得到四张步长 8 的特征图；热图大于 0.3 的格子各提出一个框，NMS 后每个字留一个框；在检出的字之间做注意力，逐字分类并给每个可能的下一个字打分，贪心串成顺序。图中的热图、字框、分数和顺序都是模型在一张 Read v1.3 裁图上的真实输出。矢量版：[samples/pipeline.pdf](samples/pipeline.pdf)。

- **输入**：保持长宽比缩放到不超过 196 个 16 × 16 像素格，用边缘中位色填充。
- **主干**：深度可分离 CNN（48 / 96 / 192 通道），在步长 16 处加一个自注意力块，融合到步长 8 的特征图。
- **稠密输出头**（步长 8）：字中心热图、字框、注音标记、64 维字向量。
- **解码**：热图峰值 > 0.3 → 字框 → NMS（IoU 0.2，并去掉中心落在已保留框内的框）→ 去掉注音 → 逐字分类（6,817 字 + 12 个縦中横 `!?` 组合）。
- **阅读顺序**：对检出的字（字向量 + 框几何）做一层自注意力，两两打"下一个字"的分，并打起点 / 终点分，贪心串成序列。列、行、多列和弯曲排列用同一个机制。

## 训练

| 阶段 | 数据 | 步数（batch 1,024） | 学习率调度 |
|---|---|---|---|
| 1 | 带逐字框的合成裁图 3,463,874 | 34,050（10 轮） | AdamW，OneCycle，峰值 4e-3 |
| 2 | 合成 4,291,646（印刷 1,992,408，手写 2,299,238）+ AnimeText-OCR 真实裁图 1,006,116（只有文本标签） | 31,068（6 轮）+ 冷却 4,660 | AdamW，恒定 2e-3，之后按 1 − √t 降到 0 |

- 真实裁图没有逐字框。参照 [PageNet](https://arxiv.org/abs/2207.14807)（Peng 等，IJCV 2022）：每轮开始时用当前模型读一遍所有真实裁图，把输出和文本标签对齐，只有与标签完全一致的字才作为训练框；各轮之间按 PageNet 的分数加权规则合并框。热图负样本只取自已匹配的相邻字之间。第 6 轮后，1,005,758 个可用裁图中有 997,401 个带伪字框，共 12,319,134 个。
- 4 × RTX 2080 Ti，fp16 混合精度。

## 局限

- **非印刷体**：Read v1.3 上 EM 70.29，manga-ocr-nar-small 为 80.96。每个字只按自身特征分类，没有语言上下文；潦草手写中的形近字是主要错误。
- **横排**：Read 的 439 个横排框上 EM 65.83，manga-ocr-nar-small 为 84.51。
- **长框**：超过 32 字时，Manga109-s 上（3,182 框）60.21，manga-ocr-nar-small 为 74.39；Read 上（169 框）71.01 对 73.96。
- 真实数据的监督只覆盖模型在该裁图上已经读对的字；从来读不对的字只能从合成数据学。
- Read 的框都来自同一个检测器（AnimeText YOLO12x，紧贴文字块），和真实训练数据是同一类裁法；Manga109-s 用的是它自己的标注框。

## 评测说明

- Read v1.2（3,142 框，属于 v1.3 的一部分）是开发集：用于选检查点、定 NMS 阈值和错误分析。v1.3 新增的 4,660 框没有参与任何决定。
- Manga109-s 没有用于训练、选模型或调参。
- 以上成绩来自发布的 ONNX 包（CPU，fp32），与 PyTorch fp32 输出在 Read 全部 7,789 框上逐框一致。CUDA 后端的稠密网络用 fp16：在 Read v1.3 上，单张输出有 5 框与 CPU 不同（EM 89.37 对 89.38），按页合批输出有 7 框不同（EM 89.38，这 7 框两边都读错）。

## 致谢

- [PageNet](https://arxiv.org/abs/2207.14807)（Peng 等，IJCV 2022）：在网格上检测字；对只有文字标签的真实裁图做弱监督（伪字框与标签匹配，跨轮按分数加权更新）。
- [CenterNet](https://arxiv.org/abs/1904.07850)（Zhou 等，2019）：字中心热图的高斯目标和减罚 focal loss，在峰值处回归框。
- [KuroNet](https://arxiv.org/abs/1910.09433)（Clanuwat 等，ICDAR 2019）：逐位置检测并分类字来读日文。
- 把阅读顺序建模为成对的后继关系：[Zhang 等](https://arxiv.org/abs/2409.19672)（EMNLP 2024）、[Hakim 等](https://arxiv.org/abs/2607.01018)（2026）；根据坐标给字排序：[Lamb 等](https://arxiv.org/abs/2106.06786)（2021）。
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)：骨干的 stem、块和下采样结构（小型 PP-OCR 识别骨干）。[CPVT](https://arxiv.org/abs/2102.10882)：注意力前的位置卷积。[MobileNetV3](https://arxiv.org/abs/1905.02244)：hard-swish 与 hard-sigmoid 的 squeeze-excite。
- [Cluster-NMS](https://arxiv.org/abs/2005.03572)：伪标注和 `model.onnx` 中的并行 NMS。
- [Hägele 等](https://arxiv.org/abs/2405.18392)（NeurIPS 2024）：恒定学习率加 1 − √t 冷却。
- [AnimeText](https://arxiv.org/abs/2510.07951) 与 [AnimeText-OCR](https://huggingface.co/datasets/JustANormalTinkerer/animetext-ocr)：真实训练裁图；合成手写用到的真实假名字形也切自 AnimeText 图片，并用 [Real-ESRGAN](https://arxiv.org/abs/2107.10833) 放大。
- [Manga109](http://www.manga109.org/)（[Matsui 等，2017](https://arxiv.org/abs/1510.04389)；[Aizawa 等，2020](https://arxiv.org/abs/2005.04425)）：Manga109-s 用于评测。
- mangaOCR、Hayai、baberuOCR、PaddleOCR-VL-For-Manga：来自 Read 排行榜的对比模型。

## 许可

代码：Apache-2.0。权重：CC BY-NC-SA 4.0（训练用到 AnimeText-OCR）。数据来源与署名见 [NOTICE.md](NOTICE.md)。
