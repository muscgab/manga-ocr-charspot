# manga-ocr-charspot-nano technical report

[English](TECHNICAL_REPORT.md) | [中文](TECHNICAL_REPORT_zh.md)

Date: 2026-10-19

manga-ocr-charspot-nano reads a cropped Japanese manga text box without a text decoder. One convolutional pass marks the centre of every character, a box is decoded at each centre, every character is classified from its own feature, and a small attention module over the detected characters scores which character comes next. Reading order is a greedy walk over these scores. The model has 1,703,263 parameters.

## Summary

1. **A decoder-free character model reaches the accuracy of much larger sequence models.** On JMangaBench_Read v1.3 (7,789 boxes), EM is 89.38%. That is between manga-ocr-nar-small-distil (30M, 89.09%) and manga-ocr-nar-small (54M, 91.36%). On printed text it is the highest of the listed models (95.59%).
2. **The weak side is non-printed text, horizontal text and long boxes.** EM is 70.29% on non-printed boxes, 65.83% on horizontal boxes and 71.01% on boxes longer than 32 characters. Each character is classified from its own visual feature plus the features and positions of the other characters in the box; there is no language model.
3. **Weak supervision on real crops with text-only labels is the largest single gain.** The synthetic-only model scores 82.50% on Read v1.3. Training on 1.0M real crops with pseudo character boxes, which the model produces for itself and refines every epoch, raises it to 89.38%. Non-printed EM rises from 53.09% to 70.29%.
4. **The reading-order head is not the bottleneck.** On printed vertical text 96.26% of boxes are read exactly. Pure ordering errors are rare: 3 of 682 wrong development-set boxes at stage-1 epoch 6. Multi-line horizontal boxes are the exception.
5. **On CPU the dense network sets the latency.** End to end per crop it takes 8.0 ms on an idle Xeon Gold 6148 (8 threads, ONNX Runtime fp32). On an M1 Pro the dense network is 4.7 of about 6 ms per crop. Decoding, NMS, the tail network and the chain together take about 0.5 ms.
6. **On GPU the CPU-side work sets the limit.** On an RTX 2080 Ti per-crop latency is 4.7 ms and throughput is at most 648 crops/s. That needs masked batching of crops with different sizes and CUDA graphs. Without them the same model reached 313 crops/s.

## 1. Task and design idea

**Task.** The input is one text box already cropped from a manga page, in any orientation. The output is its text in reading order. Onomatopoeia drawn as artwork is not a target.

**Why no decoder.** A text decoder (autoregressive or non-autoregressive) is the most expensive part of our earlier models on CPU. The task also allows a different factorisation:

- In manga, characters in a text box rarely touch or overlap.
- The order of characters is mostly determined by geometry: columns right to left, top to bottom within a column, or rows for horizontal text.

So the model first finds every character, then classifies each one, then orders them.

**Is per-character accuracy enough?** If characters were independent, box EM would be $\mathrm{EM} = \mathbb{E}[p^{n}]$, where $p$ is the probability that one character is detected, classified and ordered correctly and $n$ is the number of characters in the box. The development set (Read v1.2) has a mean of 11.23 and a median of 9 characters per box. Under this assumption, matching mangaOCR's 78.33% on the 3,124 development boxes shared with it needs $p \approx 97.7\%$.

**Prior work this design draws on.**

- KuroNet (Clanuwat et al., 2019) reads pre-modern Japanese pages by predicting at every location whether a character is present and which one.
- PageNet (Peng et al., 2022) detects and classifies characters on a grid for Chinese handwriting. It trains on real pages that have only line transcriptions, by matching its own predictions to the text.
- CenterNet (Zhou et al., 2019) detects objects as heatmap peaks and regresses the box at the peak.
- Reading order as pairwise successor relations: Zhang et al. (2024), and Hakim et al. (2026) with a directed path cover. Lamb et al. (2021) order the characters of historical Japanese documents from their coordinates.

## 2. Model

### 2.1 Input

- The crop is resized with bicubic interpolation, aspect ratio kept, to the largest size whose 16 × 16 px grid has at most 196 cells. The grid search is the same as manga-ocr-nar-small's.
- The rest of the grid is filled with the median border colour.
- Pixels are normalised with fixed RGB mean and standard deviation inside the network; the network takes raw 0–255 values.

### 2.2 Dense network (1,154,542 parameters)

| Part | Output | Parameters |
|---|---|---|
| Stem: 3×3 conv stride 2 (24 ch); two branches [2×2 conv → 2×2 conv] and [2×2 max-pool]; concat; 3×3 conv stride 2; 1×1 conv | H/4 × W/4 × 48 | 14,604 |
| Stage 1: 2 blocks (first with squeeze-excite) | H/4 × W/4 × 48 | 20,892 |
| Stage 2: downsample + 2 blocks | H/8 × W/8 × 96 | 90,720 |
| Stage 3: downsample + 4 blocks (1st and 3rd with squeeze-excite) | H/16 × W/16 × 192 | 693,792 |
| Positional conv (depthwise 3×3, added) + 1 transformer block (d 192, 3 heads, pre-LN, MLP 384, GELU) over all stride-16 cells | H/16 × W/16 × 192 | 298,944 |
| Merge: 1×1 conv 192 → 96, nearest ×2 upsampling, add stage-2 output, depthwise 3×3 + 1×1 + hard-swish | H/8 × W/8 × 96 | 28,800 |
| Heads (1×1): heatmap (1, sigmoid), box (4), furigana flag (1, sigmoid), embedding (64) | H/8 × W/8 | 6,790 |

- **Block**: $y = \mathrm{DW}_{3\times3}(x)$ (optionally followed by squeeze-excite with hard-sigmoid); $\mathrm{out} = y + W_2\,\mathrm{hswish}(W_1 y)$ with a 2× expansion.
- **Downsample**: depthwise 3×3 stride 2 → 1×1 → hard-swish → 1×1.
- The layout of stem, blocks and downsampling follows the small PP-OCR recognition backbone in PaddleOCR, with 2D strides and the stage widths of its detection backbone.
- The positional conv before attention is a conditional positional encoding (Chu et al., 2023).
- **Why one attention block at stride 16.** The intent is that every cell sees the whole crop before the heads, since furigana vs body text and small vs large kana are judged relative to the other characters. On 50,000 training rows, a second attention block (+298,944 parameters) was within seed noise on the development set (§4.4), so one was kept.

### 2.3 Character detection (no learned parameters)

Every stride-8 cell $(i, j)$ whose heatmap value exceeds 0.3 proposes one box:

$$c_x = 8\,(j + \Delta x), \quad c_y = 8\,(i + \Delta y), \quad w = 8\,e^{\ell_w}, \quad h = 8\,e^{\ell_h}, \quad \text{score} = \text{heat}_{ij}$$

- **NMS**: greedy by score. A box is removed if its IoU with a kept box is at least 0.2, or if its centre lies in the central half of a kept box.
- **Furigana**: boxes whose furigana flag is at least 0.5 are dropped. Furigana is detected like body text, so that it is not mistaken for body characters, and then removed.
- **Result**: $N$ unordered character boxes.

**Why stride 8.**

- In a sample of 2,052 handwritten glyphs, 2.10% of characters shared a stride-16 cell with another character's centre, and none shared a stride-8 cell.
- Grouped tate-chū-yoko glyphs such as 「!!」 or 「!?」 have a single box in the renderer, so all their characters have exactly the same centre. These 12 combinations of two or three `!`/`?` are one class each (6,818 → 6,830 output classes) and are expanded in the output.
- Other tate-chū-yoko groups (digits, Latin letters) are split into equal-width boxes.

### 2.4 Classification and reading order (tail, 548,721 parameters)

For the $N$ detected characters:

1. **Character feature** $e_i$: the 64-channel embedding map is sampled bilinearly at 4 points of the box (25% and 75% of its width and height), concatenated (256) and projected to 64.
2. **Geometry** $g_i = (c_x, c_y, w, h) / u$, where $u$ is the median of the boxes' longer sides.
3. **Token** $t_i = \mathrm{Linear}([e_i; g_i]) \in \mathbb{R}^{64}$.
4. **2 transformer blocks** over the $N$ tokens (d 64, 4 heads, MLP 128, GELU) give $h_i$.
5. **Class** = $\arg\max\, W_{\text{cls}}\,\mathrm{Linear}([e_i; h_i])$ over 6,830 classes: 6,817 characters, 1 empty class, and 12 combined `!?` glyphs. The two layers have 452,206 parameters; the 64 → 6,830 output layer alone has 443,950.
6. **Order scores**:

$$s(i \to j) = w^\top \mathrm{ReLU}\left(W_a h_i + W_b h_j + W_c\, \Delta g_{ij}\right) + b, \qquad \Delta g_{ij} = \left(c_{x,j} - c_{x,i},\; c_{y,j} - c_{y,i},\; \log\tfrac{w_j}{w_i},\; \log\tfrac{h_j}{h_i}\right)$$

plus a start score and an end score per character (linear in $h_i$).

7. **Chain** (no learned parameters):
   - Start at the character with the highest start score.
   - Repeatedly move to the unused character $j$ with the highest $s(\text{current} \to j)$.
   - If end(current) is higher than that score, restart at the unused character with the highest start score (a new column or line).

**What the chain does and does not handle.** Columns, rows, multi-column boxes and curved text all go through the same mechanism; there is no line or column segmentation. Because each step looks at all remaining characters, a wrong jump cannot be undone later. This shows up in multi-line horizontal text (§5.6).

### 2.5 Output

Text in reading order, plus a box and a score (the heatmap peak) for every character. A combined `!?` glyph is one entry with a two- or three-character string.

## 3. Training data

| Source | Rows | Character boxes | Used in |
|---|---|---|---|
| Synthetic printed text (project renderer) | 1,794,428 (+197,980 in stage 2) | From the renderer | Stages 1 and 2 |
| Synthetic handwriting (project renderer) | 1,669,446 (+629,792 in stage 2) | From the renderer | Stages 1 and 2 |
| [AnimeText-OCR](https://huggingface.co/datasets/JustANormalTinkerer/animetext-ocr) real crops, sequence confidence 0.90–0.98 | 1,006,116 | None (text only) | Stage 2 |

- **Synthetic boxes**:
  - The renderer records a box for every glyph. Labels are mapped to these boxes, so each label character (or combined `!?` glyph) has a box, a class and its position in reading order.
  - Furigana glyphs have boxes but are not part of the label.
  - Rows whose characters could not all be mapped were dropped, and so were rows on the manual review list of manga-ocr-nar-small (relabelled or removed there): 1,441 printed and 554 handwriting rows in stage 1.
- **Stage-1 totals**: 3,463,874 rows, 33,552,773 character units, 242,215 furigana boxes. The longest row has 63 units; the 99th percentile is 34.
- **Stage-2 synthetic rows**: 4,291,646 (printed 1,992,408, handwriting 2,299,238; 40,576,516 units). The additional 827,772 rows came from the same rendering runs as manga-ocr-nar-small's data but were not used for it.
- **AnimeText-OCR rows**: 358 rows carrying a known fixed wrong label were excluded, leaving 1,005,758 usable crops.
- **Excluded data**: no image from the JMangaBench_Read candidate pool or benchmark, and no Manga109 image, was used for training.

## 4. Training

### 4.1 Stage 1: synthetic data with full supervision

| Item | Setting |
|---|---|
| Initialisation | Random |
| Steps | 10 epochs, 34,050 steps, global batch 1,024 (4 × 256) |
| Optimiser | AdamW, weight decay 0.01, gradient clipping 5.0 |
| Learning rate | OneCycle, peak $4\times10^{-3}$, 1,000 warm-up steps |
| Precision | fp16 mixed precision with GradScaler |
| Hardware and time | 4 × RTX 2080 Ti, about 6,000 images/s, 578 s per epoch |

**Losses** (equal weights):

$$\mathcal{L} = \mathcal{L}_{\text{heat}} + \mathcal{L}_{\text{box}} + \mathcal{L}_{\text{furigana}} + \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{order}}$$

- **Heatmap**: a Gaussian target at each character centre (body and furigana) with $\sigma = \max(0.5, \min(w, h)/48)$ stride-8 cells. The loss is CenterNet's penalty-reduced focal loss ($\alpha = 2$, $\beta = 4$), normalised by the number of centres.
- **Box and furigana flag**: trained on the 3 × 3 cells around each centre, with weight 1 at the centre and 0.5 around it. A cell shared by two characters belongs to the nearer centre. Box targets are $(\Delta x, \Delta y, \log(w/8), \log(h/8))$ with an L1 loss; the furigana flag uses binary cross-entropy.
  - Why the neighbourhood: in an overfitting test, 72 detection peaks fell one cell off the true centre. Outputs trained at the centre cell only were unreliable there (31% of these were classified correctly by the per-cell classifier used then); trained on the 3 × 3 neighbourhood, all were (§4.4).
- **Classification and order**: the tail runs on the ground-truth boxes, each side jittered by up to ±10% of the box size, to imitate the imprecision of predicted boxes.
  - Classification is cross-entropy over the 6,830 classes.
  - Order: for each character, cross-entropy over [scores to every other character; its end score], with the next character as target (the end score for the last one). Plus cross-entropy over the start scores with the first character as target.

**Stage-1 curve** (training-time evaluation: GPU fp16, NMS IoU 0.5, labels in the r7 normalisation; held-out = 2,000 synthetic printed rows not trained on; dev = Read v1.2, 3,142 boxes):

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Held-out synthetic EM | 87.50 | 89.40 | 92.85 | 90.40 | 95.95 | 95.40 | 96.55 | 97.50 | 98.25 | 98.30 |
| Read v1.2 EM | 73.04 | 74.09 | 77.05 | 75.43 | 78.90 | 78.23 | 79.57 | 80.71 | 81.44 | 81.51 |

**Fit at the end of stage 1** (2,000 random training rows per source):

- EM on training rows: printed 98.75–99.00, handwriting 95.35–95.60. On held-out rows: printed 98.30, handwriting 93.78.
- Handwriting rows longer than 20 characters reach only 76.4% even on the training set.

There was no overfitting; long handwritten rows were underfitted.

**The gap to real data.** Handwriting held-out EM was 93.78%, but Read v1.2 non-printed EM was 54.30%. A model this size can fit synthetic handwriting, so the gap pointed to the difference between synthetic and real data. That motivated stage 2.

### 4.2 Stage 2: adding real crops with text-only labels

Real crops have no character boxes. Following PageNet, the model labels them itself.

**Relabelling at the start of every epoch** (including the first, with the stage-1 weights):

1. The current model reads every usable AnimeText-OCR crop (fp16, NMS IoU 0.2).
2. Its output, in reading order, is aligned to the label by edit distance (`difflib`).
3. A predicted character becomes a pseudo box only if its text equals the aligned label character. For a combined `!?` glyph, its 2–3 characters must equal consecutive label characters.
4. Crops whose alignment accuracy $\mathrm{AR} = (L - S - I - D)/L$ is below 0.3 contribute nothing.
5. **Merging across epochs** (PageNet's update rule):
   - If the label position already has a pseudo box of the same span with IoU ≥ 0.5, the two are averaged with weight $\lambda = 1 / (1 + e^{\varepsilon (B - g)})$, $\varepsilon = 10$, where $g$ is the old score and $B$ the new one. The score is updated the same way.
   - A new box is added only if its label position is not yet covered.
6. Every crop with at least one pseudo box enters that epoch's training set.

**Losses on real crops.** Only what the pseudo boxes support is supervised:

- **Heatmap**: positives at the pseudo boxes. The loss is applied only inside each box's Gaussian window, and in the rectangle spanned by two pseudo boxes that are adjacent in the label and close in the image (centre distance ≤ 1.6 × the larger box side). Elsewhere the image may contain characters without pseudo boxes, so it gets no loss.
- **Box and class**: as on synthetic data, at the pseudo boxes.
- **Order**:
  - "next" only where the next pseudo box is also the next label character;
  - "end" only where the pseudo box ends the label;
  - "start" only where the first pseudo box starts it.
- **Furigana flag**: no loss.

**Settings.**

| Item | Setting |
|---|---|
| Initialisation | End of stage 1; new AdamW state |
| Data per epoch | All 4,291,646 synthetic rows + the real crops with pseudo boxes (about 1.0M, about 19%), shuffled together |
| Learning rate | 1,000 warm-up steps, then constant $2\times10^{-3}$ (warmup-stable-decay) |
| Length | 6 epochs, 31,068 steps; then a cool-down branch of 4,660 steps (15%) with the learning rate decaying as $1 - \sqrt{t}$ to 0 (Hägele et al., 2024) |
| Batch, optimiser, precision, hardware | As in stage 1; about 20 minutes per epoch including relabelling |

**Pseudo-label growth** (real crops; "exact" = the whole output equals the label):

| Relabel before epoch | 1 | 2 | 3 | 4 | 5 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| Exact crops | 708,968 | 749,285 | 780,171 | 800,040 | 806,569 | 826,111 | 823,700 |
| Pseudo boxes (M) | 11.78 | 12.05 | 12.19 | 12.25 | 12.29 | — | — |

- The first pass already gave pseudo boxes for about 94% of the label characters of usable crops.
- New boxes per round roughly halved: about 270k, 134k, 64k and 41k for rounds 2–5, then 21k and 14k for rounds 7–8.
- After 6 epochs, 997,401 of the 1,005,758 usable crops carried 12,319,134 pseudo boxes.

**Stage-2 curve** (training-time evaluation: GPU fp16, NMS IoU 0.2, r7 labels; dev = Read v1.2, 3,142 boxes; AT val = 2,000 AnimeText-OCR rows from validation pages; HW val = 2,000 held-out synthetic handwriting rows):

| Checkpoint | Dev EM | Dev non-printed | AT val | HW val |
|---|---|---|---|---|
| End of stage 1 | 81.95 | 54.30 | 78.55 | 94.15 |
| Epoch 1 (constant LR) | 82.30 | 56.63 | 82.75 | 89.30 |
| Epoch 2 | 84.12 | 60.57 | 86.05 | 89.70 |
| Epoch 3 | 85.20 | 61.55 | 87.25 | 91.20 |
| Epoch 4 | 84.75 | 61.55 | 88.00 | 90.65 |
| Epoch 5 | 84.79 | 62.29 | 87.85 | 90.60 |
| Epoch 6 | 86.51 | 66.71 | 88.95 | 92.15 |
| Epoch 2 + 1,550-step cool-down | 85.71 | 63.02 | 86.95 | 93.80 |
| Epoch 6 + 1,550-step cool-down | 87.59 | 69.04 | 91.00 | 94.45 |
| Epoch 8 + 1,550-step cool-down | 87.46 | 69.16 | 91.10 | 94.50 |
| **Epoch 6 + 4,660-step cool-down (released)** | **87.87** | **69.16** | **91.55** | **94.95** |

- **Checkpoint choice.**
  - Short cool-down branches were used as probes. Epochs 7 and 8 at constant learning rate scored 85.01 and 84.91 on the dev set.
  - The epoch-8 probe did not beat the epoch-6 probe; paired on the dev set, 51 vs 55 boxes were read correctly by only one of the two ($p = 0.77$).
  - Training was therefore stopped, and the epoch-6 snapshot was cooled down over 4,660 steps.
- **The handwriting dip.** Synthetic handwriting EM dropped at constant learning rate and recovered after cool-down. The drop came from the learning-rate phase, not from the real data.

### 4.3 Training notes

- **Learning rate $8\times10^{-3}$ diverged.**
  - Around step 913, the first non-finite gradient appeared in stage 3 of the backbone; at step 925 the loss jumped from 0.78 to 56.7.
  - Replaying from the same step-900 state in fp32 also blew up (loss 3,685 at step 950, gradient norm $4\times10^{5}$). It did not produce NaN, but fell back to near the initial loss.
  - So it was an optimisation divergence, not an fp16 overflow; fp16 only turned it into NaN. We suspect AdamW's large relative step after a period of small gradients, in a backbone without normalisation layers.
  - Training was restarted at $4\times10^{-3}$ and stayed stable.
- **Throughput was limited by the data pipeline, not the model.**
  - A first 4-GPU run reached 848 images/s.
  - Fixes:
    - stop sending per-row target tensors (about 1,150 per batch) from loader workers through shared memory;
    - remove device synchronisations from boolean indexing in the heatmap loss;
    - compute the classifier only on valid character slots;
    - pin memory;
    - read from a random-access binary pack instead of streaming compressed shards.
  - Together these raised throughput to about 6,000 images/s.
- **Pseudo-labelling throughput.**
  - The first version decoded row by row in the main process (about 1,300 rows/s on 4 GPUs).
  - Batching crops by size across loader workers and running NMS on the GPU raised it to about 3,400 rows/s. The GPU NMS uses the fixed-point form of Cluster-NMS (Zheng et al., 2021), which equals greedy NMS.

### 4.4 Small-scale experiments behind the design

**Overfitting 256 synthetic rows** (150 epochs, EM on the same rows):

| Version | EM | Failures |
|---|---|---|
| v1: per-cell classifier, supervised at the centre cell only; tate-chū-yoko glyphs keep one shared box | 72.66 | 70 rows: 36 two characters on one cell, 23 classification, 11 length, 0 order |
| v2: equal-width split of tate-chū-yoko boxes; classifier supervised on the 3 × 3 neighbourhood; 196-cell input | 98.44 | 4 rows, all duplicate detections |
| v3: box and furigana outputs also on the 3 × 3 neighbourhood; classification from box-pooled features with 2 attention layers (the current tail) | 98.44 | — |
| v3 weights, NMS that also removes boxes centred in the central half of a kept box | 98.83 | 3 rows, all a lost `!` in 「!!」 / 「!?」 |

**50,000 printed rows, 10 epochs** (single seed unless stated; Read v1.2 EM against r7 labels):

| Setting | Held-out synthetic EM | Read v1.2 EM | Printed | Non-printed |
|---|---|---|---|---|
| Tate-chū-yoko split into equal-width boxes | 83.75 | 57.00 | 73.28 | 10.44 |
| `!?` combinations as one class (seed 0) | 83.80 | 57.73 | 74.10 | 10.93 |
| `!?` combinations as one class (seed 1) | 84.85 | 58.15 | 74.40 | 11.67 |
| + second attention block at stride 16 | 85.30 | 57.96 | 74.14 | 11.67 |

- **Seed noise.** Seeds 0 and 1 differ on 121 + 134 boxes, a net change of 13 boxes, about ±0.4 points.
- **Second attention block.** Net +7 boxes against seed 0 and −6 against seed 1, i.e. within noise; not adopted.
- **Combined `!?` class.** Net +23 boxes against the split version. On the 274 boxes whose label contains 「!!」/「!?」/「?!」/「??」, EM is 47.45 vs 41.61; adopted.

## 5. Accuracy

### 5.1 Evaluation protocol

- **Benchmark**: [JMangaBench_Read v1.3.1](https://github.com/muscgab/JMangaBench_Read), 7,789 boxes, official scorer, RIN v1.1 normalisation.
- **Metrics**:
  - EM = share of boxes matched exactly after normalisation.
  - CER = total edit distance / total label characters.
  - Composite = equal-weight mean of $1 - \mathrm{NED}$ over the printed and non-printed boxes:

$$s_i = 1 - \frac{\mathrm{ED}(p_i, g_i)}{\max(|p_i|, |g_i|)}, \qquad \text{Composite} = 100 \times \tfrac{1}{2}\left(\overline{s}_{\text{printed}} + \overline{s}_{\text{non-printed}}\right)$$

- **Development use**: Read v1.2 (3,142 boxes, of which 3,129 remain in v1.3) was the development set for checkpoint choice, the NMS threshold and error analysis. The 4,660 boxes new in v1.3 were not used for any decision.
- **Run path**: the released ONNX package, CPU fp32, batch 1. Its outputs equal PyTorch fp32 on all 7,789 boxes.
- **Other models**: predictions from the Read repository, batch 1.

### 5.2 Read v1.3

| Model | Params | Composite | CER % | EM | EM new | Printed | Non-printed | Hard |
|---|---|---|---|---|---|---|---|---|
| **manga-ocr-charspot-nano** | **1.7M** | 95.74 | 2.14 | 89.38 | 89.55 | **95.59** | 70.29 | 63.90 |
| manga-ocr-base-CPT-flex | 111M | 97.47 | 1.50 | 91.71 | 91.93 | 94.50 | 83.11 | 70.66 |
| manga-ocr-nar-small | 54M | 97.29 | 1.71 | 91.36 | 91.65 | 94.74 | 80.96 | 71.05 |
| manga-ocr-nar-small-distil | 30M | 96.40 | 2.29 | 89.09 | 88.95 | 93.21 | 76.41 | 63.90 |
| Hayai v2.5 Nova p384 | 157M | 96.64 | 2.38 | 88.18 | 88.41 | 91.93 | 76.62 | 63.39 |
| PaddleOCR-VL-For-Manga | 959M | 95.04 | 1.99 | 87.89 | 87.92 | 95.17 | 65.53 | 60.20 |
| manga-ocr-nar-preview | 196M | 95.84 | 2.92 | 85.68 | 85.62 | 89.14 | 75.05 | 60.84 |
| baberuOCR v1.1 | 113M | 93.37 | 3.62 | 81.88 | 81.61 | 88.97 | 60.09 | 53.44 |
| mangaOCR | 111M | 90.27 | 4.81 | 78.51 | 78.48 | 88.34 | 48.27 | 44.64 |

Subset sizes: new 4,660; printed 5,877; non-printed 1,912; hard 784.

**EM by length, orientation and role:**

| Model | 1–4 | 5–16 | 17–32 | > 32 | Vertical | Horizontal | Square | Dialogue | Narration | Other |
|---|---|---|---|---|---|---|---|---|---|---|
| **manga-ocr-charspot-nano** | 89.49 | 89.88 | 89.80 | 71.01 | 91.06 | 65.83 | 80.73 | 90.83 | 87.30 | 63.44 |
| manga-ocr-nar-small | 91.82 | 91.66 | 91.95 | 73.96 | 91.88 | 84.51 | 87.50 | 91.85 | 92.26 | 80.06 |
| manga-ocr-nar-small-distil | 88.63 | 89.86 | 90.01 | 63.91 | 90.03 | 76.31 | 83.33 | 89.93 | 89.09 | 72.51 |
| Hayai v2.5 Nova p384 | 89.69 | 89.48 | 86.26 | 55.03 | 88.75 | 83.60 | 77.08 | 88.40 | 90.48 | 80.97 |
| mangaOCR | 79.12 | 79.27 | 78.90 | 48.52 | 80.72 | 46.24 | 69.79 | 80.25 | 83.33 | 35.95 |

Subset sizes: 1,504 / 4,675 / 1,441 / 169 by length; 7,158 / 439 / 192 by orientation; 6,948 / 504 / 331 by role.

**Orientation × print style** (EM):

| | Vertical printed (5,561) | Vertical non-printed (1,597) | Horizontal printed (195) | Horizontal non-printed (244) |
|---|---|---|---|---|
| manga-ocr-charspot-nano | 96.26 | 72.95 | 80.51 | 54.10 |
| manga-ocr-nar-small | 94.89 | 81.40 | 92.82 | 77.87 |

- **CER**: 1.02% on printed and 7.55% on non-printed boxes.
- **Horizontal boxes** score lower for two reasons:
  - their mix: 56% of them are non-printed, against 22% of vertical boxes;
  - within each print style, horizontal is still 15.8 (printed) and 18.9 (non-printed) points below vertical.

### 5.3 Manga109-s

All 123,014 text annotations of Manga109-s (scanned commercial manga, 87 volumes), cropped with the dataset's own boxes, RIN v1.1, EM and CER as above. Models trained on Manga109-s are not listed.

| Model | Params | EM | CER % | 1–4 | 5–16 | 17–32 | > 32 |
|---|---|---|---|---|---|---|---|
| **manga-ocr-charspot-nano** | **1.7M** | 89.10 | 1.98 | 94.27 | 90.37 | 84.46 | 60.21 |
| manga-ocr-nar-small | 54M | 92.08 | 1.39 | 95.92 | 92.23 | 90.11 | 74.39 |
| baberuOCR v1.1 | 113M | 86.68 | 2.51 | 94.73 | 87.92 | 79.75 | 54.40 |
| manga-ocr-nar-preview | 196M | 86.51 | 2.46 | 91.74 | 87.30 | 83.22 | 55.59 |

Boxes per length bucket: 26,336 / 65,916 / 27,580 / 3,182.

- **Across domains**: EM is about the same as on Read (89.10 vs 89.38), although Manga109-s is a different domain: scans, older works, and boxes drawn by annotators rather than by a detector.
- **Long boxes**: the gap to manga-ocr-nar-small widens with length; above 32 characters it is 14 points.

### 5.4 Effect of stage 2 on Read v1.3

| Checkpoint | EM | EM new | Printed | Non-printed | Horizontal |
|---|---|---|---|---|---|
| End of stage 1 (synthetic only; GPU fp32, NMS 0.2) | 82.50 | 82.34 | 92.07 | 53.09 | 52.85 |
| Released | 89.38 | 89.55 | 95.59 | 70.29 | 65.83 |

The largest gain is on non-printed text (+17.2), but printed text also gains 3.5 points.

### 5.5 Decoding thresholds

Checked at the end of stage 1 (CPU fp32, Read v1.2 dev set, r7 labels); the model was not retrained.

| Setting | EM | Non-printed | Boxes with too many / too few characters | Fixed / broken vs default |
|---|---|---|---|---|
| NMS IoU 0.5 (training default) | 81.54 | 52.95 | 193 / 113 | — |
| **NMS IoU 0.2** | **81.95** | 54.30 | 151 / 136 | 14 / 1 |
| NMS IoU 0.1 | 80.71 | 49.26 | 112 / 240 | 20 / 46 |
| NMS 0.2 + heatmap threshold 0.4 | 81.38 | 54.18 | 112 / 182 | 26 / 31 |
| NMS 0.2 + heatmap threshold 0.5 | 80.74 | 54.67 | 91 / 242 | 35 / 60 |

- **Adopted**: NMS IoU 0.2 with heatmap threshold 0.3.
- **What NMS cannot fix**: most surplus detections in rough handwriting do not overlap the true character (strokes, screentone, a radical split off a character), so lowering the threshold further mainly merges true neighbours.

### 5.6 Where the errors are

Unless stated, on the 3,129 development boxes of Read v1.3, released model.

**Shape of the wrong boxes:**

| | Correct | Same length | Too long | Too short |
|---|---|---|---|---|
| Printed (2,327) | 2,223 | 56 | 21 | 27 |
| Non-printed (802) | 566 | 125 | 72 | 39 |

- 193 of the 340 wrong boxes are off by a single character edit.
- On non-printed boxes, substitutions dominate (281 characters), then insertions (120) and deletions (71).

**Non-printed text** (analysed at the end of stage 1, before real data was added):

- **Substitutions**: mostly variant handwritten shapes (良→食, 察→寮); 「ー」 read as 「!」/「1」/「〜」; 「…」 read as 「ミ」; small/large kana (い↔ぃ).
- **Insertions**: screentone, frame lines or drawn symbols detected as characters; a left-right character split into two (「初期」 detected as 4 boxes).
- **Deletions**: outlined display lettering missed or merged; slanted lines; touching characters.
- **Cross-check with other models**: of the wrong non-printed boxes, 24% were also wrong for manga-ocr-nar-small, the internal 8.5M model, baberuOCR and mangaOCR. These are hard cases or label problems.

**Small kana.**

- The size cue is used: on correctly read characters, the median relative box size is 0.79 for small kana and 1.00 for large.
- Most confusions are characters drawn at an in-between size.
- At the end of stage 1, small/large swaps were the most of 18 compared models: 211 swaps on all of Read v1.3, against 127 for mangaOCR. Fixing only these swaps would have added at most 1.54 EM points.

**Punctuation in printed text.**

- Real manga sets 「、」 and 「。」 in a compressed half-width slot close to the preceding character. The renderer gave them a full slot: in vertical text, at a 「、」 the gap from the previous character to the next is 0.75 character heights in real crops (median, from predicted boxes) and 1.41 in synthetic ones (from true boxes).
- At the 50,000-row scale, 87% of printed 「、」 were missed.
- At the end of stage 1 the miss rates were 11.4% for 「、」 (20 of 175) and 22.0% for 「。」 (13 of 59).

**Horizontal text.** Besides weaker recognition, multi-line horizontal boxes can be read in the wrong line order even when every character is detected and classified correctly. We suspect a learned vertical prior ("the next character is below"); synthetic data contains about 200,000 multi-line horizontal rows, so the cause is not a lack of examples.

**Dark backgrounds** (all of Read v1.3):

| Subset | Boxes | EM, original | EM, colours inverted |
|---|---|---|---|
| White text on a dark background | 205 | 81.46 | 80.49 |
| Dark text on a light background | 7,490 | 89.79 | 88.97 |

- Background = median luminance of the outer 3 px; text polarity = sign of the high-contrast pixels.
- **Not a polarity effect**: inverting light crops costs only 0.8 points (142 boxes read correctly only in the original vs 81 only when inverted, $p = 5\times10^{-5}$), and inverting dark crops does not help (6 vs 4). So the model is close to polarity-invariant, and the lower EM of dark boxes is not about polarity.
- **Composition**: reweighting light boxes to the dark boxes' mix of print style, role and length lowers their EM to 85.95, so about half of the gap is composition.

## 6. Speed

All times are end to end per crop (image already decoded; preprocessing, network, decoding and ordering included), batch 1, p50 over all 7,789 Read v1.3 boxes, unless stated.

### 6.1 CPU (ONNX Runtime fp32)

| Device | Threads | p50 |
|---|---|---|
| Xeon Gold 6148, idle | 8 | 8.0 ms |
| M1 Pro, under load (load average 5–12; 1.0.0 two-file package) | 8 | 6.6 ms |

The package runs one ONNX file, `model.onnx` (6.9 MB), with one session. Box decoding, NMS, furigana removal, the tail and the chain are inside the graph:

- NMS runs as a fixed-point loop (the Cluster-NMS form, equal to greedy NMS).
- The stable sort by score is a TopK (ties broken by lower index).
- Box centres are computed in float64, as numpy promotes them.

**Single file vs the 1.0.0 two-file split.** Version 1.0.0 shipped the dense network and the tail as two files and ran decoding and NMS in numpy between them; the tail session used 1 thread, because with 8 threads its pool competed with the dense session's pool (dense network 4.5 → 7.4 ms).

- **Agreement**: on the M1, text and character boxes are the same on all 7,789 boxes. On the Xeon, text is the same on all 7,789, and 10 character boxes differ by 0.1 px (rounding).
- **Speed** on the idle Xeon, one process per run, alternating two-file / single / two-file / single:

| Run | p50 | p90 | p99 | p50, > 64 characters (8 boxes) |
|---|---|---|---|---|
| Two-file, run 1 | 7.79 ms | 9.89 ms | 15.96 ms | 15.21 ms |
| Single file, run 1 | 8.06 ms | 10.19 ms | 15.99 ms | 14.96 ms |
| Two-file, run 2 | 8.16 ms | 10.31 ms | 16.80 ms | 17.21 ms |
| Single file, run 2 | 7.91 ms | 10.04 ms | 15.86 ms | 14.88 ms |

  The difference between the two forms (p50 7.98 vs 7.99 ms, mean of two runs) is smaller than the run-to-run spread (0.2–0.4 ms). Running both sessions in one process and alternating crops is not a fair test: their spinning thread pools compete, and that run showed 8.70 ms for the single file against 7.88 ms.

**Breakdown on the M1 Pro** (1.0.0 two-file split, load average 4–6, ONNX Runtime 1.23.2):

| Step | Time |
|---|---|
| Preprocessing (resize, pad) | 0.78 ms |
| Dense network | 4.67 ms |
| Box decoding and NMS (numpy) | 0.13 ms |
| Tail network | 0.35 ms |
| Chain | 0.03 ms |

- **Dense network vs threads**: 13.5 ms with 1 thread, 5.6 ms with 4, 4.5 ms with 8.
- **Start-up** (single file, idle Xeon, 3 runs):
  - Loading the session takes 236–239 ms, and the first call 16–27 ms.
  - There is no per-shape warm-up: over the first 400 Read v1.3 crops, the 37 crops with a new input size had a p50 of 7.80–8.01 ms, against 7.77–8.00 ms for repeated sizes.
- **ONNX Runtime thread spinning**: left on. Turning it off raised the dense network from 4.50 to 6.72 ms and saved only 0.04 CPU-seconds per second when idle (1.0.0 dense file).
- **Reference**: an internal 8.5M CNN with a non-autoregressive text decoder (not released) took 16.97 ms on the same Xeon (8 threads, first 100 Read v1.3 boxes), against 7.06 ms for this model at the end of stage 1.

### 6.2 NVIDIA GPU (RTX 2080 Ti)

Measured with the Read speed protocol:

- **Per image** = batch 1.
- **Per page** = all boxes of one page in one call, p50 over pages.
- **Max img/s** = best batch size, with CPU preprocessing in worker processes.

| Version | Per image | Per page | Max img/s |
|---|---|---|---|
| Crops grouped by exact size, CUDA graphs captured on first use | 4.47 ms | 38.2 ms | 313 |
| **Released**: canvas ladder, masked batching, CUDA graphs up to batch 8, `warmup()` | 4.67 ms | 28.6 ms | 648 |

**What changed.**

- **Masked batching.** Crops of different sizes are zero-padded onto a common canvas. Every layer multiplies by the valid-region mask, and squeeze-excite means and attention are computed over the valid region only. Each crop's outputs then equal its single-crop outputs (fp32 difference ≤ $1.5\times10^{-5}$).
- **Canvas ladder and graphs.**
  - A ladder of 16 canvas sizes (1 to 196 cells) and batch sizes 1–64 keep the number of distinct shapes small.
  - `warmup()` captures CUDA graphs for the 17 canvases seen on the development set, at batch sizes up to 8: 88 graphs, 10.9 s, 3.1 GB of GPU memory afterwards.
  - Each graph holds about 15 MB of memory outside PyTorch's allocator, so not every shape can be captured.
- **Tail**: one tail call per request, with rows sorted by length.
- **Ceiling**: with preprocessing done in advance, the network alone runs at about 980 crops/s at batch 64 and about 1,200 at batch 512. The remaining limit is per-call CPU work (box decoding, NMS, ordering).
- **A rejected variant**: capturing graphs also for large batches reached 674 crops/s but doubled memory.
- **Precision**: the dense network runs in fp16. Per image, the output differs from CPU on 5 of 7,789 boxes (EM 89.37 vs 89.38). Batched per page, it differs on 7 boxes, all wrong either way (EM 89.38).

### 6.3 Idle clock-down

The first crop after an idle pause is slower:

| Device | Continuous | After 1 s idle | After 5 s idle | After 20 s idle |
|---|---|---|---|---|
| Xeon Gold 6148, ONNX, 8 threads | 8.22 ms | 15.4 ms | 15.2 ms | 15.6 ms |
| Xeon Gold 6148, ONNX, 1 thread | 14.27 ms | 25.5 ms | 24.5 ms | 24.5 ms |
| RTX 2080 Ti, CUDA | 4.17 ms | 8.9 ms | 8.8 ms | 24.4 ms |

- **Cause on the Xeon**:
  - Its frequency governor is `powersave` and turbo is disabled. Idle cores sit at 1.0 GHz, reach 2.4 GHz within 50 ms of load, and drop back within 1 s.
  - A crop takes 8 ms, so the first one runs mostly at the low clock.
  - The 1-thread result shows the same slowdown, so it is clock speed, not thread wake-up.
- **Fix**: locking the clocks avoids it. We could not test this on the shared machine.

## 7. Conclusions

1. **Decoder-free OCR is viable for manga text boxes.** With 1.7M parameters, detection plus per-character classification plus a learned successor score reaches 89.38% EM on Read v1.3 and 89.10% on Manga109-s. That is close to models 18–32× larger.
2. **Printed vertical text is close to solved at this size; context-dependent reading is not.** Printed vertical boxes score 96.26%. Non-printed and horizontal text trail models with a decoder by 8–19 points. These are the cases where a character's identity or order depends on more than its own shape and position.
3. **Real data matters more than further synthetic data.** Synthetic handwriting was fitted to 94–95%, while real non-printed text stayed at 53% until real crops were added. Self-labelled real crops raised it by 17 points without any character-box annotation.
4. **For small models, the system around the network sets the speed.** On CPU, thread pools and clocks matter as much as the network. On GPU, batching crops of different sizes and CUDA graphs doubled throughput; per-call CPU work is now the limit.

## 8. Limitations

- **Benchmark**: JMangaBench_Read was built by the author of this report and consists mainly of works published online. Its boxes come from the same kind of detector as the real training crops. Manga109-s gives a second, different domain.
- **Development use**: Read v1.2 (3,129 of the 7,789 boxes) was used for checkpoint choice, the NMS threshold and error analysis. Results on the 4,660 new boxes are 89.55% EM.
- **Seeds**: the full model was trained once. Single-seed noise at the 50,000-row scale was about ±0.4 points on the development set.
- **No language context**: look-alike characters that only context can resolve, characters left out of the drawing, and characters outside the 6,817-character vocabulary cannot be read correctly.
- **Self-labelling**: real-data supervision only covers characters the model already reads correctly on that crop. Characters it never reads correctly on real crops are learned from synthetic data only.
- **Layout**: one crop is assumed to hold one text in one reading order. Rotated text and several separate texts in one crop are not handled.
- **Onomatopoeia**: not a target.

## References

1. D. Peng, L. Jin, Y. Liu, C. Luo, S. Lai. *PageNet: Towards End-to-End Weakly Supervised Page-Level Handwritten Chinese Text Recognition.* IJCV, 2022. [arXiv:2207.14807](https://arxiv.org/abs/2207.14807)
2. X. Zhou, D. Wang, P. Krähenbühl. *Objects as Points.* 2019. [arXiv:1904.07850](https://arxiv.org/abs/1904.07850)
3. T. Clanuwat, A. Lamb, A. Kitamoto. *KuroNet: Pre-Modern Japanese Kuzushiji Character Recognition with Deep Learning.* ICDAR, 2019. [arXiv:1910.09433](https://arxiv.org/abs/1910.09433)
4. A. Lamb, T. Clanuwat, S. Han, M. Bober-Irizar, A. Kitamoto. *Predicting the Ordering of Characters in Japanese Historical Documents.* 2021. [arXiv:2106.06786](https://arxiv.org/abs/2106.06786)
5. C. Zhang, Y. Tu, Y. Zhao, et al. *Modeling Layout Reading Order as Ordering Relations for Visually-rich Document Understanding.* EMNLP, 2024. [arXiv:2409.19672](https://arxiv.org/abs/2409.19672)
6. I. Hakim, S. Gogawale, O. Ventura, et al. *Reading Order Inference for Complex Document Layouts.* 2026. [arXiv:2607.01018](https://arxiv.org/abs/2607.01018)
7. X. Chu, Z. Tian, B. Zhang, X. Wang, C. Shen. *Conditional Positional Encodings for Vision Transformers.* ICLR, 2023. [arXiv:2102.10882](https://arxiv.org/abs/2102.10882)
8. A. Howard, M. Sandler, G. Chu, et al. *Searching for MobileNetV3.* ICCV, 2019. [arXiv:1905.02244](https://arxiv.org/abs/1905.02244)
9. PaddleOCR. [github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
10. Z. Zheng, P. Wang, D. Ren, et al. *Enhancing Geometric Factors in Model Learning and Inference for Object Detection and Instance Segmentation.* IEEE Transactions on Cybernetics, 2021. [arXiv:2005.03572](https://arxiv.org/abs/2005.03572)
11. A. Hägele, E. Bakouch, A. Kosson, L. Ben Allal, L. Von Werra, M. Jaggi. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations.* NeurIPS, 2024. [arXiv:2405.18392](https://arxiv.org/abs/2405.18392)
12. Z. Dong, Y. Zhang, C. Li, N. R. Golding, Q. Long. *A Large-scale Dataset for Robust Complex Anime Scene Text Detection* (AnimeText). 2025. [arXiv:2510.07951](https://arxiv.org/abs/2510.07951); AnimeText-OCR: [huggingface.co/datasets/JustANormalTinkerer/animetext-ocr](https://huggingface.co/datasets/JustANormalTinkerer/animetext-ocr)
13. X. Wang, L. Xie, C. Dong, Y. Shan. *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.* ICCVW, 2021. [arXiv:2107.10833](https://arxiv.org/abs/2107.10833)
14. Y. Matsui, K. Ito, Y. Aramaki, T. Yamasaki, K. Aizawa. *Sketch-based Manga Retrieval using Manga109 Dataset.* Multimedia Tools and Applications, 2017. [arXiv:1510.04389](https://arxiv.org/abs/1510.04389)
15. K. Aizawa, A. Fujimoto, A. Otsubo, T. Ogawa, Y. Matsui, K. Tsubota, H. Ikuta. *Building a Manga Dataset "Manga109" with Annotations for Multimedia Applications.* IEEE MultiMedia, 2020. [arXiv:2005.04425](https://arxiv.org/abs/2005.04425)

## Appendix: reproducibility

- **Training**: all settings are in §4. The training scripts depend on an unreleased data pipeline and are not part of this repository; the training data and the synthetic renderer are not released.
- **Evaluation**: Read v1.3 `eval/score.py` (RIN v1.1). Manga109-s crops use the dataset's own boxes, with labels and predictions normalised by RIN v1.1.
- **Inference**: the code in `manga_ocr_charspot_nano/` is the code that produced every score and timing above, except the stage-1 and stage-2 training-time evaluations, which used the training scripts' fp16 GPU decoder.
