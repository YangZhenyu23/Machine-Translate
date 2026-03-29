# 英译中机器翻译系统

基于 PyTorch Transformer 从零训练的英文→中文神经机器翻译模型（**不加载任何预训练模型权重**）。

---

## 一、项目目的

本项目旨在**从零构建并训练一个 Transformer Encoder-Decoder 翻译模型**，实现英文到中文的机器翻译。核心特点：

- **模型权重完全从零训练**：不使用任何预训练语言模型的权重，仅借用 BERT 的分词器（tokenizer）将文本切分为子词。
- **数据集**：使用 Meta 发布的 [FLORES-200](https://huggingface.co/datasets/facebook/flores) 多语言平行语料，其中 `dev`（997 句）用于训练，`devtest`（1012 句）用于验证。
- **数据增强**：通过同义词替换、随机插入、相邻交换、随机删除等方法对英文源句进行温和增强，在不破坏语义的前提下有效扩充小规模训练集。
- **评估指标**：使用 sacrebleu 计算中文 BLEU 分数，评估翻译质量。

---

## 二、文件架构及说明

```
Machine translate/
├── main.py            # 主入口：统一超参数、串联训练→评估→交互翻译
├── model.py           # 模型定义：Transformer Encoder-Decoder + 位置编码
├── dataset.py         # 数据加载：FLORES-200 下载、分词、Dataset & DataLoader
├── augumentation.py   # 数据增强：4 种英文增强方法（同义词替换/插入/交换/删除）
├── train.py           # 训练逻辑：训练循环、验证、Early Stopping、学习率调度
├── evaluation.py      # 评估与推理：Greedy Decoding、BLEU 评估、交互翻译
├── best_model.pt      # [训练后生成] 验证集最优模型权重
└── training_curves.png# [训练后生成] 训练/验证损失曲线和学习率曲线图
```

### 各文件详细说明

| 文件 | 说明 |
|------|------|
| **`main.py`** | 项目主入口。定义了统一的 `CONFIG` 超参数字典，按顺序执行三个阶段：① 调用 `train()` 训练模型；② 调用 `load_and_evaluate()` 计算 BLEU 并展示样例翻译；③ 启动交互式翻译循环，用户输入英文实时输出中文。 |
| **`model.py`** | 定义 `Seq2SeqTransformer` 模型。包含：正弦位置编码（`PositionalEncoding`）、源语言/目标语言 Embedding、PyTorch `nn.Transformer` 核心、线性输出层。提供 `encode()`/`decode()` 方法供推理时分步调用，以及工厂函数 `build_model()`。权重使用 Kaiming 初始化。 |
| **`dataset.py`** | 负责数据处理。使用 HuggingFace `datasets` 库下载 FLORES-200 英中平行语料；使用 `bert-base-uncased` 和 `bert-base-chinese` 的分词器对文本编码；定义 `TranslationDataset`（PyTorch Dataset）和 `collate_fn`（padding 对齐）；包含 Windows GBK 编码兼容处理。 |
| **`augumentation.py`** | 数据增强模块。对英文源句实施 4 种温和增强：① 同义词替换（WordNet）；② 随机插入同义词；③ 相邻词交换；④ 低概率随机删除。中文译文保持不变。同义词替换权重最高，确保增强后语义完整。 |
| **`train.py`** | 训练核心。实现完整训练循环：AdamW 优化器 + Warmup + CosineAnnealing 学习率调度、CrossEntropy 损失（含 label smoothing）、梯度裁剪、Early Stopping（带 warmup 保护期和容忍度）。训练结束后绘制损失/学习率曲线并保存为图片。 |
| **`evaluation.py`** | 评估与推理模块。实现 Greedy Decoding 自回归解码；`translate()` 翻译单条句子；`evaluate_bleu()` 在验证集上计算 BLEU；`show_samples()` 展示源文/参考译文/模型输出对照；`load_and_evaluate()` 加载 checkpoint 并执行完整评估流程。 |

---

## 三、环境配置

### 硬件要求

- **GPU**：需要 CUDA 兼容的 NVIDIA GPU（代码硬编码使用 `cuda` 设备）
- **显存**：建议 ≥ 4 GB

### Python 依赖

```
torch           （PyTorch，需 CUDA 版本）
transformers    （HuggingFace Transformers，仅用其 tokenizer）
datasets        （HuggingFace Datasets，下载 FLORES-200）
sacrebleu       （BLEU 评分）
nltk            （WordNet 同义词，用于数据增强）
matplotlib      （绘制训练曲线）
tqdm            （进度条）
```

### 安装步骤

```bash
# 1. 创建虚拟环境（推荐）
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 2. 安装 PyTorch（根据你的 CUDA 版本选择，参考 https://pytorch.org）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装其他依赖
pip install transformers datasets sacrebleu nltk matplotlib tqdm
```

> **注意**：首次运行时会自动下载 FLORES-200 数据集（约 50 MB）以及 BERT 分词器文件，请确保网络畅通。NLTK 的 WordNet 数据也会自动下载。

---

## 四、训练过程

### 超参数配置（`main.py` 中的 `CONFIG`）

| 参数 | 值 | 说明 |
|------|------|------|
| `batch_size` | 32 | 每批样本数 |
| `max_len` | 128 | 序列最大长度 |
| `augment_factor` | 2 | 每条样本生成 2 个增强副本，训练集约扩大 3 倍 |
| `d_model` | 256 | Transformer 隐藏维度 |
| `nhead` | 8 | 多头注意力头数 |
| `num_encoder_layers` | 3 | Encoder 层数 |
| `num_decoder_layers` | 3 | Decoder 层数 |
| `dim_feedforward` | 512 | FFN 中间层维度 |
| `dropout` | 0.3 | Dropout 比率 |
| `num_epochs` | 100 | 最大训练轮数 |
| `lr` | 3e-4 | 初始学习率 |
| `warmup_epochs` | 5 | 学习率线性预热轮数 |
| `weight_decay` | 0.01 | AdamW 权重衰减 |
| `label_smoothing` | 0.1 | 标签平滑系数 |
| `clip_grad` | 1.0 | 梯度裁剪最大范数 |
| `patience` | 15 | Early Stopping 耐心值 |
| `min_delta` | 0.02 | 验证损失需下降的最小幅度 |

### 训练流程概览

```
1. 数据准备
   ├─ 下载 FLORES-200 英中平行语料
   ├─ 使用 BERT tokenizer 对文本分词编码
   └─ 对英文训练数据执行增强 (997 → ~2991 句)

2. 模型构建
   ├─ 构建 Transformer Encoder-Decoder (随机初始化, Kaiming)
   └─ 移至 GPU

3. 训练循环 (每个 epoch)
   ├─ 前向传播 → 计算 CrossEntropy Loss (含 label smoothing)
   ├─ 反向传播 → 梯度裁剪 → AdamW 更新
   ├─ 学习率调度：Warmup (前 5 epoch) → CosineAnnealing
   ├─ 验证集评估
   └─ Early Stopping 检查（warmup 期内不计入）

4. 保存最优模型 → best_model.pt
5. 绘制训练曲线 → training_curves.png
```

### 关键训练策略

- **学习率调度**：前 5 个 epoch 线性预热（从 0.1×lr 升至 lr），之后使用余弦退火逐步降至 `eta_min=1e-6`。
- **Early Stopping**：验证损失连续 15 个 epoch 未下降超过 0.02 时提前停止，warmup 期间不计入。
- **Kaiming 初始化**：相比 Transformer 默认的 Xavier 初始化，在此小数据集上表现更好。
- **温和数据增强**：同义词替换权重最高，避免产生语义偏移的"乱码"英文。

---

## 五、训练结果

### 模型规模

- 模型参数量：约 **1500 万**（取决于具体词表大小）
- Encoder: 3 层, Decoder: 3 层, d_model=256, nhead=8

### 评估指标

- 使用 **sacrebleu** 在 devtest（1012 句）验证集上计算中文 BLEU 分数。
- 训练结束后会自动打印 BLEU 分数并展示 10 条样例翻译对照（源文 / 参考译文 / 模型输出）。

### 输出文件

| 文件 | 说明 |
|------|------|
| `best_model.pt` | 验证集损失最低时保存的模型权重 |
| `training_curves.png` | 训练/验证损失曲线 + 学习率变化曲线 |

> **注意**：由于训练集仅约 3000 句（增强后），属于极小规模数据集，BLEU 分数会受到数据量限制。此项目主要用于学习和演示 Transformer 翻译模型的完整流程。

---

## 六、如何使用

### 方式 1：完整流程（训练 + 评估 + 交互翻译）

```bash
python main.py
```

程序将依次执行：
1. **训练模型**：自动下载数据、增强、训练，保存最优权重至 `best_model.pt`
2. **评估模型**：加载最优模型，计算 BLEU 分数，展示样例翻译
3. **交互翻译**：进入命令行交互模式，输入英文句子即时翻译为中文，输入 `q` 退出

### 方式 2：仅评估已有模型

如果已有训练好的 `best_model.pt`，可直接运行评估：

```bash
python evaluation.py
```

将加载 checkpoint 并在验证集上计算 BLEU、展示样例翻译。

### 方式 3：仅训练

```bash
python train.py
```

使用默认超参数进行训练，训练结束后保存模型和曲线图。

### 方式 4：测试数据增强

```bash
python augumentation.py
```

对示例英文句子展示各种增强效果。

### 方式 5：测试数据加载

```bash
python dataset.py
```

下载数据集并打印一个 batch 的 shape，验证数据管线是否正常。

---

## 七、技术架构图

```
┌──────────────┐     ┌──────────────┐
│  英文句子     │     │  BERT-uncased │
│  (English)   │────▶│  Tokenizer   │────▶ src token ids
└──────────────┘     └──────────────┘
                                          │
                                          ▼
                            ┌──────────────────────┐
                            │   Embedding (256d)    │
                            │ + Positional Encoding │
                            └──────────┬───────────┘
                                       │
                            ┌──────────▼───────────┐
                            │   Transformer Encoder │
                            │     (3 layers)        │
                            └──────────┬───────────┘
                                       │ memory
                            ┌──────────▼───────────┐
                            │   Transformer Decoder │
                            │     (3 layers)        │
                            └──────────┬───────────┘
                                       │
                            ┌──────────▼───────────┐
                            │   Linear → Softmax    │
                            └──────────┬───────────┘
                                       │
                                       ▼
                            ┌──────────────────────┐
                            │  BERT-Chinese         │
┌──────────────┐     ◀──── │  Tokenizer (decode)   │
│  中文译文     │            └──────────────────────┘
│  (Chinese)   │
└──────────────┘
```

