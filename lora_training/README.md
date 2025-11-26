# LoRA 微调模块快速开始指南

## 📖 模块简介

`lora_training` 模块为 IGRAG 系统提供 LoRA（Low-Rank Adaptation）微调功能，通过少量参数训练提升 FLAN-T5 模型在图像描述生成任务上的表现。

**核心优势**：
- 🎯 针对 COCO 风格描述优化，BLEU-4 可提升至 0.25-0.35
- 💡 仅训练少量参数（LoRA 适配器），保持模型轻量化
- 🔄 无缝集成到现有 IGRAG 系统，无需修改推理代码

---

## 🚀 快速开始（推荐方式：使用训练脚本）

### 方式一：使用训练脚本（推荐）✨

**完整流程（构建数据 + 训练）**：

```bash
# 从项目根目录运行
python3 lora_training/train_lora.py --all
```

**仅构建训练数据**：

```bash
python3 lora_training/train_lora.py --build-data --sample-count 5000
```

**仅训练模型**（需要已有训练数据）：

```bash
python3 lora_training/train_lora.py --train
```

**自定义配置**：

```bash
python3 lora_training/train_lora.py --all \
    --main-config configs/config.yaml \
    --lora-config lora_training/config/lora_config.yaml \
    --sample-count 5000 \
    --train-ratio 0.9
```

**查看所有选项**：

```bash
python3 lora_training/train_lora.py --help
```

---

### 方式二：使用 Python API

如果你更喜欢在代码中控制流程：

**步骤 1：构建训练数据**

```python
from lora_training.data_builder import LoraTrainingDataBuilder, split_dataset

# 构建 5000 个训练样本
builder = LoraTrainingDataBuilder(
    main_config_path="configs/config.yaml",
    sample_count=5000,
    output_path="lora_training/data/coco_lora_train.jsonl",
    seed=42
)

# 生成训练数据
stats = builder.build()
print(f"✅ 生成 {stats['num_samples']} 个样本，保存至 {stats['output_path']}")

# 自动切分为训练集和验证集（9:1）
split_dataset(stats["output_path"], train_ratio=0.9, seed=42)
```

**步骤 2：配置并启动训练**

编辑 `lora_training/config/lora_config.yaml`，确认以下关键参数：

```yaml
model:
  base_model_path: "../models/flan-t5-large"  # 基础模型路径
  type: "flan-t5"

data:
  train_path: "lora_training/data/coco_lora_train_train.jsonl"
  val_path: "lora_training/data/coco_lora_train_val.jsonl"

training:
  num_train_epochs: 3          # 训练轮数
  train_batch_size: 4          # 根据 GPU 显存调整
  gradient_accumulation_steps: 8  # 有效 batch = 4 × 8 = 32
  learning_rate: 1.0e-4
  output_dir: "lora_training/checkpoints"

lora:
  r: 16                        # LoRA 秩（平衡效果与参数量）
  lora_alpha: 32              # 缩放参数（通常为 2×r）
  dropout: 0.1
  target_modules: ["q", "v"]  # 针对查询和值矩阵适配
```

启动训练：

```python
from lora_training.lora_trainer import LoraCaptionTrainer

# 初始化训练器（自动加载配置）
trainer = LoraCaptionTrainer("lora_training/config/lora_config.yaml")

# 开始训练（每个 epoch 自动保存 checkpoint 并评估 BLEU）
train_result = trainer.train()

# 运行最终评估
eval_metrics = trainer.evaluate()
print(f"✅ 最终 BLEU 分数: {eval_metrics.get('eval_bleu', 'N/A')}")
```

**输出文件**：
- `lora_training/data/coco_lora_train_train.jsonl` - 训练集（4500 样本）
- `lora_training/data/coco_lora_train_val.jsonl` - 验证集（500 样本）
- `lora_training/checkpoints/checkpoint-{step}/` - 每个 epoch 的检查点
- `lora_training/checkpoints/best/` - 最佳模型（基于验证集 BLEU）

---

### 步骤 3：集成到 IGRAG 系统

在 `configs/config.yaml` 中启用 LoRA：

```yaml
lora_config:
  enabled: true
  weights_path: "lora_training/checkpoints/best"  # 最佳 checkpoint 路径
  merge_and_unload: false  # false=动态加载，true=合并到基础模型
```

重新运行 IGRAG 系统，`CaptionGenerator` 会自动加载 LoRA 适配器：

```bash
python3 main.py --i input/802.jpg --model flan-t5
```

---

## 📁 文件结构

```
lora_training/
├── README.md                    # 本文档
├── __init__.py
├── data_builder.py              # 训练数据构建器
├── lora_trainer.py              # LoRA 训练器主类
├── config/
│   └── lora_config.yaml         # 训练配置文件
└── utils/
    ├── __init__.py
    └── training_utils.py        # 数据集、工具函数
```

---

## 🔧 核心组件说明

### `data_builder.py`

**`LoraTrainingDataBuilder`**：从 COCO 数据集构建训练样本
- 使用 IGRAG 检索器生成 prompt（包含全局描述 + 局部位置信息）
- 使用 COCO 标注作为目标 caption
- 输出 JSONL 格式：`{"image_id": ..., "prompt": ..., "caption": ..., "metadata": ...}`

**`split_dataset()`**：将数据集切分为训练/验证集

---

### `lora_trainer.py`

**`LoraCaptionTrainer`**：端到端训练入口
- 自动加载基础模型（FLAN-T5）
- 应用 LoRA 配置（PEFT）
- 使用 Transformers Trainer 进行训练
- 自动计算验证集 BLEU 分数
- 保存最佳 checkpoint

---

### `config/lora_config.yaml`

训练配置分为 4 个部分：
- **`model`**：基础模型路径和类型
- **`data`**：训练/验证数据路径和长度限制
- **`training`**：训练超参数（epochs, batch size, learning rate 等）
- **`lora`**：LoRA 特定参数（r, alpha, dropout, target_modules）

---

## 💡 使用示例

### 完整训练流程脚本

```python
#!/usr/bin/env python3
"""完整的 LoRA 微调流程"""

from pathlib import Path
from lora_training.data_builder import LoraTrainingDataBuilder, split_dataset
from lora_training.lora_trainer import LoraCaptionTrainer

# === 阶段 1：构建训练数据 ===
print("📦 阶段 1: 构建训练数据...")
builder = LoraTrainingDataBuilder(
    main_config_path="configs/config.yaml",
    sample_count=5000,
    output_path="lora_training/data/coco_lora_train.jsonl",
    seed=42
)
stats = builder.build()
print(f"✅ 生成 {stats['num_samples']} 个样本")

# 切分数据集
split_dataset(stats["output_path"], train_ratio=0.9, seed=42)
print("✅ 数据集切分完成\n")

# === 阶段 2：训练 ===
print("🚀 阶段 2: 开始 LoRA 训练...")
trainer = LoraCaptionTrainer("lora_training/config/lora_config.yaml")

# 训练 3 个 epoch
train_result = trainer.train()
print(f"✅ 训练完成，损失: {train_result.metrics.get('train_loss', 'N/A')}")

# 评估
eval_metrics = trainer.evaluate()
print(f"✅ 验证集 BLEU: {eval_metrics.get('eval_bleu', 'N/A')}\n")

# === 阶段 3：提示集成 ===
print("📝 阶段 3: 请手动在 configs/config.yaml 中启用 LoRA:")
print("""
lora_config:
  enabled: true
  weights_path: "lora_training/checkpoints/best"
  merge_and_unload: false
""")
```

---

## ⚙️ 参数调优建议

### LoRA 参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `r` | 16 | 秩，越大效果越好但参数越多（8/16/32 常见） |
| `lora_alpha` | 32 | 通常设为 `2 × r` |
| `dropout` | 0.1 | 防止过拟合（0.05-0.2 范围） |
| `target_modules` | `["q", "v"]` | 针对注意力层的查询和值矩阵 |

### 训练参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `num_train_epochs` | 3 | 通常 3-5 个 epoch 足够 |
| `learning_rate` | 1e-4 | 可尝试 5e-5 到 2e-4 |
| `train_batch_size` | 4 | 根据 GPU 显存调整（2-8） |
| `gradient_accumulation_steps` | 8 | 有效 batch = batch_size × accumulation |

---

## ❓ 常见问题

### Q1: 训练需要多少 GPU 显存？

**A**: 使用 `train_batch_size=4, gradient_accumulation_steps=8` 时：
- FLAN-T5-base: 约 6-8 GB
- FLAN-T5-large: 约 12-16 GB

如果显存不足，减小 `train_batch_size` 或增大 `gradient_accumulation_steps`。

---

### Q2: 如何选择最佳 checkpoint？

**A**: 训练器会自动选择验证集 BLEU 最高的 checkpoint 保存为 `best/`。你也可以：
- 查看 `lora_training/checkpoints/` 下的各 checkpoint
- 检查训练日志中的 `eval_bleu` 分数
- 手动测试不同 checkpoint 在验证集上的表现

---

### Q3: `merge_and_unload` 选项的作用？

**A**: 
- `false`（推荐）：动态加载 LoRA 适配器，保持基础模型不变，可灵活切换不同 LoRA
- `true`：将 LoRA 权重合并到基础模型并卸载适配器，生成新的完整模型（占用更多空间）

---

### Q4: 训练数据不够怎么办？

**A**: 可以：
- 增加 `sample_count`（如 10000）
- 使用更多 COCO 训练集图像
- 调整 `data_builder.py` 中的检索参数以获取更丰富的 prompt

---

### Q5: 如何监控训练过程？

**A**: 
- 查看控制台输出的 `eval_bleu` 和 `eval_loss`
- 检查 `lora_training/checkpoints/training_state.json` 中的训练历史
- 如果配置了 `report_to: ["tensorboard"]`，可使用 TensorBoard 可视化

---

## 📚 相关文档

- 主项目 README: `../README.md`（包含 LoRA 集成说明）
- 配置文件: `configs/config.yaml`（主系统配置）
- 训练配置: `config/lora_config.yaml`（LoRA 训练配置）

---

## 🎯 预期效果

使用推荐的 LoRA 配置（`r=16, alpha=32`）训练 3 个 epoch 后：
- **BLEU-4**: 从 0.22 提升至 **0.25-0.35**
- **描述质量**: 更符合 COCO 风格，更准确的空间关系描述
- **参数量**: 仅增加约 1-2% 的可训练参数

---

**祝训练顺利！** 🎉

