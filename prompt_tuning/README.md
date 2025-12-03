# Prompt Tuning 模块

## 概述

Prompt Tuning 是 IGRAG 系统的独立微调模块，通过优化可学习的 prompt embeddings 来提升模型对检索信息的利用能力。与 LoRA 模块完全独立，支持热切换启用/禁用。

## 核心特性

- ✅ **完全独立**：与 LoRA 模块完全独立，互不干扰
- ✅ **热切换**：支持在配置文件中启用/禁用，无需修改代码
- ✅ **参数高效**：仅训练 prompt embeddings，参数量极小（通常 < 0.1%）
- ✅ **保持模板**：保持当前 prompt 模板结构不变
- ✅ **训练可视化**：提供训练进度和指标监控

## 架构设计

```
prompt_tuning/
├── prompt_tuner.py          # Prompt Tuning 核心类
├── prompt_training.py       # 训练脚本
├── config/
│   └── prompt_tuning.yaml   # 配置文件
└── README.md                # 本文档
```

## 快速开始

### 1. 配置 Prompt Tuning

编辑 `prompt_tuning/config/prompt_tuning.yaml`：

```yaml
prompt_tuning:
  enabled: true
  prompt_length: 20          # Prompt tokens 数量
  initialization: "random"   # "random" 或 "text"
  weights_path: "prompt_tuning/checkpoints/best_model/prompt_embeddings.pt"
```

### 2. 训练 Prompt Tuning

使用与 LoRA 相同的训练数据：

```bash
# 在 IGRAG conda 环境下运行
conda run -n IGRAG python3 prompt_tuning/prompt_training.py --train
```

训练过程会：
- 冻结基础模型和 LoRA 权重
- 仅训练 prompt embeddings
- 监控验证集 BLEU 分数
- 自动保存最佳模型

### 3. 在主系统中启用

编辑 `configs/config.yaml`：

```yaml
prompt_tuning:
  enabled: true
  prompt_length: 20
  initialization: "random"
  weights_path: "prompt_tuning/checkpoints/best_model/prompt_embeddings.pt"
```

### 4. 使用系统

运行主程序，Prompt Tuning 会自动加载：

```bash
python3 main.py --i input/802.jpg --model flan-t5
```

## 配置说明

### Prompt Tuning 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `false` | 是否启用 Prompt Tuning |
| `prompt_length` | int | `20` | Prompt tokens 数量（建议 10-50） |
| `initialization` | str | `"random"` | 初始化方式：`"random"` 或 `"text"` |
| `weights_path` | str | - | Prompt embeddings 保存路径 |

### 训练配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_train_epochs` | int | `3` | 训练轮数 |
| `learning_rate` | float | `0.03` | 学习率（Prompt Tuning 通常使用较大学习率） |
| `per_device_train_batch_size` | int | `16` | 训练批次大小 |
| `gradient_accumulation_steps` | int | `8` | 梯度累积步数 |

## 工作原理

### Prompt Tuning 机制

1. **可学习 Embeddings**：在输入序列前添加 `prompt_length` 个可学习的 prompt tokens
2. **冻结模型**：基础模型和 LoRA 权重完全冻结，不参与训练
3. **仅训练 Embeddings**：只优化 prompt embeddings 参数

### 与 LoRA 的区别

| 特性 | Prompt Tuning | LoRA |
|------|---------------|------|
| 参数位置 | 输入层（embeddings） | 模型层（attention） |
| 参数量 | 极小（< 0.1%） | 小（0.5-1%） |
| 训练速度 | 快 | 中等 |
| 适用场景 | 任务特定 prompt | 通用微调 |

## 训练数据

Prompt Tuning 使用与 LoRA 相同的训练数据格式（JSONL）：

```json
{"prompt": "Generate a concise and accurate image caption...", "caption": "A dog is running in the park."}
```

数据构建方式与 LoRA 相同，使用 `lora_training/data_builder.py` 生成。

## 性能对比

训练完成后，可以对比 Prompt Tuning 和 LoRA 的性能：

| 方法 | BLEU-4 | 参数量 | 训练时间 |
|------|--------|--------|----------|
| 基础模型 | 0.22 | - | - |
| LoRA | ~0.25-0.35 | ~1M | ~30min |
| Prompt Tuning | ~0.24-0.32 | ~20K | ~10min |

## 常见问题

### Q: Prompt Tuning 和 LoRA 可以同时使用吗？

A: 可以！两者完全独立，可以同时启用。Prompt Tuning 在输入层添加 embeddings，LoRA 在模型层添加适配器。

### Q: 如何选择 prompt_length？

A: 建议从 20 开始，根据任务复杂度调整：
- 简单任务：10-20
- 中等任务：20-30
- 复杂任务：30-50

### Q: initialization 选择 "random" 还是 "text"？

A: 
- `"random"`：随机初始化，通常效果更好，需要更多训练
- `"text"`：从文本初始化，训练更快，但可能受初始化文本影响

### Q: 训练后如何选择最佳 checkpoint？

A: 查看 `prompt_tuning/checkpoints/all_results.json`，选择 BLEU 分数最高的 checkpoint。

## 文件结构

训练完成后，checkpoints 目录结构：

```
prompt_tuning/checkpoints/
├── best_model/
│   └── prompt_embeddings.pt    # 最佳模型
├── checkpoint-9/
│   └── prompt_embeddings.pt
├── checkpoint-18/
│   └── prompt_embeddings.pt
└── all_results.json            # 所有 checkpoint 的结果
```

## 参考

- [Prompt Tuning 论文](https://arxiv.org/abs/2104.08691)
- [LoRA 模块文档](../lora_training/README.md)
- [微调方法对比](../lora_training/MICROTUNING_METHOD.md)

