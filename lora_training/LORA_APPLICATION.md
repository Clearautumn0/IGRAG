# LoRA 在 IGRAG 系统中的应用说明

## 概述

本文档详细说明 LoRA (Low-Rank Adaptation) 在 IGRAG 系统中是如何应用的，以及它具体应用到模型的哪些层。

## 一、LoRA 应用流程

### 1.1 训练阶段（`lora_training/lora_trainer.py`）

在训练时，LoRA 适配器通过以下步骤创建和应用：

```106:114:lora_training/lora_trainer.py
        lora_alpha = self.lora_cfg.get("lora_alpha", self.lora_cfg.get("alpha", 32))
        lora_config = LoraConfig(
            r=int(self.lora_cfg.get("r", 16)),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(self.lora_cfg.get("dropout", 0.1)),
            target_modules=self.lora_cfg.get("target_modules", ["q", "v"]),
            bias="none",
            task_type=task_type,
        )
        self.model = get_peft_model(self.model, lora_config)
```

**关键步骤**：
1. 加载基础模型（FLAN-T5-base）
2. 创建 `LoraConfig` 配置对象，指定目标模块为 `["q", "v"]`
3. 使用 `get_peft_model()` 将基础模型包装为 PEFT 模型
4. PEFT 库会自动在匹配的模块上注入 LoRA 适配器

### 1.2 推理阶段（`core/generator.py`）

在推理时，LoRA 适配器通过以下步骤加载：

```199:203:core/generator.py
                            self.model = PeftModel.from_pretrained(
                                self.model,
                                self.lora_weights_path,
                                is_trainable=False,
                            )
```

**关键步骤**：
1. 首先加载基础模型（FLAN-T5-base）
2. 使用 `PeftModel.from_pretrained()` 加载训练好的 LoRA 适配器权重
3. LoRA 权重会自动应用到匹配的模块上
4. 设置 `is_trainable=False` 确保推理时不会更新权重

## 二、LoRA 应用到哪些层

### 2.1 目标模块配置

根据配置文件 `lora_training/config/lora_config.yaml`：

```yaml
lora:
  r: 16
  lora_alpha: 32
  dropout: 0.1
  target_modules: ["q", "v"]  # 查询和值矩阵
```

### 2.2 FLAN-T5 模型结构

FLAN-T5 是一个编码器-解码器（Encoder-Decoder）架构的 Transformer 模型：

- **Encoder**: 12 层 Transformer 块
- **Decoder**: 12 层 Transformer 块

每一层都包含：
- **Self-Attention 层**：包含查询（Q）、键（K）、值（V）矩阵
- **Cross-Attention 层**（仅在 Decoder 中）：包含查询（Q）、键（K）、值（V）矩阵
- **Feed-Forward 层**

### 2.3 实际应用的模块

当 `target_modules: ["q", "v"]` 时，PEFT 库会匹配所有名称包含 "q" 或 "v" 的线性层。

对于 FLAN-T5-base 模型，LoRA 适配器会应用到以下模块：

#### Encoder 部分（12 层）
- `encoder.block.0.layer.0.SelfAttention.q` - 第 0 层查询矩阵
- `encoder.block.0.layer.0.SelfAttention.v` - 第 0 层值矩阵
- `encoder.block.1.layer.0.SelfAttention.q` - 第 1 层查询矩阵
- `encoder.block.1.layer.0.SelfAttention.v` - 第 1 层值矩阵
- ...（共 12 层）
- `encoder.block.11.layer.0.SelfAttention.q` - 第 11 层查询矩阵
- `encoder.block.11.layer.0.SelfAttention.v` - 第 11 层值矩阵

#### Decoder 部分（12 层）
- `decoder.block.0.layer.0.SelfAttention.q` - 第 0 层自注意力查询矩阵
- `decoder.block.0.layer.0.SelfAttention.v` - 第 0 层自注意力值矩阵
- `decoder.block.0.layer.1.EncDecAttention.q` - 第 0 层交叉注意力查询矩阵
- `decoder.block.0.layer.1.EncDecAttention.v` - 第 0 层交叉注意力值矩阵
- ...（共 12 层）
- `decoder.block.11.layer.0.SelfAttention.q` - 第 11 层自注意力查询矩阵
- `decoder.block.11.layer.0.SelfAttention.v` - 第 11 层自注意力值矩阵
- `decoder.block.11.layer.1.EncDecAttention.q` - 第 11 层交叉注意力查询矩阵
- `decoder.block.11.layer.1.EncDecAttention.v` - 第 11 层交叉注意力值矩阵

**总计**（已验证）：
- Encoder Q: **12 个模块**
- Encoder V: **12 个模块**
- Decoder Q: **24 个模块**（12 个 SelfAttention + 12 个 EncDecAttention）
- Decoder V: **24 个模块**（12 个 SelfAttention + 12 个 EncDecAttention）
- **总共 72 个模块**会应用 LoRA 适配器

### 2.4 为什么选择 Q 和 V 矩阵？

1. **Q（查询矩阵）**：负责计算注意力权重，决定模型关注哪些位置
2. **V（值矩阵）**：包含实际的信息内容，是注意力机制的信息来源

选择这两个矩阵的原因：
- **参数效率**：只微调 Q 和 V 可以显著减少可训练参数（约 0.71%）
- **效果平衡**：Q 和 V 的组合通常能提供良好的微调效果
- **常见实践**：这是 LoRA 微调中的常见配置

## 三、LoRA 工作原理

### 3.1 低秩分解

对于每个目标模块的权重矩阵 W（例如 Q 或 V），LoRA 将其分解为：

```
W = W₀ + ΔW
ΔW = B × A
```

其中：
- `W₀`：原始预训练权重（冻结，不更新）
- `B`：低秩矩阵，维度为 `(d, r)`，其中 `d` 是原始维度，`r=16` 是秩
- `A`：低秩矩阵，维度为 `(r, d)`

### 3.2 参数计算示例

以 FLAN-T5-base 的注意力层为例：
- 原始 Q/V 矩阵维度：`768 × 768 = 589,824` 个参数
- LoRA 参数：
  - `B`: `768 × 16 = 12,288` 个参数
  - `A`: `16 × 768 = 12,288` 个参数
  - 总计：`24,576` 个参数（约为原始的 4.2%）

对于 72 个模块：
- 总 LoRA 参数：`72 × 24,576 = 1,769,472` 个参数
- 基础模型参数：`~249,347,328` 个参数
- 可训练参数占比：`1,769,472 / 249,347,328 ≈ 0.71%`

### 3.3 前向传播

在前向传播时，LoRA 适配器的计算方式为：

```python
# 原始计算
output = W × input

# LoRA 增强计算
output = (W₀ + B × A) × input
       = W₀ × input + B × A × input
```

其中：
- `W₀ × input`：使用原始权重（冻结）
- `B × A × input`：使用 LoRA 适配器（可训练）

## 四、配置说明

### 4.1 训练配置（`lora_training/config/lora_config.yaml`）

```yaml
lora:
  r: 16                    # 秩（rank），控制低秩矩阵的维度
  lora_alpha: 32           # 缩放参数，通常设为 2×r
  dropout: 0.1            # LoRA 层的 dropout 率
  target_modules: ["q", "v"]  # 目标模块名称模式
```

### 4.2 推理配置（`configs/config.yaml`）

```yaml
lora_config:
  enabled: true
  weights_path: "lora_training/checkpoints"  # LoRA 适配器路径
  merge_and_unload: false  # 是否将 LoRA 权重合并到基础模型
```

**`merge_and_unload` 选项**：
- `false`（默认）：保持 LoRA 适配器独立，可以热插拔
- `true`：将 LoRA 权重合并到基础模型权重中，然后卸载适配器（节省内存，但失去灵活性）

## 五、验证 LoRA 应用

### 5.1 检查适配器配置

```bash
cat lora_training/checkpoints/adapter_config.json | grep -E "peft_type|target_modules|r|lora_alpha"
```

输出示例：
```json
{
  "peft_type": "LORA",
  "target_modules": ["q", "v"],
  "r": 16,
  "lora_alpha": 32
}
```

### 5.2 查看可训练参数

在训练开始时，`lora_trainer.py` 会打印可训练参数信息：

```python
self.model.print_trainable_parameters()
```

输出示例：
```
trainable params: 1,769,472 || all params: 249,347,328 || trainable%: 0.71
```

### 5.3 检查模型结构

```python
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM

base_model = AutoModelForSeq2SeqLM.from_pretrained("/path/to/flan-t5-base")
model = PeftModel.from_pretrained(base_model, "lora_training/checkpoints")

# 查看 LoRA 适配器
for name, module in model.named_modules():
    if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
        print(f"LoRA module: {name}")
```

## 六、总结

1. **应用方式**：
   - 训练时：使用 `get_peft_model()` 创建 LoRA 模型
   - 推理时：使用 `PeftModel.from_pretrained()` 加载适配器

2. **应用层数**：
   - **Encoder**: 12 层的 SelfAttention Q/V 矩阵（24 个模块）
   - **Decoder**: 12 层的 SelfAttention 和 EncDecAttention Q/V 矩阵（48 个模块）
   - **总计**: 72 个模块应用 LoRA 适配器

3. **参数效率**：
   - 仅训练约 0.71% 的参数（1.77M / 249M）
   - 显著降低内存和计算需求

4. **灵活性**：
   - 支持热插拔（不合并时）
   - 可以针对不同任务训练不同的 LoRA 适配器
   - 可以同时加载多个适配器（如果 PEFT 版本支持）

