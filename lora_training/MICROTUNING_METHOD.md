# IGRAG 微调方法说明

## 当前使用的微调方式：**LoRA (Low-Rank Adaptation)**

### 确认依据

1. **代码实现**：
   - 使用 `peft.LoraConfig` 配置 LoRA 参数
   - 使用 `peft.get_peft_model()` 将基础模型转换为 LoRA 模型
   - 使用 `peft.PeftModel.from_pretrained()` 加载 LoRA 适配器

2. **配置参数**（`lora_training/config/lora_config.yaml`）：
   ```yaml
   lora:
     r: 16                    # LoRA 秩（rank）
     lora_alpha: 32          # LoRA 缩放参数
     dropout: 0.1            # LoRA dropout
     target_modules: ["q", "v"]  # 目标模块（查询和值矩阵）
   ```
   这些都是 **LoRA 的典型参数**，不是 Adapter 的参数。

3. **文件命名说明**：
   - `adapter_config.json` 和 `adapter_model.safetensors` 是 PEFT 库的**命名约定**
   - 虽然文件名包含 "adapter"，但实际内容是 **LoRA 配置和权重**
   - PEFT 库统一使用 "adapter" 作为通用术语，但通过 `peft_type` 字段区分具体方法

### 三种微调方式对比

| 特性 | Prompt Tuning | Adapter Tuning | **LoRA (当前使用)** |
|------|---------------|----------------|---------------------|
| **原理** | 在输入层添加可学习的 prompt tokens | 在模型中插入小的适配器层 | 对权重矩阵进行低秩分解 |
| **参数量** | 很少（仅 prompt tokens） | 中等（适配器层） | **很少（仅低秩矩阵）** |
| **训练速度** | 快 | 中等 | **快** |
| **灵活性** | 低（仅影响输入） | 中等 | **高（可针对不同模块）** |
| **实现库** | 自定义或 PEFT | PEFT (AdapterConfig) | **PEFT (LoraConfig)** |
| **当前项目** | ❌ 未使用 | ❌ 未使用 | ✅ **使用中** |

### LoRA 工作原理

LoRA 通过以下方式工作：

1. **低秩分解**：
   - 原始权重矩阵 W 被分解为：W = W₀ + ΔW
   - 其中 ΔW = BA，B 和 A 是低秩矩阵（rank = r）
   - 例如：如果 W 是 768×768，r=16，则 B 是 768×16，A 是 16×768

2. **目标模块**：
   - 当前配置针对 `["q", "v"]`（查询和值矩阵）
   - 这是 Transformer 注意力机制中的关键组件
   - 只训练这些模块的 LoRA 权重，冻结其他参数

3. **参数效率**：
   - 可训练参数：1,769,472（约 0.71%）
   - 总参数：249,347,328
   - 仅训练少量参数即可获得良好效果

### 为什么不是 Adapter Tuning？

虽然文件名包含 "adapter"，但项目**不是**使用 Adapter Tuning：

1. **配置类型**：
   - Adapter 使用 `AdapterConfig`
   - 当前使用 `LoraConfig` ✅

2. **参数不同**：
   - Adapter 需要配置 `adapter_layers`, `adapter_dim` 等
   - LoRA 使用 `r`, `lora_alpha`, `target_modules` ✅

3. **文件内容**：
   - `adapter_config.json` 中的 `peft_type` 字段应该是 `"LORA"`，不是 `"ADAPTER"`

### 验证方法

运行以下命令查看实际配置：

```bash
python3 -c "
import json
with open('lora_training/checkpoints/adapter_config.json', 'r') as f:
    config = json.load(f)
print('peft_type:', config.get('peft_type'))
print('r:', config.get('r'))
print('lora_alpha:', config.get('lora_alpha'))
"
```

如果 `peft_type` 是 `"LORA"`，则确认使用的是 LoRA。

### 总结

✅ **当前项目使用的是 LoRA (Low-Rank Adaptation) 微调方式**

- 不是 Prompt Tuning
- 不是 Adapter Tuning  
- 不是混合方式
- 是**纯 LoRA 实现**

文件命名中的 "adapter" 只是 PEFT 库的通用术语，实际微调方法是 LoRA。

