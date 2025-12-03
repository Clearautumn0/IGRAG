# BLIP-2 密集描述生成迁移说明

## 概述

本次迁移将密集描述生成从 mmdetection 模型改为使用 BLIP-2 模型，以规避复杂环境依赖问题，并利用提示词工程生成更丰富的描述性短语。

## 主要更改

### 1. 脚本重写: `scripts/build_dense_knowledge_base.py`

**原实现**: 使用 mmdetection (mm_grounding_dino_tiny_o365v1_goldg_grit_v3det)  
**新实现**: 使用 BLIP-2 (blip2-opt-2.7b)

**核心功能**:
- ✅ 使用 transformers 库加载 BLIP-2 模型
- ✅ 支持 float16 以节省显存
- ✅ 提示词工程生成短语列表
- ✅ 按分号分割输出并清理
- ✅ 批次处理（逐张处理，批量显示进度）
- ✅ 进度条显示（tqdm）
- ✅ 错误处理（跳过失败图像）
- ✅ 断点恢复（从已有pickle文件继续）

### 2. 配置更新: `configs/config.yaml`

新增/更新了 `dense_descriptor` 配置节：

```yaml
dense_descriptor:
  model_type: "blip2"  # 模型类型
  model_path: "../models/blip2-opt-2.7b/"  # 模型路径
  prompt: "Question: List the objects, scenes, and actions in this image with very short phrases. Answer: "  # 提示词
  knowledge_base_path: "./output/image_id_to_dense_captions.pkl"  # 输出路径
  embedding_model_path: "../models/all-MiniLM-L6-v2/"  # 句子嵌入模型
  batch_size: 8  # 批处理大小（用于进度显示）
  max_new_tokens: 100  # 最大生成token数
  num_beams: 5  # beam search数量
  checkpoint_interval: 100  # 检查点保存间隔
```

**关键特性**:
- ✅ 提示词可从配置文件读取，便于集中调整
- ✅ 所有参数可配置

### 3. 提示词工程

**固定提示词**:
```
"Question: List the objects, scenes, and actions in this image with very short phrases. Answer: "
```

**输出处理**:
- 模型生成文本（如: "a dog; a frisbee; a man running; green grass"）
- 按分号 `;` 分割
- 去除首尾空格
- 过滤空字符串
- 返回列表: `['a dog', 'a frisbee', 'a man running', 'green grass']`

### 4. 测试脚本: `test_blip2_dense_captions.py`

创建了测试脚本用于验证功能：
- 处理少量图像（默认10张）
- 验证输出格式
- 显示统计信息
- 输出示例结果

**使用方法**:
```bash
# 默认处理10张图像
python test_blip2_dense_captions.py

# 处理指定数量图像
python test_blip2_dense_captions.py --num-images 20
```

## 使用方法

### 1. 准备模型

确保 BLIP-2 模型已下载到指定路径：
```bash
ls -la ../models/blip2-opt-2.7b/
```

### 2. 运行测试（推荐先运行）

```bash
python test_blip2_dense_captions.py --num-images 10
```

这将处理10张图像并验证输出格式。

### 3. 构建完整知识库

```bash
python scripts/build_dense_knowledge_base.py
```

脚本会自动：
- 加载配置
- 从检查点恢复（如果存在）
- 处理所有COCO训练图像
- 定期保存检查点
- 显示进度

### 4. 验证输出

输出文件: `./output/image_id_to_dense_captions.pkl`

格式: `{image_id: [phrase1, phrase2, ...]}`

示例:
```python
{
    123: ['a dog', 'a frisbee', 'a man running', 'green grass'],
    456: ['a car', 'a street', 'traffic lights', 'urban scene']
}
```

## 技术细节

### 模型加载

```python
processor = Blip2Processor.from_pretrained(model_path)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16  # 节省显存
)
model = model.to(device).eval()
```

### 生成参数

- `max_new_tokens=100`: 最大生成100个token
- `num_beams=5`: 使用5路beam search提高质量
- `early_stopping=True`: 提前停止优化速度

### 错误处理

- 单张图像失败不会中断整个流程
- 错误会被记录到日志
- 失败的图像会被跳过
- 已处理的图像会保存到检查点

### 断点恢复

- 脚本会自动检测已有的pickle文件
- 只处理未处理的图像
- 可以随时中断和恢复

## 验证步骤

### 步骤1: 测试脚本

运行测试脚本处理少量图像：
```bash
python test_blip2_dense_captions.py --num-images 10
```

验证：
- ✅ 输出格式为 `{image_id: [str, str, ...]}`
- ✅ 短语描述具体、多样
- ✅ 无错误信息

### 步骤2: 检查配置

确认配置文件中的提示词正确：
```bash
grep -A 2 "dense_descriptor:" configs/config.yaml
```

### 步骤3: 运行完整构建

处理所有图像：
```bash
python scripts/build_dense_knowledge_base.py
```

## 优势

1. **简单依赖**: 只需 transformers 库，无需 mmdetection
2. **灵活提示**: 提示词可配置，便于调整
3. **易于调试**: 清晰的日志和进度显示
4. **容错性强**: 单图失败不影响整体
5. **可恢复**: 支持断点续传

## 注意事项

1. **显存使用**: 模型使用 float16 以节省显存，如需更高精度可改为 float32
2. **生成时间**: BLIP-2 生成需要一定时间，建议使用GPU加速
3. **提示词**: 当前提示词已优化，如需修改请谨慎测试
4. **输出格式**: 输出按分号分割，确保提示词引导模型使用分号分隔

## 故障排查

### 问题1: 模型加载失败

**错误**: `FileNotFoundError` 或 `OSError`

**解决**:
- 检查模型路径是否正确
- 确认模型文件已完整下载

### 问题2: 显存不足

**错误**: `CUDA out of memory`

**解决**:
- 确保使用 `torch_dtype=torch.float16`
- 减小 `batch_size`（虽然实际逐张处理）
- 使用CPU（较慢但可用）

### 问题3: 生成短语为空

**可能原因**:
- 提示词不匹配
- 图像质量问题
- 模型输出格式不符合预期

**解决**:
- 检查提示词是否正确
- 查看日志中的错误信息
- 尝试调整生成参数

## 下一步

1. 运行测试脚本验证功能
2. 根据测试结果调整配置
3. 运行完整构建脚本
4. 使用生成的知识库进行检索

