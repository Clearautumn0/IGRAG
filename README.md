# 基于全局与局部视觉相似性检索的图像描述生成框架

## 项目概述

本项目实现了一个基于CLIP视觉编码器和分层检索策略的图像描述生成框架。该框架通过全局和局部视觉相似性检索，结合大语言模型生成更精确细致的图像描述。

## 系统架构

```
输入查询图片 → CLIP-ViT图像编码器（自动分块处理）
                    ↓
        全局图像特征向量 + N个图像块特征向量
                    ↓
        全局图像检索 + 局部图像块检索
                    ↓
        获取K组全局描述 + 获取M组局部描述
                    ↓
        构建分层提示（指令 + 全局描述 + 局部描述）
                    ↓
        大语言模型（推理与融合）
                    ↓
        生成更精确细致的图像描述
```

## 核心特性

- **多粒度特征提取**: 使用CLIP-ViT同时提取全局特征和局部patch特征
- **分层检索策略**: 实现全局图像检索和局部图像块检索
- **智能关键块选择**: 支持基于特征范数、随机选择、注意力机制的关键块选择
- **高效向量索引**: 基于FAISS的向量数据库，支持多种索引类型
- **灵活生成器**: 支持FLAN-T5和GPT-2等多种大语言模型
- **完整评估体系**: 实现BLEU、METEOR、CIDEr、SPICE等标准评估指标

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 构建向量索引

首先需要从COCO数据集构建向量索引：

```bash
python main.py build_indexes --coco_root /path/to/coco2017 --output_dir cache/faiss_indexes --max_images 1000
```

### 2. 生成图像描述

```python
from feature_extractor import CLIPFeatureExtractor
from index_builder import FAISSIndexBuilder
from retriever import HierarchicalRetriever
from generator import ImageCaptionGenerator, create_generator
from main import ImageCaptionPipeline

# 初始化组件
feature_extractor = CLIPFeatureExtractor(device="cuda")
index_builder = FAISSIndexBuilder(feature_dim=512)
index_builder.load_indexes("cache/faiss_indexes")

retriever = HierarchicalRetriever(feature_extractor, index_builder)
generator = create_generator("flan-t5", device="cuda")
caption_generator = ImageCaptionGenerator(retriever, generator)

# 初始化管道
pipeline = ImageCaptionPipeline(feature_extractor, index_builder, retriever, caption_generator)

# 生成描述
result = pipeline.generate_caption("path/to/image.jpg")
print(f"生成的描述: {result['caption']}")
```

### 3. 批量处理

```python
# 批量生成描述
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = pipeline.generator.batch_generate_captions(image_paths)

for i, result in enumerate(results):
    print(f"图像 {i+1}: {result['caption']}")
```

### 4. 评估性能

```bash
python main.py evaluate --coco_root /path/to/coco2017 --index_dir cache/faiss_indexes --split val --max_samples 100
```

## 配置说明

### 模型配置 (config.py)

```python
MODEL_CONFIG = {
    "clip_model": "ViT-B/32",  # CLIP模型版本
    "llm_model": "google/flan-t5-large",  # 大语言模型
    "device": "cuda",  # 计算设备
}
```

### 检索配置

```python
RETRIEVAL_CONFIG = {
    "global_top_k": 5,    # 全局检索Top-K
    "local_top_m": 10,    # 局部检索Top-M
    "key_patch_strategy": "norm",  # 关键块选择策略
    "num_key_patches": 3,  # 选择的关键块数量
    "similarity_threshold": 0.7,  # 相似度阈值
}
```

### 生成配置

```python
GENERATION_CONFIG = {
    "max_length": 256,     # 生成文本最大长度
    "beam_size": 3,        # beam search大小
    "temperature": 0.7,    # 生成温度
    "do_sample": True,     # 是否采样
}
```

## 核心模块说明

### 1. 特征提取模块 (feature_extractor.py)

- **CLIPFeatureExtractor**: CLIP特征提取器
- 支持全局特征和局部patch特征提取
- 提供关键patch选择策略

### 2. 索引构建模块 (index_builder.py)

- **FAISSIndexBuilder**: FAISS向量索引构建器
- 支持多种索引类型：Flat、IVF、HNSW
- 提供索引保存和加载功能

### 3. 检索模块 (retriever.py)

- **HierarchicalRetriever**: 分层检索器
- **AdaptiveRetriever**: 自适应检索器
- 实现全局和局部检索策略

### 4. 生成模块 (generator.py)

- **FLANT5Generator**: FLAN-T5生成器
- **GPT2Generator**: GPT-2生成器
- **ImageCaptionGenerator**: 图像描述生成器

### 5. 主流程模块 (main.py)

- **CaptionEvaluator**: 评估器
- **ImageCaptionPipeline**: 图像描述生成管道
- 提供完整的评估和基准测试功能

## 性能优化

### 1. GPU加速

```python
# 启用GPU加速
MODEL_CONFIG["device"] = "cuda"
INDEX_CONFIG["use_gpu"] = True
```

### 2. 批量处理

```python
# 批量特征提取
features_list = feature_extractor.batch_extract_features(image_paths)

# 批量生成
captions = generator.batch_generate(prompts)
```

### 3. 内存优化

```python
# 使用磁盘索引
INDEX_CONFIG["index_type"] = "IVF"  # 倒排文件索引
INDEX_CONFIG["nlist"] = 1000  # 聚类中心数
```

## 评估指标

系统实现了以下评估指标：

- **BLEU-4**: 基于n-gram重叠的评估指标
- **METEOR**: 考虑同义词的评估指标
- **CIDEr**: 基于TF-IDF权重的评估指标
- **SPICE**: 基于语义解析的评估指标

## 使用示例

### 示例1: 单张图像描述生成

```python
import torch
from PIL import Image

# 创建测试图像
test_image = Image.new('RGB', (224, 224), color='red')
test_image.save('test_image.jpg')

# 生成描述
result = pipeline.generate_caption('test_image.jpg')
print(f"描述: {result['caption']}")
print(f"检索统计: {result['retrieval_stats']}")
```

### 示例2: 检索结果分析

```python
# 获取详细分析
detailed_result = pipeline.generator.generate_with_analysis('test_image.jpg')

print("检索分析:")
for key, value in detailed_result['retrieval_analysis'].items():
    print(f"  {key}: {value}")

print("\n全局检索结果:")
for result in detailed_result['global_results'][:3]:
    print(f"  图像ID: {result['image_id']}, 相似度: {result['score']:.3f}")
    print(f"  描述: {result['captions'][0]}")

print("\n局部检索结果:")
for result in detailed_result['local_results'][:3]:
    print(f"  图像ID: {result['image_id']}, 相似度: {result['score']:.3f}")
    print(f"  描述: {result['captions'][0]}")
```

### 示例3: 性能基准测试

```python
# 性能测试
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
performance = pipeline.benchmark_performance(image_paths)

print("性能指标:")
print(f"  平均时间: {performance['avg_time']:.2f}秒")
print(f"  成功率: {performance['success_rate']:.2%}")
print(f"  总时间: {performance['total_time']:.2f}秒")
```

## 故障排除

### 1. 内存不足

```python
# 减少batch size
GENERATION_CONFIG["beam_size"] = 1

# 使用CPU
MODEL_CONFIG["device"] = "cpu"
```

### 2. 索引构建失败

```python
# 减少处理图像数量
python main.py build_indexes --max_images 100

# 使用更简单的索引类型
INDEX_CONFIG["index_type"] = "Flat"
```

### 3. 生成质量不佳

```python
# 调整检索参数
RETRIEVAL_CONFIG["global_top_k"] = 10
RETRIEVAL_CONFIG["local_top_m"] = 20

# 调整生成参数
GENERATION_CONFIG["temperature"] = 0.5
GENERATION_CONFIG["beam_size"] = 5
```

## 扩展功能

### 1. 自定义提示模板

```python
# 修改config.py中的PROMPT_TEMPLATE
PROMPT_TEMPLATE = """
你是一个专业的图像描述生成器。

参考描述：
{global_descriptions}

局部相似描述：
{local_descriptions}

请生成一个准确、详细的图像描述。
"""
```

### 2. 添加新的生成器

```python
class CustomGenerator(LLMGenerator):
    def __init__(self, model_name, device="cuda"):
        super().__init__(model_name, device)
        # 自定义初始化逻辑
    
    def generate(self, prompt, **kwargs):
        # 自定义生成逻辑
        pass
```

### 3. 自定义评估指标

```python
class CustomEvaluator(CaptionEvaluator):
    def evaluate_single(self, predicted, references):
        # 添加自定义评估指标
        metrics = super().evaluate_single(predicted, references)
        metrics['custom_score'] = self._compute_custom_score(predicted, references)
        return metrics
```

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request来改进本项目。

## 联系方式

如有问题，请通过GitHub Issues联系。
