# 项目结构说明

## 文件结构

```
IGRAG/
├── config.py              # 配置文件
├── feature_extractor.py   # CLIP特征提取模块
├── index_builder.py       # FAISS向量索引构建
├── retriever.py           # 分层检索策略
├── generator.py           # 大语言模型生成器
├── main.py               # 主流程和评估
├── examples.py           # 使用示例
├── test_system.py        # 系统测试
├── requirements.txt      # 依赖包列表
├── README.md            # 项目文档
└── cache/               # 缓存目录
    ├── faiss_indexes/   # FAISS索引文件
    ├── outputs/         # 输出结果
    └── checkpoints/     # 模型检查点
```

## 核心模块说明

### 1. config.py
- 项目配置管理
- 模型参数设置
- 检索和生成参数
- 路径配置

### 2. feature_extractor.py
- CLIPFeatureExtractor类
- 全局特征提取
- 局部patch特征提取
- 关键patch选择策略

### 3. index_builder.py
- FAISSIndexBuilder类
- 全局索引构建
- 局部索引构建
- 索引保存和加载

### 4. retriever.py
- HierarchicalRetriever类
- AdaptiveRetriever类
- 全局检索策略
- 局部检索策略
- 检索结果分析

### 5. generator.py
- FLANT5Generator类
- GPT2Generator类
- ImageCaptionGenerator类
- 文本生成和后处理

### 6. main.py
- CaptionEvaluator类
- ImageCaptionPipeline类
- 评估指标实现
- 性能基准测试

## 数据流程

1. **输入**: 查询图像
2. **特征提取**: CLIP提取全局和局部特征
3. **检索**: 全局检索 + 局部检索
4. **提示构建**: 分层提示模板
5. **生成**: 大语言模型生成描述
6. **输出**: 图像描述文本

## 使用方法

### 快速开始
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 构建索引
python main.py build_indexes --coco_root /path/to/coco --output_dir cache/faiss_indexes

# 3. 运行示例
python examples.py

# 4. 系统测试
python test_system.py
```

### 编程接口
```python
from main import ImageCaptionPipeline

# 初始化管道
pipeline = ImageCaptionPipeline(...)

# 生成描述
result = pipeline.generate_caption("image.jpg")
print(result['caption'])
```

## 技术特点

- **多粒度特征**: 全局 + 局部特征提取
- **分层检索**: 全局检索 + 局部检索
- **智能选择**: 关键patch选择策略
- **高效索引**: FAISS向量数据库
- **灵活生成**: 支持多种LLM模型
- **完整评估**: 标准评估指标
- **性能优化**: GPU加速和批量处理
