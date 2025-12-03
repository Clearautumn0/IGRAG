# 密集描述混合检索重构说明

## 概述

本次重构将检索的匹配信号从"物体标签"升级为"密集描述短语"，以提升检索的语义相关性和检索效果。

## 已完成的工作

### 1. 密集描述知识库构建脚本
- **文件**: `scripts/build_dense_knowledge_base.py`
- **功能**: 
  - 使用 `mm_grounding_dino_tiny_o365v1_goldg_grit_v3det` 模型为每张图像生成5-10个密集描述短语
  - 支持多GPU并行处理
  - 支持批处理加速
  - 包含进度条显示
  - 错误处理（跳过失败图像）
  - 从断点恢复
- **输出**: `./output/image_id_to_dense_captions.pkl`

### 2. 新的密集描述混合检索器
- **文件**: `core/dense_descriptor_retriever.py`
- **类名**: `DenseDescriptorHybridRetriever`
- **功能**:
  - 两阶段混合检索：CLIP召回 + 密集描述语义相似度重排序
  - 使用句子嵌入模型（all-MiniLM-L6-v2）计算描述相似度
  - 采用"平均最大余弦相似度"策略
  - 向后兼容原有接口

### 3. 配置文件更新
- **文件**: `configs/config.yaml`
- **新增配置节**:
  - `dense_descriptor`: 密集描述相关配置
  - `retrieval_config.use_dense_retrieval`: 启用/禁用开关
  - `hybrid_retrieval.dense_object_weight`: 密集描述融合权重

### 4. 模块导出
- **文件**: `core/__init__.py`
- 已添加 `DenseDescriptorHybridRetriever` 的导出

## 需要下载的模型

请将以下模型下载并保存到 `../models/{模型名称}/` 目录下：

### 1. 密集描述生成模型
- **模型名称**: `mm_grounding_dino_tiny_o365v1_goldg_grit_v3det`
- **路径**: `../models/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det/`
- **说明**: 用于生成图像的密集描述短语
- **来源**: OpenMMLab Community

### 2. 句子嵌入模型（已存在，检查路径）
- **模型名称**: `all-MiniLM-L6-v2`
- **路径**: `../models/all-MiniLM-L6-v2/`
- **说明**: 用于计算描述短语之间的语义相似度
- **注意**: 如果已存在于 `description_optimization.embedding_model` 配置的路径，则无需重复下载

## 使用步骤

### 步骤1: 构建密集描述知识库

```bash
python scripts/build_dense_knowledge_base.py
```

这个脚本会：
1. 加载密集描述生成模型
2. 遍历COCO训练集图像
3. 为每张图像生成5-10个描述短语
4. 保存到 `./output/image_id_to_dense_captions.pkl`

**注意**: 
- 如果模型输出格式与预期不同，可能需要调整 `extract_dense_captions_mmdet` 函数
- 处理大量图像可能需要较长时间，脚本支持断点恢复
- 建议使用多GPU加速处理

### 步骤2: 启用密集描述检索

在 `configs/config.yaml` 中设置：

```yaml
retrieval_config:
  use_dense_retrieval: true  # 启用密集描述检索

hybrid_retrieval:
  dense_object_weight: 0.7  # 调整融合权重（0.0-1.0）
```

### 步骤3: 使用新的检索器

系统会根据配置自动选择使用 `DenseDescriptorHybridRetriever` 或 `ObjectAwareHybridRetriever`。

## 模型推理适配说明

由于 `mm_grounding_dino_tiny_o365v1_goldg_grit_v3det` 模型的实际API可能因版本而异，`build_dense_knowledge_base.py` 中的推理逻辑可能需要根据实际情况调整。

主要需要适配的部分在 `extract_dense_captions_mmdet` 函数中。如果模型输出格式不同，请：

1. 检查模型的实际输出格式
2. 修改 `extract_dense_captions_mmdet` 函数以适配实际输出
3. 确保能够正确提取描述短语列表

## 回滚方案

如果需要切换回旧的物体标签检索方案：

1. 在 `configs/config.yaml` 中设置：
   ```yaml
   retrieval_config:
     use_dense_retrieval: false  # 禁用密集描述检索
   ```

2. 系统将自动使用 `ObjectAwareHybridRetriever`

## 预期效果

此次重构预期能够：
- 大幅提升检索结果与查询图像在语义层面的相关性
- 提升 BLEU-4 和 CIDEr 分数
- 改善 concept_recall（从当前的 ~0.31 提升）

## 故障排查

### 模型加载失败

#### 问题：找不到config.py文件
**错误信息**: `No mmdetection config.py found` 或 `No config file found`

**解决方案**:
1. HuggingFace仓库通常只有`config.json`，但mmdetection需要`config.py`
2. 从OpenMMLab官方获取config.py：
   ```bash
   # 方法1：从OpenMMLab Model Zoo下载对应配置文件
   # 访问 https://github.com/open-mmlab/mmdetection/tree/main/configs
   # 查找对应的grounding dino配置文件
   
   # 方法2：如果模型来自OpenMMLab，检查是否有对应的config文件仓库
   # 通常配置文件名与模型相关，如：
   # - grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py
   ```

3. 将下载的config.py文件放入模型目录：
   ```bash
   cp /path/to/downloaded/config.py ../models/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det/
   ```

4. 确保目录结构如下：
   ```
   ../models/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det/
   ├── config.py          # mmdetection配置文件（必需）
   ├── config.json        # HuggingFace配置（可选）
   ├── pytorch_model.bin  # 或 *.pth, *.ckpt
   └── ...                # 其他文件
   ```

#### 其他常见问题
- 检查模型路径是否正确
- 确认模型文件完整性
- 检查 mmdetection 和相关依赖是否正确安装

### 描述生成失败
- 检查模型输出格式
- 查看日志了解具体错误信息
- 可能需要调整推理逻辑以适配实际模型API

### 检索效果不佳
- 调整 `dense_object_weight` 参数
- 检查密集描述知识库是否完整
- 确认句子嵌入模型是否正确加载

## 联系方式

如遇到问题，请检查日志输出并参考上述故障排查指南。

