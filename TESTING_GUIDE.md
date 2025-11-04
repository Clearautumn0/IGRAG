# IGRAG 系统测试指南

## 测试前准备

### 1. 检查依赖安装

确保已安装所有依赖：

```bash
pip install -r requirements.txt
```

或者使用conda：

```bash
conda activate IGRAG
pip install -r requirements.txt
```

### 2. 检查配置文件

确认 `configs/config.yaml` 中的路径配置正确：

- **模型路径**：确保 CLIP 和 FLAN-T5 模型已下载到指定路径
  - CLIP: `../models/clip-vit-base-patch32/`
  - FLAN-T5: `../models/flan-t5-base/`

- **COCO数据路径**（用于构建知识库）：
  - 图像目录: `data_config.coco_images_dir`
  - 标注文件: `data_config.coco_annotations_path`

### 3. 构建知识库（首次运行必须）

如果还没有构建知识库，需要先运行：

```bash
python scripts/build_knowledge_base.py
```

这会生成：
- `./output/coco_knowledge_base.faiss` - FAISS向量索引
- `./output/image_id_to_captions.pkl` - 图像ID到描述的映射

**注意**：构建过程可能需要较长时间，取决于COCO数据集大小和硬件配置。

### 4. 准备测试图像

将你要测试的图像放到 `input/` 目录下：

```bash
# 方式1：使用默认文件名
cp your_test_image.jpg input/test_image.jpg

# 方式2：使用命令行参数指定
# （图像可以是任意路径）
```

支持的图像格式：`.jpg`, `.jpeg`, `.png`

## 运行测试

### 方式1：使用默认测试图像

```bash
python main.py
```

默认会使用 `input/test_image.jpg`

### 方式2：指定测试图像路径

```bash
python main.py --i path/to/your/image.jpg
```

### 方式3：指定运行模式

```bash
# 测试模式（生成失败时报告错误）
python main.py --mode test

# 部署模式（生成失败时使用回退方案）
python main.py --mode deploy
```

### 完整示例

```bash
python main.py --i input/my_test_image.jpg --mode test
```

## 预期输出

### 分块检索模式（use_patch_retrieval: true）

```
Using patch retrieval mode...

Retrieval completed in 2.35s
Global descriptions: 3
Local regions detected: 4

Global similar images and their captions:
- Image ID: 12345  Score: 0.9123
    • a man riding a skateboard on a street
    • person on skateboard in urban setting

Detected local regions:
  Region 1: person (confidence: 0.892)
    Retrieved descriptions: 3
      • a person standing
      • man in casual clothes
  Region 2: skateboard (confidence: 0.856)
    Retrieved descriptions: 2
      • skateboard on pavement
      • board sports equipment

Timing: Prompt building: 0.12s, Generation: 1.45s

Generated caption:
A man riding a skateboard on a city street, wearing casual clothing, with pedestrians and buildings in the background.

Total pipeline time: 4.12s
```

### 全局检索模式（use_patch_retrieval: false）

```
Using global retrieval mode...
Retrieved images and their captions:
- Image ID: 12345  Score: 0.9123
    • a man riding a skateboard on a street
    • person on skateboard in urban setting

Generated caption:
A man riding a skateboard on a city street.

Total pipeline time: 2.35s
```

## 调试功能

### 1. 保存检测到的局部区域图像

在 `configs/config.yaml` 中设置：

```yaml
patch_config:
  save_debug_patches: true
```

检测到的区域图像会保存到 `output/debug_patches/` 目录。

### 2. 调整日志级别

在 `configs/config.yaml` 中设置：

```yaml
log_config:
  log_level: "INFO"  # 或 "DEBUG" 查看更多信息
```

### 3. 验证知识库文件

```bash
# Windows
dir output\*.faiss output\*.pkl

# Linux/Mac
ls -lh output/*.faiss output/*.pkl
```

### 4. 测试单个组件

#### 测试检索器

```python
python -c "
from core.retriever import ImageRetriever
from utils.image_utils import load_image
cfg = 'configs/config.yaml'
r = ImageRetriever(cfg)
img = load_image('input/test_image.jpg')
result = r.get_retrieved_captions(img, top_k=3)
print(result)
"
```

#### 测试生成器

```python
python -c "
from core.generator import CaptionGenerator
cfg = 'configs/config.yaml'
g = CaptionGenerator(cfg)
sample = [{'captions': ['A man riding a bike.', 'Man on bicycle.']}]
prompt = g.build_prompt(sample)
print('PROMPT:', prompt)
print('GENERATED:', g.generate_caption(prompt))
"
```

## 常见问题

### 1. 知识库文件不存在

**错误**：`FAISS index file not found`

**解决**：运行 `python scripts/build_knowledge_base.py` 构建知识库

### 2. 模型加载失败

**错误**：`Failed to load CLIP model` 或 `Failed to load LLM`

**解决**：
- 检查模型路径是否正确
- 确保模型文件已下载
- 检查网络连接（首次下载需要网络）

### 3. 图像文件未找到

**错误**：`Test image not found`

**解决**：
- 确保图像文件存在
- 使用 `--i` 参数指定完整路径
- 检查文件路径中的空格和特殊字符

### 4. 内存不足

**错误**：CUDA out of memory 或系统内存不足

**解决**：
- 降低 `scripts/build_knowledge_base.py` 中的 `batch_size`（默认64）
- 使用CPU模式（确保CUDA不可用时会自动使用CPU）
- 减少 `patch_config.max_local_regions` 的值

### 5. 分块检索失败

如果分块检索失败，系统会自动回退到全局检索模式，不会中断程序运行。

## 性能优化建议

1. **GPU加速**：如果有CUDA GPU，确保安装了GPU版本的PyTorch和FAISS
2. **批量处理**：如果测试多张图像，可以编写脚本批量处理
3. **调整配置**：
   - 降低 `retrieval_config.top_k` 减少检索数量
   - 降低 `patch_config.max_local_regions` 减少检测区域
   - 降低 `generation_config.max_length` 减少生成长度

## 下一步

- 尝试不同的测试图像
- 调整配置参数观察效果
- 启用调试模式查看详细信息
- 保存检测区域图像进行分析

