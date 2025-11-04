# 快速测试步骤

## 1. 准备测试图像

将你的测试图像放到 `input/` 目录，并命名为 `test_image.jpg`：

```bash
# 将你的图像复制到input目录
copy 你的图像路径 input\test_image.jpg
```

或者使用其他文件名，运行时指定路径。

## 2. 检查知识库（首次运行必须）

确认知识库文件是否存在：

```bash
# Windows PowerShell
Test-Path output\coco_knowledge_base.faiss
Test-Path output\image_id_to_captions.pkl
```

如果文件不存在，需要先构建知识库：

```bash
python scripts/build_knowledge_base.py
```

## 3. 运行测试

### 最简单的方式（使用默认图像）

```bash
python main.py
```

### 指定图像路径

```bash
python main.py --i input\你的图像.jpg
```

### 指定图像路径并设置模式

```bash
# 测试模式
python main.py --i input\你的图像.jpg --mode test

# 部署模式（失败时自动回退）
python main.py --i input\你的图像.jpg --mode deploy
```

## 4. 查看输出

程序会输出：
- 检索到的相似图像和描述
- 检测到的局部区域（如果启用分块检索）
- 最终生成的图像描述
- 各阶段耗时统计

## 示例输出

```
Using patch retrieval mode...

Retrieval completed in 2.35s
Global descriptions: 3
Local regions detected: 4

Global similar images and their captions:
- Image ID: 12345  Score: 0.9123
    • a man riding a skateboard on a street

Detected local regions:
  Region 1: person (confidence: 0.892)
    Retrieved descriptions: 3
      • a person standing

Generated caption:
A man riding a skateboard on a city street.

Total pipeline time: 4.12s
```

## 常见问题快速解决

### 问题1：找不到图像
**解决**：确保图像路径正确，或使用绝对路径

### 问题2：知识库文件不存在
**解决**：运行 `python scripts/build_knowledge_base.py`

### 问题3：模型加载失败
**解决**：检查 `configs/config.yaml` 中的模型路径是否正确

### 问题4：想关闭分块检索
**解决**：在 `configs/config.yaml` 中设置 `use_patch_retrieval: false`

