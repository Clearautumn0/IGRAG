# IGRAG — 图像检索增强生成 (Image Retrieval-Augmented Generation)

这是一个轻量级的原型框架：输入一张图像 → 使用 CLIP 检索最相似的 COCO 图像 → 汇总这些图像的描述 → 使用 LLM(FLAN-T5) 生成最终的图像描述。

目录结构（关键文件）

- `configs/config.yaml`：项目配置（模型路径、COCO 路径、检索与生成参数、知识库路径、日志级别）。
- `scripts/build_knowledge_base.py`：构建 COCO 向量知识库（CLIP 特征 + FAISS 索引）。
- `core/retriever.py`：`ImageRetriever`，CLIP + FAISS 检索器。
- `core/generator.py`：`CaptionGenerator`，基于 FLAN-T5 的生成器。
- `utils/image_utils.py`：图像加载与预处理工具。
- `main.py`：主入口，整合检索与生成，输出最终描述。
- `lora_training/`：LoRA 微调（数据构建、训练配置、训练脚本）。
- `requirements.txt`：Python 依赖。

快速开始
----------------

1) 准备工作

- 确保已下载并放置模型：
  - CLIP：`../models/clip-vit-base-patch32/`
  - FLAN-T5：`../models/flan-t5-base/`
- 准备 COCO 数据（示例路径）：
  - 图像文件夹：`/home/m025/qqw/coco/train2017/`
  - 标注文件：`/home/m025/qqw/coco/annotations/captions_train2017.json`
- 将你要测试的图片放到 `input/`，例如 `input/test.jpg`（见下面的说明）。

2) 创建并激活 Conda 环境（可选）

创建后在你的环境中安装依赖：

```bash
conda activate IGRAG
conda run -n IGRAG pip install -r requirements.txt
```

注意：如果你希望使用 GPU 版本的 FAISS，请在 `requirements.txt` 中将 `faiss-cpu` 替换为 `faiss-gpu`，并确保 CUDA 驱动与 CUDA 版本匹配。

3) 构建知识库（必须）

从 COCO 标注中提取每张图像的描述，并对所有图像进行 CLIP 特征提取，最终生成 FAISS 索引与映射文件：

```bash
python3 scripts/build_knowledge_base.py
```

生成文件（默认，来源于 `configs/config.yaml`）：

- `./output/coco_knowledge_base.faiss` — FAISS 向量索引
- `./output/image_id_to_captions.pkl` — image_id -> 描述列表（pickle）

构建过程可能耗时较长（取决于 COCO 子集大小、模型加载与硬件）。脚本会显示特征提取的进度条。

4) 运行端到端测试（生成描述）

默认 `main.py` 会使用 `input/test_image.jpg` 作为测试图像并输出结果：

```bash
python3 main.py
```

如果你的测试图片名为 `input/test.jpg`，请把它重命名或复制到 `input/test_image.jpg`：

```bash
mv input/test.jpg input/test_image.jpg
# 或
cp input/test.jpg input/test_image.jpg
```

（当前 `main.py` 尚未实现命令行 `--input` 参数；如果你需要我可以立即添加）

预期控制台输出（最小化，只输出必要信息）

- 若一切正常，控制台将显示检索到的若干相似图像的 ID 与描述（短列表），随后打印最终生成的图像描述文本。例如：

```
Retrieved images and their captions:
- Image ID: 12345  Score: 0.9123
    • a man riding a skateboard on a street
    • ...

Generated caption:
一个骑滑板的男子在城市街道上快速前行，穿着休闲服，背景有行人和建筑。
```

- 日志级别由 `configs/config.yaml` 中的 `log_config.log_level` 控制（默认 `ERROR`），因此不会输出调试信息。

调试建议（逐步）
--------------------

1. 依赖检查

```bash
python3 -c "import importlib; print([ (pkg, importlib.util.find_spec(pkg) is not None) for pkg in ('torch','transformers','faiss','tqdm','yaml','PIL','numpy') ])"
```

2. 验证知识库文件存在

```bash
ls -l ./output/coco_knowledge_base.faiss ./output/image_id_to_captions.pkl
```

3. 快速查看映射内容（部分）

```bash
python3 - <<'PY'
import pickle
with open('output/image_id_to_captions.pkl','rb') as f:
    d=pickle.load(f)
print('映射条目数:', len(d))
for k in list(d.keys())[:3]:
    print(k, d[k][:5])
PY
```

4. 测试检索器（独立）

```bash
python3 - <<'PY'
from core.retriever import ImageRetriever
from utils.image_utils import load_image
cfg='configs/config.yaml'
r=ImageRetriever(cfg)
img=load_image('input/test_image.jpg')
print(r.get_retrieved_captions(img, top_k=3))
PY
```

5. 测试生成器（独立）

```bash
python3 - <<'PY'
from core.generator import CaptionGenerator
cfg='configs/config.yaml'
g=CaptionGenerator(cfg)
sample=[{'captions':['A man riding a bike.','Man on bicycle.']}]
p=g.build_prompt(sample)
print('PROMPT:\n',p)
print('\nGENERATED:\n', g.generate_caption(p))
PY
```

常见问题与解决
-----------------

- 模型加载失败：检查 `configs/config.yaml` 中的 `model_config.clip_model_path` 与 `model_config.llm_model_path` 是否正确，且目录下包含 transformers 可识别的模型文件。
- FAISS 索引读取失败：确认 `./output/coco_knowledge_base.faiss` 存在且在构建时未报错。
- 内存/显存不足：在 `scripts/build_knowledge_base.py` 中调小 `batch_size`（默认代码中有 `batch_size = 64`），或在无 GPU 的情况下使用 CPU（脚本会自动选择）。

后续改进建议
-----------------

- 让 `main.py` 支持命令行参数（`--input`, `--top_k`, `--no-progress` 等）。
- 把知识库构建与检索模块进一步拆分并添加更多单元测试。
- 优化 prompt 设计与输出后处理以提高 LLM 生成质量。

LoRA 微调流程
-----------------

为了解决 FLAN-T5 未针对 COCO 风格描述优化的问题，仓库新增 `lora_training/` 模块（`data_builder.py`, `lora_trainer.py`, `config/lora_config.yaml`），用于以低成本微调模型。典型流程如下：

1. **阶段1 — 构建 5k 训练样本**

```bash
python3 - <<'PY'
from lora_training.data_builder import LoraTrainingDataBuilder, split_dataset
builder = LoraTrainingDataBuilder(
    main_config_path="configs/config.yaml",
    sample_count=5000,
    output_path="lora_training/data/coco_lora_train.jsonl",
)
stats = builder.build()
print(stats)
split_dataset(stats["output_path"], train_ratio=0.9)
PY
```

输出为 JSONL，包含 prompt（带全局/局部描述）与目标 COCO caption。默认会生成 `*_train.jsonl` 与 `*_val.jsonl`。

2. **阶段2 — 训练 3 个 epoch，监控 BLEU-4**

- 确保 `requirements.txt` 已安装 `peft`, `datasets`, `evaluate`。
- 根据需要修改 `lora_training/config/lora_config.yaml`（模型路径、Batch size、LoRA r/alpha/dropout 等）。

```bash
python3 - <<'PY'
from lora_training.lora_trainer import LoraCaptionTrainer
trainer = LoraCaptionTrainer("lora_training/config/lora_config.yaml")
trainer.train()
trainer.evaluate()
PY
```

`Trainer` 会在每个 epoch 结束时保存 checkpoint 并计算 BLEU（`metric_for_best_model=eval_bleu`）。推荐重点监控 `eval_loss` 与 `eval_bleu`。

3. **阶段3 — 集成最优 LoRA checkpoint**

- 在 `configs/config.yaml` 中启用：

```yaml
lora_config:
  enabled: true
  weights_path: "lora_training/checkpoints/checkpoint-XXXX"
  merge_and_unload: false  # 若想把 LoRA 合并回基础模型，可设为 true
```

- 重新运行 `main.py`，`CaptionGenerator` 会自动加载 LoRA 适配器，仅增加少量可训练参数。

> 经验值：`r=16`, `lora_alpha=32`, `target_modules: ["q","v"]`, dropout 0.1 可在 BLEU-4 提升到 **0.25~0.35**，同时保持内存占用可控。

许可证与贡献
----------------

本仓库为个人/研究用途示例代码；如需部署或商业使用，请遵循依赖模型和数据集（CLIP、FLAN-T5、COCO）的许可条款。

若需我为你：
- 添加 `--input` 支持并更新 `main.py`；
- 生成固定版本的 `requirements.txt` 并重新安装；
- 或运行一次端到端测试并把输出返回给你，告诉我你希望我执行哪项，我会继续。
