# IGRAG 评估模块使用说明

本评估模块已重构为独立、简化的版本，使用 `pycocoevalcap` 计算标准评估指标。

## 1. 环境准备

安装评估所需依赖：

```bash
pip install pycocotools pycocoevalcap nltk
```

首次使用 NLTK 可能需要下载额外数据集：

```python
import nltk
nltk.download("punkt")
```

注意：SPICE 指标需要 Java 运行环境，请确保系统已安装 Java。

## 2. 目录结构

```
evaluation/
├── __init__.py
├── config.yaml              # 评估模块独立配置文件
├── metrics_calculator.py    # 指标计算器（使用pycocoevalcap）
├── evaluator.py             # 简化的评估器
├── run_evaluation.py        # 命令行入口
└── README.md                # 本文件
```

## 3. 配置文件

评估模块使用独立的配置文件 `evaluation/config.yaml`，与主项目配置分离。

主要配置项：

```yaml
# COCO验证集路径
data:
  val_images_dir: "/home/m025/qqw/coco/val2017/"
  val_annotations_path: "/home/m025/qqw/coco/annotations/captions_val2017.json"

# 评估输出配置
output:
  output_dir: "./evaluation_results/"
  output_file: null  # 如果为null则自动生成时间戳文件名

# IGRAG模型配置
igrag:
  main_config_path: "configs/config.yaml"  # IGRAG主配置文件路径
  retrieval_mode: "global_local"  # "global_only" 或 "global_local"

# 评估指标配置
metrics:
  bleu_1: true
  bleu_2: true
  bleu_4: true
  meteor: true
  rouge: true
  cider: true
  spice: true
```

## 4. 运行评估

### 基本用法

```bash
# 评估所有图片
python evaluation/run_evaluation.py

# 评估前100张图片（用于快速测试）
python evaluation/run_evaluation.py --subset 100

# 使用自定义配置文件
python evaluation/run_evaluation.py --config evaluation/config.yaml --subset 50
```

### 命令行参数

- `--config`: 评估模块配置文件路径（默认: `evaluation/config.yaml`）
- `--subset`: 评估子集大小，只评估前N张图片（用于快速测试）

## 5. 评估结果

评估结果保存在 JSON 文件中，包含以下信息：

```json
{
  "generated_at": "2024-01-01T12:00:00Z",
  "num_images": 100,
  "aggregate_metrics": {
    "BLEU-1": 0.7234,
    "BLEU-2": 0.5678,
    "BLEU-4": 0.3456,
    "METEOR": 0.4567,
    "ROUGE-L": 0.5123,
    "CIDEr": 0.6789,
    "SPICE": 0.2345
  },
  "results": [
    {
      "image_id": 12345,
      "file_name": "COCO_val2017_000000012345.jpg",
      "generated_caption": "IGRAG生成的caption",
      "coco_captions": [
        "COCO参考caption 1",
        "COCO参考caption 2",
        ...
      ],
      "metrics": {
        "BLEU-1": 0.7500,
        "BLEU-2": 0.6000,
        "BLEU-4": 0.4000,
        "METEOR": 0.5000,
        "ROUGE-L": 0.5500,
        "CIDEr": 0.7000,
        "SPICE": 0.3000
      }
    },
    ...
  ]
}
```

每个结果项包含：
1. **IGRAG生成的caption** (`generated_caption`)
2. **COCO参考caption** (`coco_captions`)
3. **各项指标得分** (`metrics`)

## 6. 支持的评估指标

本模块使用 `pycocoevalcap` 计算以下标准指标：

- **BLEU-1, BLEU-2, BLEU-4**: n-gram 精确度指标
- **METEOR**: 考虑同义词的评估指标
- **ROUGE-L**: 基于最长公共子序列的指标
- **CIDEr**: 共识导向的图像描述评估指标
- **SPICE**: 基于语义命题的评估指标

所有指标均通过 `pycocoevalcap` 计算，确保与标准评估流程一致。

## 7. 常见问题

- **依赖报错**: 确认 `pycocotools`、`pycocoevalcap` 已正确安装
- **SPICE 失败**: 检查 Java 环境，必要时设置 `JAVA_HOME`
- **评估缓慢**: 使用 `--subset` 参数进行快速测试
- **配置文件路径**: 确保评估配置文件和IGRAG主配置文件路径正确

## 8. 模块独立性

评估模块与主项目保持相对独立：

- 使用独立的配置文件 `evaluation/config.yaml`
- 通过 `main.generate_caption()` 函数调用IGRAG生成caption，不直接依赖内部实现
- 评估结果格式简单，易于解析和分析
