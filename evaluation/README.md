# IGRAG 评估使用说明

本说明文档介绍如何使用 `evaluation/` 目录下的脚本与模块，对 IGRAG 图像描述系统在 COCO 验证集上进行性能评估。

## 1. 环境准备

在项目根目录执行以下命令安装评估所需依赖（建议使用虚拟环境）：

```bash
pip install pycocotools nltk rouge-score tqdm matplotlib
```

> **提示**  
> - `pycocotools` 与 `rouge-score` 用于计算 CIDEr-D、ROUGE-L 等指标。  
> - `matplotlib` 用于生成指标分布图，可选装。  
> - SPICE 依赖 Java 运行环境，请确保系统已安装 Java（例如 OpenJDK）。

首次使用 NLTK 可能需要下载额外数据集：

```python
import nltk
nltk.download("punkt")
```

## 2. 目录结构

```
evaluation/
├── __init__.py
├── evaluator.py
├── metrics_calculator.py
├── results_analyzer.py
├── run_evaluation.py
└── README.md
```

- `metrics_calculator.py`：封装 BLEU-4、ROUGE-L、CIDEr-D、SPICE 等指标计算。  
- `evaluator.py`：`IGRAGEvaluator` 类，负责对整批图像生成描述并记录指标。  
- `results_analyzer.py`：基于评估结果 JSON 生成报告、挖掘高低分样本。  
- `run_evaluation.py`：命令行脚本，一键执行完整评估流程。  
- `README.md`：当前说明文件。

## 3. 配置说明

评估相关配置位于 `configs/config.yaml`：

```yaml
evaluation:
  val_images_dir: "/home/m025/qqw/coco/val2017/"
  val_annotations_path: "/home/m025/qqw/coco/annotations/captions_val2017.json"
  subset_size: null              # 可设为整数快速验证
  output_dir: "./evaluation_results/"
  save_individual_results: true  # 是否保存逐图结果

metrics:
  bleu: true
  rouge: true
  cider: true
  spice: true
```

- **val_images_dir / val_annotations_path**：COCO 验证集路径，已在本地准备。  
- **subset_size**：设置为整数可仅评估前 N 张图像，便于调试。  
- **output_dir**：评估结果 JSON 与图表的保存路径。  
- **save_individual_results**：评估过程中保留每张图像的中间结果，方便断点续跑。  
- **metrics**：按需启停指标，若关闭某项指标请确保上层逻辑能正确处理 `None`。

## 4. 运行评估

命令行示例：

```bash
python evaluation/run_evaluation.py \
  --config configs/config.yaml \
  --mode global_local \
  --subset 500 \
  --output ./evaluation_results/coco_val_subset.json
```

参数说明：

- `--config`：配置文件路径。  
- `--mode`：检索模式；`global_only` 仅使用全局检索，`global_local` 同时启用局部分块检索。  
- `--subset`：评估图像数量；缺省时遍历验证集全部图像。  
- `--output`：结果文件路径；未指定时按时间戳自动生成。

脚本会输出：

1. JSON 结果文件，包含总体统计与逐图指标。  
2. 终端文本报告（由 `ResultsAnalyzer` 生成），列出均值、分位数及高低分案例。  
3. 若安装了 `matplotlib`，额外保存对应直方图图片。

## 5. 结果分析

手动分析现有结果：

```python
from pathlib import Path
from evaluation.results_analyzer import ResultsAnalyzer

analyzer = ResultsAnalyzer()
analyzer.load_results(Path("evaluation_results/coco_val_subset.json"))
report = analyzer.generate_report(top_k=10)
print(report)
```

可按需调高 `top_k`，查看更多高分与低分样本，用于误差分析。

## 6. 常见问题

- **依赖报错**：确认 `pycocotools`、`rouge-score` 已安装，Linux 环境需提前安装 `gcc`、`python3-dev` 等编译依赖。  
- **SPICE 失败**：检查 Java 环境，必要时设置 `JAVA_HOME`。若仍报错，可在配置中将 `metrics.spice` 设为 `false`。  
- **评估缓慢**：使用 `--subset` 或在配置中设置 `subset_size`；检查是否启用了 `global_local` 模式（分块检索更耗时）。  
- **磁盘占用**：若不需要中间结果，将 `save_individual_results` 设为 `false`。

## 7. 后续扩展

- 在 `MetricsCalculator` 中新增其它指标（如 METEOR、BERTScore）。  
- 在 `ResultsAnalyzer` 中加入图像可视化、错误案例筛选等功能。  
- 将评估脚本整合至 CI/CD 流程，实现自动化回归评估。

如需进一步定制，可直接扩展相应模块或在 `evaluation/` 下新增辅助脚本。

