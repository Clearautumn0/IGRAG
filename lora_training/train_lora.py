#!/usr/bin/env python3
"""LoRA 训练脚本：从数据构建到模型训练的完整流程

用法:
    # 完整流程（构建数据 + 训练）
    python3 lora_training/train_lora.py --all

    # 仅构建训练数据
    python3 lora_training/train_lora.py --build-data --sample-count 5000

    # 仅训练（需要已有训练数据）
    python3 lora_training/train_lora.py --train

    # 自定义配置
    python3 lora_training/train_lora.py --all --lora-config lora_training/config/lora_config.yaml

输出:
    - lora_training/data/coco_lora_train.jsonl (原始数据)
    - lora_training/data/coco_lora_train_train.jsonl (训练集)
    - lora_training/data/coco_lora_train_val.jsonl (验证集)
    - lora_training/checkpoints/ (训练检查点)
"""
import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径，以便导入模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lora_training.data_builder import LoraTrainingDataBuilder, split_dataset
from lora_training.lora_trainer import LoraCaptionTrainer


def setup_logging(level: str = "INFO"):
    """设置日志级别"""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    logging.basicConfig(
        level=level_map.get(level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_training_data(
    main_config_path: str = "configs/config.yaml",
    sample_count: int = 5000,
    output_path: str = "lora_training/data/coco_lora_train.jsonl",
    train_ratio: float = 0.9,
    seed: int = 42,
    skip_if_exists: bool = True,
) -> dict:
    """构建训练数据"""
    print("=" * 70)
    print("阶段 1: 构建 LoRA 训练数据")
    print("=" * 70)

    output_path_obj = Path(output_path)
    if skip_if_exists and output_path_obj.exists():
        print(f"⚠️  数据文件已存在: {output_path}")
        response = input("是否重新构建? (y/N): ").strip().lower()
        if response != "y":
            print("跳过数据构建，使用现有文件")
            # 检查是否已有切分后的文件
            train_path = output_path_obj.with_name(output_path_obj.stem + "_train.jsonl")
            val_path = output_path_obj.with_name(output_path_obj.stem + "_val.jsonl")
            if train_path.exists() and val_path.exists():
                return {
                    "output_path": str(output_path_obj),
                    "train_path": str(train_path),
                    "val_path": str(val_path),
                    "num_samples": "unknown (using existing)",
                }
            else:
                print("现有文件未切分，进行切分...")
                split_dataset(str(output_path_obj), train_ratio=train_ratio, seed=seed)
                train_path = output_path_obj.with_name(output_path_obj.stem + "_train.jsonl")
                val_path = output_path_obj.with_name(output_path_obj.stem + "_val.jsonl")
                return {
                    "output_path": str(output_path_obj),
                    "train_path": str(train_path),
                    "val_path": str(val_path),
                    "num_samples": "unknown (using existing)",
                }

    print(f"📦 开始构建 {sample_count} 个训练样本...")
    print(f"   主配置文件: {main_config_path}")
    print(f"   输出路径: {output_path}")
    print()

    try:
        builder = LoraTrainingDataBuilder(
            main_config_path=main_config_path,
            sample_count=sample_count,
            output_path=output_path,
            seed=seed,
        )
        stats = builder.build()
        print(f"✅ 成功生成 {stats['num_samples']} 个训练样本")
        print(f"   保存位置: {stats['output_path']}")
        print()

        # 自动切分数据集
        print("📊 切分数据集 (训练集/验证集 = {:.0%}/{:.0%})...".format(train_ratio, 1 - train_ratio))
        split_result = split_dataset(stats["output_path"], train_ratio=train_ratio, seed=seed)
        print(f"✅ 数据集切分完成")
        print(f"   训练集: {split_result['train_path']}")
        print(f"   验证集: {split_result['val_path']}")
        print()

        return {
            **stats,
            **split_result,
        }
    except Exception as e:
        logging.error(f"数据构建失败: {e}", exc_info=True)
        raise


def train_lora(
    lora_config_path: str = "lora_training/config/lora_config.yaml",
    train_path: str = None,
    val_path: str = None,
) -> dict:
    """训练 LoRA 模型"""
    print("=" * 70)
    print("阶段 2: LoRA 模型训练")
    print("=" * 70)

    print(f"📋 配置文件: {lora_config_path}")
    if train_path:
        print(f"   训练集: {train_path}")
    if val_path:
        print(f"   验证集: {val_path}")
    print()

    try:
        # 如果提供了数据路径，临时更新配置
        if train_path or val_path:
            import yaml
            with open(lora_config_path, "r") as f:
                config = yaml.safe_load(f)
            if train_path:
                config["data"]["train_path"] = train_path
            if val_path:
                config["data"]["val_path"] = val_path
            # 保存临时配置
            temp_config_path = lora_config_path.replace(".yaml", "_temp.yaml")
            with open(temp_config_path, "w") as f:
                yaml.dump(config, f)
            lora_config_path = temp_config_path

        trainer = LoraCaptionTrainer(lora_config_path)
        print("✅ 训练器初始化成功")
        print()

        # 显示训练参数摘要
        print("📊 训练配置摘要:")
        print(f"   基础模型: {trainer.base_model_path}")
        print(f"   训练轮数: {trainer.training_cfg.get('num_train_epochs', 3)}")
        print(f"   批次大小: {trainer.training_cfg.get('train_batch_size', 4)}")
        print(f"   梯度累积: {trainer.training_cfg.get('gradient_accumulation_steps', 4)}")
        print(f"   学习率: {trainer.training_cfg.get('learning_rate', 5e-5)}")
        print(f"   LoRA r: {trainer.lora_cfg.get('r', 16)}")
        print(f"   LoRA alpha: {trainer.lora_cfg.get('lora_alpha', 32)}")
        print()

        # 开始训练
        print("🚀 开始训练...")
        print("-" * 70)
        train_result = trainer.train()
        print("-" * 70)
        print("✅ 训练完成")
        print(f"   最终训练损失: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
        print()

        # 运行评估
        print("📈 运行验证集评估...")
        eval_metrics = trainer.evaluate()
        print("✅ 评估完成")
        print(f"   验证集 BLEU: {eval_metrics.get('eval_bleu', 'N/A'):.4f}")
        print(f"   验证集损失: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
        print()

        # 显示最佳模型位置
        output_dir = trainer.training_cfg.get("output_dir", "lora_training/checkpoints")
        best_model_path = Path(output_dir) / "best"
        if best_model_path.exists():
            print(f"✅ 最佳模型已保存至: {best_model_path}")
        else:
            print(f"📁 检查点保存在: {output_dir}")

        # 清理临时配置
        if train_path or val_path:
            temp_config = Path(lora_config_path)
            if temp_config.exists():
                temp_config.unlink()

        return {
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_metrics,
            "best_model_path": str(best_model_path) if best_model_path.exists() else output_dir,
        }
    except Exception as e:
        logging.error(f"训练失败: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="LoRA 训练脚本：从数据构建到模型训练的完整流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流程
  python3 lora_training/train_lora.py --all

  # 仅构建数据
  python3 lora_training/train_lora.py --build-data --sample-count 10000

  # 仅训练
  python3 lora_training/train_lora.py --train

  # 自定义配置
  python3 lora_training/train_lora.py --all \\
      --main-config configs/config.yaml \\
      --lora-config lora_training/config/lora_config.yaml \\
      --sample-count 5000
        """,
    )

    # 主要操作选项
    parser.add_argument(
        "--all",
        action="store_true",
        help="执行完整流程：构建数据 + 训练",
    )
    parser.add_argument(
        "--build-data",
        action="store_true",
        help="仅构建训练数据",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="仅训练模型（需要已有训练数据）",
    )

    # 配置文件
    parser.add_argument(
        "--main-config",
        type=str,
        default="configs/config.yaml",
        help="主配置文件路径（用于数据构建）",
    )
    parser.add_argument(
        "--lora-config",
        type=str,
        default="lora_training/config/lora_config.yaml",
        help="LoRA 训练配置文件路径",
    )

    # 数据构建参数
    parser.add_argument(
        "--sample-count",
        type=int,
        default=5000,
        help="训练样本数量（默认: 5000）",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        default="lora_training/data/coco_lora_train.jsonl",
        help="训练数据输出路径",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="训练集比例（默认: 0.9，即 90%% 训练，10%% 验证）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="强制重新构建数据（即使文件已存在）",
    )

    # 训练数据路径（用于仅训练模式）
    parser.add_argument(
        "--train-path",
        type=str,
        help="训练集路径（覆盖配置文件中的设置）",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        help="验证集路径（覆盖配置文件中的设置）",
    )

    # 日志级别
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认: INFO）",
    )

    args = parser.parse_args()

    # 如果没有指定任何操作，显示帮助
    if not (args.all or args.build_data or args.train):
        parser.print_help()
        sys.exit(1)

    # 设置日志
    setup_logging(args.log_level)

    try:
        # 执行操作
        data_stats = None
        train_stats = None

        if args.all or args.build_data:
            data_stats = build_training_data(
                main_config_path=args.main_config,
                sample_count=args.sample_count,
                output_path=args.output_data,
                train_ratio=args.train_ratio,
                seed=args.seed,
                skip_if_exists=not args.force_rebuild,
            )

        if args.all or args.train:
            # 如果数据刚构建完成，使用新的路径
            train_path = args.train_path
            val_path = args.val_path
            if data_stats:
                train_path = data_stats.get("train_path") or train_path
                val_path = data_stats.get("val_path") or val_path

            train_stats = train_lora(
                lora_config_path=args.lora_config,
                train_path=train_path,
                val_path=val_path,
            )

        # 总结
        print("=" * 70)
        print("✅ 所有操作完成")
        print("=" * 70)
        if data_stats:
            print(f"📦 数据: {data_stats.get('num_samples', 'N/A')} 个样本")
        if train_stats:
            print(f"📈 训练: BLEU = {train_stats.get('eval_metrics', {}).get('eval_bleu', 'N/A'):.4f}")
            print(f"📁 模型: {train_stats.get('best_model_path', 'N/A')}")
        print()
        print("💡 下一步: 在 configs/config.yaml 中启用 LoRA:")
        print("   lora_config:")
        print("     enabled: true")
        if train_stats:
            print(f"     weights_path: \"{train_stats.get('best_model_path', 'lora_training/checkpoints/best')}\"")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        logging.error(f"执行失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

