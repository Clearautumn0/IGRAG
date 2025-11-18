#!/usr/bin/env python3
"""评估模块命令行入口"""

from __future__ import annotations

import argparse
import logging
import yaml
from pathlib import Path

from evaluation.evaluator import SimpleEvaluator
from main import setup_logging, load_config


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="运行IGRAG caption评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 评估前100张图片
  python evaluation/run_evaluation.py --subset 100
  
  # 使用自定义评估配置文件
  python evaluation/run_evaluation.py --config evaluation/config.yaml --subset 50
  
  # 使用flan-t5模型进行评估
  python evaluation/run_evaluation.py --model flan-t5 --subset 100
  
  # 使用qwen模型进行评估
  python evaluation/run_evaluation.py --model qwen --subset 100
  
  # 评估所有图片
  python evaluation/run_evaluation.py
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="evaluation/config.yaml",
        help="评估模块配置文件路径（默认: evaluation/config.yaml）"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="评估子集大小（只评估前N张图片，用于快速测试）"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["qwen", "flan-t5"],
        help="模型类型（qwen 或 flan-t5）"
    )
    return parser.parse_args()


def main() -> None:
    """主函数"""
    args = parse_args()
    
    # 加载评估配置
    eval_config_path = Path(args.config)
    if not eval_config_path.exists():
        logging.error(f"评估配置文件不存在: {eval_config_path}")
        return
    
    # 加载IGRAG主配置（如果需要根据--model参数修改）
    igrag_config = None
    if args.model:
        # 从评估配置中获取主配置路径，或使用默认路径
        with open(eval_config_path, "r", encoding="utf-8") as f:
            eval_config = yaml.safe_load(f)
        main_config_path = eval_config.get("igrag", {}).get("main_config_path", "configs/config.yaml")
        igrag_config = load_config(main_config_path)
        
        # 根据--model参数设置模型路径和类型
        model_config = igrag_config.setdefault("model_config", {})
        if args.model == "flan-t5":
            model_config["llm_model_path"] = "../models/flan-t5-base/"
            model_config["model_type"] = "flan-t5"
        elif args.model == "qwen":
            model_config["llm_model_path"] = "../models/Qwen2.5-3B-instruct/"
            model_config["model_type"] = "qwen"
        
        logging.info(f"使用模型: {args.model}, 模型路径: {model_config.get('llm_model_path')}")
    
    # 初始化评估器
    try:
        evaluator = SimpleEvaluator(eval_config_path, igrag_config=igrag_config)
    except Exception as e:
        logging.error(f"初始化评估器失败: {e}")
        return
    
    # 获取所有图片ID
    image_ids = sorted(evaluator.image_id_to_file.keys())
    
    if not image_ids:
        logging.error("没有可评估的图片")
        return
    
    # 显示评估信息
    subset_size = args.subset
    if subset_size is not None:
        logging.info(f"将评估前 {subset_size} 张图片（共 {len(image_ids)} 张）")
    else:
        logging.info(f"将评估所有 {len(image_ids)} 张图片")
    
    # 运行评估
    try:
        results = evaluator.evaluate_dataset(image_ids, subset_size=subset_size)
        
        # 打印汇总信息
        if results:
            print("\n" + "=" * 60)
            print("评估完成")
            print("=" * 60)
            print(f"评估图片数: {results.get('num_images', 0)}")
            
            aggregate_metrics = results.get("aggregate_metrics", {})
            if aggregate_metrics:
                print("\n总体指标:")
                for metric_name, score in aggregate_metrics.items():
                    print(f"  {metric_name}: {score:.4f}")
            
            print("=" * 60)
    except Exception as e:
        logging.error(f"评估过程出错: {e}", exc_info=True)


if __name__ == "__main__":
    # 设置基本日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    main()
