#!/usr/bin/env python3
"""评估模块命令行入口"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from evaluation.evaluator import SimpleEvaluator
from main import setup_logging


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
    return parser.parse_args()


def main() -> None:
    """主函数"""
    args = parse_args()
    
    # 加载评估配置
    eval_config_path = Path(args.config)
    if not eval_config_path.exists():
        logging.error(f"评估配置文件不存在: {eval_config_path}")
        return
    
    # 初始化评估器
    try:
        evaluator = SimpleEvaluator(eval_config_path)
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
