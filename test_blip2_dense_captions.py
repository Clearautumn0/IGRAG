#!/usr/bin/env python3
"""测试BLIP-2密集描述生成功能。

处理少量图像（默认10张）以验证输出格式。
"""
import os
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm
import yaml
from PIL import Image

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from scripts.build_dense_knowledge_base import (
    load_config,
    load_coco_image_mapping,
    image_paths_and_ids,
    init_blip2_model,
    extract_dense_captions_blip2
)
from utils.image_utils import load_image


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def test_blip2_dense_captions(num_images: int = 10):
    """测试BLIP-2密集描述生成。
    
    Args:
        num_images: 要处理的图像数量
    """
    setup_logging()
    
    # 加载配置
    cfg = load_config()
    
    dense_config = cfg.get("dense_descriptor", {})
    model_path = dense_config.get("model_path", "../models/blip2-opt-2.7b/")
    prompt = dense_config.get("prompt", "Question: List the objects, scenes, and actions in this image with very short phrases. Answer: ")
    max_new_tokens = dense_config.get("max_new_tokens", 100)
    num_beams = dense_config.get("num_beams", 5)
    
    data_config = cfg.get("data_config", {})
    images_dir = data_config.get("coco_images_dir")
    annotations_path = data_config.get("coco_annotations_path")
    
    print("=" * 60)
    print("BLIP-2 密集描述生成测试")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"提示词: {prompt}")
    print(f"测试图像数量: {num_images}")
    print()
    
    # 检查模型路径
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        print("请先下载BLIP-2模型到该路径")
        return False
    
    # 加载COCO图像映射
    print("加载COCO图像映射...")
    image_id_to_filename = load_coco_image_mapping(annotations_path)
    image_ids, image_paths = image_paths_and_ids(images_dir, image_id_to_filename)
    
    if len(image_paths) == 0:
        print(f"❌ 未找到图像文件: {images_dir}")
        return False
    
    print(f"✅ 找到 {len(image_paths)} 张图像")
    print()
    
    # 限制测试图像数量
    test_image_ids = image_ids[:num_images]
    test_image_paths = image_paths[:num_images]
    
    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print()
    
    # 加载模型
    print("正在加载BLIP-2模型...")
    try:
        processor, model = init_blip2_model(model_path, device=device)
        print("✅ 模型加载成功")
        print()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 处理测试图像
    print("=" * 60)
    print("开始处理测试图像...")
    print("=" * 60)
    print()
    
    results = {}
    
    for i, (image_id, image_path) in enumerate(zip(test_image_ids, test_image_paths), 1):
        print(f"[{i}/{num_images}] 处理图像 ID {image_id}: {os.path.basename(image_path)}")
        
        try:
            phrases = extract_dense_captions_blip2(
                model,
                processor,
                image_path,
                prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams
            )
            
            results[image_id] = phrases
            
            if phrases:
                print(f"  ✅ 生成了 {len(phrases)} 个短语:")
                for j, phrase in enumerate(phrases[:5], 1):  # 只显示前5个
                    print(f"     {j}. {phrase}")
                if len(phrases) > 5:
                    print(f"     ... 还有 {len(phrases) - 5} 个短语")
            else:
                print(f"  ⚠️  未生成短语")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            results[image_id] = []
        
        print()
    
    # 验证结果格式
    print("=" * 60)
    print("验证输出格式")
    print("=" * 60)
    
    all_valid = True
    for image_id, phrases in results.items():
        # 检查是否为列表
        if not isinstance(phrases, list):
            print(f"❌ 图像 {image_id}: 输出不是列表类型，而是 {type(phrases)}")
            all_valid = False
            continue
        
        # 检查列表元素是否为字符串
        if phrases and not all(isinstance(p, str) for p in phrases):
            print(f"❌ 图像 {image_id}: 列表包含非字符串元素")
            all_valid = False
            continue
        
        # 检查是否有非空字符串
        non_empty_phrases = [p for p in phrases if p and p.strip()]
        if non_empty_phrases:
            print(f"✅ 图像 {image_id}: {len(non_empty_phrases)} 个有效短语")
        else:
            print(f"⚠️  图像 {image_id}: 没有有效短语")
    
    print()
    
    # 统计信息
    images_with_phrases = sum(1 for phrases in results.values() if phrases)
    total_phrases = sum(len(phrases) for phrases in results.values())
    avg_phrases = total_phrases / len(results) if results else 0
    
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"处理的图像数量: {len(results)}")
    print(f"包含短语的图像: {images_with_phrases}")
    print(f"总短语数: {total_phrases}")
    print(f"平均每张图像短语数: {avg_phrases:.2f}")
    print(f"输出格式验证: {'✅ 通过' if all_valid else '❌ 失败'}")
    print()
    
    # 显示示例输出格式
    print("=" * 60)
    print("示例输出格式 (字典形式)")
    print("=" * 60)
    
    example_output = {k: v for k, v in list(results.items())[:3]}
    for image_id, phrases in example_output.items():
        print(f"  {image_id}: {phrases}")
    
    print()
    
    if all_valid and images_with_phrases > 0:
        print("🎉 测试通过！输出格式正确，可以继续处理全部图像。")
        return True
    else:
        print("❌ 测试未完全通过，请检查错误信息。")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="测试BLIP-2密集描述生成")
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="要处理的测试图像数量 (默认: 10)"
    )
    
    args = parser.parse_args()
    
    success = test_blip2_dense_captions(num_images=args.num_images)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

