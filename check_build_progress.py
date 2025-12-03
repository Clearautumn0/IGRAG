#!/usr/bin/env python3
"""查看密集描述知识库构建进度。

显示当前处理进度、统计信息和示例数据。
"""
import os
import sys
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# 添加项目根目录到路径
script_dir = Path(__file__).parent
project_root = script_dir
sys.path.insert(0, str(project_root))


def load_config(config_path: str = "configs/config.yaml"):
    """加载配置文件。"""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(kb_path: str) -> Dict[int, List[str]]:
    """加载检查点文件。"""
    if not os.path.exists(kb_path):
        return {}
    
    try:
        with open(kb_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"❌ 无法加载检查点文件: {e}")
        return {}


def get_total_images(config_path: str = "configs/config.yaml") -> int:
    """获取需要处理的总图像数量。"""
    try:
        cfg = load_config(config_path)
        data_config = cfg.get("data_config", {})
        images_dir = data_config.get("coco_images_dir")
        annotations_path = data_config.get("coco_annotations_path")
        
        if not annotations_path or not os.path.exists(annotations_path):
            return 0
        
        import json
        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 获取所有图像ID
        image_ids = {item["id"] for item in data.get("images", [])}
        
        # 检查实际存在的图像文件
        if images_dir and os.path.exists(images_dir):
            image_id_to_filename = {item["id"]: item["file_name"] 
                                   for item in data.get("images", [])}
            existing_count = 0
            for img_id, filename in image_id_to_filename.items():
                img_path = os.path.join(images_dir, filename)
                if os.path.exists(img_path):
                    existing_count += 1
            return existing_count
        
        return len(image_ids)
    except Exception as e:
        return 0


def format_size(size_bytes: int) -> str:
    """格式化文件大小。"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def show_progress(kb_path: str, config_path: str = "configs/config.yaml", 
                  watch: bool = False, interval: int = 5):
    """显示构建进度。"""
    
    if watch:
        import time
        print("进入监控模式，每 {} 秒刷新一次。按 Ctrl+C 退出。\n".format(interval))
    
    while True:
        # 清屏（如果在监控模式）
        if watch:
            os.system('clear' if os.name != 'nt' else 'cls')
            print("=" * 70)
            print("密集描述知识库构建进度监控")
            print("=" * 70)
            print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 检查文件是否存在
        if not os.path.exists(kb_path):
            print(f"❌ 检查点文件不存在: {kb_path}")
            print("   脚本可能还未开始运行，或者检查点文件路径配置错误。")
            if not watch:
                return
            time.sleep(interval)
            continue
        
        # 加载数据
        data = load_checkpoint(kb_path)
        
        if not data:
            print("⚠️  检查点文件存在但为空。")
            if not watch:
                return
            time.sleep(interval)
            continue
        
        # 获取文件信息
        file_size = os.path.getsize(kb_path)
        file_mtime = datetime.fromtimestamp(os.path.getmtime(kb_path))
        
        # 统计数据
        total_processed = len(data)
        images_with_captions = sum(1 for v in data.values() if v)
        images_empty = total_processed - images_with_captions
        total_phrases = sum(len(v) for v in data.values() if v)
        avg_phrases = total_phrases / images_with_captions if images_with_captions > 0 else 0
        
        # 获取总图像数
        total_images = get_total_images(config_path)
        if total_images > 0:
            progress_percent = (total_processed / total_images) * 100
            remaining = total_images - total_processed
        else:
            progress_percent = 0
            remaining = 0
        
        # 显示进度
        print("📊 构建进度统计")
        print("-" * 70)
        
        if total_images > 0:
            print(f"总图像数:           {total_images:,}")
            print(f"已处理:             {total_processed:,}  ({progress_percent:.2f}%)")
            print(f"剩余:               {remaining:,}")
        else:
            print(f"已处理图像:         {total_processed:,}")
        
        print(f"有描述的图像:       {images_with_captions:,}")
        print(f"空描述图像:         {images_empty:,}")
        print(f"总短语数:           {total_phrases:,}")
        if images_with_captions > 0:
            print(f"平均短语数/图像:    {avg_phrases:.2f}")
        
        print()
        print("📁 文件信息")
        print("-" * 70)
        print(f"检查点文件:         {kb_path}")
        print(f"文件大小:           {format_size(file_size)}")
        print(f"最后更新:           {file_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 计算时间差
        time_diff = datetime.now() - file_mtime
        if time_diff.total_seconds() < 60:
            print(f"更新时间:           {int(time_diff.total_seconds())} 秒前")
        elif time_diff.total_seconds() < 3600:
            print(f"更新时间:           {int(time_diff.total_seconds() / 60)} 分钟前")
        else:
            print(f"更新时间:           {int(time_diff.total_seconds() / 3600)} 小时前")
        
        # 显示示例数据
        if images_with_captions > 0:
            print()
            print("📝 示例数据")
            print("-" * 70)
            
            # 找到前3个有描述的图像
            examples = []
            for img_id, phrases in data.items():
                if phrases:
                    examples.append((img_id, phrases))
                if len(examples) >= 3:
                    break
            
            for i, (img_id, phrases) in enumerate(examples, 1):
                print(f"示例 {i} - 图像 ID {img_id}:")
                # 显示前5个短语
                for j, phrase in enumerate(phrases[:5], 1):
                    print(f"  {j}. {phrase}")
                if len(phrases) > 5:
                    print(f"  ... 还有 {len(phrases) - 5} 个短语")
                print()
        
        # 显示进度条（如果有总图像数）
        if total_images > 0 and total_processed > 0:
            print("📈 进度条")
            print("-" * 70)
            bar_width = 50
            filled = int(bar_width * progress_percent / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"[{bar}] {progress_percent:.1f}%")
            print()
        
        # 如果不在监控模式，退出
        if not watch:
            break
        
        # 等待下一次刷新
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(
        description="查看密集描述知识库构建进度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看一次进度
  python check_build_progress.py
  
  # 持续监控（每5秒刷新）
  python check_build_progress.py --watch
  
  # 持续监控（自定义刷新间隔）
  python check_build_progress.py --watch --interval 10
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径 (默认: configs/config.yaml)"
    )
    
    parser.add_argument(
        "--kb-path",
        type=str,
        default=None,
        help="检查点文件路径（默认从配置文件读取）"
    )
    
    parser.add_argument(
        "--watch",
        action="store_true",
        help="持续监控模式，定期刷新显示"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="监控模式下的刷新间隔（秒，默认: 5）"
    )
    
    args = parser.parse_args()
    
    # 获取检查点文件路径
    if args.kb_path:
        kb_path = args.kb_path
    else:
        try:
            cfg = load_config(args.config)
            kb_path = cfg.get("dense_descriptor", {}).get(
                "knowledge_base_path",
                "./output/image_id_to_dense_captions.pkl"
            )
        except Exception as e:
            print(f"❌ 无法加载配置文件: {e}")
            print(f"   使用默认路径: ./output/image_id_to_dense_captions.pkl")
            kb_path = "./output/image_id_to_dense_captions.pkl"
    
    # 显示进度
    try:
        show_progress(kb_path, args.config, args.watch, args.interval)
    except KeyboardInterrupt:
        if args.watch:
            print("\n\n监控已停止。")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

