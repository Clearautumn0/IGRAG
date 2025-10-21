#!/usr/bin/env python3
"""
快速启动脚本 - 图像描述生成系统
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_environment():
    """设置环境"""
    logger.info("设置环境...")
    
    # 创建必要目录
    directories = [
        "cache",
        "cache/faiss_indexes", 
        "outputs",
        "checkpoints"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {dir_path}")


def check_dependencies():
    """检查依赖"""
    logger.info("检查依赖...")
    
    required_packages = [
        "torch",
        "transformers", 
        "open_clip",  # open_clip_torch 的安装包提供的模块名为 open_clip
        "faiss-cpu",
        "PIL",
        "numpy",
        "tqdm"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "PIL":
                import PIL
            elif package == "faiss-cpu":
                import faiss
            else:
                __import__(package)
            logger.info(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package}")
    
    if missing_packages:
        logger.error(f"缺少依赖包: {missing_packages}")
        logger.error("请运行: pip install -r requirements.txt")
        return False
    
    logger.info("所有依赖检查通过")
    return True


def build_indexes(coco_root: str, output_dir: str, max_images: int = None):
    """构建索引"""
    logger.info("构建FAISS索引...")
    
    try:
        from feature_extractor import CLIPFeatureExtractor
        from index_builder import build_indexes_from_coco
        
        # 初始化特征提取器
        feature_extractor = CLIPFeatureExtractor(device="cpu")
        
        # 构建索引
        build_indexes_from_coco(
            coco_root=coco_root,
            output_dir=output_dir,
            feature_extractor=feature_extractor,
            max_images=max_images
        )
        
        logger.info("索引构建完成")
        return True
        
    except Exception as e:
        logger.error(f"索引构建失败: {e}")
        return False


def run_examples():
    """运行示例"""
    logger.info("运行使用示例...")
    
    try:
        from examples import main as run_examples_main
        run_examples_main()
        return True
        
    except Exception as e:
        logger.error(f"示例运行失败: {e}")
        return False


def run_tests():
    """运行测试"""
    logger.info("运行系统测试...")
    
    try:
        from test_system import run_all_tests
        return run_all_tests()
        
    except Exception as e:
        logger.error(f"测试运行失败: {e}")
        return False


def generate_caption(image_path: str):
    """生成图像描述"""
    logger.info(f"生成图像描述: {image_path}")
    
    try:
        from feature_extractor import CLIPFeatureExtractor
        from index_builder import FAISSIndexBuilder
        from retriever import HierarchicalRetriever
        from generator import ImageCaptionGenerator, create_generator
        from main import ImageCaptionPipeline
        
        # 检查索引是否存在
        index_dir = Path("cache/faiss_indexes")
        if not index_dir.exists():
            logger.error("索引不存在，请先构建索引")
            return False
        
        # 初始化组件
        feature_extractor = CLIPFeatureExtractor(device="cpu")
        index_builder = FAISSIndexBuilder(feature_dim=512)
        index_builder.load_indexes(index_dir)
        
        retriever = HierarchicalRetriever(feature_extractor, index_builder)
        generator = create_generator("flan-t5", device="cpu")
        caption_generator = ImageCaptionGenerator(retriever, generator)
        
        # 初始化管道
        pipeline = ImageCaptionPipeline(feature_extractor, index_builder, retriever, caption_generator)
        
        # 生成描述
        result = pipeline.generate_caption(image_path)
        
        print(f"\n生成的描述: {result['caption']}")
        print(f"\n检索统计:")
        print(result['retrieval_stats'])
        
        return True
        
    except Exception as e:
        logger.error(f"描述生成失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="图像描述生成系统快速启动")
    parser.add_argument("--action", choices=["setup", "check", "build", "example", "test", "generate"], 
                       default="setup", help="执行的操作")
    parser.add_argument("--coco_root", help="COCO数据集根目录")
    parser.add_argument("--output_dir", default="cache/faiss_indexes", help="输出目录")
    parser.add_argument("--max_images", type=int, help="最大处理图像数量")
    parser.add_argument("--image_path", help="要生成描述的图像路径")
    
    args = parser.parse_args()
    
    logger.info(f"执行操作: {args.action}")
    
    # 设置环境
    setup_environment()
    
    success = True
    
    if args.action == "setup":
        logger.info("环境设置完成")
        
    elif args.action == "check":
        success = check_dependencies()
        
    elif args.action == "build":
        if not args.coco_root:
            logger.error("构建索引需要指定 --coco_root")
            success = False
        else:
            success = build_indexes(args.coco_root, args.output_dir, args.max_images)
            
    elif args.action == "example":
        success = run_examples()
        
    elif args.action == "test":
        success = run_tests()
        
    elif args.action == "generate":
        if not args.image_path:
            logger.error("生成描述需要指定 --image_path")
            success = False
        else:
            success = generate_caption(args.image_path)
    
    if success:
        logger.info("操作完成!")
        sys.exit(0)
    else:
        logger.error("操作失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()
