"""
使用示例脚本
"""
import os
import logging
from pathlib import Path
from PIL import Image
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_single_image():
    """示例1: 单张图像描述生成"""
    logger.info("=== 示例1: 单张图像描述生成 ===")
    
    try:
        from feature_extractor import CLIPFeatureExtractor
        from index_builder import FAISSIndexBuilder
        from retriever import HierarchicalRetriever
        from generator import ImageCaptionGenerator, create_generator
        from main import ImageCaptionPipeline
        
        # 检查索引是否存在
        index_dir = Path("cache/faiss_indexes")
        if not index_dir.exists():
            logger.error("索引不存在，请先运行: python main.py build_indexes --coco_root /path/to/coco --output_dir cache/faiss_indexes")
            return
        
        # 初始化组件
        logger.info("初始化特征提取器...")
        feature_extractor = CLIPFeatureExtractor(device="cpu")  # 使用CPU避免GPU依赖
        
        logger.info("加载索引...")
        index_builder = FAISSIndexBuilder(feature_dim=512)
        index_builder.load_indexes(index_dir)
        
        logger.info("初始化检索器...")
        retriever = HierarchicalRetriever(feature_extractor, index_builder)
        
        logger.info("初始化生成器...")
        generator = create_generator("flan-t5", device="cpu")
        caption_generator = ImageCaptionGenerator(retriever, generator)
        
        # 初始化管道
        pipeline = ImageCaptionPipeline(feature_extractor, index_builder, retriever, caption_generator)
        
        # 创建测试图像
        test_image = Image.new('RGB', (224, 224), color='red')
        test_image_path = "test_image.jpg"
        test_image.save(test_image_path)
        
        try:
            # 生成描述
            logger.info("生成图像描述...")
            result = pipeline.generate_caption(test_image_path)
            
            print(f"\n生成的描述: {result['caption']}")
            print(f"\n检索统计:")
            print(result['retrieval_stats'])
            
        finally:
            # 清理测试文件
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
                
    except Exception as e:
        logger.error(f"示例1执行失败: {e}")


def example_batch_processing():
    """示例2: 批量处理"""
    logger.info("=== 示例2: 批量处理 ===")
    
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
            return
        
        # 初始化组件
        feature_extractor = CLIPFeatureExtractor(device="cpu")
        index_builder = FAISSIndexBuilder(feature_dim=512)
        index_builder.load_indexes(index_dir)
        retriever = HierarchicalRetriever(feature_extractor, index_builder)
        generator = create_generator("flan-t5", device="cpu")
        caption_generator = ImageCaptionGenerator(retriever, generator)
        pipeline = ImageCaptionPipeline(feature_extractor, index_builder, retriever, caption_generator)
        
        # 创建多个测试图像
        test_images = []
        colors = ['red', 'blue', 'green', 'yellow', 'purple']
        
        for i, color in enumerate(colors):
            test_image = Image.new('RGB', (224, 224), color=color)
            test_image_path = f"test_image_{i}.jpg"
            test_image.save(test_image_path)
            test_images.append(test_image_path)
        
        try:
            # 批量生成描述
            logger.info("批量生成图像描述...")
            results = pipeline.generator.batch_generate_captions(test_images)
            
            print("\n批量生成结果:")
            for i, result in enumerate(results):
                print(f"\n图像 {i+1} ({colors[i]}):")
                print(f"  描述: {result['caption']}")
                print(f"  全局检索结果: {result['global_results_count']} 个")
                print(f"  局部检索结果: {result['local_results_count']} 个")
                
        finally:
            # 清理测试文件
            for test_image_path in test_images:
                if os.path.exists(test_image_path):
                    os.remove(test_image_path)
                    
    except Exception as e:
        logger.error(f"示例2执行失败: {e}")


def example_retrieval_analysis():
    """示例3: 检索结果分析"""
    logger.info("=== 示例3: 检索结果分析 ===")
    
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
            return
        
        # 初始化组件
        feature_extractor = CLIPFeatureExtractor(device="cpu")
        index_builder = FAISSIndexBuilder(feature_dim=512)
        index_builder.load_indexes(index_dir)
        retriever = HierarchicalRetriever(feature_extractor, index_builder)
        generator = create_generator("flan-t5", device="cpu")
        caption_generator = ImageCaptionGenerator(retriever, generator)
        pipeline = ImageCaptionPipeline(feature_extractor, index_builder, retriever, caption_generator)
        
        # 创建测试图像
        test_image = Image.new('RGB', (224, 224), color='blue')
        test_image_path = "test_image_analysis.jpg"
        test_image.save(test_image_path)
        
        try:
            # 获取详细分析
            logger.info("获取检索结果分析...")
            detailed_result = pipeline.generator.generate_with_analysis(test_image_path)
            
            print(f"\n生成的描述: {detailed_result['caption']}")
            
            print("\n检索分析:")
            analysis = detailed_result['retrieval_analysis']
            for key, value in analysis.items():
                print(f"  {key}: {value}")
            
            print("\n全局检索结果 (Top 3):")
            for i, result in enumerate(detailed_result['global_results'][:3]):
                print(f"  {i+1}. 图像ID: {result['image_id']}")
                print(f"     相似度: {result['score']:.3f}")
                print(f"     描述: {result['captions'][0]}")
            
            print("\n局部检索结果 (Top 3):")
            for i, result in enumerate(detailed_result['local_results'][:3]):
                print(f"  {i+1}. 图像ID: {result['image_id']}")
                print(f"     相似度: {result['score']:.3f}")
                print(f"     Patch索引: {result['patch_idx']}")
                print(f"     描述: {result['captions'][0]}")
                
        finally:
            # 清理测试文件
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
                
    except Exception as e:
        logger.error(f"示例3执行失败: {e}")


def example_performance_benchmark():
    """示例4: 性能基准测试"""
    logger.info("=== 示例4: 性能基准测试 ===")
    
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
            return
        
        # 初始化组件
        feature_extractor = CLIPFeatureExtractor(device="cpu")
        index_builder = FAISSIndexBuilder(feature_dim=512)
        index_builder.load_indexes(index_dir)
        retriever = HierarchicalRetriever(feature_extractor, index_builder)
        generator = create_generator("flan-t5", device="cpu")
        caption_generator = ImageCaptionGenerator(retriever, generator)
        pipeline = ImageCaptionPipeline(feature_extractor, index_builder, retriever, caption_generator)
        
        # 创建测试图像
        test_images = []
        for i in range(5):
            test_image = Image.new('RGB', (224, 224), color=np.random.choice(['red', 'blue', 'green', 'yellow']))
            test_image_path = f"test_image_perf_{i}.jpg"
            test_image.save(test_image_path)
            test_images.append(test_image_path)
        
        try:
            # 性能测试
            logger.info("执行性能基准测试...")
            performance = pipeline.benchmark_performance(test_images)
            
            print("\n性能指标:")
            print(f"  平均时间: {performance['avg_time']:.2f}秒")
            print(f"  标准差: {performance['std_time']:.2f}秒")
            print(f"  最小时间: {performance['min_time']:.2f}秒")
            print(f"  最大时间: {performance['max_time']:.2f}秒")
            print(f"  总时间: {performance['total_time']:.2f}秒")
            print(f"  成功率: {performance['success_rate']:.2%}")
            
        finally:
            # 清理测试文件
            for test_image_path in test_images:
                if os.path.exists(test_image_path):
                    os.remove(test_image_path)
                    
    except Exception as e:
        logger.error(f"示例4执行失败: {e}")


def example_custom_configuration():
    """示例5: 自定义配置"""
    logger.info("=== 示例5: 自定义配置 ===")
    
    try:
        from feature_extractor import CLIPFeatureExtractor
        from index_builder import FAISSIndexBuilder
        from retriever import HierarchicalRetriever, AdaptiveRetriever
        from generator import ImageCaptionGenerator, create_generator
        from main import ImageCaptionPipeline
        
        # 检查索引是否存在
        index_dir = Path("cache/faiss_indexes")
        if not index_dir.exists():
            logger.error("索引不存在，请先构建索引")
            return
        
        # 初始化组件
        feature_extractor = CLIPFeatureExtractor(device="cpu")
        index_builder = FAISSIndexBuilder(feature_dim=512)
        index_builder.load_indexes(index_dir)
        
        # 使用自适应检索器
        logger.info("使用自适应检索器...")
        retriever = AdaptiveRetriever(feature_extractor, index_builder)
        
        generator = create_generator("flan-t5", device="cpu")
        caption_generator = ImageCaptionGenerator(retriever, generator)
        pipeline = ImageCaptionPipeline(feature_extractor, index_builder, retriever, caption_generator)
        
        # 创建测试图像
        test_image = Image.new('RGB', (224, 224), color='green')
        test_image_path = "test_image_custom.jpg"
        test_image.save(test_image_path)
        
        try:
            # 使用自定义生成参数
            logger.info("使用自定义生成参数...")
            custom_params = {
                'max_length': 128,
                'temperature': 0.5,
                'beam_size': 2
            }
            
            result = pipeline.generate_caption(test_image_path, **custom_params)
            
            print(f"\n生成的描述: {result['caption']}")
            print(f"\n检索统计:")
            print(result['retrieval_stats'])
            
        finally:
            # 清理测试文件
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
                
    except Exception as e:
        logger.error(f"示例5执行失败: {e}")


def main():
    """主函数"""
    logger.info("开始运行使用示例...")
    
    # 运行所有示例
    example_single_image()
    print("\n" + "="*50 + "\n")
    
    example_batch_processing()
    print("\n" + "="*50 + "\n")
    
    example_retrieval_analysis()
    print("\n" + "="*50 + "\n")
    
    example_performance_benchmark()
    print("\n" + "="*50 + "\n")
    
    example_custom_configuration()
    
    logger.info("所有示例运行完成!")


if __name__ == "__main__":
    main()
