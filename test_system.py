"""
测试脚本 - 验证系统功能
"""
import os
import sys
import logging
import numpy as np
from pathlib import Path
from PIL import Image
import torch

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_feature_extractor():
    """测试特征提取器"""
    logger.info("=== 测试特征提取器 ===")
    
    try:
        from feature_extractor import CLIPFeatureExtractor
        
        # 创建特征提取器
        extractor = CLIPFeatureExtractor(device="cpu")
        
        # 创建测试图像
        test_image = Image.new('RGB', (224, 224), color='red')
        test_image_path = "test_feature.jpg"
        test_image.save(test_image_path)
        
        try:
            # 测试特征提取
            features = extractor.extract_features_from_path(test_image_path)
            
            print(f"全局特征形状: {features['global'].shape}")
            print(f"Patch特征形状: {features['patches'].shape}")
            print(f"Patch位置数量: {len(features['patch_positions'])}")
            
            # 测试关键patch选择
            key_patches = extractor.select_key_patches(features['patches'], strategy="norm")
            print(f"关键patch索引: {key_patches}")
            
            logger.info("特征提取器测试通过")
            
        finally:
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
                
    except Exception as e:
        logger.error(f"特征提取器测试失败: {e}")
        return False
    
    return True


def test_index_builder():
    """测试索引构建器"""
    logger.info("=== 测试索引构建器 ===")
    
    try:
        from index_builder import FAISSIndexBuilder
        
        # 创建测试数据
        feature_dim = 512
        num_images = 10
        num_patches = 49
        
        # 生成随机特征
        global_features = [np.random.randn(feature_dim) for _ in range(num_images)]
        patch_features_list = [np.random.randn(num_patches, feature_dim) for _ in range(num_images)]
        image_ids = [f"img_{i:06d}" for i in range(num_images)]
        captions_list = [[f"caption {j} for image {i}" for j in range(5)] for i in range(num_images)]
        
        # 构建索引
        index_builder = FAISSIndexBuilder(feature_dim, "Flat", False)
        index_builder.build_global_index(global_features, image_ids, captions_list)
        index_builder.build_local_index(patch_features_list, image_ids, captions_list)
        
        # 测试搜索
        query_global = np.random.randn(feature_dim)
        query_patches = np.random.randn(num_patches, feature_dim)
        
        global_results = index_builder.search_global(query_global, top_k=3)
        local_results = index_builder.search_local(query_patches, top_m=5)
        
        print(f"全局搜索结果: {len(global_results)}")
        print(f"局部搜索结果: {len(local_results)}")
        
        # 测试保存和加载
        test_dir = Path("test_indexes")
        index_builder.save_indexes(test_dir)
        
        new_builder = FAISSIndexBuilder(feature_dim)
        new_builder.load_indexes(test_dir)
        
        logger.info("索引构建器测试通过")
        
        # 清理
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
            
    except Exception as e:
        logger.error(f"索引构建器测试失败: {e}")
        return False
    
    return True


def test_retriever():
    """测试检索器"""
    logger.info("=== 测试检索器 ===")
    
    try:
        from feature_extractor import CLIPFeatureExtractor
        from index_builder import FAISSIndexBuilder
        from retriever import HierarchicalRetriever
        
        # 创建模拟组件
        class MockFeatureExtractor:
            def __init__(self):
                self.feature_dim = 512
            
            def extract_features_from_path(self, image_path):
                return {
                    'global': torch.randn(1, 512),
                    'patches': torch.randn(49, 512)
                }
            
            def select_key_patches(self, patch_features, strategy="norm", num_patches=3):
                return list(range(num_patches))
        
        class MockIndexBuilder:
            def search_global(self, query_feature, top_k=5):
                return [
                    {
                        'image_id': f'img_{i}',
                        'captions': [f'caption {j} for image {i}' for j in range(5)],
                        'score': 0.9 - i * 0.1
                    }
                    for i in range(top_k)
                ]
            
            def search_local(self, query_features, top_m=10):
                return [
                    {
                        'image_id': f'img_{i}',
                        'captions': [f'caption {j} for image {i}' for j in range(5)],
                        'score': 0.8 - i * 0.05,
                        'patch_idx': i % 3
                    }
                    for i in range(top_m)
                ]
        
        # 测试检索器
        feature_extractor = MockFeatureExtractor()
        index_builder = MockIndexBuilder()
        retriever = HierarchicalRetriever(feature_extractor, index_builder)
        
        # 测试检索
        retrieval_results = retriever.retrieve("test_image.jpg")
        
        # 测试分析
        analysis = retriever.analyze_retrieval_results(retrieval_results)
        print(f"检索分析结果: {analysis}")
        
        # 测试提示构建
        prompt = retriever.build_prompt(retrieval_results)
        print(f"构建的提示长度: {len(prompt)}")
        
        logger.info("检索器测试通过")
        
    except Exception as e:
        logger.error(f"检索器测试失败: {e}")
        return False
    
    return True


def test_generator():
    """测试生成器"""
    logger.info("=== 测试生成器 ===")
    
    try:
        from generator import FLANT5Generator, GPT2Generator
        
        # 测试FLAN-T5生成器
        print("测试FLAN-T5生成器...")
        flan_generator = FLANT5Generator(device="cpu")
        test_prompt = "你是一个专业的图像描述生成器。整体相似的图片描述：- A cat sitting on a chair. - A dog running in the park. 在关键局部区域相似的图片描述：- The cat's eyes are green. - The dog has brown fur. 请综合分析以上描述，生成一个全新、准确且详尽的图片描述。"
        
        result = flan_generator.generate(test_prompt, max_length=50)
        print(f"FLAN-T5生成结果: {result[:100]}...")
        
        # 测试GPT-2生成器
        print("测试GPT-2生成器...")
        gpt2_generator = GPT2Generator(device="cpu")
        test_prompt = "This is a beautiful image showing"
        
        result = gpt2_generator.generate(test_prompt, max_length=30)
        print(f"GPT-2生成结果: {result[:100]}...")
        
        logger.info("生成器测试通过")
        
    except Exception as e:
        logger.error(f"生成器测试失败: {e}")
        return False
    
    return True


def test_evaluator():
    """测试评估器"""
    logger.info("=== 测试评估器 ===")
    
    try:
        from main import CaptionEvaluator
        
        evaluator = CaptionEvaluator()
        
        # 测试单个评估
        predicted = "A cat is sitting on a chair"
        references = [
            "A cat sits on a chair",
            "The chair has a cat on it",
            "A feline is resting on furniture"
        ]
        
        metrics = evaluator.evaluate_single(predicted, references)
        print(f"单个评估结果: {metrics}")
        
        # 测试批量评估
        predictions = [
            "A cat is sitting on a chair",
            "A dog is running in the park"
        ]
        references_list = [
            ["A cat sits on a chair", "The chair has a cat on it"],
            ["A dog runs in the park", "The park has a running dog"]
        ]
        
        batch_metrics = evaluator.evaluate_batch(predictions, references_list)
        print(f"批量评估结果: {batch_metrics}")
        
        logger.info("评估器测试通过")
        
    except Exception as e:
        logger.error(f"评估器测试失败: {e}")
        return False
    
    return True


def test_integration():
    """测试系统集成"""
    logger.info("=== 测试系统集成 ===")
    
    try:
        from feature_extractor import CLIPFeatureExtractor
        from index_builder import FAISSIndexBuilder
        from retriever import HierarchicalRetriever
        from generator import ImageCaptionGenerator, create_generator
        from main import ImageCaptionPipeline
        
        # 创建测试数据
        feature_dim = 512
        num_images = 5
        num_patches = 49
        
        # 生成随机特征
        global_features = [np.random.randn(feature_dim) for _ in range(num_images)]
        patch_features_list = [np.random.randn(num_patches, feature_dim) for _ in range(num_images)]
        image_ids = [f"img_{i:06d}" for i in range(num_images)]
        captions_list = [[f"caption {j} for image {i}" for j in range(5)] for i in range(num_images)]
        
        # 初始化组件
        feature_extractor = CLIPFeatureExtractor(device="cpu")
        index_builder = FAISSIndexBuilder(feature_dim, "Flat", False)
        index_builder.build_global_index(global_features, image_ids, captions_list)
        index_builder.build_local_index(patch_features_list, image_ids, captions_list)
        
        retriever = HierarchicalRetriever(feature_extractor, index_builder)
        generator = create_generator("flan-t5", device="cpu")
        caption_generator = ImageCaptionGenerator(retriever, generator)
        
        # 初始化管道
        pipeline = ImageCaptionPipeline(feature_extractor, index_builder, retriever, caption_generator)
        
        # 创建测试图像
        test_image = Image.new('RGB', (224, 224), color='blue')
        test_image_path = "test_integration.jpg"
        test_image.save(test_image_path)
        
        try:
            # 测试完整流程
            result = pipeline.generate_caption(test_image_path)
            print(f"集成测试结果: {result['caption'][:100]}...")
            print(f"检索统计: {result['retrieval_stats']}")
            
            logger.info("系统集成测试通过")
            
        finally:
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
                
    except Exception as e:
        logger.error(f"系统集成测试失败: {e}")
        return False
    
    return True


def run_all_tests():
    """运行所有测试"""
    logger.info("开始运行所有测试...")
    
    tests = [
        ("特征提取器", test_feature_extractor),
        ("索引构建器", test_index_builder),
        ("检索器", test_retriever),
        ("生成器", test_generator),
        ("评估器", test_evaluator),
        ("系统集成", test_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n运行测试: {test_name}")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} 测试通过")
            else:
                logger.error(f"✗ {test_name} 测试失败")
        except Exception as e:
            logger.error(f"✗ {test_name} 测试异常: {e}")
    
    logger.info(f"\n测试完成: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过!")
        return True
    else:
        logger.error(f"❌ {total - passed} 个测试失败")
        return False


def main():
    """主函数"""
    logger.info("开始系统测试...")
    
    success = run_all_tests()
    
    if success:
        logger.info("系统测试全部通过!")
        sys.exit(0)
    else:
        logger.error("系统测试失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()
