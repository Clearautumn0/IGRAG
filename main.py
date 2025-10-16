"""
评估指标和主流程实现
"""
import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm

# 评估指标相关导入
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOEval
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except ImportError as e:
    print(f"评估指标依赖未安装: {e}")
    print("请安装: pip install pycocotools nltk")

from config import (
    MODEL_CONFIG, DATA_CONFIG, EVALUATION_CONFIG, 
    OUTPUT_CONFIG, RETRIEVAL_CONFIG, GENERATION_CONFIG
)
from feature_extractor import CLIPFeatureExtractor
from index_builder import FAISSIndexBuilder, build_indexes_from_coco
from retriever import HierarchicalRetriever
from generator import ImageCaptionGenerator, create_generator

logger = logging.getLogger(__name__)


class CaptionEvaluator:
    """图像描述评估器"""
    
    def __init__(self):
        """初始化评估器"""
        self.smoothing_function = SmoothingFunction().method1
        
    def evaluate_single(self, predicted: str, references: List[str]) -> Dict[str, float]:
        """
        评估单个预测结果
        
        Args:
            predicted: 预测描述
            references: 参考描述列表
            
        Returns:
            评估指标字典
        """
        # 预处理文本
        predicted_tokens = self._tokenize(predicted)
        reference_tokens = [self._tokenize(ref) for ref in references]
        
        # 计算BLEU-4
        bleu4 = sentence_bleu(
            reference_tokens, 
            predicted_tokens, 
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=self.smoothing_function
        )
        
        # 计算METEOR
        try:
            meteor = meteor_score(reference_tokens, predicted_tokens)
        except:
            meteor = 0.0
        
        # 计算CIDEr（简化版本）
        cider = self._compute_cider(predicted_tokens, reference_tokens)
        
        # 计算SPICE（简化版本）
        spice = self._compute_spice(predicted_tokens, reference_tokens)
        
        return {
            'bleu4': bleu4,
            'meteor': meteor,
            'cider': cider,
            'spice': spice
        }
    
    def evaluate_batch(self, predictions: List[str], references_list: List[List[str]]) -> Dict[str, float]:
        """
        批量评估
        
        Args:
            predictions: 预测描述列表
            references_list: 参考描述列表的列表
            
        Returns:
            平均评估指标
        """
        all_metrics = []
        
        for pred, refs in zip(predictions, references_list):
            metrics = self.evaluate_single(pred, refs)
            all_metrics.append(metrics)
        
        # 计算平均值
        avg_metrics = {}
        for metric_name in all_metrics[0].keys():
            avg_metrics[metric_name] = np.mean([m[metric_name] for m in all_metrics])
        
        return avg_metrics
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        return text.lower().split()
    
    def _compute_cider(self, predicted: List[str], references: List[List[str]]) -> float:
        """
        计算CIDEr分数（简化版本）
        
        Args:
            predicted: 预测分词列表
            references: 参考分词列表的列表
            
        Returns:
            CIDEr分数
        """
        # 简化的CIDEr计算
        # 实际实现需要更复杂的n-gram匹配和TF-IDF权重
        
        # 计算n-gram重叠
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        cider_scores = []
        for n in range(1, 5):  # 1-gram到4-gram
            pred_ngrams = set(get_ngrams(predicted, n))
            ref_ngrams = set()
            for ref in references:
                ref_ngrams.update(get_ngrams(ref, n))
            
            if len(ref_ngrams) > 0:
                precision = len(pred_ngrams & ref_ngrams) / len(pred_ngrams) if pred_ngrams else 0
                recall = len(pred_ngrams & ref_ngrams) / len(ref_ngrams)
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    cider_scores.append(f1)
        
        return np.mean(cider_scores) if cider_scores else 0.0
    
    def _compute_spice(self, predicted: List[str], references: List[List[str]]) -> float:
        """
        计算SPICE分数（简化版本）
        
        Args:
            predicted: 预测分词列表
            references: 参考分词列表的列表
            
        Returns:
            SPICE分数
        """
        # 简化的SPICE计算
        # 实际实现需要语义解析和场景图匹配
        
        # 基于词汇重叠的简化版本
        pred_words = set(predicted)
        ref_words = set()
        for ref in references:
            ref_words.update(ref)
        
        if len(ref_words) > 0:
            precision = len(pred_words & ref_words) / len(pred_words) if pred_words else 0
            recall = len(pred_words & ref_words) / len(ref_words)
            
            if precision + recall > 0:
                return 2 * precision * recall / (precision + recall)
        
        return 0.0


class ImageCaptionPipeline:
    """图像描述生成管道"""
    
    def __init__(self, 
                 feature_extractor: CLIPFeatureExtractor,
                 index_builder: FAISSIndexBuilder,
                 retriever: HierarchicalRetriever,
                 generator: ImageCaptionGenerator):
        """
        初始化管道
        
        Args:
            feature_extractor: 特征提取器
            index_builder: 索引构建器
            retriever: 检索器
            generator: 生成器
        """
        self.feature_extractor = feature_extractor
        self.index_builder = index_builder
        self.retriever = retriever
        self.generator = generator
        
        logger.info("图像描述生成管道初始化完成")
    
    def generate_caption(self, image_path: str, **kwargs) -> Dict[str, str]:
        """
        生成单张图像的描述
        
        Args:
            image_path: 图像路径
            **kwargs: 生成参数
            
        Returns:
            生成结果
        """
        return self.generator.generate_caption(image_path, **kwargs)
    
    def evaluate_on_coco(self, 
                        coco_root: str, 
                        split: str = "val",
                        max_samples: Optional[int] = None) -> Dict[str, float]:
        """
        在COCO数据集上评估
        
        Args:
            coco_root: COCO数据集根目录
            split: 数据集划分 ("val", "test")
            max_samples: 最大评估样本数
            
        Returns:
            评估结果
        """
        logger.info(f"开始在COCO {split}集上评估...")
        
        # 加载COCO数据
        coco = COCO(os.path.join(coco_root, "annotations", f"captions_{split}2017.json"))
        
        # 获取图像ID
        img_ids = coco.getImgIds()
        if max_samples:
            img_ids = img_ids[:max_samples]
        
        logger.info(f"评估 {len(img_ids)} 张图像...")
        
        # 生成描述
        predictions = []
        references_list = []
        
        for img_id in tqdm(img_ids, desc="生成描述"):
            try:
                # 获取图像信息
                img_info = coco.loadImgs(img_id)[0]
                img_path = os.path.join(coco_root, f"{split}2017", img_info['file_name'])
                
                if not os.path.exists(img_path):
                    continue
                
                # 生成描述
                result = self.generate_caption(img_path)
                predictions.append(result['caption'])
                
                # 获取参考描述
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                references = [ann['caption'] for ann in anns]
                references_list.append(references)
                
            except Exception as e:
                logger.error(f"处理图像 {img_id} 失败: {e}")
                continue
        
        # 评估
        evaluator = CaptionEvaluator()
        metrics = evaluator.evaluate_batch(predictions, references_list)
        
        logger.info(f"评估完成，结果: {metrics}")
        return metrics
    
    def benchmark_performance(self, image_paths: List[str]) -> Dict[str, float]:
        """
        性能基准测试
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            性能指标
        """
        logger.info(f"开始性能基准测试，测试 {len(image_paths)} 张图像...")
        
        times = []
        
        for img_path in tqdm(image_paths, desc="性能测试"):
            start_time = time.time()
            
            try:
                result = self.generate_caption(img_path)
                end_time = time.time()
                
                if result['caption']:  # 成功生成
                    times.append(end_time - start_time)
                    
            except Exception as e:
                logger.error(f"性能测试失败 {img_path}: {e}")
                continue
        
        if times:
            performance_metrics = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'total_time': np.sum(times),
                'success_rate': len(times) / len(image_paths)
            }
        else:
            performance_metrics = {
                'avg_time': 0,
                'std_time': 0,
                'min_time': 0,
                'max_time': 0,
                'total_time': 0,
                'success_rate': 0
            }
        
        logger.info(f"性能基准测试完成: {performance_metrics}")
        return performance_metrics


def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("开始图像描述生成系统...")
    
    # 检查配置
    device = MODEL_CONFIG['device']
    logger.info(f"使用设备: {device}")
    
    # 初始化组件
    logger.info("初始化特征提取器...")
    feature_extractor = CLIPFeatureExtractor(
        model_name=MODEL_CONFIG['clip_model'],
        device=device
    )
    
    logger.info("初始化索引构建器...")
    index_builder = FAISSIndexBuilder(
        feature_dim=feature_extractor.feature_dim,
        index_type=INDEX_CONFIG['index_type'],
        use_gpu=INDEX_CONFIG['use_gpu']
    )
    
    # 检查是否存在预构建的索引
    index_dir = OUTPUT_CONFIG['cache_dir'] / "faiss_indexes"
    if index_dir.exists():
        logger.info(f"加载预构建的索引: {index_dir}")
        index_builder.load_indexes(index_dir)
    else:
        logger.info("未找到预构建的索引，需要先构建索引")
        logger.info("请运行: python index_builder.py --coco_root /path/to/coco --output_dir cache/faiss_indexes")
        return
    
    logger.info("初始化检索器...")
    retriever = HierarchicalRetriever(feature_extractor, index_builder)
    
    logger.info("初始化生成器...")
    generator = create_generator(
        model_type="flan-t5",
        model_name=MODEL_CONFIG['llm_model'],
        device=device
    )
    
    logger.info("初始化图像描述生成器...")
    caption_generator = ImageCaptionGenerator(retriever, generator)
    
    # 初始化管道
    pipeline = ImageCaptionPipeline(
        feature_extractor, index_builder, retriever, caption_generator
    )
    
    # 示例使用
    logger.info("系统初始化完成，开始示例测试...")
    
    # 创建测试图像
    from PIL import Image
    test_image = Image.new('RGB', (224, 224), color='blue')
    test_image_path = "test_image.jpg"
    test_image.save(test_image_path)
    
    try:
        # 生成描述
        result = pipeline.generate_caption(test_image_path)
        logger.info(f"生成的描述: {result['caption']}")
        logger.info(f"检索统计: {result['retrieval_stats']}")
        
        # 性能测试
        performance = pipeline.benchmark_performance([test_image_path])
        logger.info(f"性能指标: {performance}")
        
    finally:
        # 清理测试文件
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
    
    logger.info("图像描述生成系统运行完成")


def build_indexes_main():
    """构建索引的主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="构建FAISS索引")
    parser.add_argument("--coco_root", required=True, help="COCO数据集根目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--max_images", type=int, help="最大处理图像数量")
    parser.add_argument("--device", default="cuda", help="计算设备")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 初始化特征提取器
    feature_extractor = CLIPFeatureExtractor(
        model_name=MODEL_CONFIG['clip_model'],
        device=args.device
    )
    
    # 构建索引
    build_indexes_from_coco(
        coco_root=args.coco_root,
        output_dir=args.output_dir,
        feature_extractor=feature_extractor,
        max_images=args.max_images
    )


def evaluate_main():
    """评估的主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="评估图像描述生成")
    parser.add_argument("--coco_root", required=True, help="COCO数据集根目录")
    parser.add_argument("--index_dir", required=True, help="索引目录")
    parser.add_argument("--split", default="val", help="数据集划分")
    parser.add_argument("--max_samples", type=int, help="最大评估样本数")
    parser.add_argument("--device", default="cuda", help="计算设备")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 初始化组件
    feature_extractor = CLIPFeatureExtractor(
        model_name=MODEL_CONFIG['clip_model'],
        device=args.device
    )
    
    index_builder = FAISSIndexBuilder(
        feature_dim=feature_extractor.feature_dim,
        index_type=INDEX_CONFIG['index_type'],
        use_gpu=INDEX_CONFIG['use_gpu']
    )
    index_builder.load_indexes(Path(args.index_dir))
    
    retriever = HierarchicalRetriever(feature_extractor, index_builder)
    generator = create_generator(
        model_type="flan-t5",
        model_name=MODEL_CONFIG['llm_model'],
        device=args.device
    )
    caption_generator = ImageCaptionGenerator(retriever, generator)
    
    pipeline = ImageCaptionPipeline(
        feature_extractor, index_builder, retriever, caption_generator
    )
    
    # 评估
    metrics = pipeline.evaluate_on_coco(
        coco_root=args.coco_root,
        split=args.split,
        max_samples=args.max_samples
    )
    
    print(f"评估结果: {metrics}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "build_indexes":
        build_indexes_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        evaluate_main()
    else:
        main()
