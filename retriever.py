"""
分层检索策略实现
"""
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

from config import RETRIEVAL_CONFIG, PROMPT_TEMPLATE
from feature_extractor import CLIPFeatureExtractor
from index_builder import FAISSIndexBuilder

logger = logging.getLogger(__name__)


class HierarchicalRetriever:
    """分层检索器 - 实现全局和局部检索策略"""
    
    def __init__(self, 
                 feature_extractor: CLIPFeatureExtractor,
                 index_builder: FAISSIndexBuilder):
        """
        初始化分层检索器
        
        Args:
            feature_extractor: 特征提取器
            index_builder: 索引构建器
        """
        self.feature_extractor = feature_extractor
        self.index_builder = index_builder
        
        # 检索配置
        self.global_top_k = RETRIEVAL_CONFIG["global_top_k"]
        self.local_top_m = RETRIEVAL_CONFIG["local_top_m"]
        self.key_patch_strategy = RETRIEVAL_CONFIG["key_patch_strategy"]
        self.num_key_patches = RETRIEVAL_CONFIG["num_key_patches"]
        self.similarity_threshold = RETRIEVAL_CONFIG["similarity_threshold"]
        
        logger.info(f"初始化分层检索器 - 全局Top-K: {self.global_top_k}, "
                   f"局部Top-M: {self.local_top_m}, 关键块策略: {self.key_patch_strategy}")
    
    def retrieve(self, image_path: str) -> Dict[str, List[Dict]]:
        """
        执行分层检索
        
        Args:
            image_path: 查询图像路径
            
        Returns:
            包含全局和局部检索结果的字典
        """
        logger.info(f"开始检索图像: {image_path}")
        
        # 提取查询图像特征
        features = self.feature_extractor.extract_features_from_path(image_path)
        global_features = features['global'].cpu().numpy()
        patch_features = features['patches'].cpu().numpy()
        
        # 全局检索
        global_results = self._retrieve_global(global_features)
        
        # 局部检索
        local_results = self._retrieve_local(patch_features)
        
        # 构建检索结果
        retrieval_results = {
            'global': global_results,
            'local': local_results,
            'query_features': {
                'global': global_features,
                'patches': patch_features
            }
        }
        
        logger.info(f"检索完成 - 全局结果: {len(global_results)}, "
                   f"局部结果: {len(local_results)}")
        
        return retrieval_results
    
    def _retrieve_global(self, query_global_features: np.ndarray) -> List[Dict]:
        """
        执行全局检索
        
        Args:
            query_global_features: 查询全局特征
            
        Returns:
            全局检索结果列表
        """
        try:
            results = self.index_builder.search_global(
                query_global_features, 
                top_k=self.global_top_k
            )
            
            # 过滤低相似度结果
            filtered_results = [
                result for result in results 
                if result['score'] >= self.similarity_threshold
            ]
            
            logger.info(f"全局检索完成，返回 {len(filtered_results)} 个结果")
            return filtered_results
            
        except Exception as e:
            logger.error(f"全局检索失败: {e}")
            return []
    
    def _retrieve_local(self, query_patch_features: np.ndarray) -> List[Dict]:
        """
        执行局部检索
        
        Args:
            query_patch_features: 查询patch特征
            
        Returns:
            局部检索结果列表
        """
        try:
            # 选择关键patch
            key_patch_indices = self.feature_extractor.select_key_patches(
                torch.from_numpy(query_patch_features),
                strategy=self.key_patch_strategy,
                num_patches=self.num_key_patches
            )
            
            # 使用关键patch进行检索
            key_patch_features = query_patch_features[key_patch_indices]
            
            results = self.index_builder.search_local(
                key_patch_features,
                top_m=self.local_top_m
            )
            
            # 过滤低相似度结果
            filtered_results = [
                result for result in results 
                if result['score'] >= self.similarity_threshold
            ]
            
            logger.info(f"局部检索完成，使用 {len(key_patch_indices)} 个关键patch，"
                       f"返回 {len(filtered_results)} 个结果")
            return filtered_results
            
        except Exception as e:
            logger.error(f"局部检索失败: {e}")
            return []
    
    def build_prompt(self, retrieval_results: Dict[str, List[Dict]]) -> str:
        """
        构建分层提示
        
        Args:
            retrieval_results: 检索结果
            
        Returns:
            构建的提示文本
        """
        # 提取全局描述
        global_descriptions = []
        for result in retrieval_results['global']:
            for caption in result['captions']:
                global_descriptions.append(f"- {caption}")
        
        # 提取局部描述
        local_descriptions = []
        for result in retrieval_results['local']:
            for caption in result['captions']:
                local_descriptions.append(f"- {caption}")
        
        # 构建提示
        prompt = PROMPT_TEMPLATE.format(
            global_descriptions="\n".join(global_descriptions[:10]),  # 限制数量
            local_descriptions="\n".join(local_descriptions[:10])
        )
        
        return prompt
    
    def analyze_retrieval_results(self, retrieval_results: Dict[str, List[Dict]]) -> Dict:
        """
        分析检索结果
        
        Args:
            retrieval_results: 检索结果
            
        Returns:
            分析结果字典
        """
        global_results = retrieval_results['global']
        local_results = retrieval_results['local']
        
        # 统计信息
        analysis = {
            'global_count': len(global_results),
            'local_count': len(local_results),
            'global_avg_score': np.mean([r['score'] for r in global_results]) if global_results else 0,
            'local_avg_score': np.mean([r['score'] for r in local_results]) if local_results else 0,
            'global_max_score': max([r['score'] for r in global_results]) if global_results else 0,
            'local_max_score': max([r['score'] for r in local_results]) if local_results else 0,
        }
        
        # 重叠分析
        global_image_ids = set(r['image_id'] for r in global_results)
        local_image_ids = set(r['image_id'] for r in local_results)
        overlap_ids = global_image_ids.intersection(local_image_ids)
        
        analysis.update({
            'overlap_count': len(overlap_ids),
            'overlap_ratio': len(overlap_ids) / max(len(global_image_ids), len(local_image_ids), 1),
            'unique_global': len(global_image_ids - local_image_ids),
            'unique_local': len(local_image_ids - global_image_ids)
        })
        
        return analysis
    
    def get_retrieval_statistics(self, retrieval_results: Dict[str, List[Dict]]) -> str:
        """
        获取检索统计信息
        
        Args:
            retrieval_results: 检索结果
            
        Returns:
            统计信息字符串
        """
        analysis = self.analyze_retrieval_results(retrieval_results)
        
        stats = f"""
检索统计信息:
- 全局检索结果: {analysis['global_count']} 个
- 局部检索结果: {analysis['local_count']} 个
- 全局平均相似度: {analysis['global_avg_score']:.3f}
- 局部平均相似度: {analysis['local_avg_score']:.3f}
- 全局最高相似度: {analysis['global_max_score']:.3f}
- 局部最高相似度: {analysis['local_max_score']:.3f}
- 重叠图像数量: {analysis['overlap_count']} 个
- 重叠比例: {analysis['overlap_ratio']:.3f}
- 仅全局检索到: {analysis['unique_global']} 个
- 仅局部检索到: {analysis['unique_local']} 个
        """
        
        return stats.strip()
    
    def visualize_retrieval_results(self, 
                                 retrieval_results: Dict[str, List[Dict]], 
                                 save_path: Optional[str] = None) -> None:
        """
        可视化检索结果
        
        Args:
            retrieval_results: 检索结果
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 设置样式
            plt.style.use('seaborn-v0_8')
            
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('检索结果分析', fontsize=16)
            
            # 1. 相似度分布
            global_scores = [r['score'] for r in retrieval_results['global']]
            local_scores = [r['score'] for r in retrieval_results['local']]
            
            axes[0, 0].hist(global_scores, alpha=0.7, label='全局', bins=10)
            axes[0, 0].hist(local_scores, alpha=0.7, label='局部', bins=10)
            axes[0, 0].set_title('相似度分布')
            axes[0, 0].set_xlabel('相似度分数')
            axes[0, 0].set_ylabel('频次')
            axes[0, 0].legend()
            
            # 2. Top-K结果对比
            k_values = range(1, min(len(global_scores), len(local_scores)) + 1)
            global_top_k_scores = [max(global_scores[:k]) for k in k_values]
            local_top_k_scores = [max(local_scores[:k]) for k in k_values]
            
            axes[0, 1].plot(k_values, global_top_k_scores, 'o-', label='全局')
            axes[0, 1].plot(k_values, local_top_k_scores, 's-', label='局部')
            axes[0, 1].set_title('Top-K最高相似度')
            axes[0, 1].set_xlabel('K值')
            axes[0, 1].set_ylabel('最高相似度')
            axes[0, 1].legend()
            
            # 3. 检索结果数量对比
            categories = ['全局检索', '局部检索', '重叠结果']
            counts = [
                len(retrieval_results['global']),
                len(retrieval_results['local']),
                len(set(r['image_id'] for r in retrieval_results['global']).intersection(
                    set(r['image_id'] for r in retrieval_results['local'])))
            ]
            
            axes[1, 0].bar(categories, counts, color=['skyblue', 'lightcoral', 'lightgreen'])
            axes[1, 0].set_title('检索结果数量对比')
            axes[1, 0].set_ylabel('数量')
            
            # 4. 相似度箱线图
            all_scores = [global_scores, local_scores]
            labels = ['全局', '局部']
            axes[1, 1].boxplot(all_scores, labels=labels)
            axes[1, 1].set_title('相似度分布箱线图')
            axes[1, 1].set_ylabel('相似度分数')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"检索结果可视化已保存到: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib或seaborn未安装，跳过可视化")
        except Exception as e:
            logger.error(f"可视化失败: {e}")


class AdaptiveRetriever(HierarchicalRetriever):
    """自适应检索器 - 根据查询图像动态调整检索策略"""
    
    def __init__(self, 
                 feature_extractor: CLIPFeatureExtractor,
                 index_builder: FAISSIndexBuilder):
        super().__init__(feature_extractor, index_builder)
        
        # 自适应参数
        self.min_global_results = 3
        self.min_local_results = 5
        self.score_decay_factor = 0.1
    
    def adaptive_retrieve(self, image_path: str) -> Dict[str, List[Dict]]:
        """
        自适应检索
        
        Args:
            image_path: 查询图像路径
            
        Returns:
            检索结果
        """
        logger.info(f"开始自适应检索: {image_path}")
        
        # 提取特征
        features = self.feature_extractor.extract_features_from_path(image_path)
        global_features = features['global'].cpu().numpy()
        patch_features = features['patches'].cpu().numpy()
        
        # 初始检索
        global_results = self._retrieve_global(global_features)
        local_results = self._retrieve_local(patch_features)
        
        # 自适应调整
        if len(global_results) < self.min_global_results:
            # 降低全局检索阈值
            original_threshold = self.similarity_threshold
            self.similarity_threshold *= (1 - self.score_decay_factor)
            global_results = self._retrieve_global(global_features)
            self.similarity_threshold = original_threshold
            
            logger.info(f"全局检索结果不足，降低阈值后获得 {len(global_results)} 个结果")
        
        if len(local_results) < self.min_local_results:
            # 增加局部检索数量
            original_top_m = self.local_top_m
            self.local_top_m = int(self.local_top_m * 1.5)
            local_results = self._retrieve_local(patch_features)
            self.local_top_m = original_top_m
            
            logger.info(f"局部检索结果不足，增加检索数量后获得 {len(local_results)} 个结果")
        
        return {
            'global': global_results,
            'local': local_results,
            'query_features': {
                'global': global_features,
                'patches': patch_features
            }
        }


def test_retriever():
    """测试检索器"""
    import numpy as np
    from pathlib import Path
    
    # 创建测试数据
    feature_dim = 512
    num_patches = 49
    
    # 模拟特征提取器
    class MockFeatureExtractor:
        def __init__(self):
            self.feature_dim = feature_dim
        
        def extract_features_from_path(self, image_path):
            return {
                'global': torch.randn(1, feature_dim),
                'patches': torch.randn(num_patches, feature_dim)
            }
        
        def select_key_patches(self, patch_features, strategy="norm", num_patches=3):
            return list(range(num_patches))
    
    # 模拟索引构建器
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
    print("检索分析结果:", analysis)
    
    # 测试提示构建
    prompt = retriever.build_prompt(retrieval_results)
    print("构建的提示:", prompt[:200] + "...")
    
    # 测试统计信息
    stats = retriever.get_retrieval_statistics(retrieval_results)
    print("检索统计:", stats)
    
    print("检索器测试完成")


if __name__ == "__main__":
    test_retriever()
