import os
import logging
from typing import List, Dict, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class DescriptionOptimizer:
    """智能描述聚类优化器，用于对相似图片的多个描述进行聚类并选择代表性描述。"""

    def __init__(self, config: Union[dict, str]):
        """初始化描述优化器。
        
        Args:
            config: 配置字典或配置文件路径
        """
        if isinstance(config, str):
            import yaml
            with open(config, "r") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = config
        
        # 获取描述优化配置
        opt_config = cfg.get("description_optimization", {})
        
        if not opt_config.get("enabled", False):
            logger.warning("Description optimization is disabled in config")
            self.enabled = False
            return
        
        self.enabled = True
        
        # 加载句子嵌入模型
        embedding_model_path = opt_config.get("embedding_model", "../models/all-MiniLM-L6-v2/")
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(embedding_model_path)
            logger.info(f"Loaded embedding model from {embedding_model_path}")
        except Exception as e:
            logger.error(f"Failed to load embedding model from {embedding_model_path}: {e}")
            raise
        
        # 配置聚类参数
        clustering_algorithm = opt_config.get("clustering_algorithm", "hdbscan")
        min_cluster_size = opt_config.get("min_cluster_size", 2)
        
        if clustering_algorithm == "hdbscan":
            try:
                from hdbscan import HDBSCAN
                hdbscan_config = opt_config.get("hdbscan", {})
                min_samples = hdbscan_config.get("min_samples", 1)
                cluster_selection_epsilon = hdbscan_config.get("cluster_selection_epsilon", 0.1)
                self.cluster_model = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=cluster_selection_epsilon
                )
                logger.info(f"Initialized HDBSCAN with min_cluster_size={min_cluster_size}")
            except ImportError:
                logger.warning("HDBSCAN not available, falling back to DBSCAN")
                clustering_algorithm = "dbscan"
        
        if clustering_algorithm == "dbscan":
            try:
                from sklearn.cluster import DBSCAN
                self.cluster_model = DBSCAN(eps=0.5, min_samples=min_cluster_size)
                logger.info(f"Initialized DBSCAN with min_samples={min_cluster_size}")
            except ImportError:
                logger.error("Neither HDBSCAN nor sklearn.DBSCAN available")
                raise
        
        # 配置分数权重
        score_weights = opt_config.get("score_weights", {})
        self.w1 = float(score_weights.get("cluster_similarity", 0.4))
        self.w2 = float(score_weights.get("image_similarity", 0.4))
        self.w3 = float(score_weights.get("brevity", 0.2))
        
        # 归一化权重
        total_weight = self.w1 + self.w2 + self.w3
        if total_weight > 0:
            self.w1 /= total_weight
            self.w2 /= total_weight
            self.w3 /= total_weight
        
        # 最大返回描述数量
        self.max_final_descriptions = int(opt_config.get("max_final_descriptions", 5))
        
        logger.info(f"DescriptionOptimizer initialized with weights: cluster={self.w1:.2f}, "
                   f"image={self.w2:.2f}, brevity={self.w3:.2f}")

    def cluster_descriptions(self, descriptions: List[str], image_similarities: List[float]) -> Tuple[np.ndarray, List[int]]:
        """对描述进行聚类分析。
        
        Args:
            descriptions: 描述文本列表
            image_similarities: 对应的图像相似度分数列表
            
        Returns:
            (embeddings, cluster_labels): 嵌入向量和聚类标签
        """
        if not descriptions:
            return np.array([]), []
        
        # 将描述转换为嵌入向量
        try:
            embeddings = self.embedding_model.encode(descriptions, convert_to_numpy=True, show_progress_bar=False)
            logger.debug(f"Generated embeddings for {len(descriptions)} descriptions, shape: {embeddings.shape}")
        except Exception as e:
            logger.error(f"Failed to encode descriptions: {e}")
            raise
        
        # 执行聚类
        try:
            cluster_labels = self.cluster_model.fit_predict(embeddings)
            logger.debug(f"Clustering completed, found {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters")
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            raise
        
        return embeddings, cluster_labels.tolist()

    def select_representative_descriptions(
        self, 
        descriptions: List[str], 
        embeddings: np.ndarray,
        cluster_labels: List[int],
        image_similarities: List[float]
    ) -> List[Dict]:
        """为每个聚类选择代表性描述。
        
        Args:
            descriptions: 描述文本列表
            embeddings: 描述嵌入向量
            cluster_labels: 聚类标签列表（-1表示噪声点）
            image_similarities: 对应的图像相似度分数列表
            
        Returns:
            优化后的描述列表，每个元素包含描述和评分信息
        """
        if not descriptions or len(descriptions) == 0:
            return []
        
        # 获取所有聚类ID（排除噪声点-1）
        unique_clusters = set(cluster_labels)
        unique_clusters.discard(-1)
        
        if not unique_clusters:
            # 如果没有有效聚类，返回所有描述（按图像相似度排序）
            logger.warning("No valid clusters found, returning all descriptions sorted by image similarity")
            results = []
            for desc, img_sim in zip(descriptions, image_similarities):
                # 计算简洁性分数（句子越短分数越高）
                brevity_score = self._calculate_brevity_score(desc)
                combined_score = self.w2 * img_sim + self.w3 * brevity_score
                results.append({
                    'description': desc,
                    'cluster_score': 0.0,
                    'image_similarity': img_sim,
                    'brevity_score': brevity_score,
                    'combined_score': combined_score,
                    'cluster_size': 1
                })
            # 按综合分数排序并限制数量
            results.sort(key=lambda x: x['combined_score'], reverse=True)
            return results[:self.max_final_descriptions]
        
        # 为每个聚类选择代表性描述
        representative_descriptions = []
        
        for cluster_id in unique_clusters:
            # 获取该聚类中的所有描述索引
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            if not cluster_indices:
                continue
            
            cluster_size = len(cluster_indices)
            cluster_embeddings = embeddings[cluster_indices]
            cluster_descriptions = [descriptions[i] for i in cluster_indices]
            cluster_image_sims = [image_similarities[i] for i in cluster_indices]
            
            # 计算聚类中心（均值向量）
            cluster_center = np.mean(cluster_embeddings, axis=0)
            
            # 为每个描述计算综合分数
            best_desc_idx = None
            best_score = -float('inf')
            best_metrics = {}
            
            for local_idx, global_idx in enumerate(cluster_indices):
                desc = descriptions[global_idx]
                emb = embeddings[global_idx]
                img_sim = image_similarities[global_idx]
                
                # 1. 聚类中心距离分数（余弦相似度）
                cluster_sim = self._cosine_similarity(emb, cluster_center)
                
                # 2. 图像相似度分数（已归一化到[0,1]）
                normalized_img_sim = max(0.0, min(1.0, img_sim))
                
                # 3. 简洁性分数
                brevity_score = self._calculate_brevity_score(desc)
                
                # 综合分数
                combined_score = (
                    self.w1 * cluster_sim +
                    self.w2 * normalized_img_sim +
                    self.w3 * brevity_score
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_desc_idx = global_idx
                    best_metrics = {
                        'cluster_score': cluster_sim,
                        'image_similarity': normalized_img_sim,
                        'brevity_score': brevity_score,
                        'combined_score': combined_score
                    }
            
            if best_desc_idx is not None:
                representative_descriptions.append({
                    'description': descriptions[best_desc_idx],
                    'cluster_score': best_metrics['cluster_score'],
                    'image_similarity': best_metrics['image_similarity'],
                    'brevity_score': best_metrics['brevity_score'],
                    'combined_score': best_metrics['combined_score'],
                    'cluster_size': cluster_size
                })
        
        # 按综合分数排序
        representative_descriptions.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # 限制返回数量
        return representative_descriptions[:self.max_final_descriptions]

    def optimize_descriptions(
        self, 
        descriptions: List[str], 
        image_similarities: List[float]
    ) -> List[Dict]:
        """主优化方法：对描述进行聚类并选择代表性描述。
        
        Args:
            descriptions: 描述文本列表
            image_similarities: 对应的图像相似度分数列表
            
        Returns:
            优化后的描述列表，每个元素包含描述和评分信息
        """
        if not self.enabled:
            logger.warning("Description optimization is disabled, returning original descriptions")
            # 返回原始格式的简化版本
            return [
                {
                    'description': desc,
                    'cluster_score': 0.0,
                    'image_similarity': sim,
                    'brevity_score': self._calculate_brevity_score(desc),
                    'combined_score': sim,
                    'cluster_size': 1
                }
                for desc, sim in zip(descriptions, image_similarities)
            ]
        
        if not descriptions or len(descriptions) == 0:
            return []
        
        if len(descriptions) == 1:
            # 只有一个描述，直接返回
            return [{
                'description': descriptions[0],
                'cluster_score': 1.0,
                'image_similarity': image_similarities[0] if image_similarities else 0.0,
                'brevity_score': self._calculate_brevity_score(descriptions[0]),
                'combined_score': image_similarities[0] if image_similarities else 0.0,
                'cluster_size': 1
            }]
        
        try:
            # 1. 聚类分析
            embeddings, cluster_labels = self.cluster_descriptions(descriptions, image_similarities)
            
            # 2. 选择代表性描述
            optimized = self.select_representative_descriptions(
                descriptions, embeddings, cluster_labels, image_similarities
            )
            
            logger.info(f"Optimized {len(descriptions)} descriptions to {len(optimized)} representative descriptions")
            return optimized
            
        except Exception as e:
            logger.error(f"Description optimization failed: {e}, returning original descriptions")
            # 失败时返回原始描述
            return [
                {
                    'description': desc,
                    'cluster_score': 0.0,
                    'image_similarity': sim,
                    'brevity_score': self._calculate_brevity_score(desc),
                    'combined_score': sim,
                    'cluster_size': 1
                }
                for desc, sim in zip(descriptions, image_similarities)
            ]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度。"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    def _calculate_brevity_score(self, description: str) -> float:
        """计算简洁性分数：句子越短分数越高。
        
        使用反比例函数：score = 1 / (1 + length / base_length)
        其中base_length是参考长度（如20个词）
        """
        if not description:
            return 0.0
        
        # 计算词数（简单按空格分割）
        words = description.strip().split()
        word_count = len(words)
        
        # 参考长度：20个词
        base_length = 20.0
        
        # 使用反比例函数，确保分数在[0, 1]范围内
        # 当word_count=0时，score=1.0
        # 当word_count=base_length时，score≈0.5
        # 当word_count→∞时，score→0
        score = 1.0 / (1.0 + word_count / base_length)
        
        return float(score)

