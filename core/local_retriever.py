import logging
from typing import List, Dict, Tuple, Union
from PIL import Image
import numpy as np

from core.retriever import ImageRetriever

logger = logging.getLogger(__name__)


class LocalRetriever:
    """对局部图像块进行检索，复用全局检索器的知识库和CLIP模型。"""

    def __init__(self, global_retriever: ImageRetriever, config: dict):
        """初始化局部检索器。
        
        Args:
            global_retriever: ImageRetriever实例，复用其CLIP模型和知识库
            config: 配置字典，包含局部检索参数
        """
        self.retriever = global_retriever
        patch_config = config.get("patch_config", {})
        self.local_top_k = int(patch_config.get("local_retrieval_top_k", 2))
        
        logger.info(f"LocalRetriever initialized with top_k={self.local_top_k}")

    def retrieve_local_descriptions(self, image_patches: List[Tuple[Image.Image, Dict]], 
                                   exclude_image_ids: set = None) -> List[Dict]:
        """对每个图像块进行检索。
        
        Args:
            image_patches: (裁剪图像, 检测信息) 元组列表
            exclude_image_ids: 要排除的image_id集合（通常是全局检索已使用的image_id）
            
        Returns:
            局部检索结果列表，每个元素包含：
            - 'bbox': 边界框坐标
            - 'class_label': 类别标签
            - 'confidence': 检测置信度
            - 'descriptions': 检索到的描述列表
            - 'retrieval_scores': 检索分数列表
        """
        if exclude_image_ids is None:
            exclude_image_ids = set()
        
        local_results = []
        
        for idx, (patch_image, detection_info) in enumerate(image_patches):
            try:
                # 使用全局检索器提取特征并检索
                # 增加检索数量，以便在排除后仍有足够的候选
                # 如果有很多要排除的image_id，需要检索更多
                base_search_k = self.local_top_k + len(exclude_image_ids)
                search_top_k = min(base_search_k + 5, 20)  # 最多检索20个，确保有足够候选
                hits = self.retriever.retrieve_similar_images(patch_image, top_k=search_top_k)
                
                # 只取第一条不在全局检索结果中的描述
                description = None
                score = None
                excluded_count = 0
                
                for image_id, hit_score in hits:
                    # 跳过已经在全局检索结果中的image_id
                    if image_id in exclude_image_ids:
                        excluded_count += 1
                        continue
                    
                    # 获取该image_id的第一个caption作为描述
                    captions = self.retriever.image_id_to_captions.get(image_id, [])
                    if captions and len(captions) > 0:
                        description = captions[0].strip()  # 只取第一条描述
                        score = hit_score
                        break
                
                logger.debug(f"Local region {idx+1}: excluded {excluded_count} image_ids that were in global results")
                
                # 如果没有找到描述（全部被排除），记录警告
                if description is None:
                    logger.warning(f"Local region {idx+1}: all retrieved images were excluded, "
                                 f"no unique description found")
                    descriptions = []
                    scores = []
                else:
                    descriptions = [description]  # 只返回一条描述
                    scores = [score] if score is not None else []
                
                local_results.append({
                    'bbox': detection_info['bbox'],
                    'class_label': detection_info['class_label'],
                    'confidence': detection_info['confidence'],
                    'descriptions': descriptions,
                    'retrieval_scores': scores
                })
                
                logger.debug(f"Local region {idx+1}: {detection_info['class_label']} - "
                           f"retrieved {len(descriptions)} description(s) "
                           f"(excluded {excluded_count} global image_ids)")
                
            except Exception as e:
                logger.warning(f"Failed to retrieve descriptions for local region {idx+1}: {e}")
                # 即使检索失败，也保留检测信息
                local_results.append({
                    'bbox': detection_info['bbox'],
                    'class_label': detection_info['class_label'],
                    'confidence': detection_info['confidence'],
                    'descriptions': [],
                    'retrieval_scores': []
                })
        
        return local_results

    def merge_local_descriptions(self, local_results: List[Dict]) -> List[Dict]:
        """合并多个局部检索结果。
        
        Args:
            local_results: 局部检索结果列表
            
        Returns:
            合并后的结果（当前实现直接返回，保留接口以便后续扩展）
        """
        # 当前实现直接返回，后续可以添加去重、排序等逻辑
        return local_results

