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

    def retrieve_local_descriptions(self, image_patches: List[Tuple[Image.Image, Dict]]) -> List[Dict]:
        """对每个图像块进行检索。
        
        Args:
            image_patches: (裁剪图像, 检测信息) 元组列表
            
        Returns:
            局部检索结果列表，每个元素包含：
            - 'bbox': 边界框坐标
            - 'class_label': 类别标签
            - 'confidence': 检测置信度
            - 'descriptions': 检索到的描述列表
            - 'retrieval_scores': 检索分数列表
        """
        local_results = []
        
        for idx, (patch_image, detection_info) in enumerate(image_patches):
            try:
                # 使用全局检索器提取特征并检索
                hits = self.retriever.retrieve_similar_images(patch_image, top_k=self.local_top_k)
                
                # 提取描述
                descriptions = []
                scores = []
                for image_id, score in hits:
                    captions = self.retriever.image_id_to_captions.get(image_id, [])
                    if captions:
                        descriptions.extend(captions)
                        scores.append(score)
                
                # 去重并保留前N个描述
                if descriptions:
                    # 简单去重：保留每个描述的第一个出现
                    seen = set()
                    unique_descriptions = []
                    for desc in descriptions:
                        desc_lower = desc.lower().strip()
                        if desc_lower not in seen:
                            seen.add(desc_lower)
                            unique_descriptions.append(desc)
                    descriptions = unique_descriptions[:self.local_top_k * 3]  # 每个检索结果最多3个描述
                
                local_results.append({
                    'bbox': detection_info['bbox'],
                    'class_label': detection_info['class_label'],
                    'confidence': detection_info['confidence'],
                    'descriptions': descriptions,
                    'retrieval_scores': scores[:self.local_top_k]
                })
                
                logger.debug(f"Local region {idx+1}: {detection_info['class_label']} - "
                           f"retrieved {len(descriptions)} descriptions")
                
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

