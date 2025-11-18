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
        """对每个图像块进行处理，返回位置信息而非描述。
        
        Args:
            image_patches: (裁剪图像, 检测信息) 元组列表
                        检测信息应包含 'position' 字段（由retriever计算）
            exclude_image_ids: 要排除的image_id集合（已不再使用，保留以兼容接口）
            
        Returns:
            局部结果列表，每个元素包含：
            - 'bbox': 边界框坐标
            - 'class_label': 类别标签
            - 'confidence': 检测置信度
            - 'position': 相对位置字符串（如 "top left", "center"）
        """
        local_results = []
        
        for idx, (patch_image, detection_info) in enumerate(image_patches):
            try:
                # 从detection_info中获取位置信息（由retriever预先计算）
                position = detection_info.get('position', 'unknown')
                
                local_results.append({
                    'bbox': detection_info['bbox'],
                    'class_label': detection_info['class_label'],
                    'confidence': detection_info['confidence'],
                    'position': position
                })
                
                logger.debug(f"Local region {idx+1}: {detection_info['class_label']} - "
                           f"position: {position}")
                
            except Exception as e:
                logger.warning(f"Failed to process local region {idx+1}: {e}")
                # 即使处理失败，也保留检测信息
                local_results.append({
                    'bbox': detection_info.get('bbox', []),
                    'class_label': detection_info.get('class_label', 'unknown'),
                    'confidence': detection_info.get('confidence', 0.0),
                    'position': detection_info.get('position', 'unknown')
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

