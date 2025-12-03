"""物体感知的混合检索模块。

该模块包装并增强现有的CLIP检索器，通过两阶段混合检索提升检索结果的实体相关性：
1. 第一阶段（召回）：使用CLIP检索器初步召回视觉相似的候选图像
2. 第二阶段（重排序）：利用物体匹配度对候选图像进行重排序
"""
import os
import json
import logging
from typing import List, Dict, Set, Union, Optional
from pathlib import Path
from PIL import Image

from core.retriever import ImageRetriever
from core.patch_detector import PatchDetector
from core.retrieval_enhancer import reorder_by_objects
from utils.image_utils import load_image

logger = logging.getLogger(__name__)


class ObjectAwareHybridRetriever:
    """物体感知的混合检索器。
    
    该检索器包装并增强现有的CLIP检索器，通过引入物体感知能力，
    使检索结果不仅在场景层面相似，更在关键实体层面匹配。
    """
    
    def __init__(self, base_retriever: ImageRetriever, detector: PatchDetector, config: dict):
        """初始化物体感知混合检索器。
        
        Args:
            base_retriever: 基础的CLIP检索器实例
            detector: Faster R-CNN检测器实例
            config: 配置字典，包含混合检索相关参数
        """
        self.base_retriever = base_retriever
        self.detector = detector
        self.config = config
        
        # 从配置中读取混合检索参数
        hybrid_config = config.get("hybrid_retrieval", {})
        self.object_weight = float(hybrid_config.get("object_weight", 0.5))
        self.initial_recall_k = int(hybrid_config.get("initial_recall_k", 20))
        
        # 尝试加载预构建的物体标签映射（优先使用）
        self.image_id_to_objects = self._load_object_tags_mapping(config)
        
        # 如果预构建的映射不存在，则使用实时检测（回退方案）
        if not self.image_id_to_objects:
            logger.warning(
                "Pre-built object tags mapping not found. Will use real-time detection for candidate images. "
                "This is slower. Consider running build_knowledge_base.py to generate image_id_to_objects.pkl"
            )
            # 加载image_id到文件名的映射（用于从image_id加载图像进行实时检测）
            self.image_id_to_filename = self._load_image_id_mapping(config)
            self.images_dir = Path(config.get("data_config", {}).get("coco_images_dir", ""))
        else:
            logger.info(f"Loaded pre-built object tags for {len(self.image_id_to_objects)} images")
            self.image_id_to_filename = {}
            self.images_dir = None
        
        logger.info(
            f"ObjectAwareHybridRetriever initialized: "
            f"object_weight={self.object_weight}, initial_recall_k={self.initial_recall_k}"
        )
    
    def _load_object_tags_mapping(self, config: dict) -> Dict[int, List[str]]:
        """从预构建的pickle文件中加载image_id到物体类别列表的映射。
        
        Args:
            config: 配置字典
            
        Returns:
            image_id到物体类别列表的映射字典，如果文件不存在则返回空字典
        """
        import pickle
        
        objects_path = config.get("knowledge_base_config", {}).get("image_id_to_objects_path", "")
        
        if not objects_path or not os.path.exists(objects_path):
            return {}
        
        try:
            with open(objects_path, "rb") as f:
                image_id_to_objects = pickle.load(f)
            
            # 转换为列表格式（如果存储的是列表）或保持集合格式
            # 统一转换为列表以便后续处理
            result = {}
            for img_id, objects in image_id_to_objects.items():
                if isinstance(objects, (list, tuple)):
                    result[img_id] = list(objects)
                elif isinstance(objects, set):
                    result[img_id] = sorted(list(objects))
                else:
                    result[img_id] = []
            
            logger.info(f"Loaded object tags mapping from {objects_path}")
            return result
        except Exception as e:
            logger.warning(f"Failed to load object tags mapping from {objects_path}: {e}")
            return {}
    
    def _load_image_id_mapping(self, config: dict) -> Dict[int, str]:
        """从COCO标注文件中加载image_id到文件名的映射。
        
        Args:
            config: 配置字典
            
        Returns:
            image_id到文件名的映射字典
        """
        annotations_path = config.get("data_config", {}).get("coco_annotations_path", "")
        
        if not annotations_path or not os.path.exists(annotations_path):
            logger.warning(
                f"COCO annotations file not found at {annotations_path}. "
                f"Candidate image object detection will be disabled."
            )
            return {}
        
        try:
            with open(annotations_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            image_id_to_filename = {
                item["id"]: item["file_name"] 
                for item in data.get("images", [])
            }
            
            logger.info(f"Loaded {len(image_id_to_filename)} image_id mappings")
            return image_id_to_filename
        except Exception as e:
            logger.error(f"Failed to load image_id mapping: {e}")
            return {}
    
    def _load_image_by_id(self, image_id: int) -> Optional[Image.Image]:
        """根据image_id加载图像。
        
        Args:
            image_id: 图像ID
            
        Returns:
            PIL.Image对象，如果加载失败则返回None
        """
        filename = self.image_id_to_filename.get(image_id)
        if not filename:
            logger.debug(f"Image ID {image_id} not found in mapping")
            return None
        
        image_path = self.images_dir / filename
        if not image_path.exists():
            logger.debug(f"Image file not found: {image_path}")
            return None
        
        try:
            return load_image(str(image_path))
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None
    
    def _extract_object_classes(self, detections: List[Dict]) -> Set[str]:
        """从检测结果中提取物体类别集合。
        
        Args:
            detections: 检测结果列表，每个元素包含'class_label'字段
            
        Returns:
            物体类别集合（去除'N/A'和'__background__'）
        """
        classes = set()
        for det in detections:
            class_label = det.get("class_label", "").strip()
            if class_label and class_label not in ["N/A", "__background__"]:
                classes.add(class_label)
        return classes
    
    def _get_candidate_objects(self, candidate_image_ids: List[int]) -> Dict[int, Set[str]]:
        """获取候选图像的物体类别。
        
        优先使用预构建的物体标签映射，如果不存在则使用实时检测。
        
        Args:
            candidate_image_ids: 候选图像ID列表
            
        Returns:
            候选图像ID到物体类别集合的映射
        """
        candidate_objects_map = {}
        
        # 如果存在预构建的物体标签映射，直接使用
        if self.image_id_to_objects:
            for image_id in candidate_image_ids:
                object_list = self.image_id_to_objects.get(image_id, [])
                candidate_objects_map[image_id] = set(object_list)
            return candidate_objects_map
        
        # 否则使用实时检测（回退方案）
        logger.debug("Using real-time detection for candidate images (slower)")
        for image_id in candidate_image_ids:
            # 加载图像
            image = self._load_image_by_id(image_id)
            if image is None:
                candidate_objects_map[image_id] = set()
                continue
            
            # 检测物体
            try:
                detections = self.detector.detect_objects(image)
                object_classes = self._extract_object_classes(detections)
                candidate_objects_map[image_id] = object_classes
            except Exception as e:
                logger.warning(f"Failed to detect objects for image {image_id}: {e}")
                candidate_objects_map[image_id] = set()
        
        return candidate_objects_map
    
    def retrieve(self, query_image: Union[str, Image.Image], top_k: int = 5) -> List[Dict]:
        """执行物体感知的混合检索。
        
        流程：
        1. 检测查询图像的物体类别
        2. 使用基础检索器进行初步召回（扩大召回数）
        3. 检测候选图像的物体类别
        4. 根据物体匹配度进行重排序
        5. 返回重排序后的Top-K结果
        
        Args:
            query_image: 查询图像（路径或PIL.Image）
            top_k: 最终返回的Top-K结果数量
            
        Returns:
            检索结果列表，每个元素包含：
            - "image_id": 图像ID
            - "score": 原始CLIP分数
            - "object_match_score": 物体匹配度分数
            - "hybrid_score": 混合分数
            - "captions": 图像描述列表
            - "_query_objects": 查询图像的物体类别列表
            - "_candidate_objects": 候选图像的物体类别列表
        """
        # 加载查询图像
        if isinstance(query_image, str):
            img = load_image(query_image)
        elif isinstance(query_image, Image.Image):
            img = query_image
        else:
            raise ValueError("query_image must be a file path or PIL.Image.Image")
        
        # 第一阶段：检测查询图像的物体
        try:
            query_detections = self.detector.detect_objects(img)
            query_objects = self._extract_object_classes(query_detections)
            logger.debug(f"Query image objects: {query_objects}")
        except Exception as e:
            logger.warning(f"Failed to detect objects in query image: {e}")
            query_objects = set()
        
        # 如果查询图像没有检测到物体，回退到基础检索器
        if not query_objects:
            logger.info("No objects detected in query image, falling back to base retriever")
            return self.base_retriever.get_retrieved_captions(query_image, top_k=top_k)
        
        # 第二阶段：使用基础检索器进行初步召回（扩大召回数）
        candidates = self.base_retriever.get_retrieved_captions(
            query_image, 
            top_k=self.initial_recall_k
        )
        
        if not candidates:
            logger.warning("No candidates retrieved from base retriever")
            return []
        
        # 提取候选图像ID列表
        candidate_image_ids = [item.get("image_id") for item in candidates if item.get("image_id") is not None]
        
        if not candidate_image_ids:
            logger.warning("No valid candidate image IDs found")
            return candidates[:top_k]  # 返回原始结果
        
        # 第三阶段：获取候选图像的物体类别（优先使用预构建映射，否则实时检测）
        logger.debug(f"Getting object tags for {len(candidate_image_ids)} candidate images...")
        candidate_objects_map = self._get_candidate_objects(candidate_image_ids)
        
        # 第四阶段：根据物体匹配度进行重排序
        reordered = reorder_by_objects(
            candidates,
            query_objects,
            candidate_objects_map,
            self.object_weight
        )
        
        # 返回Top-K结果
        return reordered[:top_k]
    
    # 向后兼容：实现与ImageRetriever相同的主要接口
    def get_retrieved_captions(self, query_image: Union[str, Image.Image], top_k: int = None) -> List[Dict]:
        """获取检索到的图像描述（向后兼容接口）。
        
        Args:
            query_image: 查询图像（路径或PIL.Image）
            top_k: 返回的Top-K结果数量（如果为None，使用配置中的默认值）
            
        Returns:
            检索结果列表，格式与ImageRetriever.get_retrieved_captions相同
        """
        if top_k is None:
            top_k = self.config.get("retrieval_config", {}).get("top_k", 5)
        
        return self.retrieve(query_image, top_k=top_k)
    
    def retrieve_similar_images(self, query_image: Union[str, Image.Image], top_k: int = 3) -> List[tuple]:
        """检索相似图像（向后兼容接口）。
        
        Args:
            query_image: 查询图像（路径或PIL.Image）
            top_k: 返回的Top-K结果数量
            
        Returns:
            (image_id, score)元组列表，score为混合分数
        """
        results = self.retrieve(query_image, top_k=top_k)
        return [
            (item.get("image_id"), item.get("hybrid_score", item.get("score", 0.0)))
            for item in results
            if item.get("image_id") is not None
        ]
    
    def extract_features(self, image: Union[str, Image.Image]) -> 'np.ndarray':
        """提取图像特征（向后兼容接口）。
        
        委托给基础检索器。
        """
        return self.base_retriever.extract_features(image)
    
    def enable_patch_retrieval(self):
        """启用分块检索模式（向后兼容接口）。
        
        委托给基础检索器。
        """
        return self.base_retriever.enable_patch_retrieval()
    
    def retrieve_with_patches(self, query_image: Union[str, Image.Image]) -> Dict:
        """执行包含分块的完整检索流程（向后兼容接口）。
        
        注意：此方法使用基础检索器的全局检索部分，但会利用混合检索器增强全局检索结果。
        
        Args:
            query_image: 查询图像（路径或PIL.Image）
            
        Returns:
            结构化检索结果，包含全局描述和局部区域信息
        """
        # 使用混合检索器获取增强的全局描述
        retrieval_config = self.config.get("retrieval_config", {})
        global_top_k = retrieval_config.get("top_k", 3)
        global_descriptions = self.retrieve(query_image, top_k=global_top_k)
        
        # 加载查询图像
        if isinstance(query_image, str):
            img = load_image(query_image)
        elif isinstance(query_image, Image.Image):
            img = query_image
        else:
            raise ValueError("query_image must be a file path or PIL.Image.Image")
        
        # 执行目标检测和局部检索（复用基础检索器的逻辑）
        local_regions = []
        try:
            # 确保基础检索器的分块检索已启用
            if not hasattr(self.base_retriever, 'use_patch_retrieval') or not self.base_retriever.use_patch_retrieval:
                self.base_retriever.enable_patch_retrieval()
            
            # 检测显著物体
            detections = self.detector.detect_objects(img)
            filtered_detections = self.detector.filter_detections(detections)
            
            if filtered_detections:
                # 裁剪区域
                image_patches = self.detector.crop_regions(img, filtered_detections)
                
                # 获取图像尺寸
                image_size = img.size  # (width, height)
                
                # 为每个检测结果计算位置信息
                image_patches_with_position = []
                for patch_image, detection_info in image_patches:
                    # 计算相对位置
                    bbox = detection_info['bbox']
                    position = self.detector.calculate_relative_position(bbox, image_size)
                    # 将位置信息添加到detection_info中
                    detection_info_with_position = detection_info.copy()
                    detection_info_with_position['position'] = position
                    image_patches_with_position.append((patch_image, detection_info_with_position))
                
                # 提取全局检索到的image_id集合
                global_image_ids = {item.get('image_id') for item in global_descriptions if item.get('image_id') is not None}
                
                # 对每个局部区域进行处理
                if self.base_retriever.local_retriever:
                    local_results = self.base_retriever.local_retriever.retrieve_local_descriptions(
                        image_patches_with_position, 
                        exclude_image_ids=global_image_ids
                    )
                    local_regions = self.base_retriever.local_retriever.merge_local_descriptions(local_results)
                
                logger.info(f"Processed {len(local_regions)} local regions with positions")
            else:
                logger.warning("No objects detected, using global descriptions only")
        except Exception as e:
            logger.error(f"Patch retrieval failed: {e}, falling back to global retrieval only")
        
        return {
            "global_descriptions": global_descriptions,
            "local_regions": local_regions
        }

