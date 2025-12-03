"""基于密集描述子的混合检索模块。

该模块包装并增强现有的CLIP检索器，通过两阶段混合检索提升检索结果的语义相关性：
1. 第一阶段（召回）：使用CLIP检索器初步召回视觉相似的候选图像
2. 第二阶段（重排序）：利用密集描述短语的语义相似度对候选图像进行重排序
"""
import os
import json
import logging
from typing import List, Dict, Set, Union, Optional

import torch
import numpy as np
from pathlib import Path
from PIL import Image

from core.retriever import ImageRetriever
from core.patch_detector import PatchDetector
from utils.image_utils import load_image

# 句子嵌入模型
try:
    from sentence_transformers import SentenceTransformer
    _has_sentence_transformers = True
except ImportError:
    _has_sentence_transformers = False
    logging.warning("sentence-transformers not available. Please install it for dense descriptor retrieval.")

logger = logging.getLogger(__name__)


def compute_descriptor_similarity(
    query_descriptors: List[str],
    candidate_descriptors: List[str],
    embedding_model
) -> float:
    """计算两组密集描述列表之间的相似度。
    
    采用"平均最大余弦相似度"策略：
    1. 对查询图像的每个描述，找到候选图像中与之最相似的描述
    2. 取所有最大相似度的平均值
    
    Args:
        query_descriptors: 查询图像的密集描述列表
        candidate_descriptors: 候选图像的密集描述列表
        embedding_model: 句子嵌入模型
        
    Returns:
        相似度分数，范围[0.0, 1.0]
    """
    if not query_descriptors or not candidate_descriptors:
        return 0.0
    
    if not _has_sentence_transformers:
        logger.warning("sentence-transformers not available, returning 0.0 similarity")
        return 0.0
    
    # 编码所有描述
    try:
        all_descriptors = query_descriptors + candidate_descriptors
        embeddings = embedding_model.encode(all_descriptors, convert_to_numpy=True, normalize_embeddings=True)
        
        query_embeddings = embeddings[:len(query_descriptors)]
        candidate_embeddings = embeddings[len(query_descriptors):]
        
        # 计算每个查询描述与所有候选描述的最大相似度
        max_similarities = []
        for query_emb in query_embeddings:
            # 计算余弦相似度（归一化后的点积）
            similarities = np.dot(candidate_embeddings, query_emb)
            max_sim = float(np.max(similarities))
            max_similarities.append(max_sim)
        
        # 返回平均最大相似度
        if max_similarities:
            return float(np.mean(max_similarities))
        else:
            return 0.0
            
    except Exception as e:
        logger.warning(f"Failed to compute descriptor similarity: {e}")
        return 0.0


def compute_hybrid_score(
    clip_score: float,
    descriptor_match_score: float,
    descriptor_weight: float = 0.7
) -> float:
    """计算混合分数，融合CLIP相似度和密集描述匹配度。
    
    Args:
        clip_score: CLIP相似度分数（FAISS内积，对于归一化向量范围在[-1, 1]，实际通常为[0, 1]）
        descriptor_match_score: 密集描述匹配度分数（范围[0.0, 1.0]）
        descriptor_weight: 密集描述匹配度的权重（范围[0.0, 1.0]），clip权重为(1 - descriptor_weight)
        
    Returns:
        混合分数
    """
    # 将CLIP分数归一化到[0, 1]范围
    if clip_score < 0:
        normalized_clip_score = (clip_score + 1.0) / 2.0
    else:
        normalized_clip_score = min(1.0, max(0.0, clip_score))
    
    # 确保归一化后的分数在[0, 1]范围内
    normalized_clip_score = max(0.0, min(1.0, normalized_clip_score))
    
    # 线性融合
    hybrid_score = (1.0 - descriptor_weight) * normalized_clip_score + descriptor_weight * descriptor_match_score
    
    return hybrid_score


def reorder_by_descriptors(
    candidates: List[Dict],
    query_descriptors: List[str],
    candidate_descriptors_map: Dict[int, List[str]],
    embedding_model,
    descriptor_weight: float = 0.7
) -> List[Dict]:
    """根据密集描述相似度对候选结果进行重排序。
    
    Args:
        candidates: 候选结果列表，每个元素包含至少 {"image_id": int, "score": float}
        query_descriptors: 查询图像的密集描述列表
        candidate_descriptors_map: 候选图像ID到密集描述列表的映射
        embedding_model: 句子嵌入模型
        descriptor_weight: 密集描述匹配度的权重
        
    Returns:
        重排序后的候选结果列表，每个元素增加了 "descriptor_match_score" 和 "hybrid_score" 字段
    """
    if not query_descriptors:
        logger.debug("Query image has no dense descriptors, returning original order")
        return candidates
    
    enhanced_candidates = []
    
    for candidate in candidates:
        image_id = candidate.get("image_id")
        clip_score = candidate.get("score", 0.0)
        
        # 获取候选图像的密集描述列表
        candidate_descriptors = candidate_descriptors_map.get(image_id, [])
        
        # 计算密集描述相似度
        descriptor_match_score = compute_descriptor_similarity(
            query_descriptors,
            candidate_descriptors,
            embedding_model
        )
        
        # 计算混合分数
        hybrid_score = compute_hybrid_score(clip_score, descriptor_match_score, descriptor_weight)
        
        # 创建增强的候选结果
        enhanced = candidate.copy()
        enhanced["descriptor_match_score"] = descriptor_match_score
        enhanced["hybrid_score"] = hybrid_score
        enhanced["_query_descriptors"] = query_descriptors
        enhanced["_candidate_descriptors"] = candidate_descriptors
        
        enhanced_candidates.append(enhanced)
    
    # 按混合分数降序排序
    enhanced_candidates.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    
    return enhanced_candidates


class DenseDescriptorHybridRetriever:
    """基于密集描述子的混合检索器。
    
    该检索器包装并增强现有的CLIP检索器，通过引入密集描述语义匹配能力，
    使检索结果不仅在场景层面相似，更在语义细节层面匹配。
    """
    
    def __init__(self, base_retriever: ImageRetriever, detector: PatchDetector, config: dict):
        """初始化密集描述混合检索器。
        
        Args:
            base_retriever: 基础的CLIP检索器实例
            detector: 检测器实例（用于查询图像的密集描述生成）
            config: 配置字典，包含混合检索相关参数
        """
        self.base_retriever = base_retriever
        self.detector = detector
        self.config = config
        
        # 从配置中读取混合检索参数
        dense_config = config.get("dense_descriptor", {})
        hybrid_config = config.get("hybrid_retrieval", {})
        
        self.descriptor_weight = float(hybrid_config.get("dense_object_weight", dense_config.get("descriptor_weight", 0.7)))
        self.initial_recall_k = int(hybrid_config.get("initial_recall_k", 100))
        
        # 加载句子嵌入模型
        embedding_model_path = dense_config.get("embedding_model_path", "../models/all-MiniLM-L6-v2/")
        self.embedding_model = None
        
        if _has_sentence_transformers:
            try:
                self.embedding_model = SentenceTransformer(embedding_model_path)
                logger.info(f"Loaded sentence embedding model from {embedding_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load sentence embedding model from {embedding_model_path}: {e}")
                logger.warning("Dense descriptor retrieval will be disabled")
        else:
            logger.warning("sentence-transformers not available, dense descriptor retrieval will be disabled")
        
        # 加载预构建的密集描述知识库
        self.image_id_to_descriptors = self._load_dense_descriptors_mapping(config)
        
        if not self.image_id_to_descriptors:
            logger.warning(
                "Pre-built dense descriptors mapping not found. Will use real-time generation for candidate images. "
                "This is slower. Consider running build_dense_knowledge_base.py to generate image_id_to_dense_captions.pkl"
            )
            # 加载image_id到文件名的映射（用于从image_id加载图像进行实时生成）
            self.image_id_to_filename = self._load_image_id_mapping(config)
            self.images_dir = Path(config.get("data_config", {}).get("coco_images_dir", ""))
        else:
            logger.info(f"Loaded pre-built dense descriptors for {len(self.image_id_to_descriptors)} images")
            self.image_id_to_filename = {}
            self.images_dir = None
        
        # 加载密集描述生成模型（用于查询图像的实时生成）
        self.dense_caption_model = None
        self.dense_caption_cfg = None
        self.dense_model_path = dense_config.get("model_path", "../models/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det/")
        
        # 查询图像的实时生成将在需要时延迟加载，避免循环导入
        self._dense_model_device = None
        if os.path.exists(self.dense_model_path):
            self._dense_model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Dense caption model path available: {self.dense_model_path}, will load on demand")
        else:
            logger.warning(f"Dense caption model path does not exist: {self.dense_model_path}")
        
        logger.info(
            f"DenseDescriptorHybridRetriever initialized: "
            f"descriptor_weight={self.descriptor_weight}, initial_recall_k={self.initial_recall_k}"
        )
    
    def _load_dense_descriptors_mapping(self, config: dict) -> Dict[int, List[str]]:
        """从预构建的pickle文件中加载image_id到密集描述列表的映射。
        
        Args:
            config: 配置字典
            
        Returns:
            image_id到密集描述列表的映射字典，如果文件不存在则返回空字典
        """
        import pickle
        
        dense_config = config.get("dense_descriptor", {})
        descriptors_path = dense_config.get("knowledge_base_path", "./output/image_id_to_dense_captions.pkl")
        
        if not descriptors_path or not os.path.exists(descriptors_path):
            return {}
        
        try:
            with open(descriptors_path, "rb") as f:
                image_id_to_descriptors = pickle.load(f)
            
            # 确保格式为列表
            result = {}
            for img_id, descriptors in image_id_to_descriptors.items():
                if isinstance(descriptors, (list, tuple)):
                    result[img_id] = list(descriptors)
                elif isinstance(descriptors, str):
                    result[img_id] = [descriptors]
                else:
                    result[img_id] = []
            
            logger.info(f"Loaded dense descriptors mapping from {descriptors_path}")
            return result
        except Exception as e:
            logger.warning(f"Failed to load dense descriptors mapping from {descriptors_path}: {e}")
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
                f"Candidate image dense descriptor generation will be disabled."
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
    
    def _load_dense_caption_model(self):
        """延迟加载密集描述生成模型（避免循环导入）。"""
        if self.dense_caption_model is not None:
            return
        
        if self._dense_model_device is None:
            return
        
        try:
            # 延迟导入以避免循环依赖
            import sys
            import importlib.util
            build_script_path = Path(__file__).parent.parent / "scripts" / "build_dense_knowledge_base.py"
            
            if build_script_path.exists():
                spec = importlib.util.spec_from_file_location("build_dense_knowledge_base", build_script_path)
                build_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(build_module)
                
                self.dense_caption_model, self.dense_caption_cfg = build_module.init_dense_caption_model(
                    self.dense_model_path,
                    device=str(self._dense_model_device)
                )
                logger.info(f"Loaded dense caption model from {self.dense_model_path}")
            else:
                logger.warning(f"Could not find build_dense_knowledge_base.py script")
        except Exception as e:
            logger.warning(f"Failed to load dense caption model from {self.dense_model_path}: {e}")
            logger.warning("Query image dense descriptor generation will be disabled")
    
    def _generate_dense_descriptors(self, image: Image.Image) -> List[str]:
        """为图像生成密集描述短语。
        
        Args:
            image: PIL.Image对象
            
        Returns:
            密集描述短语列表
        """
        # 延迟加载模型
        self._load_dense_caption_model()
        
        if self.dense_caption_model is None:
            logger.debug("Dense caption model not available, returning empty descriptors")
            return []
        
        try:
            # 延迟导入以避免循环依赖
            import sys
            import importlib.util
            import tempfile
            build_script_path = Path(__file__).parent.parent / "scripts" / "build_dense_knowledge_base.py"
            
            if build_script_path.exists():
                spec = importlib.util.spec_from_file_location("build_dense_knowledge_base", build_script_path)
                build_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(build_module)
                
                # 临时保存图像到文件（因为模型可能接受文件路径）
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    image.save(tmp_file.name)
                    tmp_path = tmp_file.name
                
                try:
                    descriptors = build_module.extract_dense_captions_mmdet(
                        self.dense_caption_model,
                        tmp_path,
                        min_phrases=5,
                        max_phrases=10,
                        confidence_threshold=0.3
                    )
                finally:
                    # 清理临时文件
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                
                return descriptors
            else:
                logger.warning("Could not find build_dense_knowledge_base.py script")
                return []
        except Exception as e:
            logger.warning(f"Failed to generate dense descriptors: {e}")
            return []
    
    def _get_candidate_descriptors(self, candidate_image_ids: List[int]) -> Dict[int, List[str]]:
        """获取候选图像的密集描述。
        
        优先使用预构建的密集描述映射，如果不存在则使用实时生成。
        
        Args:
            candidate_image_ids: 候选图像ID列表
            
        Returns:
            候选图像ID到密集描述列表的映射
        """
        candidate_descriptors_map = {}
        
        # 如果存在预构建的密集描述映射，直接使用
        if self.image_id_to_descriptors:
            for image_id in candidate_image_ids:
                descriptors = self.image_id_to_descriptors.get(image_id, [])
                candidate_descriptors_map[image_id] = descriptors
            return candidate_descriptors_map
        
        # 否则使用实时生成（回退方案，较慢）
        logger.debug("Using real-time dense descriptor generation for candidate images (slower)")
        for image_id in candidate_image_ids:
            # 加载图像
            image = self._load_image_by_id(image_id)
            if image is None:
                candidate_descriptors_map[image_id] = []
                continue
            
            # 生成密集描述
            try:
                descriptors = self._generate_dense_descriptors(image)
                candidate_descriptors_map[image_id] = descriptors
            except Exception as e:
                logger.warning(f"Failed to generate dense descriptors for image {image_id}: {e}")
                candidate_descriptors_map[image_id] = []
        
        return candidate_descriptors_map
    
    def retrieve(self, query_image: Union[str, Image.Image], top_k: int = 5) -> List[Dict]:
        """执行基于密集描述子的混合检索。
        
        流程：
        1. 为查询图像生成密集描述短语
        2. 使用基础检索器进行初步召回（扩大召回数）
        3. 获取候选图像的密集描述
        4. 根据密集描述语义相似度进行重排序
        5. 返回重排序后的Top-K结果
        
        Args:
            query_image: 查询图像（路径或PIL.Image）
            top_k: 最终返回的Top-K结果数量
            
        Returns:
            检索结果列表，每个元素包含：
            - "image_id": 图像ID
            - "score": 原始CLIP分数
            - "descriptor_match_score": 密集描述匹配度分数
            - "hybrid_score": 混合分数
            - "captions": 图像描述列表
            - "_query_descriptors": 查询图像的密集描述列表
            - "_candidate_descriptors": 候选图像的密集描述列表
        """
        # 加载查询图像
        if isinstance(query_image, str):
            img = load_image(query_image)
        elif isinstance(query_image, Image.Image):
            img = query_image
        else:
            raise ValueError("query_image must be a file path or PIL.Image.Image")
        
        # 第一阶段：为查询图像生成密集描述
        query_descriptors = []
        if self.dense_caption_model is not None:
            try:
                query_descriptors = self._generate_dense_descriptors(img)
                logger.debug(f"Query image dense descriptors: {len(query_descriptors)} phrases")
            except Exception as e:
                logger.warning(f"Failed to generate dense descriptors for query image: {e}")
        
        # 如果查询图像没有密集描述，回退到基础检索器
        if not query_descriptors:
            logger.info("No dense descriptors generated for query image, falling back to base retriever")
            return self.base_retriever.get_retrieved_captions(query_image, top_k=top_k)
        
        # 如果没有嵌入模型，也无法使用密集描述检索
        if self.embedding_model is None:
            logger.info("Embedding model not available, falling back to base retriever")
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
        
        # 第三阶段：获取候选图像的密集描述（优先使用预构建映射，否则实时生成）
        logger.debug(f"Getting dense descriptors for {len(candidate_image_ids)} candidate images...")
        candidate_descriptors_map = self._get_candidate_descriptors(candidate_image_ids)
        
        # 第四阶段：根据密集描述相似度进行重排序
        reordered = reorder_by_descriptors(
            candidates,
            query_descriptors,
            candidate_descriptors_map,
            self.embedding_model,
            self.descriptor_weight
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

