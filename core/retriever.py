import os
import logging
import pickle
from typing import List, Tuple, Union, Dict

import torch
import faiss
import yaml
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from utils.image_utils import load_image


logger = logging.getLogger(__name__)


class ImageRetriever:
    """Image retriever using CLIP + FAISS.

    Notes:
    - Expects a FAISS index built from normalized CLIP image embeddings.
    - Expects the mapping pickle to be a dict mapping image_id -> [captions],
      and relies on the insertion order of that dict to map FAISS row ids -> image ids.
    """

    def __init__(self, config: Union[dict, str]):
        """Initialize retriever.

        Args:
            config: dict or path to YAML config (same structure as configs/config.yaml)
        """
        if isinstance(config, str):
            with open(config, "r") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = config

        # logging level
        log_level = cfg.get("log_config", {}).get("log_level", "ERROR")
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.ERROR))

        clip_model_path = cfg.get("model_config", {}).get("clip_model_path")
        kb_path = cfg.get("knowledge_base_config", {}).get("knowledge_base_path")
        mapping_path = cfg.get("knowledge_base_config", {}).get("image_id_to_captions_path")

        if not clip_model_path or not kb_path or not mapping_path:
            logger.error("Missing required paths in config for retriever.")
            raise ValueError("Missing required config values for ImageRetriever")

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load CLIP
        try:
            self.processor = CLIPProcessor.from_pretrained(clip_model_path)
            self.model = CLIPModel.from_pretrained(clip_model_path).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load CLIP model from {clip_model_path}: {e}")
            raise

        # load faiss index
        if not os.path.exists(kb_path):
            logger.error(f"FAISS index file not found: {kb_path}")
            raise FileNotFoundError(kb_path)
        try:
            self.index = faiss.read_index(kb_path)
        except Exception as e:
            logger.error(f"Failed to read FAISS index from {kb_path}: {e}")
            raise

        # load mapping (image_id -> captions list)
        if not os.path.exists(mapping_path):
            logger.error(f"Mapping file not found: {mapping_path}")
            raise FileNotFoundError(mapping_path)
        try:
            with open(mapping_path, "rb") as f:
                self.image_id_to_captions: Dict[int, List[str]] = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load mapping from {mapping_path}: {e}")
            raise

        # preserve ordering to map faiss indices -> image ids
        self.index_id_to_image_id = list(self.image_id_to_captions.keys())
        
        # 保存配置用于分块检索
        self.config = cfg
        self.use_patch_retrieval = False
        self.patch_detector = None
        self.local_retriever = None
        
        # 初始化描述优化器（如果启用）
        self.description_optimizer = None
        opt_config = cfg.get("description_optimization", {})
        if opt_config.get("enabled", False):
            try:
                from core.description_optimizer import DescriptionOptimizer
                self.description_optimizer = DescriptionOptimizer(cfg)
                logger.info("Description optimizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize description optimizer: {e}, continuing without optimization")

    def extract_features(self, image: Union[str, Image.Image]) -> 'np.ndarray':
        """Extract CLIP image features for a single image.

        Args:
            image: PIL.Image or path to image file

        Returns:
            numpy array (1, dim) normalized
        """
        import numpy as np

        if isinstance(image, str):
            img = load_image(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError("image must be a file path or PIL.Image.Image")

        inputs = self.processor(images=[img], return_tensors="pt")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        with torch.no_grad():
            embeds = self.model.get_image_features(**inputs)

        embeds = embeds.cpu().numpy()
        # normalize
        norms = (embeds ** 2).sum(axis=1, keepdims=True) ** 0.5
        norms[norms == 0] = 1.0
        embeds = embeds / norms
        return embeds

    def retrieve_similar_images(self, query_image: Union[str, Image.Image], top_k: int = 3) -> List[Tuple[int, float]]:
        """Retrieve most similar images to query_image.

        Returns a list of (image_id, score) sorted by score desc.
        Scores are FAISS inner-product values (with normalized vectors ~ cosine similarity).
        """
        import numpy as np

        qv = self.extract_features(query_image)
        if qv.shape[0] == 0:
            return []

        # faiss expects float32
        qv = qv.astype("float32")
        D, I = self.index.search(qv, top_k)

        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self.index_id_to_image_id):
                continue
            image_id = self.index_id_to_image_id[idx]
            results.append((image_id, float(score)))
        return results

    def get_retrieved_captions(self, query_image: Union[str, Image.Image], top_k: int = None) -> List[Dict]:
        """Get captions for retrieved images.

        Returns a list of dicts: {"image_id": id, "score": float, "captions": [..]}
        如果启用了描述优化，返回优化后的代表性描述。
        """
        if top_k is None:
            # fall back to config default if available
            top_k = 3

        hits = self.retrieve_similar_images(query_image, top_k=top_k)
        out = []
        for image_id, score in hits:
            captions = self.image_id_to_captions.get(image_id, [])
            out.append({"image_id": image_id, "score": score, "captions": captions})
        
        # 如果启用了描述优化，对描述进行聚类优化
        if self.description_optimizer and self.description_optimizer.enabled:
            try:
                # 收集所有描述和对应的图像相似度
                all_descriptions = []
                all_similarities = []
                
                for item in out:
                    captions = item.get("captions", [])
                    score = item.get("score", 0.0)
                    # 将每个图像的多个描述都加入列表
                    for caption in captions:
                        if isinstance(caption, str) and caption.strip():
                            all_descriptions.append(caption.strip())
                            all_similarities.append(score)
                
                if len(all_descriptions) > 1:
                    # 执行描述优化
                    optimized_descriptions = self.description_optimizer.optimize_descriptions(
                        all_descriptions, all_similarities
                    )
                    
                    if optimized_descriptions:
                        # 将优化后的描述转换回原始格式
                        # 创建描述到优化元数据的映射
                        desc_to_metadata = {
                            opt_desc.get("description", ""): opt_desc
                            for opt_desc in optimized_descriptions
                        }
                        
                        # 构建优化后的输出，保持原有结构但使用优化后的描述
                        optimized_out = []
                        used_descriptions = set()
                        
                        # 按图像相似度排序原始结果
                        sorted_out = sorted(out, key=lambda x: x.get("score", 0.0), reverse=True)
                        
                        for item in sorted_out:
                            image_id = item.get("image_id")
                            score = item.get("score", 0.0)
                            
                            # 尝试找到匹配的优化描述
                            matching_desc = None
                            for opt_desc in optimized_descriptions:
                                desc_text = opt_desc.get("description", "")
                                if desc_text not in used_descriptions:
                                    # 使用第一个未使用的优化描述
                                    matching_desc = desc_text
                                    used_descriptions.add(desc_text)
                                    break
                            
                            # 如果没有找到匹配的优化描述，使用原始描述的第一个
                            if not matching_desc:
                                original_captions = item.get("captions", [])
                                if original_captions:
                                    matching_desc = original_captions[0]
                            
                            if matching_desc:
                                opt_metadata = desc_to_metadata.get(matching_desc, {})
                                optimized_out.append({
                                    "image_id": image_id,
                                    "score": score,
                                    "captions": [matching_desc],
                                    "_optimized": True,
                                    "_optimization_metadata": {
                                        "cluster_score": opt_metadata.get("cluster_score", 0.0),
                                        "image_similarity": opt_metadata.get("image_similarity", 0.0),
                                        "brevity_score": opt_metadata.get("brevity_score", 0.0),
                                        "combined_score": opt_metadata.get("combined_score", 0.0),
                                        "cluster_size": opt_metadata.get("cluster_size", 1)
                                    }
                                })
                        
                        # 如果优化后没有结果，回退到原始结果
                        if optimized_out:
                            logger.info(f"Description optimization: {len(all_descriptions)} descriptions -> "
                                      f"{len(optimized_descriptions)} representative descriptions")
                            return optimized_out
                        else:
                            logger.warning("Description optimization produced no results, using original")
                    else:
                        logger.warning("Description optimization returned empty result, using original")
                else:
                    logger.debug("Not enough descriptions for optimization (need > 1)")
            except Exception as e:
                logger.warning(f"Description optimization failed: {e}, using original descriptions")
        
        return out

    def enable_patch_retrieval(self):
        """启用分块检索模式，初始化PatchDetector和LocalRetriever。"""
        if self.use_patch_retrieval:
            return  # 已经启用
        
        try:
            from core.patch_detector import PatchDetector
            from core.local_retriever import LocalRetriever
            
            self.patch_detector = PatchDetector(self.config)
            self.local_retriever = LocalRetriever(self, self.config)
            self.use_patch_retrieval = True
            logger.info("Patch retrieval enabled")
        except Exception as e:
            logger.error(f"Failed to enable patch retrieval: {e}")
            self.use_patch_retrieval = False
            raise

    def retrieve_with_patches(self, query_image: Union[str, Image.Image]) -> Dict:
        """执行包含分块的完整检索流程。
        
        Args:
            query_image: 查询图像（路径或PIL.Image）
            
        Returns:
            结构化检索结果：
            {
                "global_descriptions": [{"image_id": ..., "score": ..., "captions": [...]}],
                "local_regions": [
                    {
                        "bbox": [x1, y1, x2, y2],
                        "class_label": "类别名称",
                        "confidence": 置信度,
                        "descriptions": [描述列表]
                    }
                ]
            }
        """
        if isinstance(query_image, str):
            img = load_image(query_image)
        elif isinstance(query_image, Image.Image):
            img = query_image
        else:
            raise ValueError("query_image must be a file path or PIL.Image.Image")
        
        # 确保分块检索已启用
        if not self.use_patch_retrieval:
            self.enable_patch_retrieval()
        
        # 1. 执行全局检索
        retrieval_config = self.config.get("retrieval_config", {})
        global_top_k = retrieval_config.get("top_k", 3)
        global_descriptions = self.get_retrieved_captions(img, top_k=global_top_k)
        
        # 2. 执行目标检测和局部检索
        local_regions = []
        try:
            # 检测显著物体
            detections = self.patch_detector.detect_objects(img)
            filtered_detections = self.patch_detector.filter_detections(detections)
            
            if filtered_detections:
                # 裁剪区域
                image_patches = self.patch_detector.crop_regions(img, filtered_detections)
                
                # 提取全局检索到的image_id集合，用于排除重复
                global_image_ids = {item.get('image_id') for item in global_descriptions if item.get('image_id') is not None}
                
                # 对每个局部区域进行检索，排除全局检索到的image_id
                local_results = self.local_retriever.retrieve_local_descriptions(
                    image_patches, 
                    exclude_image_ids=global_image_ids
                )
                local_regions = self.local_retriever.merge_local_descriptions(local_results)
                
                logger.info(f"Retrieved {len(local_regions)} local regions with descriptions")
            else:
                logger.warning("No objects detected, using global descriptions only")
        except Exception as e:
            logger.error(f"Patch retrieval failed: {e}, falling back to global retrieval only")
            # 失败时回退到全局检索
        
        return {
            "global_descriptions": global_descriptions,
            "local_regions": local_regions
        }
