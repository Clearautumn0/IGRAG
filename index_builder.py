"""
FAISS向量索引构建模块
"""
import faiss
import numpy as np
import torch
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import os

from config import INDEX_CONFIG, OUTPUT_CONFIG
from feature_extractor import CLIPFeatureExtractor

logger = logging.getLogger(__name__)


class FAISSIndexBuilder:
    """FAISS向量索引构建器"""
    
    def __init__(self, feature_dim: int, index_type: str = "IVF", use_gpu: bool = False):
        """
        初始化索引构建器
        
        Args:
            feature_dim: 特征维度
            index_type: 索引类型 ("Flat", "IVF", "HNSW")
            use_gpu: 是否使用GPU
        """
        self.feature_dim = feature_dim
        self.index_type = index_type
        self.use_gpu = use_gpu
        
        # 初始化索引
        self.global_index = None
        self.local_index = None
        
        # 元数据存储
        self.global_metadata = []
        self.local_metadata = []
        
        logger.info(f"初始化FAISS索引构建器 - 特征维度: {feature_dim}, "
                   f"索引类型: {index_type}, GPU: {use_gpu}")
    
    def _create_index(self, feature_dim: int) -> faiss.Index:
        """
        创建FAISS索引
        
        Args:
            feature_dim: 特征维度
            
        Returns:
            FAISS索引对象
        """
        if self.index_type == "Flat":
            # 精确搜索索引
            index = faiss.IndexFlatIP(feature_dim)  # 内积相似度
            
        elif self.index_type == "IVF":
            # 倒排文件索引
            quantizer = faiss.IndexFlatIP(feature_dim)
            index = faiss.IndexIVFFlat(quantizer, feature_dim, INDEX_CONFIG["nlist"])
            
        elif self.index_type == "HNSW":
            # 分层导航小世界图索引
            index = faiss.IndexHNSWFlat(feature_dim, 32)
            
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")
        
        # GPU加速
        if self.use_gpu and faiss.get_num_gpus() > 0:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("使用GPU加速索引")
            except Exception as e:
                logger.warning(f"GPU加速失败，使用CPU: {e}")
        
        return index
    
    def build_global_index(self, global_features: List[np.ndarray], 
                          image_ids: List[str], 
                          captions: List[List[str]]) -> None:
        """
        构建全局特征索引
        
        Args:
            global_features: 全局特征列表
            image_ids: 图像ID列表
            captions: 对应的描述列表
        """
        logger.info("开始构建全局特征索引...")
        
        # 创建索引
        self.global_index = self._create_index(self.feature_dim)
        
        # 准备数据
        features_array = np.vstack(global_features).astype(np.float32)
        
        # 归一化特征（用于余弦相似度）
        faiss.normalize_L2(features_array)
        
        # 训练索引（IVF需要）
        if self.index_type == "IVF":
            logger.info("训练IVF索引...")
            self.global_index.train(features_array)
        
        # 添加特征到索引
        logger.info(f"添加 {len(features_array)} 个全局特征到索引...")
        self.global_index.add(features_array)
        
        # 设置搜索参数
        if self.index_type == "IVF":
            self.global_index.nprobe = INDEX_CONFIG["nprobe"]
        
        # 存储元数据
        self.global_metadata = []
        for i, (img_id, caption_list) in enumerate(zip(image_ids, captions)):
            self.global_metadata.append({
                'index_id': i,
                'image_id': img_id,
                'captions': caption_list
            })
        
        logger.info(f"全局索引构建完成，包含 {len(features_array)} 个特征")
    
    def build_local_index(self, patch_features_list: List[np.ndarray], 
                         image_ids: List[str], 
                         captions: List[List[str]]) -> None:
        """
        构建局部特征索引
        
        Args:
            patch_features_list: patch特征列表，每个元素是 [num_patches, feature_dim]
            image_ids: 图像ID列表
            captions: 对应的描述列表
        """
        logger.info("开始构建局部特征索引...")
        
        # 创建索引
        self.local_index = self._create_index(self.feature_dim)
        
        # 准备数据
        all_patch_features = []
        self.local_metadata = []
        
        for img_idx, (patch_features, img_id, caption_list) in enumerate(
            zip(patch_features_list, image_ids, captions)):
            
            # 归一化patch特征
            patch_features_norm = patch_features.astype(np.float32)
            faiss.normalize_L2(patch_features_norm)
            
            # 添加每个patch特征
            for patch_idx in range(patch_features_norm.shape[0]):
                all_patch_features.append(patch_features_norm[patch_idx])
                
                # 存储元数据
                self.local_metadata.append({
                    'index_id': len(all_patch_features) - 1,
                    'image_id': img_id,
                    'patch_idx': patch_idx,
                    'image_idx': img_idx,
                    'captions': caption_list
                })
        
        # 转换为numpy数组
        features_array = np.vstack(all_patch_features).astype(np.float32)
        
        # 训练索引（IVF需要）
        if self.index_type == "IVF":
            logger.info("训练IVF索引...")
            self.local_index.train(features_array)
        
        # 添加特征到索引
        logger.info(f"添加 {len(features_array)} 个patch特征到索引...")
        self.local_index.add(features_array)
        
        # 设置搜索参数
        if self.index_type == "IVF":
            self.local_index.nprobe = INDEX_CONFIG["nprobe"]
        
        logger.info(f"局部索引构建完成，包含 {len(features_array)} 个patch特征")
    
    def save_indexes(self, save_dir: Path) -> None:
        """
        保存索引和元数据
        
        Args:
            save_dir: 保存目录
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存全局索引
        if self.global_index is not None:
            global_index_path = save_dir / "global_index.faiss"
            faiss.write_index(self.global_index, str(global_index_path))
            
            global_metadata_path = save_dir / "global_metadata.pkl"
            with open(global_metadata_path, 'wb') as f:
                pickle.dump(self.global_metadata, f)
            
            logger.info(f"全局索引已保存到: {global_index_path}")
        
        # 保存局部索引
        if self.local_index is not None:
            local_index_path = save_dir / "local_index.faiss"
            faiss.write_index(self.local_index, str(local_index_path))
            
            local_metadata_path = save_dir / "local_metadata.pkl"
            with open(local_metadata_path, 'wb') as f:
                pickle.dump(self.local_metadata, f)
            
            logger.info(f"局部索引已保存到: {local_index_path}")
        
        # 保存配置信息
        config_path = save_dir / "index_config.json"
        config_info = {
            'feature_dim': self.feature_dim,
            'index_type': self.index_type,
            'use_gpu': self.use_gpu,
            'global_count': len(self.global_metadata) if self.global_metadata else 0,
            'local_count': len(self.local_metadata) if self.local_metadata else 0
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_info, f, indent=2)
        
        logger.info(f"索引配置已保存到: {config_path}")
    
    def load_indexes(self, load_dir: Path) -> None:
        """
        加载索引和元数据
        
        Args:
            load_dir: 加载目录
        """
        # 加载配置
        config_path = load_dir / "index_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_info = json.load(f)
            
            self.feature_dim = config_info['feature_dim']
            self.index_type = config_info['index_type']
            self.use_gpu = config_info['use_gpu']
            
            logger.info(f"加载索引配置: {config_info}")
        
        # 加载全局索引
        global_index_path = load_dir / "global_index.faiss"
        if global_index_path.exists():
            self.global_index = faiss.read_index(str(global_index_path))
            
            global_metadata_path = load_dir / "global_metadata.pkl"
            with open(global_metadata_path, 'rb') as f:
                self.global_metadata = pickle.load(f)
            
            logger.info(f"全局索引已加载，包含 {len(self.global_metadata)} 个特征")
        
        # 加载局部索引
        local_index_path = load_dir / "local_index.faiss"
        if local_index_path.exists():
            self.local_index = faiss.read_index(str(local_index_path))
            
            local_metadata_path = load_dir / "local_metadata.pkl"
            with open(local_metadata_path, 'rb') as f:
                self.local_metadata = pickle.load(f)
            
            logger.info(f"局部索引已加载，包含 {len(self.local_metadata)} 个特征")
    
    def search_global(self, query_feature: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        全局特征搜索
        
        Args:
            query_feature: 查询特征 [feature_dim]
            top_k: 返回top-k结果
            
        Returns:
            搜索结果列表
        """
        if self.global_index is None:
            raise ValueError("全局索引未构建")
        
        # 归一化查询特征
        query_feature = query_feature.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_feature)
        
        # 搜索
        scores, indices = self.global_index.search(query_feature, top_k)
        
        # 构建结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # 有效索引
                metadata = self.global_metadata[idx]
                results.append({
                    'image_id': metadata['image_id'],
                    'captions': metadata['captions'],
                    'score': float(score),
                    'index_id': idx
                })
        
        return results
    
    def search_local(self, query_features: np.ndarray, top_m: int = 10) -> List[Dict]:
        """
        局部特征搜索
        
        Args:
            query_features: 查询特征 [num_patches, feature_dim]
            top_m: 返回top-m结果
            
        Returns:
            搜索结果列表
        """
        if self.local_index is None:
            raise ValueError("局部索引未构建")
        
        # 归一化查询特征
        query_features = query_features.astype(np.float32)
        faiss.normalize_L2(query_features)
        
        # 搜索
        scores, indices = self.local_index.search(query_features, top_m)
        
        # 构建结果
        results = []
        for patch_idx, (patch_scores, patch_indices) in enumerate(zip(scores, indices)):
            for score, idx in zip(patch_scores, patch_indices):
                if idx >= 0:  # 有效索引
                    metadata = self.local_metadata[idx]
                    results.append({
                        'image_id': metadata['image_id'],
                        'captions': metadata['captions'],
                        'score': float(score),
                        'patch_idx': patch_idx,
                        'source_patch_idx': metadata['patch_idx'],
                        'index_id': idx
                    })
        
        # 按分数排序并去重
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        # 去重（相同图像ID只保留最高分）
        seen_images = set()
        unique_results = []
        for result in results:
            if result['image_id'] not in seen_images:
                unique_results.append(result)
                seen_images.add(result['image_id'])
                if len(unique_results) >= top_m:
                    break
        
        return unique_results


def build_indexes_from_coco(coco_root: str, 
                          output_dir: str,
                          feature_extractor: CLIPFeatureExtractor,
                          max_images: Optional[int] = None) -> None:
    """
    从COCO数据集构建索引
    
    Args:
        coco_root: COCO数据集根目录
        output_dir: 输出目录
        feature_extractor: 特征提取器
        max_images: 最大处理图像数量（用于测试）
    """
    from pycocotools.coco import COCO
    
    logger.info(f"开始从COCO数据集构建索引: {coco_root}")
    
    # 加载COCO数据
    coco = COCO(os.path.join(coco_root, "annotations", "captions_train2017.json"))
    
    # 获取图像ID和描述
    img_ids = coco.getImgIds()
    if max_images:
        img_ids = img_ids[:max_images]
    
    logger.info(f"处理 {len(img_ids)} 张图像...")
    
    # 提取特征
    global_features = []
    patch_features_list = []
    image_ids = []
    captions_list = []
    
    for img_id in tqdm(img_ids, desc="提取特征"):
        try:
            # 获取图像信息
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(coco_root, "train2017", img_info['file_name'])
            
            if not os.path.exists(img_path):
                continue
            
            # 获取描述
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            captions = [ann['caption'] for ann in anns]
            
            # 提取特征
            features = feature_extractor.extract_features_from_path(img_path)
            
            global_features.append(features['global'].cpu().numpy())
            patch_features_list.append(features['patches'].cpu().numpy())
            image_ids.append(str(img_id))
            captions_list.append(captions)
            
        except Exception as e:
            logger.error(f"处理图像 {img_id} 失败: {e}")
            continue
    
    logger.info(f"成功提取 {len(global_features)} 张图像的特征")
    
    # 构建索引
    index_builder = FAISSIndexBuilder(
        feature_dim=feature_extractor.feature_dim,
        index_type=INDEX_CONFIG["index_type"],
        use_gpu=INDEX_CONFIG["use_gpu"]
    )
    
    # 构建全局索引
    index_builder.build_global_index(global_features, image_ids, captions_list)
    
    # 构建局部索引
    index_builder.build_local_index(patch_features_list, image_ids, captions_list)
    
    # 保存索引
    output_path = Path(output_dir)
    index_builder.save_indexes(output_path)
    
    logger.info(f"索引构建完成，保存到: {output_path}")


if __name__ == "__main__":
    # 测试索引构建
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    feature_dim = 512
    num_images = 100
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
    
    global_results = index_builder.search_global(query_global, top_k=5)
    local_results = index_builder.search_local(query_patches, top_m=10)
    
    print(f"全局搜索结果: {len(global_results)}")
    print(f"局部搜索结果: {len(local_results)}")
    
    # 保存和加载测试
    test_dir = Path("test_indexes")
    index_builder.save_indexes(test_dir)
    
    new_builder = FAISSIndexBuilder(feature_dim)
    new_builder.load_indexes(test_dir)
    
    print("索引保存和加载测试完成")
