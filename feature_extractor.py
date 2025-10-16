"""
CLIP特征提取模块 - 支持全局和局部特征提取
"""
import torch
import torch.nn as nn
import open_clip
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from config import MODEL_CONFIG, FEATURE_CONFIG

logger = logging.getLogger(__name__)


class CLIPFeatureExtractor:
    """CLIP特征提取器，支持全局和局部特征提取"""
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        """
        初始化CLIP特征提取器
        
        Args:
            model_name: CLIP模型名称
            device: 计算设备
        """
        self.device = device
        self.model_name = model_name
        
        # 加载CLIP模型
        logger.info(f"正在加载CLIP模型: {model_name}")
        self.model, self.preprocess, self.tokenizer = open_clip.create_model_and_transforms(
            model_name, pretrained='openai'
        )
        self.model = self.model.to(device)
        self.model.eval()
        
        # 获取模型配置信息
        self._get_model_config()
        
        logger.info(f"CLIP模型加载完成，设备: {device}")
    
    def _get_model_config(self):
        """获取模型配置信息"""
        # 获取patch信息
        if hasattr(self.model.visual, 'patch_embed'):
            self.patch_size = self.model.visual.patch_embed.patch_size[0]
            self.num_patches_h = self.model.visual.patch_embed.img_size[0] // self.patch_size
            self.num_patches_w = self.model.visual.patch_embed.img_size[1] // self.patch_size
            self.num_patches = self.num_patches_h * self.num_patches_w
        else:
            # 默认配置
            self.patch_size = 32
            self.num_patches_h = 7
            self.num_patches_w = 7
            self.num_patches = 49
        
        # 获取特征维度
        self.feature_dim = self.model.visual.output_dim
        
        logger.info(f"模型配置 - Patch大小: {self.patch_size}, "
                   f"Patch数量: {self.num_patches}, 特征维度: {self.feature_dim}")
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            预处理后的图像张量
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            return image_tensor
        except Exception as e:
            logger.error(f"图像预处理失败 {image_path}: {e}")
            raise
    
    def extract_features(self, image_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取全局和局部特征
        
        Args:
            image_tensor: 预处理后的图像张量 [1, 3, H, W]
            
        Returns:
            包含全局和局部特征的字典
        """
        with torch.no_grad():
            # 获取视觉特征
            visual_features = self.model.encode_image(image_tensor)
            
            # 提取patch特征（局部特征）
            patch_features = self._extract_patch_features(image_tensor)
            
            # 全局特征（CLS token）
            global_features = visual_features
            
            return {
                'global': global_features,
                'patches': patch_features,
                'patch_positions': self._get_patch_positions()
            }
    
    def _extract_patch_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        提取patch特征（局部特征）
        
        Args:
            image_tensor: 图像张量 [1, 3, H, W]
            
        Returns:
            patch特征张量 [num_patches, feature_dim]
        """
        with torch.no_grad():
            # 获取patch embeddings
            x = self.model.visual.conv1(image_tensor)  # [1, embed_dim, H', W']
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [1, embed_dim, H'*W']
            x = x.permute(0, 2, 1)  # [1, H'*W', embed_dim]
            
            # 添加位置编码
            x = x + self.model.visual.positional_embedding[:, 1:, :]  # 跳过CLS token
            
            # 通过transformer层
            x = self.model.visual.transformer(x)
            
            # 获取patch特征（跳过CLS token）
            patch_features = x[0]  # [num_patches, feature_dim]
            
            return patch_features
    
    def _get_patch_positions(self) -> List[Tuple[int, int]]:
        """
        获取patch位置信息
        
        Returns:
            patch位置列表 [(row, col), ...]
        """
        positions = []
        for i in range(self.num_patches_h):
            for j in range(self.num_patches_w):
                positions.append((i, j))
        return positions
    
    def extract_global_features(self, image_path: str) -> torch.Tensor:
        """
        仅提取全局特征
        
        Args:
            image_path: 图像路径
            
        Returns:
            全局特征向量
        """
        image_tensor = self.preprocess_image(image_path)
        features = self.extract_features(image_tensor)
        return features['global'].cpu()
    
    def extract_patch_features(self, image_path: str) -> torch.Tensor:
        """
        仅提取patch特征
        
        Args:
            image_path: 图像路径
            
        Returns:
            patch特征矩阵 [num_patches, feature_dim]
        """
        image_tensor = self.preprocess_image(image_path)
        features = self.extract_features(image_tensor)
        return features['patches'].cpu()
    
    def select_key_patches(self, patch_features: torch.Tensor, 
                          strategy: str = "norm", 
                          num_patches: int = 3) -> List[int]:
        """
        选择关键patch
        
        Args:
            patch_features: patch特征 [num_patches, feature_dim]
            strategy: 选择策略 ("norm", "random", "attention")
            num_patches: 选择的patch数量
            
        Returns:
            关键patch的索引列表
        """
        if strategy == "norm":
            # 基于特征范数选择
            norms = torch.norm(patch_features, dim=1)
            _, indices = torch.topk(norms, num_patches)
            return indices.tolist()
        
        elif strategy == "random":
            # 随机选择
            indices = torch.randperm(patch_features.shape[0])[:num_patches]
            return indices.tolist()
        
        elif strategy == "attention":
            # 基于注意力机制选择（简化版本）
            # 计算每个patch与全局特征的相似度
            global_feature = torch.mean(patch_features, dim=0, keepdim=True)
            similarities = torch.mm(patch_features, global_feature.T).squeeze()
            _, indices = torch.topk(similarities, num_patches)
            return indices.tolist()
        
        else:
            raise ValueError(f"未知的选择策略: {strategy}")
    
    def batch_extract_features(self, image_paths: List[str]) -> List[Dict[str, torch.Tensor]]:
        """
        批量提取特征
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            特征字典列表
        """
        features_list = []
        
        for image_path in image_paths:
            try:
                features = self.extract_features_from_path(image_path)
                features_list.append(features)
            except Exception as e:
                logger.error(f"批量特征提取失败 {image_path}: {e}")
                continue
        
        return features_list
    
    def extract_features_from_path(self, image_path: str) -> Dict[str, torch.Tensor]:
        """
        从图像路径提取特征
        
        Args:
            image_path: 图像路径
            
        Returns:
            特征字典
        """
        image_tensor = self.preprocess_image(image_path)
        return self.extract_features(image_tensor)


def test_feature_extractor():
    """测试特征提取器"""
    import matplotlib.pyplot as plt
    
    # 创建特征提取器
    extractor = CLIPFeatureExtractor(device="cpu")
    
    # 创建测试图像
    test_image = Image.new('RGB', (224, 224), color='red')
    test_image.save('test_image.jpg')
    
    try:
        # 提取特征
        features = extractor.extract_features_from_path('test_image.jpg')
        
        print(f"全局特征形状: {features['global'].shape}")
        print(f"Patch特征形状: {features['patches'].shape}")
        print(f"Patch位置数量: {len(features['patch_positions'])}")
        
        # 选择关键patch
        key_patches = extractor.select_key_patches(features['patches'], strategy="norm")
        print(f"关键patch索引: {key_patches}")
        
    finally:
        # 清理测试文件
        import os
        if os.path.exists('test_image.jpg'):
            os.remove('test_image.jpg')


if __name__ == "__main__":
    test_feature_extractor()
