import os
import logging
from typing import List, Dict, Tuple, Optional
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# COCO类别标签（80类）
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class PatchDetector:
    """使用Faster R-CNN进行显著目标检测，识别图像中的关键区域。"""

    def __init__(self, config: dict):
        """初始化Faster R-CNN检测器。
        
        Args:
            config: 配置字典，包含检测相关参数
        """
        patch_config = config.get("patch_config", {})
        self.confidence_threshold = float(patch_config.get("detection_confidence_threshold", 0.7))
        self.max_regions = int(patch_config.get("max_local_regions", 5))
        self.expand_ratio = float(patch_config.get("expand_ratio", 0.1))  # 边界扩展比例
        
        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载预训练的Faster R-CNN模型
        try:
            logger.info("Loading Faster R-CNN model...")
            # 兼容新旧版本的torchvision API
            try:
                # 新版本API (torchvision >= 0.13)
                self.model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
            except TypeError:
                # 旧版本API (torchvision < 0.13)
                self.model = fasterrcnn_resnet50_fpn(pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Faster R-CNN model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Faster R-CNN model: {e}")
            raise

    def detect_objects(self, image: Image.Image) -> List[Dict]:
        """检测图像中的显著物体/区域。
        
        Args:
            image: PIL.Image对象
            
        Returns:
            检测结果列表，每个元素包含：
            - 'bbox': [x1, y1, x2, y2] 边界框坐标
            - 'confidence': 置信度分数
            - 'class_id': 类别ID
            - 'class_label': 类别名称
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 转换为tensor
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # 执行检测
        with torch.no_grad():
            try:
                outputs = self.model(img_tensor)
            except Exception as e:
                logger.error(f"Object detection failed: {e}")
                return []
        
        # 解析结果
        detections = []
        boxes = outputs[0]['boxes'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            if score >= self.confidence_threshold:
                x1, y1, x2, y2 = box.astype(int).tolist()
                class_id = int(label)
                class_label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(score),
                    'class_id': class_id,
                    'class_label': class_label
                })
        
        return detections

    def filter_detections(self, detections: List[Dict], 
                         confidence_threshold: Optional[float] = None,
                         max_num: Optional[int] = None) -> List[Dict]:
        """过滤低置信度检测结果，保留前N个。
        
        Args:
            detections: 检测结果列表
            confidence_threshold: 置信度阈值（如果为None，使用初始化时的阈值）
            max_num: 最大保留数量（如果为None，使用初始化时的max_regions）
            
        Returns:
            过滤后的检测结果列表
        """
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        max_n = max_num if max_num is not None else self.max_regions
        
        # 按置信度排序
        filtered = [d for d in detections if d['confidence'] >= threshold]
        filtered.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 保留前N个
        return filtered[:max_n]

    def expand_bbox(self, bbox: List[int], image_width: int, image_height: int) -> List[int]:
        """扩展边界框以确保包含完整物体。
        
        Args:
            bbox: [x1, y1, x2, y2] 原始边界框
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            扩展后的边界框 [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # 计算扩展量
        expand_w = int(width * self.expand_ratio)
        expand_h = int(height * self.expand_ratio)
        
        # 扩展边界框
        x1 = max(0, x1 - expand_w)
        y1 = max(0, y1 - expand_h)
        x2 = min(image_width, x2 + expand_w)
        y2 = min(image_height, y2 + expand_h)
        
        return [x1, y1, x2, y2]

    def crop_regions(self, image: Image.Image, detections: List[Dict]) -> List[Tuple[Image.Image, Dict]]:
        """根据检测结果裁剪区域。
        
        Args:
            image: 原始PIL.Image对象
            detections: 检测结果列表
            
        Returns:
            (裁剪后的图像, 检测信息) 元组列表
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_width, image_height = image.size
        cropped_regions = []
        
        for det in detections:
            bbox = det['bbox'].copy()
            # 扩展边界框
            expanded_bbox = self.expand_bbox(bbox, image_width, image_height)
            x1, y1, x2, y2 = expanded_bbox
            
            # 裁剪区域
            try:
                cropped = image.crop((x1, y1, x2, y2))
                cropped_regions.append((cropped, det))
            except Exception as e:
                logger.warning(f"Failed to crop region {bbox}: {e}")
                continue
        
        return cropped_regions

