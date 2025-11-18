"""使用pycocoevalcap计算caption评估指标"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from pycocotools.coco import COCO
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


class MetricsCalculator:
    """使用pycocoevalcap计算caption评估指标"""

    def __init__(
        self, 
        annotations_path: Union[str, Path], 
        metrics_config: Optional[Dict[str, bool]] = None
    ) -> None:
        """
        初始化指标计算器
        
        Args:
            annotations_path: COCO标注文件路径
            metrics_config: 指标配置字典，例如 {"bleu_1": True, "bleu_2": True, ...}
        """
        self.annotations_path = Path(annotations_path)
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"标注文件不存在: {self.annotations_path}")

        self.coco = COCO(str(self.annotations_path))
        self.metrics_config = metrics_config or {}
        
        # 默认启用所有指标
        self.enabled_metrics = self._resolve_enabled_metrics()

    def _resolve_enabled_metrics(self) -> List[str]:
        """解析启用的指标列表"""
        default_enabled = {
            "bleu_1": True,
            "bleu_2": True,
            "bleu_4": True,
            "meteor": True,
            "rouge": True,
            "cider": True,
            "spice": True,
        }
        
        enabled = []
        for metric_name, default_value in default_enabled.items():
            if self.metrics_config.get(metric_name, default_value):
                enabled.append(metric_name)
        
        if not enabled:
            logging.warning("没有启用任何指标，默认启用BLEU-4")
            enabled = ["bleu_4"]
        
        return enabled

    def evaluate(
        self, 
        predictions: Sequence[Dict[str, Union[int, str]]]
    ) -> Dict[str, Dict]:
        """
        评估预测结果
        
        Args:
            predictions: 预测列表，每个元素为 {"image_id": int, "caption": str}
            
        Returns:
            包含总体指标和每张图片指标的字典
        """
        predictions = list(predictions)
        if not predictions:
            logging.warning("没有提供预测结果，跳过指标计算")
            return {
                "aggregate": {},
                "per_image": {},
                "enabled_metrics": self.enabled_metrics
            }

        # 格式化预测结果
        formatted_predictions = self._format_predictions(predictions)
        
        # 过滤掉没有ground truth的图片
        valid_predictions = []
        for item in formatted_predictions:
            image_id = int(item["image_id"])
            anns = self.coco.imgToAnns.get(image_id, [])
            if not anns:
                logging.warning(f"图片 {image_id} 没有ground truth标注，跳过")
                continue
            valid_predictions.append(item)

        if not valid_predictions:
            logging.warning("没有有效的预测结果与COCO标注重叠，无法计算指标")
            return {
                "aggregate": {},
                "per_image": {},
                "enabled_metrics": self.enabled_metrics
            }

        # 加载为COCO格式
        coco_results = self.coco.loadRes(valid_predictions)
        img_ids = coco_results.getImgIds()

        # 准备ground truth和预测结果
        gts = {img_id: self.coco.imgToAnns[img_id] for img_id in img_ids}
        res = {img_id: coco_results.imgToAnns[img_id] for img_id in img_ids}

        # 使用PTB tokenizer进行分词
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # 计算各项指标
        aggregate: Dict[str, float] = {}
        per_image: Dict[str, Dict[str, float]] = {str(img_id): {} for img_id in img_ids}
        active_metrics: List[str] = []

        # BLEU指标（计算BLEU-1, BLEU-2, BLEU-4）
        if any(m in self.enabled_metrics for m in ["bleu_1", "bleu_2", "bleu_4"]):
            try:
                scorer = Bleu(4)  # 计算到BLEU-4
                score, scores = scorer.compute_score(gts, res)
                
                # score是长度为4的列表，分别对应BLEU-1到BLEU-4
                # scores也是长度为4的列表，每个元素是对应BLEU的每张图片得分
                if "bleu_1" in self.enabled_metrics:
                    aggregate["BLEU-1"] = float(score[0])
                    for img_id, s in zip(img_ids, scores[0]):
                        per_image[str(img_id)]["BLEU-1"] = float(s)
                    active_metrics.append("BLEU-1")
                
                if "bleu_2" in self.enabled_metrics:
                    aggregate["BLEU-2"] = float(score[1])
                    for img_id, s in zip(img_ids, scores[1]):
                        per_image[str(img_id)]["BLEU-2"] = float(s)
                    active_metrics.append("BLEU-2")
                
                if "bleu_4" in self.enabled_metrics:
                    aggregate["BLEU-4"] = float(score[3])
                    for img_id, s in zip(img_ids, scores[3]):
                        per_image[str(img_id)]["BLEU-4"] = float(s)
                    active_metrics.append("BLEU-4")
            except Exception as exc:
                logging.warning(f"BLEU计算失败: {exc}")

        # METEOR指标
        if "meteor" in self.enabled_metrics:
            try:
                scorer = Meteor()
                score, scores = scorer.compute_score(gts, res)
                aggregate["METEOR"] = float(score)
                for img_id, s in zip(img_ids, scores):
                    per_image[str(img_id)]["METEOR"] = float(s)
                active_metrics.append("METEOR")
            except Exception as exc:
                logging.warning(f"METEOR计算失败: {exc}")

        # ROUGE-L指标
        if "rouge" in self.enabled_metrics:
            try:
                scorer = Rouge()
                score, scores = scorer.compute_score(gts, res)
                aggregate["ROUGE-L"] = float(score)
                for img_id, s in zip(img_ids, scores):
                    per_image[str(img_id)]["ROUGE-L"] = float(s)
                active_metrics.append("ROUGE-L")
            except Exception as exc:
                logging.warning(f"ROUGE-L计算失败: {exc}")

        # CIDEr指标
        if "cider" in self.enabled_metrics:
            try:
                scorer = Cider()
                score, scores = scorer.compute_score(gts, res)
                aggregate["CIDEr"] = float(score)
                for img_id, s in zip(img_ids, scores):
                    per_image[str(img_id)]["CIDEr"] = float(s)
                active_metrics.append("CIDEr")
            except Exception as exc:
                logging.warning(f"CIDEr计算失败: {exc}")

        # SPICE指标
        if "spice" in self.enabled_metrics:
            try:
                scorer = Spice()
                score, scores = scorer.compute_score(gts, res)
                # SPICE返回的score是字典，scores是每张图片的字典列表
                overall = score.get("All", 0.0) if isinstance(score, dict) else score
                aggregate["SPICE"] = float(overall)
                for img_id, s in zip(img_ids, scores):
                    spice_score = s.get("All", 0.0) if isinstance(s, dict) else s
                    per_image[str(img_id)]["SPICE"] = float(spice_score)
                active_metrics.append("SPICE")
            except Exception as exc:
                logging.warning(f"SPICE计算失败: {exc}")

        return {
            "aggregate": aggregate,
            "per_image": per_image,
            "enabled_metrics": active_metrics
        }

    def _format_predictions(
        self, 
        predictions: Sequence[Dict[str, Union[int, str]]]
    ) -> List[Dict[str, Union[int, str]]]:
        """格式化预测结果，去除重复的image_id"""
        seen = set()
        formatted = []
        for item in predictions:
            image_id = int(item["image_id"])
            if image_id in seen:
                continue
            caption = str(item.get("caption", "")).strip()
            formatted.append({"image_id": image_id, "caption": caption})
            seen.add(image_id)
        return formatted
