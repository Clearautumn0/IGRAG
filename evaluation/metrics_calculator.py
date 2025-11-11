"""COCO-style captioning metrics calculator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from pycocotools.coco import COCO
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


class MetricsCalculator:
    """Wrapper around COCOEvalCap for computing captioning metrics."""

    _METRIC_KEY_MAP: Dict[str, str] = {
        "Bleu_4": "BLEU-4",
        "ROUGE_L": "ROUGE-L",
        "CIDEr": "CIDEr",
        "SPICE": "SPICE",
    }

    _CONFIG_MAP: Dict[str, str] = {
        "bleu": "BLEU-4",
        "rouge": "ROUGE-L",
        "cider": "CIDEr",
        "spice": "SPICE",
    }

    def __init__(self, annotations_path: Union[str, Path], metrics_config: Optional[Dict[str, bool]] = None) -> None:
        self.annotations_path = Path(annotations_path)
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_path}")

        self.coco = COCO(str(self.annotations_path))
        self.enabled_metrics = self._resolve_enabled_metrics(metrics_config)

    def _resolve_enabled_metrics(self, metrics_config: Optional[Dict[str, bool]]) -> List[str]:
        if not metrics_config:
            return list(self._METRIC_KEY_MAP.values())
        enabled = []
        for key, friendly in self._CONFIG_MAP.items():
            flag = metrics_config.get(key, True)
            if flag:
                enabled.append(friendly)
        if not enabled:
            logging.warning("No metrics enabled; defaulting to BLEU-4.")
            enabled = ["BLEU-4"]
        return enabled

    def evaluate(self, predictions: Sequence[Dict[str, Union[int, str]]]) -> Dict[str, Dict]:
        """Evaluate predictions against COCO references using official metrics.

        Args:
            predictions: iterable of {"image_id": int, "caption": str}
        """
        predictions = list(predictions)
        if not predictions:
            logging.warning("No predictions provided; skipping metric evaluation.")
            return {"aggregate": {}, "per_image": {}, "enabled_metrics": self.enabled_metrics}

        formatted_predictions = self._format_predictions(predictions)
        valid_predictions = []
        for item in formatted_predictions:
            image_id = int(item["image_id"])
            anns = self.coco.imgToAnns.get(image_id, [])
            if not anns:
                logging.warning("Skipping image %s due to missing ground-truth captions.", image_id)
                continue
            valid_predictions.append(item)

        if not valid_predictions:
            logging.warning("No valid predictions overlap with COCO annotations; cannot compute metrics.")
            return {"aggregate": {}, "per_image": {}, "enabled_metrics": self.enabled_metrics}

        coco_results = self.coco.loadRes(valid_predictions)
        img_ids = coco_results.getImgIds()

        gts = {img_id: self.coco.imgToAnns[img_id] for img_id in img_ids}
        res = {img_id: coco_results.imgToAnns[img_id] for img_id in img_ids}

        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        aggregate: Dict[str, float] = {}
        per_image: Dict[str, Dict[str, float]] = {str(img_id): {} for img_id in img_ids}
        active_metrics: List[str] = []

        def _store(metric_name: str, overall: float, scores: Sequence[float]) -> None:
            aggregate[metric_name] = float(overall)
            for img_id, score in zip(img_ids, scores):
                per_image[str(img_id)][metric_name] = float(score)
            active_metrics.append(metric_name)

        if "BLEU-4" in self.enabled_metrics:
            try:
                scorer = Bleu(4)
                score, scores = scorer.compute_score(gts, res)
                _store("BLEU-4", score[3], scores[3])
            except Exception as exc:
                logging.warning("BLEU computation failed: %s", exc)

        if "ROUGE-L" in self.enabled_metrics:
            try:
                scorer = Rouge()
                score, scores = scorer.compute_score(gts, res)
                _store("ROUGE-L", score, scores)
            except Exception as exc:
                logging.warning("ROUGE-L computation failed: %s", exc)

        if "CIDEr" in self.enabled_metrics:
            try:
                scorer = Cider()
                score, scores = scorer.compute_score(gts, res)
                _store("CIDEr", score, scores)
            except Exception as exc:
                logging.warning("CIDEr computation failed: %s", exc)

        if "SPICE" in self.enabled_metrics:
            try:
                scorer = Spice()
                score, scores = scorer.compute_score(gts, res)
                # SPICE returns tuple(score, scores) where scores is list of dicts; extract "All"
                overall = score
                per_scores = [item.get("All", 0.0) if isinstance(item, dict) else item for item in scores]
                _store("SPICE", overall, per_scores)
            except Exception as exc:
                logging.warning("SPICE computation failed: %s", exc)

        return {"aggregate": aggregate, "per_image": per_image, "enabled_metrics": active_metrics}

    def _format_predictions(self, predictions: Iterable[Dict[str, Union[int, str]]]) -> List[Dict[str, Union[int, str]]]:
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

    # _extract_* helpers removed since metrics are computed inline.

