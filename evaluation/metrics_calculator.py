"""Metric calculation helpers for caption evaluation."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

try:
    from pycocoevalcap.cider.cider import Cider
except ImportError:  # pragma: no cover - dependency missing at runtime
    Cider = None  # type: ignore

try:
    from pycocoevalcap.spice.spice import Spice
except ImportError:  # pragma: no cover - dependency missing at runtime
    Spice = None  # type: ignore


class MetricsCalculator:
    """Compute a collection of captioning metrics for a single prediction."""

    def __init__(self) -> None:
        self._bleu_weights = (0.25, 0.25, 0.25, 0.25)
        self._smoothing = SmoothingFunction().method1
        self._rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        self._cider_scorer = Cider() if Cider is not None else None
        if Cider is None:
            logging.warning("CIDEr scorer not available. Install pycocoevalcap to enable CIDEr.")

        self._spice_scorer = Spice() if Spice is not None else None
        if Spice is None:
            logging.warning("SPICE scorer not available. Install pycocoevalcap to enable SPICE.")

    def calculate_all_metrics(self, generated_caption: str, reference_captions: List[str]) -> Dict[str, Optional[float]]:
        """Compute BLEU-4, ROUGE-L, CIDEr-D, and SPICE scores."""
        references = [r.strip() for r in reference_captions if r and r.strip()]
        generated = generated_caption.strip()

        if not references:
            logging.warning("No reference captions provided; metrics will be None.")
            return {"BLEU-4": None, "ROUGE-L": None, "CIDEr": None, "SPICE": None}

        scores: Dict[str, Optional[float]] = {
            "BLEU-4": self._calculate_bleu(generated, references),
            "ROUGE-L": self._calculate_rouge(generated, references),
            "CIDEr": self._calculate_cider(generated, references),
            "SPICE": self._calculate_spice(generated, references),
        }
        return scores

    def _tokenize(self, text: str) -> List[str]:
        return text.split()

    def _calculate_bleu(self, generated: str, references: List[str]) -> Optional[float]:
        try:
            hypothesis_tokens = self._tokenize(generated)
            reference_tokens = [self._tokenize(ref) for ref in references]
            if not hypothesis_tokens or not any(reference_tokens):
                return 0.0
            score = sentence_bleu(
                reference_tokens,
                hypothesis_tokens,
                weights=self._bleu_weights,
                smoothing_function=self._smoothing,
            )
            return float(score)
        except Exception as exc:  # pragma: no cover - defensive
            logging.error("BLEU calculation failed: %s", exc)
            return None

    def _calculate_rouge(self, generated: str, references: List[str]) -> Optional[float]:
        try:
            scores = [
                self._rouge_scorer.score(ref, generated)["rougeL"].fmeasure
                for ref in references
            ]
            if not scores:
                return 0.0
            return float(sum(scores) / len(scores))
        except Exception as exc:  # pragma: no cover - defensive
            logging.error("ROUGE-L calculation failed: %s", exc)
            return None

    def _prepare_coco_format(self, generated: str, references: List[str]):
        res = {0: [generated]}
        gts = {0: references}
        return gts, res

    def _calculate_cider(self, generated: str, references: List[str]) -> Optional[float]:
        if self._cider_scorer is None:
            return None
        try:
            gts, res = self._prepare_coco_format(generated, references)
            score, _ = self._cider_scorer.compute_score(gts, res)
            return float(score)
        except Exception as exc:  # pragma: no cover - external dependency
            logging.error("CIDEr calculation failed: %s", exc)
            return None

    def _calculate_spice(self, generated: str, references: List[str]) -> Optional[float]:
        if self._spice_scorer is None:
            return None
        try:
            gts, res = self._prepare_coco_format(generated, references)
            score, _ = self._spice_scorer.compute_score(gts, res)
            return float(score)
        except Exception as exc:  # pragma: no cover - external dependency
            logging.error("SPICE calculation failed: %s", exc)
            return None

