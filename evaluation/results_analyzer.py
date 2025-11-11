"""Utilities to analyse evaluation results and produce human-readable reports."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple


class ResultsAnalyzer:
    """Load evaluation JSON results and create summaries."""

    def __init__(self) -> None:
        self.data: Optional[Dict] = None

    def load_results(self, results_path: Path) -> Dict:
        path = Path(results_path)
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        return self.data

    def generate_report(self, top_k: int = 5) -> str:
        if not self.data:
            raise ValueError("No evaluation data loaded. Call load_results first.")

        summary = self.data.get("summary", {})
        metrics = summary.get("metrics", {})
        per_image = self.data.get("per_image", {})
        num_failures = summary.get("num_failures", 0)
        num_images = summary.get("num_images", len(per_image))

        lines: List[str] = []
        lines.append("IGRAG Captioning Evaluation Report")
        lines.append("=" * 40)
        lines.append(f"Images evaluated: {num_images}")
        lines.append(f"Failed samples: {num_failures}")
        lines.append("")
        lines.append("Aggregate metrics:")

        for name, stats in metrics.items():
            official = stats.get("official")
            mean_val = stats.get("mean")
            median = stats.get("median")
            p25 = stats.get("p25")
            p75 = stats.get("p75")

            def _fmt(value: Optional[float]) -> str:
                return f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"

            lines.append(
                f"- {name}: official={_fmt(official)} mean={_fmt(mean_val)} "
                f"median={_fmt(median)} [p25={_fmt(p25)}, p75={_fmt(p75)}]"
            )

        low_samples, high_samples = self._identify_extremes(per_image, top_k)
        if high_samples:
            lines.append("")
            lines.append(f"Top {len(high_samples)} samples:")
            for entry in high_samples:
                lines.append(self._format_sample(entry))

        if low_samples:
            lines.append("")
            lines.append(f"Bottom {len(low_samples)} samples:")
            for entry in low_samples:
                lines.append(self._format_sample(entry))

        return "\n".join(lines)

    def _identify_extremes(
        self, per_image: Dict[str, Dict], top_k: int
    ) -> Tuple[List[Tuple[str, Dict]], List[Tuple[str, Dict]]]:
        scored_items: List[Tuple[str, Dict, float]] = []
        for image_id, entry in per_image.items():
            metrics = entry.get("metrics", {})
            scores = [v for v in metrics.values() if isinstance(v, (int, float))]
            if not scores:
                continue
            avg_score = mean(scores)
            scored_items.append((image_id, entry, avg_score))

        scored_items.sort(key=lambda x: x[2])
        low = [(img_id, data) for img_id, data, _ in scored_items[:top_k]]
        high = [(img_id, data) for img_id, data, _ in scored_items[-top_k:][::-1]]
        return low, high

    def _format_sample(self, item: Tuple[str, Dict]) -> str:
        image_id, data = item
        caption = data.get("generated_caption", "")
        metrics = data.get("metrics", {})
        status = data.get("metadata", {}).get("status", "unknown")
        metric_items = []
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metric_items.append(f"{k}={v:.4f}")
        metric_str = ", ".join(metric_items) if metric_items else "no metrics"
        return f"[{image_id}] status={status} caption='{caption}' ({metric_str})"

