#!/usr/bin/env python3
"""CLI entrypoint for running IGRAG caption evaluation."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from core.generator import CaptionGenerator
from core.retriever import ImageRetriever
from evaluation.evaluator import IGRAGEvaluator
from evaluation.results_analyzer import ResultsAnalyzer
from main import load_config, setup_logging

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IGRAG caption evaluation on COCO validation set.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file.")
    parser.add_argument("--subset", type=int, default=None, help="Number of validation images to evaluate.")
    parser.add_argument("--output", type=str, default=None, help="Path to write JSON evaluation results.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["global_only", "global_local"],
        default=None,
        help="Evaluation retrieval mode override.",
    )
    return parser.parse_args()


def adjust_config(config: dict, args: argparse.Namespace) -> None:
    evaluation_cfg = config.setdefault("evaluation", {})
    if args.subset is not None:
        evaluation_cfg["subset_size"] = args.subset

    retrieval_cfg = config.setdefault("retrieval_config", {})
    if args.mode == "global_only":
        retrieval_cfg["use_patch_retrieval"] = False
    elif args.mode == "global_local":
        retrieval_cfg["use_patch_retrieval"] = True


def resolve_output_path(config: dict, output_arg: Optional[str]) -> Path:
    evaluation_cfg = config.get("evaluation", {})
    base_dir = Path(evaluation_cfg.get("output_dir", "./evaluation_results"))
    base_dir.mkdir(parents=True, exist_ok=True)

    if output_arg:
        return Path(output_arg)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"evaluation_{timestamp}.json"


def select_image_ids(evaluator: IGRAGEvaluator, subset: Optional[int]) -> List[int]:
    image_ids = sorted(int(image_id) for image_id in evaluator.image_id_to_file.keys())
    if subset is not None:
        return image_ids[:subset]
    return image_ids


def save_visualizations(results_path: Path, results: dict) -> None:
    if plt is None:
        logging.info("matplotlib not available; skipping visualization export.")
        return

    metrics = results.get("per_image", {})
    keys = ["BLEU-4", "ROUGE-L", "CIDEr", "SPICE"]
    data = {k: [] for k in keys}
    for entry in metrics.values():
        metric = entry.get("metrics", {})
        for k in keys:
            val = metric.get(k)
            if isinstance(val, (int, float)):
                data[k].append(float(val))

    plt.figure(figsize=(12, 8))
    for idx, key in enumerate(keys, start=1):
        values = data[key]
        plt.subplot(2, 2, idx)
        if values:
            plt.hist(values, bins=30, alpha=0.7, color="steelblue")
            plt.title(f"{key} distribution")
            plt.xlabel("Score")
            plt.ylabel("Frequency")
        else:
            plt.text(0.5, 0.5, "No data", ha="center", va="center")
            plt.title(f"{key} distribution")
            plt.axis("off")

    plt.tight_layout()
    figure_path = results_path.with_suffix(".png")
    plt.savefig(figure_path)
    plt.close()
    logging.info("Saved metric histograms to %s", figure_path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    adjust_config(config, args)
    setup_logging(config)

    retriever = ImageRetriever(config)
    generator = CaptionGenerator(config)
    evaluator = IGRAGEvaluator(config, retriever, generator)

    subset_size = config.get("evaluation", {}).get("subset_size")
    image_ids = select_image_ids(evaluator, subset_size)
    if not image_ids:
        logging.error("No validation images available for evaluation.")
        return

    output_path = resolve_output_path(config, args.output)
    results = evaluator.evaluate_dataset(image_ids, output_path)

    analyzer = ResultsAnalyzer()
    analyzer.load_results(output_path)
    report = analyzer.generate_report()
    print(report)

    save_visualizations(output_path, results)


if __name__ == "__main__":
    main()

