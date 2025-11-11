"""Dataset-level evaluator for IGRAG captioning."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from utils.image_utils import load_image
from core.retriever import ImageRetriever  # type: ignore
from core.generator import CaptionGenerator  # type: ignore
from evaluation.metrics_calculator import MetricsCalculator
from main import CaptionResult  # reuse dataclass

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore


@dataclass
class EvaluationSample:
    image_id: int
    file_name: str
    generated_caption: str
    references: List[str]
    metrics: Dict[str, Optional[float]]
    metadata: Dict[str, Union[str, float, bool, Dict]]


class IGRAGEvaluator:
    """Evaluate caption quality against COCO references."""

    def __init__(self, config: Dict, retriever: ImageRetriever, generator: CaptionGenerator) -> None:
        self.config = deepcopy(config)
        self.retriever = retriever
        self.generator = generator
        self.metrics = MetricsCalculator()

        evaluation_cfg = self.config.get("evaluation", {})
        self.val_images_dir = Path(evaluation_cfg.get("val_images_dir", ""))
        self.annotations_path = evaluation_cfg.get("val_annotations_path")
        self.output_dir = Path(evaluation_cfg.get("output_dir", "./evaluation_results"))
        self.save_individual = evaluation_cfg.get("save_individual_results", True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.val_images_dir.exists():
            raise FileNotFoundError(f"Validation images directory not found: {self.val_images_dir}")
        if not self.annotations_path or not Path(self.annotations_path).exists():
            raise FileNotFoundError(f"Validation annotations not found: {self.annotations_path}")

        self.runtime_cfg = self.config.setdefault("runtime_config", {})
        self.verbose = self.runtime_cfg.get("verbose", False)
        self.mode = self.runtime_cfg.get("mode", "deploy")

        gt_data = self._load_ground_truth(self.annotations_path)
        self.references = gt_data["references"]
        self.image_id_to_file = gt_data["file_map"]

    def _load_ground_truth(self, annotations_path: Union[str, Path]) -> Dict[str, Dict]:
        """Load COCO-style annotations into memory."""
        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_id_to_file = {item["id"]: item["file_name"] for item in data.get("images", [])}
        references: Dict[int, List[str]] = defaultdict(list)
        for ann in data.get("annotations", []):
            references[ann["image_id"]].append(ann["caption"])

        return {"references": references, "file_map": image_id_to_file}

    def evaluate_single_image(self, image_path: Union[str, Path, Image.Image], image_id: int) -> EvaluationSample:
        """Generate a caption for a single image and compute metrics."""
        if isinstance(image_path, (str, Path)):
            img = load_image(str(image_path))
        elif isinstance(image_path, Image.Image):
            img = image_path
        else:
            raise ValueError("image_path must be a path or PIL.Image.Image")

        references = self.references.get(image_id, [])
        image_meta = self._run_generation_pipeline(img)
        caption_result = image_meta["caption_result"]
        caption = caption_result.caption

        metrics = self.metrics.calculate_all_metrics(caption, references)

        sample = EvaluationSample(
            image_id=image_id,
            file_name=self.image_id_to_file.get(image_id, Path(image_path).name if isinstance(image_path, (str, Path)) else str(image_id)),
            generated_caption=caption,
            references=references,
            metrics=metrics,
            metadata={
                "pipeline": caption_result.metadata,
                "status": image_meta["status"],
                "error": image_meta.get("error"),
            },
        )
        return sample

    def _run_generation_pipeline(self, img: Image.Image) -> Dict:
        """Run generation using existing retriever and generator instances."""
        start_time = datetime.utcnow()
        runtime_cfg = self.runtime_cfg
        verbose = self.verbose
        use_patch_retrieval = self.config.get("retrieval_config", {}).get("use_patch_retrieval", False)

        metadata: Dict[str, Union[str, float, bool, Dict]] = {
            "runtime_mode": runtime_cfg.get("mode", "deploy"),
            "retrieval": {"mode": "patch" if use_patch_retrieval else "global"},
            "generation": {"fallback_used": False, "valid": True},
            "timestamps": {"started_at": start_time.isoformat() + "Z"},
        }

        retrieval_time = prompt_time = gen_time = None
        retrieved_for_fallback: List[Dict] = []

        try:
            if use_patch_retrieval:
                retrieval_start = datetime.utcnow()
                try:
                    retrieved_data = self.retriever.retrieve_with_patches(img)
                except Exception as exc:
                    logging.error("Patch retrieval failed: %s, fallback to global mode", exc)
                    metadata["retrieval"]["mode"] = "patch_fallback_global"  # type: ignore[index]
                    metadata["retrieval"]["patch_error"] = str(exc)  # type: ignore[index]
                    retrieved_data = {
                        "global_descriptions": self.retriever.get_retrieved_captions(
                            img, top_k=self.config.get("retrieval_config", {}).get("top_k", 3)
                        ),
                        "local_regions": [],
                    }

                retrieval_time = (datetime.utcnow() - retrieval_start).total_seconds()
                global_descriptions = retrieved_data.get("global_descriptions", [])
                local_regions = retrieved_data.get("local_regions", [])
                metadata["retrieval"].update(  # type: ignore[call-arg]
                    {
                        "global_candidates": len(global_descriptions),
                        "local_regions_detected": len(local_regions),
                        "retrieval_time_sec": retrieval_time,
                    }
                )

                prompt_start = datetime.utcnow()
                prompt = self.generator.build_prompt(retrieved_data)
                prompt_time = (datetime.utcnow() - prompt_start).total_seconds()

                gen_start = datetime.utcnow()
                caption = self.generator.generate_caption(prompt, debug=verbose)
                gen_time = (datetime.utcnow() - gen_start).total_seconds()

                retrieved_for_fallback = global_descriptions
            else:
                retrieval_start = datetime.utcnow()
                retrieved_for_fallback = self.retriever.get_retrieved_captions(
                    img, top_k=self.config.get("retrieval_config", {}).get("top_k", 3)
                )
                retrieval_time = (datetime.utcnow() - retrieval_start).total_seconds()
                metadata["retrieval"].update(  # type: ignore[call-arg]
                    {
                        "global_candidates": len(retrieved_for_fallback),
                        "retrieval_time_sec": retrieval_time,
                    }
                )

                if not retrieved_for_fallback:
                    metadata["generation"]["valid"] = False  # type: ignore[index]
                    return {
                        "caption_result": CaptionResult(caption="", metadata=metadata),
                        "status": "no_retrieval",
                    }

                prompt_start = datetime.utcnow()
                prompt = self.generator.build_prompt(retrieved_for_fallback)
                prompt_time = (datetime.utcnow() - prompt_start).total_seconds()

                gen_start = datetime.utcnow()
                caption = self.generator.generate_caption(prompt, debug=verbose)
                gen_time = (datetime.utcnow() - gen_start).total_seconds()

            def _is_valid(s: str) -> bool:
                if not s:
                    return False
                import re

                core = re.sub(r"[\W_]+", "", s.strip())
                return len(core) >= 6

            if not _is_valid(caption):
                metadata["generation"]["valid"] = False  # type: ignore[index]
                mode = runtime_cfg.get("mode", "deploy")
                if mode in {"deploy", "dev"}:
                    parts = []
                    for item in retrieved_for_fallback:
                        caps = item.get("captions", [])
                        if caps:
                            parts.append(caps[0].strip().rstrip("."))
                    if parts:
                        caption = ";".join(parts)
                        metadata["generation"]["fallback_used"] = True  # type: ignore[index]
                elif mode == "test":
                    return {
                        "caption_result": CaptionResult(caption="", metadata=metadata),
                        "status": "generation_failed",
                    }
                else:
                    parts = []
                    for item in retrieved_for_fallback:
                        caps = item.get("captions", [])
                        if caps:
                            parts.append(caps[0].strip().rstrip("."))
                    if parts:
                        caption = ";".join(parts)
                        metadata["generation"]["fallback_used"] = True  # type: ignore[index]

            total_time = (datetime.utcnow() - start_time).total_seconds()
            metadata["timing"] = {
                "total_sec": total_time,
                "retrieval_sec": retrieval_time,
                "prompt_sec": prompt_time,
                "generation_sec": gen_time,
            }

            return {
                "caption_result": CaptionResult(caption=caption, metadata=metadata),
                "status": "success",
            }

        except Exception as exc:  # pragma: no cover - safeguard
            logging.error("Generation pipeline failed: %s", exc)
            metadata["generation"]["valid"] = False  # type: ignore[index]
            metadata["error"] = str(exc)  # type: ignore[index]
            return {
                "caption_result": CaptionResult(caption="", metadata=metadata),
                "status": "error",
                "error": str(exc),
            }

    def evaluate_dataset(self, image_ids: Iterable[int], output_path: Union[str, Path]) -> Dict:
        """Run evaluation on a list of image IDs."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        existing = self._load_existing_results(output_path) if output_path.exists() else {}
        results: Dict[str, Dict] = {}
        results.update(existing.get("per_image", {}))

        if tqdm is not None:
            iterator = tqdm(list(image_ids), desc="Evaluating captions", unit="image")
        else:
            iterator = image_ids

        for image_id in iterator:
            if str(image_id) in results:
                continue
            try:
                file_name = self.image_id_to_file.get(image_id)
                if not file_name:
                    logging.warning("Image id %s not found in annotations; skipping.", image_id)
                    continue
                image_path = self.val_images_dir / file_name
                sample = self.evaluate_single_image(image_path, image_id)
                results[str(image_id)] = self._sample_to_dict(sample)
            except Exception as exc:
                logging.error("Failed to evaluate image %s: %s", image_id, exc)
                results[str(image_id)] = {
                    "image_id": image_id,
                    "file_name": self.image_id_to_file.get(image_id, ""),
                    "generated_caption": "",
                    "references": self.references.get(image_id, []),
                    "metrics": {
                        "BLEU-4": None,
                        "ROUGE-L": None,
                        "CIDEr": None,
                        "SPICE": None,
                    },
                    "metadata": {"status": "error", "error": str(exc)},
                }

            if self.save_individual:
                self._write_partial_results(output_path, results)

        summary = self._aggregate_results(results)
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "config": {
                "retrieval_mode": self.config.get("retrieval_config", {}).get("use_patch_retrieval", False),
                "metrics": self.config.get("metrics", {}),
                "runtime_mode": self.mode,
            },
            "summary": summary,
            "per_image": results,
        }

        self._write_json(output_path, payload)
        return payload

    def _sample_to_dict(self, sample: EvaluationSample) -> Dict:
        return {
            "image_id": sample.image_id,
            "file_name": sample.file_name,
            "generated_caption": sample.generated_caption,
            "references": sample.references,
            "metrics": sample.metrics,
            "metadata": sample.metadata,
        }

    def _load_existing_results(self, output_path: Path) -> Dict:
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except Exception as exc:
            logging.warning("Failed to load existing results %s: %s", output_path, exc)
            return {}

    def _write_partial_results(self, output_path: Path, results: Dict[str, Dict]) -> None:
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "per_image": results,
        }
        self._write_json(output_path, payload)

    def _write_json(self, path: Path, payload: Dict) -> None:
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp_path.replace(path)

    def _aggregate_results(self, results: Dict[str, Dict]) -> Dict:
        metrics_keys = ["BLEU-4", "ROUGE-L", "CIDEr", "SPICE"]
        metrics_arrays: Dict[str, List[float]] = {k: [] for k in metrics_keys}
        failures = 0

        for item in results.values():
            meta = item.get("metadata", {})
            status = meta.get("status")
            if status != "success":
                failures += 1
            metrics = item.get("metrics", {})
            for key in metrics_keys:
                val = metrics.get(key)
                if isinstance(val, (int, float)):
                    metrics_arrays[key].append(float(val))

        summary_metrics = {}
        for key, values in metrics_arrays.items():
            if not values:
                summary_metrics[key] = {"mean": None, "median": None, "p25": None, "p75": None}
                continue
            arr = np.array(values)
            summary_metrics[key] = {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
            }

        return {
            "num_images": len(results),
            "num_failures": failures,
            "metrics": summary_metrics,
        }

