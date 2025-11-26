import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from tqdm import tqdm

from core.generator import CaptionGenerator
from core.retriever import ImageRetriever
from utils.image_utils import load_image

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATCH = CaptionGenerator.PROMPT_TEMPLATE_PATCH


@dataclass
class TrainingSample:
    """Single prompt-target pair used for LoRA fine-tuning."""

    image_id: int
    prompt: str
    caption: str
    metadata: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {
                "image_id": self.image_id,
                "prompt": self.prompt,
                "caption": self.caption,
                "metadata": self.metadata,
            },
            ensure_ascii=False,
        )


class CocoCaptionIndex:
    """Utility to sample COCO captions per image id."""

    def __init__(self, annotations_path: Path):
        with annotations_path.open("r") as f:
            payload = json.load(f)
        self._images = {item["id"]: item for item in payload.get("images", [])}

        caps: Dict[int, List[str]] = {}
        for ann in payload.get("annotations", []):
            image_id = ann.get("image_id")
            caption = (ann.get("caption") or "").strip()
            if not caption:
                continue
            caps.setdefault(image_id, []).append(caption)
        self._captions = caps

    @property
    def available_ids(self) -> List[int]:
        return [img_id for img_id in self._images.keys() if img_id in self._captions]

    def get_image_info(self, image_id: int) -> Optional[Dict[str, Any]]:
        return self._images.get(image_id)

    def sample_caption(self, image_id: int, rng: random.Random) -> Optional[str]:
        captions = self._captions.get(image_id) or []
        if not captions:
            return None
        return rng.choice(captions)


def build_patch_prompt(
    retrieved_data: Dict[str, Any], template: str = PROMPT_TEMPLATE_PATCH
) -> str:
    """Format retrieval outputs into the patched prompt template."""
    global_descriptions: List[str] = []
    for item in retrieved_data.get("global_descriptions", []):
        caps = item.get("captions", [])
        if isinstance(caps, (list, tuple)):
            joined = "; ".join([c.strip() for c in caps if isinstance(c, str) and c.strip()])
            if joined:
                global_descriptions.append(joined)
        elif isinstance(caps, str) and caps.strip():
            global_descriptions.append(caps.strip())
    if len(global_descriptions) > 10:
        global_descriptions = global_descriptions[:10]
    global_block = "\n".join(f"- {desc}" for desc in global_descriptions) or "None"

    grouped_positions: Dict[str, List[str]] = {}
    for region in retrieved_data.get("local_regions", []):
        cls = region.get("class_label", "unknown")
        pos = region.get("position", "unknown")
        grouped_positions.setdefault(cls, []).append(pos)

    local_lines = []
    for cls, positions in grouped_positions.items():
        if positions:
            local_lines.append(f"{cls}: {', '.join(positions)}")
    local_block = "\n".join(local_lines) or "None"

    return template.format(
        global_descriptions=global_block,
        local_descriptions=local_block,
    )


class LoraTrainingDataBuilder:
    """Generate prompt-target pairs for LoRA fine-tuning."""

    def __init__(
        self,
        main_config_path: str = "configs/config.yaml",
        *,
        sample_count: int = 5_000,
        seed: int = 42,
        output_path: str = "lora_training/data/coco_lora_train.jsonl",
    ):
        self.main_cfg = self._load_yaml(main_config_path)
        self.sample_count = sample_count
        self.seed = seed
        self.output_path = Path(output_path)

        data_cfg = self.main_cfg.get("data_config", {})
        self.images_dir = Path(data_cfg.get("coco_images_dir", ".")).expanduser()
        annotations_path = Path(data_cfg.get("coco_annotations_path", "")).expanduser()
        if not annotations_path.exists():
            raise FileNotFoundError(f"COCO annotations not found: {annotations_path}")
        self.caption_index = CocoCaptionIndex(annotations_path)

        retrieval_cfg = dict(self.main_cfg.get("retrieval_config", {}))
        retrieval_cfg["use_patch_retrieval"] = True
        self.main_cfg["retrieval_config"] = retrieval_cfg
        self.retriever = ImageRetriever(self.main_cfg)
        self.retriever.enable_patch_retrieval()

    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _build_single_sample(self, image_id: int, rng: random.Random) -> Optional[TrainingSample]:
        img_info = self.caption_index.get_image_info(image_id)
        if not img_info:
            return None
        image_path = self.images_dir / img_info["file_name"]
        if not image_path.exists():
            logger.warning("Image missing on disk: %s", image_path)
            return None

        pil_img = load_image(str(image_path))
        try:
            retrieved = self.retriever.retrieve_with_patches(pil_img)
        except Exception as err:
            logger.warning("Patch retrieval failed for %s: %s", image_path, err)
            retrieved = {"global_descriptions": self.retriever.get_retrieved_captions(pil_img), "local_regions": []}

        prompt = build_patch_prompt(retrieved)
        caption = self.caption_index.sample_caption(image_id, rng)
        if not caption:
            return None

        metadata = {
            "image_path": str(image_path),
            "global_candidates": len(retrieved.get("global_descriptions", [])),
            "local_regions": len(retrieved.get("local_regions", [])),
        }
        return TrainingSample(
            image_id=image_id,
            prompt=prompt,
            caption=caption.strip(),
            metadata=metadata,
        )

    def build(self) -> Dict[str, Any]:
        rng = random.Random(self.seed)
        image_ids = self.caption_index.available_ids
        if not image_ids:
            raise RuntimeError("No COCO image ids found for training.")

        selected = rng.sample(image_ids, k=min(self.sample_count, len(image_ids)))
        samples: List[TrainingSample] = []

        desc = f"Building LoRA dataset ({len(selected)} images)"
        for image_id in tqdm(selected, desc=desc):
            sample = self._build_single_sample(image_id, rng)
            if sample:
                samples.append(sample)

        if not samples:
            raise RuntimeError("Failed to build any training samples.")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(sample.to_json() + "\n")

        stats = {
            "output_path": str(self.output_path),
            "num_samples": len(samples),
            "seed": self.seed,
            "images_dir": str(self.images_dir),
        }
        logger.info("LoRA training data written to %s (%d samples)", self.output_path, len(samples))
        return stats


def split_dataset(input_path: str, train_ratio: float = 0.9, seed: int = 42) -> Dict[str, str]:
    """Split a JSONL dataset into train/val shards."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with input_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    rng = random.Random(seed)
    rng.shuffle(lines)
    split_idx = int(len(lines) * train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    train_path = input_path.with_name(input_path.stem + "_train.jsonl")
    val_path = input_path.with_name(input_path.stem + "_val.jsonl")

    train_path.write_text("".join(train_lines), encoding="utf-8")
    val_path.write_text("".join(val_lines), encoding="utf-8")

    logger.info(
        "Split LoRA dataset -> train: %s (%d)  val: %s (%d)",
        train_path,
        len(train_lines),
        val_path,
        len(val_lines),
    )
    return {"train_path": str(train_path), "val_path": str(val_path)}

