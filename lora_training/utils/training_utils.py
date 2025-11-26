import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PromptCaptionDataset(Dataset):
    """Dataset that stores prompt-target caption pairs in-memory."""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_source_length: int = 512,
        max_target_length: int = 64,
    ):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(data_path)

        with path.open("r", encoding="utf-8") as f:
            self.samples: List[Dict[str, Any]] = [json.loads(line) for line in f if line.strip()]

        logger.info("Loaded %d samples from %s", len(self.samples), data_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        prompt = sample["prompt"]
        caption = sample["caption"]

        model_inputs = self.tokenizer(
            prompt,
            max_length=self.max_source_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = self.tokenizer(
            text_target=caption,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": model_inputs["input_ids"][0],
            "attention_mask": model_inputs["attention_mask"][0],
            "labels": labels["input_ids"][0],
        }
        if "metadata" in sample:
            item["metadata"] = sample["metadata"]
        return item

