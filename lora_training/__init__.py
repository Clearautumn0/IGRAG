"""
LoRA training package for IGRAG.

This namespace contains helpers to build COCO-style caption prompts,
prepare lightweight training datasets, and fine-tune the base caption
LLM using Low-Rank Adaptation (LoRA).
"""

from .data_builder import LoraTrainingDataBuilder, PROMPT_TEMPLATE_PATCH
from .lora_trainer import LoraCaptionTrainer

__all__ = [
    "LoraTrainingDataBuilder",
    "LoraCaptionTrainer",
    "PROMPT_TEMPLATE_PATCH",
]

