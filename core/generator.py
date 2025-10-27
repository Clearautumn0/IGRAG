import os
import logging
import re
from typing import List, Union

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


logger = logging.getLogger(__name__)


class CaptionGenerator:
    """Generate a consolidated caption from retrieved captions using FLAN-T5.

    Usage:
        gen = CaptionGenerator(config)
        prompt = gen.build_prompt(retrieved_captions)
        caption = gen.generate_caption(prompt)
    """

    PROMPT_TEMPLATE = (
        "基于以下相似图像的描述，请生成一个准确且全面的图像描述：\n\n"
        "相似图像描述：\n{descriptions}\n\n"
        "请综合分析以上描述，生成一个新的图像描述：\n"
        "\n(如果你更习惯英文，请用英文回答。)"
    )

    PROMPT_TEMPLATE_EN = (
        "Based on the following similar image descriptions, generate ONE concise and comprehensive caption for the query image. "
        "Do not copy the input sentences verbatim; synthesize and paraphrase.\n\n"
        "Similar image descriptions:\n{descriptions}\n\n"
        "Please synthesize the above and produce ONE new image description (one sentence):\n"
    )

    def __init__(self, config: Union[dict, str]):
        """Load FLAN-T5 model and tokenizer.

        Args:
            config: dict or path to YAML config (same structure as configs/config.yaml)
        """
        if isinstance(config, str):
            with open(config, "r") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = config

        log_level = cfg.get("log_config", {}).get("log_level", "ERROR")
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.ERROR))

        llm_path = cfg.get("model_config", {}).get("llm_model_path")
        gen_cfg = cfg.get("generation_config", {})
        self.max_length = int(gen_cfg.get("max_length", 100))
        self.num_beams = int(gen_cfg.get("num_beams", 3))
        # additional generation knobs
        self.min_length = int(gen_cfg.get("min_length", 10))
        self.repetition_penalty = float(gen_cfg.get("repetition_penalty", 1.2))

        if not llm_path:
            logger.error("LLM model path missing in config")
            raise ValueError("llm_model_path required in config")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
            # ensure pad token exists (T5 sometimes doesn't set pad_token)
            if getattr(self.tokenizer, "pad_token_id", None) is None:
                try:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                except Exception:
                    pass
            self.model = AutoModelForSeq2SeqLM.from_pretrained(llm_path).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load LLM from {llm_path}: {e}")
            raise

    def build_prompt(self, retrieved_captions: List[Union[str, dict]]) -> str:
        """Construct the prompt for the LLM from retrieved captions.

        Accepts either a list of strings or a list of dicts with a 'captions' key.
        Each entry becomes one description block.
        """
        descriptions = []
        for item in retrieved_captions:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    descriptions.append(text)
            elif isinstance(item, dict):
                # expect {'image_id': ..., 'score': ..., 'captions': [...]}
                caps = item.get("captions", []) if item is not None else []
                # join multiple captions per image with Chinese semicolon
                if isinstance(caps, (list, tuple)):
                    joined = "；".join([c.strip() for c in caps if c and isinstance(c, str)])
                    if joined:
                        descriptions.append(joined)
                elif isinstance(caps, str):
                    s = caps.strip()
                    if s:
                        descriptions.append(s)
        # limit number of descriptions to avoid overly long prompts
        if len(descriptions) > 20:
            descriptions = descriptions[:20]

        desc_block = "\n".join(descriptions)
        # prefer English prompt to avoid tokenizer <unk> issues with Chinese instructions
        prompt = self.PROMPT_TEMPLATE_EN.format(descriptions=desc_block)
        return prompt

    def _clean_output(self, text: str) -> str:
        """Basic cleaning: remove repeated whitespace, strip special tokens, ensure ending punctuation."""
        if not isinstance(text, str):
            text = str(text)
        # remove tokenizer artifacts like <pad> etc if present
        text = re.sub(r"<[^>]+>", "", text)
        # collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # ensure ending punctuation for Chinese/English
        if text and text[-1] not in set("。！？.!?"):
            text = text + "。"
        return text

    def _is_valid_caption(self, text: str) -> bool:
        """Basic validity: non-empty, longer than a threshold and contains letters/numbers/Chinese chars."""
        if not text:
            return False
        s = text.strip()
        # remove punctuation
        import re

        core = re.sub(r"[\p{P}\p{S}]", "", s) if False else re.sub(r"[\W_]+", "", s)
        return len(core) >= 4

    def generate_caption(self, prompt: str, debug: bool = False) -> str:
        """Generate a caption from a prompt using beam search.

        Returns the cleaned string.
        """
        # debug: print prompt being sent to LLM
        if debug:
            print("\n=== LLM PROMPT ===\n")
            print(prompt)
            try:
                # show tokenized input (decoded back) and token ids
                tmp = self.tokenizer(prompt, return_tensors="pt", truncation=True)
                ids = tmp.get("input_ids")
                if ids is not None:
                    print("\n--- token ids ---")
                    print(ids[0].tolist()[:200])
                    print("--- decoded tokens (first 200 tokens) ---")
                    print(self.tokenizer.decode(ids[0][:200], skip_special_tokens=False))
            except Exception as e:
                print(f"Failed to show tokenized prompt: {e}")

        # primary generation attempt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

        gen_kwargs = dict(
            max_length=self.max_length,
            min_length=self.min_length,
            num_beams=self.num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            repetition_penalty=self.repetition_penalty,
        )

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
        cleaned = self._clean_output(decoded)

        if debug:
            try:
                print("\n--- raw generated token ids ---")
                print(out[0].tolist()[:200])
                print("--- raw decoded (skip_special_tokens=False) ---")
                print(self.tokenizer.decode(out[0], skip_special_tokens=False))
            except Exception as e:
                print(f"Failed to print generated ids: {e}")

        # if output is clearly invalid or too short, retry with stronger beams and English prompt
        if not self._is_valid_caption(cleaned) or len(cleaned) <= 4:
            # retry 1: increase beams and min length
            try:
                gen_kwargs["num_beams"] = max(self.num_beams, 5)
                gen_kwargs["min_length"] = max(self.min_length, 15)
                with torch.no_grad():
                    out2 = self.model.generate(**inputs, **gen_kwargs)
                decoded2 = self.tokenizer.decode(out2[0], skip_special_tokens=True)
                cleaned2 = self._clean_output(decoded2)
                if self._is_valid_caption(cleaned2) and len(cleaned2) > len(cleaned):
                    return cleaned2
            except Exception:
                pass

            # retry 2: use English prompt as fallback
            try:
                # build english prompt using descriptions extracted from original prompt
                # attempt to extract {descriptions} block
                if "{descriptions}" in self.PROMPT_TEMPLATE_EN:
                    # naive: reuse prompt but treat as english
                    en_prompt = self.PROMPT_TEMPLATE_EN.format(descriptions=prompt)
                else:
                    en_prompt = self.PROMPT_TEMPLATE_EN.format(descriptions=prompt)

                inputs2 = self.tokenizer(en_prompt, return_tensors="pt", truncation=True).to(self.device)
                with torch.no_grad():
                    out3 = self.model.generate(**inputs2, **gen_kwargs)
                decoded3 = self.tokenizer.decode(out3[0], skip_special_tokens=True)
                cleaned3 = self._clean_output(decoded3)
                if self._is_valid_caption(cleaned3):
                    return cleaned3
            except Exception:
                pass

        return cleaned
