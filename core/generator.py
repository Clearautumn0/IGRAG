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
        "Based on the following information, generate an accurate and comprehensive image description:\n\n"
        "Overall similar image descriptions:\n{global_descriptions}\n\n"
        "Key local region descriptions:\n{patch_descriptions_block}\n\n"
        "Please synthesize the global and local information and generate a new image description:\n"
    )

    PROMPT_TEMPLATE_EN = (
        "You will receive both global and local observations about the query image. Prioritize the overall/global description to form a coherent scene understanding, and only use local observations to refine the quantity, color, details of small objects, and other subtle attributes. \n\n"
        "Global information:Overall similar image descriptions (most important):\n{global_descriptions}\n\n"
        "Local information:Key local observations (Use to refine details and quantity information.):\n{local_details}\n\n"
        "Requirement:Please pay special attention to the quantity information provided in the local parts. For example, if there are three sentences in the local information describing the existence of a bird, then the global information should be three birds."
        "Please synthesize the global and local information and produce ONE concise, natural-sounding image description (one sentences). Do NOT include any tags or list the observations verbatim; instead, integrate them into a single descriptive caption.Ultimately, form a sentence that describes a single picture.\n"
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

    def build_prompt(self, retrieved_captions: Union[List[Union[str, dict]], dict]) -> str:
        """Construct the prompt for the LLM from retrieved captions.

        Supports two input formats:
          - legacy: list[str] or list[dict] (each dict: {'image_id', 'score', 'captions'})
          - new: dict with keys 'global' and 'patches' returned by ImageRetriever.get_retrieved_captions

        The resulting prompt will be fully English. Local (patch) descriptions are limited by
        `generation_config.max_patch_descriptions` in the config to avoid an overly long prompt.
        Optionally filters patch descriptions by a similarity threshold `generation_config.patch_similarity_threshold`.
        """
        # load config-driven limits (set in __init__)
        max_patch = int(getattr(self, "max_patch_descriptions", 4))
        patch_similarity_threshold = float(getattr(self, "patch_similarity_threshold", 0.0))

        global_descriptions_list = []
        patch_descriptions_map = {}

        # normalize input
        if isinstance(retrieved_captions, dict):
            # expected shape: {'global': [{image_id, score, captions}], 'patches': {coord_key: [captions]}}
            global_entries = retrieved_captions.get("global", [])
            for item in global_entries:
                if isinstance(item, dict):
                    caps = item.get("captions", [])
                    if isinstance(caps, (list, tuple)):
                        joined = "; ".join([c.strip() for c in caps if c and isinstance(c, str)])
                        if joined:
                            global_descriptions_list.append(joined)
                    elif isinstance(caps, str):
                        s = caps.strip()
                        if s:
                            global_descriptions_list.append(s)
                elif isinstance(item, str):
                    s = item.strip()
                    if s:
                        global_descriptions_list.append(s)

                patches = retrieved_captions.get("patches", {})
                # patches is expected mapping like 'x,y'-> [captions or {caption,score}]
                # collect local details across patches (no region labels)
                local_details = []
                seen_local = set()
                for k in sorted(patches.keys()):
                    vals = patches[k]
                    if not vals:
                        continue
                    for v in vals:
                        cap_text = None
                        score = 0.0
                        if isinstance(v, dict):
                            cap_text = v.get("caption") or v.get("captions") or v.get("text")
                            try:
                                score = float(v.get("score", 0.0))
                            except Exception:
                                score = 0.0
                        elif isinstance(v, str):
                            cap_text = v
                        if not cap_text:
                            continue
                        cap_text = cap_text.strip()
                        # filter by similarity threshold if set
                        if score < patch_similarity_threshold:
                            continue
                        if cap_text in seen_local:
                            continue
                        seen_local.add(cap_text)
                        local_details.append(cap_text)
                # limit local details to max_patch entries
                if len(local_details) > max_patch:
                    local_details = local_details[:max_patch]
                # build a short local details block without region tags
                patch_descriptions_map = {"local_summaries": local_details}

        else:
            # legacy list input
            legacy_list = retrieved_captions if isinstance(retrieved_captions, list) else []
            for item in legacy_list:
                if isinstance(item, str):
                    s = item.strip()
                    if s:
                        global_descriptions_list.append(s)
                elif isinstance(item, dict):
                    caps = item.get("captions", [])
                    if isinstance(caps, (list, tuple)):
                        joined = "; ".join([c.strip() for c in caps if c and isinstance(c, str)])
                        if joined:
                            global_descriptions_list.append(joined)
                    elif isinstance(caps, str):
                        s = caps.strip()
                        if s:
                            global_descriptions_list.append(s)

        # limit overall descriptions to avoid huge prompt
        if len(global_descriptions_list) > 20:
            global_descriptions_list = global_descriptions_list[:20]

        # Build patch description block limited to max_patch entries
        global_block = "\n".join(global_descriptions_list)
        local_block = "\n".join(patch_descriptions_map.get("local_summaries", [])) if patch_descriptions_map.get("local_summaries") else "None"

        # prefer English template and instruct model to integrate local details without region tags
        prompt = self.PROMPT_TEMPLATE_EN.format(global_descriptions=global_block, local_details=local_block)
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

        # if debug:
        #     try:
        #         print("\n--- raw generated token ids ---")
        #         print(out[0].tolist()[:200])
        #         print("--- raw decoded (skip_special_tokens=False) ---")
        #         print(self.tokenizer.decode(out[0], skip_special_tokens=False))
        #     except Exception as e:
        #         print(f"Failed to print generated ids: {e}")

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
