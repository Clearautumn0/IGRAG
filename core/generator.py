import os
import logging
import re
from typing import Dict, List, Optional, Union

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM


logger = logging.getLogger(__name__)


class CaptionGenerator:
    """Generate a consolidated caption from retrieved captions using FLAN-T5.

    Usage:
        gen = CaptionGenerator(config)
        prompt = gen.build_prompt(retrieved_captions)
        caption = gen.generate_caption(prompt)
    """
    PROMPT_TEMPLATE_EN = (
        "Based on the following similar image descriptions, generate ONE concise and comprehensive caption for the query image. "
        "Do not copy the input sentences verbatim; synthesize and paraphrase.\n\n"
        "Similar image descriptions:\n{descriptions}\n\n"
        "Please describe the image in one sentence based on the overall description and the number of entities in the image (do not directly copy the description above).:\n"
    )
    
    PROMPT_TEMPLATE_PATCH = (
        "Generate a concise image caption in COCO dataset style.\n"
        "Requirements:\n"
        "- One sentence only\n" 
        "- 8-12 words maximum\n"
        "- Describe only what is visually apparent\n"
        "- Use simple, factual language\n"
        "- Avoid inferences or background information\n\n"
        
        "Similar images show:\n"
        "{global_descriptions}\n\n"
        
        "Detected objects:\n"
        "{local_descriptions}\n\n"
        
        "Caption:"
    )

    def __init__(self, config: Union[dict, str]):
        """Load Qwen causal model and tokenizer.

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
        self.repetition_penalty = float(gen_cfg.get("repetition_penalty", 1.3))
        self.temperature = float(gen_cfg.get("temperature", 0.7))
        self.top_p = float(gen_cfg.get("top_p", 0.9))
        
        # 保存配置用于提示词构建
        self.config = cfg
        patch_config = cfg.get("patch_config", {})
        self.global_prompt_section = patch_config.get("global_prompt_section", "Overall similar image descriptions:")
        self.local_prompt_section = patch_config.get("local_prompt_section", "Key local regions with similar descriptions:")
        # self.final_instruction = patch_config.get("final_instruction", 
        #     "Please synthesize all the above descriptions, paying special attention to the consistency of local details, and generate a new, accurate and comprehensive image description.")

        if not llm_path:
            logger.error("LLM model path missing in config")
            raise ValueError("llm_model_path required in config")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.system_prompt = cfg.get(
            "model_config", {}
        ).get(
            "system_prompt",
            "You are a helpful assistant that specializes in writing concise, comprehensive image captions.",
        )

        try:
            tokenizer_kwargs = dict(trust_remote_code=True, use_fast=False)
            self.tokenizer = AutoTokenizer.from_pretrained(llm_path, **tokenizer_kwargs)
            if getattr(self.tokenizer, "pad_token_id", None) is None:
                pad_token = getattr(self.tokenizer, "eos_token", None)
                if pad_token:
                    self.tokenizer.pad_token = pad_token
            model_kwargs = {"trust_remote_code": True}
            if self.device.type == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(llm_path, **model_kwargs)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load LLM from {llm_path}: {e}")
            raise

    def build_prompt(self, retrieved_captions: Union[List[Union[str, dict]], dict]) -> str:
        """Construct the prompt for the LLM from retrieved captions.
        
        支持两种输入格式：
        1. 传统格式：List[dict] 或 List[str] - 仅全局描述
        2. 分块格式：dict - 包含全局描述和局部区域描述
        
        Args:
            retrieved_captions: 
                - 传统格式: List[dict] 每个dict包含 'captions' 键
                - 分块格式: {"global_descriptions": [...], "local_regions": [...]}
        """
        # 检查是否为分块格式
        if isinstance(retrieved_captions, dict) and "global_descriptions" in retrieved_captions:
            return self._build_patch_prompt(retrieved_captions)
        else:
            return self._build_standard_prompt(retrieved_captions)
    
    def _build_standard_prompt(self, retrieved_captions: List[Union[str, dict]]) -> str:
        """构建标准提示词（仅全局描述）。"""
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
    
    def _build_patch_prompt(self, retrieved_data: dict) -> str:
        """构建包含全局和局部描述的分层提示词。
        
        Args:
            retrieved_data: {
                "global_descriptions": [{"image_id": ..., "score": ..., "captions": [...]}],
                "local_regions": [
                    {
                        "bbox": [x1, y1, x2, y2],
                        "class_label": "类别名称",
                        "confidence": 置信度,
                        "descriptions": [描述列表]
                    }
                ]
            }
        """
        # 处理全局描述
        global_descriptions = []
        for item in retrieved_data.get("global_descriptions", []):
            caps = item.get("captions", [])
            if isinstance(caps, (list, tuple)):
                joined = "; ".join([c.strip() for c in caps if c and isinstance(c, str)])
                if joined:
                    global_descriptions.append(joined)
            elif isinstance(caps, str) and caps.strip():
                global_descriptions.append(caps.strip())
        
        # 限制全局描述数量
        if len(global_descriptions) > 10:
            global_descriptions = global_descriptions[:10]
        
        global_block = "\n".join([f"- {desc}" for desc in global_descriptions]) if global_descriptions else "None"
        
        # 处理局部描述：按类别分组，结构化呈现以方便模型推断实例个数与复数
        local_regions = retrieved_data.get("local_regions", [])
        grouped = self.group_local_descriptions_by_class(local_regions)

        local_sections = []
        local_entrys = []
        for class_label, descs in grouped.items():
            if not descs:
                continue
            # 只输出类别及实例数量，例如 "bird:3"
            # count = len(descs)
            # count = sum(1 for region in local_regions if region['class_label'] == class_label)
            # local_sections.append(f"{class_label}:{count}")

            # 生成结构化分组块
            section_lines = [f"OBJECT TYPE: {class_label}"]
            for i, d in enumerate(descs, start=1):
                section_lines.append(f"- Instance {i}: {d}")
            local_sections.append("\n".join(section_lines))
        # 使用单行换行连接每个类别:数量，示例："bird:3\ndog:2"
        local_block = "\n".join(local_sections) if local_sections else "None"
        
        # 构建完整提示词
        prompt = self.PROMPT_TEMPLATE_PATCH.format(
            global_prompt_section=self.global_prompt_section,
            global_descriptions=global_block,
            local_prompt_section=self.local_prompt_section,
            local_descriptions=local_block
        )
        
        # 检查提示词长度，如果过长则截断
        max_prompt_length = 2000  # 字符数限制
        if len(prompt) > max_prompt_length:
            logger.warning(f"Prompt too long ({len(prompt)} chars), truncating...")
            # 简单截断到最大长度
            prompt = prompt[:max_prompt_length] + "..."
        
        return prompt

    def group_local_descriptions_by_class(self, local_regions: List[dict]) -> dict:
        """按检测类别分组局部描述，并去重、规范化。

        返回形如：
        {
            "bird": ["desc1", "desc2", ...],
            "person": ["descA", ...]
        }
        """
        grouped: dict = {}

        def _normalize(text: str) -> str:
            # 基本规范化：小写、去多余空白与尾部标点
            s = (text or "").strip()
            s = re.sub(r"\s+", " ", s).strip().lower()
            s = s.rstrip(".!")
            return s

        def _token_set(s: str) -> set:
            # 简单分词并移除非字母数字字符
            s = re.sub(r"[^a-z0-9\s]", " ", s)
            toks = [t for t in s.split() if t]
            return set(toks)

        def _similar(a: str, b: str, threshold: float = 0.78) -> bool:
            # 基于 Jaccard 相似度的简单模糊合并
            A, B = _token_set(_normalize(a)), _token_set(_normalize(b))
            if not A or not B:
                return False
            inter = len(A & B)
            union = len(A | B)
            jacc = inter / union if union else 0.0
            # 若短句高度包含于长句也视为相似（overlap coefficient）
            overlap = inter / min(len(A), len(B)) if min(len(A), len(B)) else 0.0
            return jacc >= threshold or overlap >= 0.85

        # 维护插入顺序：按出现顺序组织类别
        class_order = []
        # 每个类别维护代表描述列表，用于模糊合并
        class_reps: dict = {}
        for idx, region in enumerate(local_regions):
            class_label = region.get("class_label") or f"object_{idx+1}"
            # 统一类别大小写
            class_key = str(class_label).strip()
            if class_key not in grouped:
                grouped[class_key] = []
                class_order.append(class_key)
                class_reps[class_key] = []

            descs = region.get("descriptions", [])
            # 当前我们保证每个区域只有一条描述，但为兼容性保留遍历
            for d in descs if isinstance(descs, list) else [descs]:
                if not isinstance(d, str):
                    d = str(d)
                norm = _normalize(d)
                if not norm:
                    continue
                # 与已有代表进行相似度比较，极相似则合并：
                reps = class_reps[class_key]
                is_similar = False
                for i, rep in enumerate(reps):
                    if _similar(d, rep):
                        # 选择信息量更大的那条（更长的句子）作为代表
                        better = d if len(d) > len(rep) else rep
                        reps[i] = better
                        # 同步更新 grouped 中对应代表（最后一个元素即最新代表）
                        # 简化处理：若存在旧代表，替换第一次出现的旧代表
                        for j, existed in enumerate(grouped[class_key]):
                            if _similar(existed, rep):
                                grouped[class_key][j] = better.strip()
                                break
                        is_similar = True
                        break
                if not is_similar:
                    reps.append(d)
                    grouped[class_key].append(d.strip())
    
        # 保持类别顺序
        ordered_grouped = {k: grouped[k] for k in class_order}
        return ordered_grouped

    def _extract_caption_segment(self, text: str) -> str:
        """Extract the portion of the model output that follows the caption marker."""
        if not isinstance(text, str):
            text = str(text)

        markers = ["Caption:", "caption:", "说明:", "回答:", "答复:", "caption：", "Caption："]
        for marker in markers:
            idx = text.rfind(marker)
            if idx != -1:
                text = text[idx + len(marker) :]
                break

        # remove assistant or other speaker prefixes
        text = re.sub(r"^(?:\s*(Assistant|assistant|助手)[：:]\s*)", "", text)

        # truncate when another speaker prompt appears again
        stop_markers = ["\nHuman:", "\nUser:", "\nAssistant:", "\nSystem:", " Human:", " User:", " Assistant:", " System:"]
        for marker in stop_markers:
            idx = text.find(marker)
            if idx != -1 and idx > 0:
                text = text[:idx]
                break

        # strip stray quotes
        text = text.strip().strip('"').strip()
        return text

    def _clean_output(self, text: str) -> str:
        """Basic cleaning: remove repeated whitespace, strip special tokens, ensure ending punctuation."""
        if not isinstance(text, str):
            text = str(text)
        text = self._extract_caption_segment(text)
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

            except Exception as e:
                print(f"Failed to show tokenized prompt: {e}")

        # primary generation attempt
        def _build_chat_inputs(user_prompt: str):
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            try:
                chat_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except AttributeError:
                chat_text = user_prompt
            tokenized = self.tokenizer(
                chat_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            return tokenized

        def _run_generation(user_prompt: str, override_kwargs: Optional[Dict] = None) -> str:
            chat_inputs = _build_chat_inputs(user_prompt)
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id
            if pad_token_id is None:
                pad_token_id = getattr(self.model.config, "pad_token_id", None) or getattr(
                    self.model.config, "eos_token_id", None
                )
            if eos_token_id is None:
                eos_token_id = getattr(self.model.config, "eos_token_id", None)

            prompt_len = chat_inputs["input_ids"].shape[1]
            total_min_length = min(prompt_len + self.min_length, prompt_len + self.max_length)

            gen_kwargs = dict(
                max_new_tokens=self.max_length,
                min_length=total_min_length,
                num_beams=self.num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.5,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True,
            )
            if override_kwargs:
                overrides = dict(override_kwargs)
                min_length_offset = overrides.pop("min_length_offset", None)
                if min_length_offset is not None:
                    gen_kwargs["min_length"] = min(
                        prompt_len + min_length_offset, prompt_len + self.max_length
                    )
                gen_kwargs.update(overrides)
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=chat_inputs["input_ids"],
                    attention_mask=chat_inputs.get("attention_mask"),
                    **gen_kwargs,
                )
            generated_sequence = output[0]
            if generated_sequence.shape[0] > prompt_len:
                generated_sequence = generated_sequence[prompt_len:]
            return self.tokenizer.decode(generated_sequence, skip_special_tokens=True)

        decoded = _run_generation(prompt)
        cleaned = self._clean_output(decoded)

        if not self._is_valid_caption(cleaned) or len(cleaned) <= 4:
            # retry 1: increase beams and min length
            try:
                alt_kwargs = {
                    "num_beams": max(self.num_beams, 5),
                    "min_length_offset": max(self.min_length, 15),
                    "temperature": min(self.temperature, 0.7),
                    "top_p": max(self.top_p, 0.9),
                }
                decoded2 = _run_generation(prompt, alt_kwargs)
                cleaned2 = self._clean_output(decoded2)
                if self._is_valid_caption(cleaned2) and len(cleaned2) > len(cleaned):
                    return cleaned2
            except Exception:
                pass

            # retry 2: use English prompt as fallback
            try:
                en_prompt = self.PROMPT_TEMPLATE_EN.format(descriptions=prompt)
                decoded3 = _run_generation(en_prompt)
                cleaned3 = self._clean_output(decoded3)
                if self._is_valid_caption(cleaned3):
                    return cleaned3
            except Exception:
                pass

        return cleaned
