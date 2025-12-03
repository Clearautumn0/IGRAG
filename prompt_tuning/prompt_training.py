"""Prompt Tuning训练脚本 - 训练可学习的prompt embeddings."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from prompt_tuning.prompt_tuner import PromptTuner

# 尝试导入 BLEU 计算库
try:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    _has_pycocoevalcap = True
except ImportError:
    _has_pycocoevalcap = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _has_nltk = True
except ImportError:
    _has_nltk = False

logger = logging.getLogger(__name__)


class PromptCaptionDataset(Dataset):
    """Prompt-Caption数据集，用于Prompt Tuning训练。"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_source_length: int = 512,
        max_target_length: int = 64,
    ):
        """初始化数据集。
        
        Args:
            data_path: JSONL数据文件路径
            tokenizer: Tokenizer
            max_source_length: 最大源序列长度
            max_target_length: 最大目标序列长度
        """
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        # 加载数据
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    import json
                    sample = json.loads(line)
                    self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = sample["prompt"]
        caption = sample["caption"]
        
        # Tokenize prompt
        prompt_encoded = self.tokenizer(
            prompt,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Tokenize caption
        caption_encoded = self.tokenizer(
            caption,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": prompt_encoded["input_ids"].squeeze(0),
            "attention_mask": prompt_encoded["attention_mask"].squeeze(0),
            "labels": caption_encoded["input_ids"].squeeze(0),
        }


class PromptTuningTrainer:
    """Prompt Tuning训练器。"""
    
    def __init__(self, config_path: str = "prompt_tuning/config/prompt_tuning.yaml"):
        """初始化训练器。
        
        Args:
            config_path: 配置文件路径
        """
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        
        self.model_cfg = self.cfg.get("model", {})
        self.data_cfg = self.cfg.get("data", {})
        self.training_cfg = self.cfg.get("training", {})
        self.prompt_cfg = self.cfg.get("prompt_tuning", {})
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型和tokenizer
        self._load_model()
        
        # 创建PromptTuner
        self.prompt_tuner = PromptTuner(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_length=self.prompt_cfg.get("prompt_length", 20),
            initialization=self.prompt_cfg.get("initialization", "random"),
            device=self.device,
        )
        
        # 加载数据集
        self._load_datasets()
        
        # 初始化BLEU计算器
        self._init_bleu_metric()
        
        # 创建Trainer
        self._create_trainer()
    
    def _load_model(self):
        """加载基础模型和tokenizer。"""
        base_model_path = self.model_cfg.get("base_model_path")
        if not base_model_path:
            raise ValueError("base_model_path is required in config")
        
        logger.info(f"Loading base model from {base_model_path}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 加载模型（使用float32以避免混合精度问题）
        model_type = self.model_cfg.get("model_type", "seq2seq")
        if model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # 使用float32以避免混合精度问题
            )
        else:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # 使用float32以避免混合精度问题
            )
        
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式，因为参数将被冻结
        
        logger.info("Model and tokenizer loaded successfully")
    
    def _load_datasets(self):
        """加载训练和验证数据集。"""
        train_path = self.data_cfg.get("train_data_path")
        val_path = self.data_cfg.get("val_data_path")
        
        max_source_length = self.data_cfg.get("max_source_length", 512)
        max_target_length = self.data_cfg.get("max_target_length", 64)
        
        self.train_dataset = PromptCaptionDataset(
            train_path,
            self.tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )
        
        self.eval_dataset = PromptCaptionDataset(
            val_path,
            self.tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )
        
        logger.info(f"Loaded {len(self.train_dataset)} training samples")
        logger.info(f"Loaded {len(self.eval_dataset)} validation samples")
    
    def _init_bleu_metric(self):
        """初始化BLEU计算器。"""
        if _has_pycocoevalcap:
            self.bleu_scorer = Bleu(4)
            self.ptb_tokenizer = PTBTokenizer()
            self.bleu_method = "pycocoevalcap"
            logger.info("Using pycocoevalcap for BLEU calculation")
        elif _has_nltk:
            self.smoothing = SmoothingFunction().method1
            self.bleu_method = "nltk"
            logger.info("Using nltk for BLEU calculation")
        else:
            self.bleu_method = "simple"
            logger.warning("No BLEU library found, using simple implementation")
    
    def _compute_bleu(self, predictions: list, references: list) -> float:
        """计算BLEU-4分数。"""
        if self.bleu_method == "pycocoevalcap":
            # 格式化数据：pycocoevalcap期望的格式是 {image_id: [{"caption": "text"}, ...]}
            gts = {}
            res = {}
            for i, (pred, refs) in enumerate(zip(predictions, references)):
                # 确保refs是列表格式
                ref_list = refs if isinstance(refs, list) else [refs]
                # 转换为pycocoevalcap期望的格式：每个caption是一个字典
                gts[i] = [{"caption": ref} if isinstance(ref, str) else ref for ref in ref_list]
                res[i] = [{"caption": pred} if isinstance(pred, str) else pred]
            
            # Tokenize
            gts = self.ptb_tokenizer.tokenize(gts)
            res = self.ptb_tokenizer.tokenize(res)
            
            # 计算BLEU
            score, _ = self.bleu_scorer.compute_score(gts, res)
            return float(score[3])  # BLEU-4
        
        elif self.bleu_method == "nltk":
            scores = []
            for pred, refs in zip(predictions, references):
                ref_list = refs if isinstance(refs, list) else [refs]
                pred_tokens = pred.split()
                ref_tokens_list = [ref.split() for ref in ref_list]
                score = sentence_bleu(
                    ref_tokens_list,
                    pred_tokens,
                    smoothing_function=self.smoothing,
                )
                scores.append(score)
            return float(np.mean(scores))
        
        else:
            # 简单的BLEU实现
            def simple_bleu(pred_tokens, ref_tokens, n=4):
                if len(pred_tokens) == 0:
                    return 0.0
                
                precisions = []
                for i in range(1, n + 1):
                    pred_ngrams = {}
                    ref_ngrams = {}
                    
                    for j in range(len(pred_tokens) - i + 1):
                        ngram = tuple(pred_tokens[j:j + i])
                        pred_ngrams[ngram] = pred_ngrams.get(ngram, 0) + 1
                    
                    for j in range(len(ref_tokens) - i + 1):
                        ngram = tuple(ref_tokens[j:j + i])
                        ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1
                    
                    matches = sum(
                        min(pred_ngrams.get(ngram, 0), ref_ngrams.get(ngram, 0))
                        for ngram in pred_ngrams
                    )
                    total = sum(pred_ngrams.values())
                    precisions.append(matches / total if total > 0 else 0.0)
                
                if min(precisions) == 0:
                    return 0.0
                
                return (precisions[0] * precisions[1] * precisions[2] * precisions[3]) ** 0.25
            
            scores = []
            for pred, refs in zip(predictions, references):
                ref_list = refs if isinstance(refs, list) else [refs]
                pred_tokens = pred.split()
                ref_tokens = ref_list[0].split()
                score = simple_bleu(pred_tokens, ref_tokens)
                scores.append(score)
            return float(np.mean(scores))
    
    def _compute_metrics(self, eval_preds: EvalPrediction) -> Dict[str, float]:
        """计算评估指标。"""
        preds, labels = eval_preds
        
        # 解码预测
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # 处理preds（可能是logits或token IDs）
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)
        
        preds = preds.astype(np.int64)
        
        # 过滤无效的token IDs（超出词汇表范围）
        vocab_size = len(self.tokenizer)
        decoded_preds = []
        for i in range(len(preds)):
            sample = preds[i]
            # 过滤无效的token IDs
            valid_mask = (sample >= 0) & (sample < vocab_size)
            valid_tokens = sample[valid_mask]
            
            try:
                if len(valid_tokens) > 0:
                    decoded = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
                    decoded_preds.append(decoded)
                else:
                    decoded_preds.append("")
            except Exception as decode_err:
                logger.debug(f"Failed to decode prediction {i}: {decode_err}")
                decoded_preds.append("")
        
        # 处理labels
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        labels = labels.astype(np.int64)
        
        # 过滤labels中的无效token IDs
        decoded_labels = []
        for i in range(len(labels)):
            sample = labels[i]
            valid_mask = (sample >= 0) & (sample < vocab_size)
            valid_tokens = sample[valid_mask]
            
            try:
                if len(valid_tokens) > 0:
                    decoded = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
                    decoded_labels.append(decoded)
                else:
                    decoded_labels.append("")
            except Exception as decode_err:
                logger.debug(f"Failed to decode label {i}: {decode_err}")
                decoded_labels.append("")
        
        # 过滤空字符串
        decoded_preds = [p.strip() for p in decoded_preds if p and p.strip()]
        decoded_labels = [[l.strip()] for l in decoded_labels if l and l.strip()]
        
        # 确保长度匹配
        min_len = min(len(decoded_preds), len(decoded_labels))
        if min_len == 0:
            logger.warning("No valid predictions or labels for BLEU calculation")
            return {"bleu": 0.0}
        
        decoded_preds = decoded_preds[:min_len]
        decoded_labels = decoded_labels[:min_len]
        
        # 计算BLEU
        bleu_score = self._compute_bleu(decoded_preds, decoded_labels)
        
        return {"bleu": bleu_score}
    
    def _create_trainer(self):
        """创建Trainer。"""
        # 数据整理器
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
        )
        
        # 训练参数
        output_dir = self.training_cfg.get("output_dir", "prompt_tuning/checkpoints")
        eval_strategy = self.training_cfg.get("eval_strategy", "epoch")
        
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.training_cfg.get("num_train_epochs", 3),
            per_device_train_batch_size=self.training_cfg.get("per_device_train_batch_size", 16),
            per_device_eval_batch_size=self.training_cfg.get("per_device_eval_batch_size", 16),
            gradient_accumulation_steps=self.training_cfg.get("gradient_accumulation_steps", 8),
            learning_rate=self.training_cfg.get("learning_rate", 0.03),
            weight_decay=self.training_cfg.get("weight_decay", 0.01),
            warmup_steps=self.training_cfg.get("warmup_steps", 100),
            logging_steps=self.training_cfg.get("logging_steps", 10),
            eval_strategy=eval_strategy,
            eval_steps=self.training_cfg.get("eval_steps", 100),
            save_strategy=self.training_cfg.get("save_strategy", "epoch"),
            save_steps=self.training_cfg.get("save_steps", 100),
            load_best_model_at_end=self.training_cfg.get("load_best_model_at_end", True),
            metric_for_best_model=self.training_cfg.get("metric_for_best_model", "bleu"),
            greater_is_better=self.training_cfg.get("greater_is_better", True),
            save_total_limit=self.training_cfg.get("save_total_limit", 3),
            seed=self.training_cfg.get("seed", 42),
            fp16=self.training_cfg.get("fp16", False),  # 暂时禁用fp16以避免混合精度问题
            dataloader_num_workers=self.training_cfg.get("dataloader_num_workers", 4),
            report_to="none",  # 禁用wandb等日志记录
        )
        
        # 创建自定义Trainer，使用PromptTuner包装的模型
        class PromptTuningTrainerWrapper(Trainer):
            def __init__(self, prompt_tuner, generation_max_length=64, generation_num_beams=1, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.prompt_tuner = prompt_tuner
                self.generation_max_length = generation_max_length
                self.generation_num_beams = generation_num_beams
            
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                """计算损失，使用PromptTuner的forward方法。"""
                labels = inputs.get("labels")
                # 创建inputs的副本，移除labels用于forward
                forward_inputs = {k: v for k, v in inputs.items() if k != "labels"}
                
                # 对于seq2seq模型，需要设置decoder_input_ids和labels
                if hasattr(self.prompt_tuner.model, "decoder") and labels is not None:
                    # 为decoder准备输入（shift right）
                    # T5模型使用prepare_decoder_input_ids_from_labels
                    if hasattr(self.prompt_tuner.model, "prepare_decoder_input_ids_from_labels"):
                        decoder_input_ids = self.prompt_tuner.model.prepare_decoder_input_ids_from_labels(labels)
                    else:
                        # 手动shift right（在开头添加decoder_start_token_id）
                        decoder_start_token_id = getattr(self.prompt_tuner.model.config, "decoder_start_token_id", None)
                        if decoder_start_token_id is None:
                            decoder_start_token_id = self.prompt_tuner.tokenizer.pad_token_id
                        decoder_input_ids = torch.cat([
                            torch.full((labels.shape[0], 1), decoder_start_token_id, device=labels.device, dtype=labels.dtype),
                            labels[:, :-1]
                        ], dim=1)
                    forward_inputs["decoder_input_ids"] = decoder_input_ids
                    forward_inputs["labels"] = labels
                
                # 确保prompt_tuner在训练模式
                self.prompt_tuner.train()
                outputs = self.prompt_tuner(**forward_inputs)
                loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else None
                
                if loss is None and labels is not None:
                    # 如果没有loss，需要手动计算
                    # 对于seq2seq模型，logits在decoder输出中
                    if hasattr(outputs, "logits"):
                        logits = outputs.logits
                    elif isinstance(outputs, tuple) and len(outputs) > 0:
                        logits = outputs[0] if hasattr(outputs[0], "shape") else None
                    else:
                        logits = None
                    
                    if logits is not None:
                        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                    else:
                        # 如果无法计算loss，返回0（不应该发生）
                        loss = torch.tensor(0.0, device=labels.device, requires_grad=True)
                
                return (loss, outputs) if return_outputs else loss
            
            def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
                """重写预测步骤，在评估时生成文本。"""
                has_labels = "labels" in inputs
                inputs = self._prepare_inputs(inputs)
                
                # 计算损失
                loss = None
                if has_labels:
                    with torch.no_grad():
                        outputs = self.prompt_tuner(**inputs)
                        loss = outputs.loss if hasattr(outputs, "loss") else None
                
                # 如果是评估阶段且需要生成文本，使用generate
                if not prediction_loss_only and has_labels:
                    # 获取tokenizer
                    tokenizer = self.tokenizer if hasattr(self, 'tokenizer') else self.processing_class
                    
                    with torch.no_grad():
                        # 生成文本
                        generation_kwargs = {
                            "input_ids": inputs["input_ids"],
                            "max_length": self.generation_max_length,
                            "num_beams": self.generation_num_beams,
                            "pad_token_id": tokenizer.pad_token_id,
                            "eos_token_id": tokenizer.eos_token_id,
                            "do_sample": False,
                        }
                        
                        if "attention_mask" in inputs:
                            generation_kwargs["attention_mask"] = inputs["attention_mask"]
                        
                        generated_ids = self.prompt_tuner.generate(**generation_kwargs)
                        preds = generated_ids.cpu()
                else:
                    if prediction_loss_only:
                        return (loss, None, None) if loss is not None else super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
                    return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
                
                # 获取labels
                labels = inputs.get("labels")
                if labels is not None:
                    labels = labels.detach().cpu()
                else:
                    labels = None
                
                return (loss, preds, labels)
        
        self.trainer = PromptTuningTrainerWrapper(
            prompt_tuner=self.prompt_tuner,
            generation_max_length=self.data_cfg.get("max_target_length", 64),
            generation_num_beams=1,
            model=self.model,  # 虽然不使用，但Trainer需要
            processing_class=self.tokenizer,  # 使用processing_class替代tokenizer
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=self._compute_metrics,
        )
    
    def train(self):
        """开始训练。"""
        logger.info("Starting Prompt Tuning training...")
        
        # 打印可训练参数数量
        trainable_params = sum(p.numel() for p in self.prompt_tuner.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.prompt_tuner.parameters())
        trainable_percentage = 100 * trainable_params / all_params if all_params > 0 else 0
        
        logger.info(
            f"Trainable parameters: {trainable_params:,} / {all_params:,} "
            f"({trainable_percentage:.4f}%)"
        )
        
        # 训练
        train_result = self.trainer.train()
        
        # 保存最佳模型
        best_model_path = Path(self.training_args.output_dir) / "best_model"
        best_model_path.mkdir(parents=True, exist_ok=True)
        self.prompt_tuner.save_prompt_embeddings(best_model_path / "prompt_embeddings.pt")
        
        logger.info(f"Training completed. Best model saved to {best_model_path}")
        
        return train_result


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description="Train Prompt Tuning for IGRAG")
    parser.add_argument(
        "--config",
        type=str,
        default="prompt_tuning/config/prompt_tuning.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Start training",
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    if args.train:
        trainer = PromptTuningTrainer(config_path=args.config)
        trainer.train()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

