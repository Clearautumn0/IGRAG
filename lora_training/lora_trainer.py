import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction

from lora_training.utils import PromptCaptionDataset, load_yaml_config, set_seed

# 尝试导入 BLEU 计算库
try:
    from evaluate import load as load_metric
    _has_evaluate = True
except ImportError:
    _has_evaluate = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _has_nltk = True
except ImportError:
    _has_nltk = False

# 使用 pycocoevalcap 计算 BLEU（与评估模块一致）
try:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    _has_pycocoevalcap = True
except ImportError:
    _has_pycocoevalcap = False

logger = logging.getLogger(__name__)


class LoraCaptionTrainer:
    """End-to-end LoRA fine-tuning entrypoint."""

    def __init__(self, config_path: str = "lora_training/config/lora_config.yaml"):
        self.cfg = load_yaml_config(config_path)
        self.model_cfg = self.cfg.get("model", {})
        self.data_cfg = self.cfg.get("data", {})
        self.training_cfg = self.cfg.get("training", {})
        self.lora_cfg = self.cfg.get("lora", {})
        self.seed = int(self.training_cfg.get("seed", 42))

        set_seed(self.seed)

        self.base_model_path = self.model_cfg.get("base_model_path")
        if not self.base_model_path:
            raise ValueError("base_model_path must be specified in lora config.")

        # 将相对路径转换为绝对路径
        base_model_path_obj = Path(self.base_model_path)
        if not base_model_path_obj.is_absolute():
            # 相对于配置文件所在目录
            config_path_obj = Path(config_path)
            if config_path_obj.is_absolute():
                config_dir = config_path_obj.parent
            else:
                # 配置文件也是相对路径，需要从当前工作目录解析
                config_dir = config_path_obj.resolve().parent
            self.base_model_path = str((config_dir / base_model_path_obj).resolve())
        else:
            self.base_model_path = str(base_model_path_obj.resolve())

        # 验证路径是否存在
        if not Path(self.base_model_path).exists():
            raise FileNotFoundError(
                f"Base model path does not exist: {self.base_model_path}\n"
                f"Please check the path in {config_path}\n"
                f"Resolved from: {self.model_cfg.get('base_model_path')}"
            )
        
        logger.info(f"Using base model path: {self.base_model_path}")

        tokenizer_kwargs = {"trust_remote_code": True, "use_fast": False}
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, **tokenizer_kwargs)
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            pad_token = getattr(self.tokenizer, "eos_token", None)
            if pad_token:
                self.tokenizer.pad_token = pad_token

        self.model_type = self.model_cfg.get("type", "flan-t5").lower()
        if self.model_type != "flan-t5":
            raise NotImplementedError("LoRA trainer currently targets FLAN-T5 (seq2seq) models only.")

        model_cls = AutoModelForSeq2SeqLM
        task_type = "SEQ_2_SEQ_LM"

        logger.info("Loading base model %s as %s", self.base_model_path, task_type)
        model_kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
        self.model = model_cls.from_pretrained(self.base_model_path, **model_kwargs)
        self.model.train()

        lora_alpha = self.lora_cfg.get("lora_alpha", self.lora_cfg.get("alpha", 32))
        lora_config = LoraConfig(
            r=int(self.lora_cfg.get("r", 16)),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(self.lora_cfg.get("dropout", 0.1)),
            target_modules=self.lora_cfg.get("target_modules", ["q", "v"]),
            bias="none",
            task_type=task_type,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        max_src = int(self.data_cfg.get("max_source_length", 512))
        max_tgt = int(self.data_cfg.get("max_target_length", 64))
        train_path = self.data_cfg.get("train_path")
        val_path = self.data_cfg.get("val_path")
        if not train_path or not Path(train_path).exists():
            raise FileNotFoundError(f"Train dataset missing: {train_path}")
        if not val_path or not Path(val_path).exists():
            raise FileNotFoundError(f"Validation dataset missing: {val_path}")

        self.train_dataset = PromptCaptionDataset(train_path, self.tokenizer, max_source_length=max_src, max_target_length=max_tgt)
        self.eval_dataset = PromptCaptionDataset(val_path, self.tokenizer, max_source_length=max_src, max_target_length=max_tgt)

        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model, padding="longest")

        output_dir = self.training_cfg.get("output_dir", "lora_training/checkpoints")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 兼容不同版本的 transformers
        # 新版本 (>=4.21) 使用 eval_strategy，旧版本使用 evaluation_strategy
        eval_strategy_value = self.training_cfg.get("eval_strategy") or self.training_cfg.get("evaluation_strategy", "epoch")
        
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=float(self.training_cfg.get("num_train_epochs", 3)),
            per_device_train_batch_size=int(self.training_cfg.get("train_batch_size", 4)),
            per_device_eval_batch_size=int(self.training_cfg.get("eval_batch_size", 4)),
            gradient_accumulation_steps=int(self.training_cfg.get("gradient_accumulation_steps", 4)),
            learning_rate=float(self.training_cfg.get("learning_rate", 5e-5)),
            weight_decay=float(self.training_cfg.get("weight_decay", 0.01)),
            warmup_steps=int(self.training_cfg.get("warmup_steps", 100)),
            logging_steps=int(self.training_cfg.get("logging_steps", 50)),
            eval_strategy=eval_strategy_value,  # 新版本参数名
            save_strategy=self.training_cfg.get("save_strategy", "epoch"),
            save_total_limit=int(self.training_cfg.get("save_total_limit", 2)),
            bf16=self.training_cfg.get("bf16", False),
            fp16=self.training_cfg.get("fp16", True),
            seed=self.seed,
            report_to=self.training_cfg.get("report_to", []),
            load_best_model_at_end=True,
            metric_for_best_model="eval_bleu",
            greater_is_better=True,
        )
        
        # 保存生成参数，用于在 compute_metrics 中手动生成
        self.generation_max_length = int(self.data_cfg.get("max_target_length", 64))
        self.generation_num_beams = 1

        # 初始化 BLEU 计算器
        self._init_bleu_metric()

        def _compute_metrics(eval_preds):
            preds, labels = eval_preds
            
            # 对于 seq2seq 模型，如果设置了 predict_with_generate=True，
            # preds 应该是生成的 token IDs（2D 数组: batch_size, seq_len）
            # 如果没有设置，preds 可能是 logits（3D 数组），需要 argmax
            
            # 处理 preds：确保是 numpy 数组格式
            if isinstance(preds, tuple):
                preds = preds[0]
            
            # 转换为 numpy 数组（如果不是）
            if not isinstance(preds, np.ndarray):
                try:
                    preds = np.array(preds)
                except Exception as e:
                    logger.error(f"Failed to convert preds to numpy array: {e}, preds type: {type(preds)}")
                    return {"bleu": 0.0}
            
            # 如果 preds 是 3D（logits），需要取 argmax 得到 token IDs
            if preds.ndim == 3:
                # 形状应该是 (batch_size, seq_len, vocab_size)
                logger.warning("preds is 3D (logits), taking argmax to get token IDs. Consider setting predict_with_generate=True")
                preds = np.argmax(preds, axis=-1)
            
            # 处理嵌套列表的情况
            if preds.dtype == object:
                try:
                    # 尝试将每个元素转换为整数数组
                    processed_preds = []
                    for p in preds:
                        if isinstance(p, (list, np.ndarray)):
                            # 转换为整数数组
                            arr = np.array(p, dtype=np.int64)
                            processed_preds.append(arr)
                        elif isinstance(p, (int, np.integer)):
                            processed_preds.append(np.array([p], dtype=np.int64))
                    if processed_preds:
                        # 找到最大长度并填充
                        max_len = max(len(arr) for arr in processed_preds)
                        padded = []
                        for arr in processed_preds:
                            if len(arr) < max_len:
                                # 用 pad_token_id 填充
                                pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                                arr = np.pad(arr, (0, max_len - len(arr)), constant_values=pad_id)
                            padded.append(arr)
                        preds = np.array(padded)
                    else:
                        logger.warning("No valid token IDs in preds after processing")
                        return {"bleu": 0.0}
                except Exception as e:
                    logger.error(f"Failed to process nested preds: {e}")
                    return {"bleu": 0.0}
            
            # 确保 preds 是 2D 数组 (batch_size, seq_len)
            if preds.ndim == 1:
                preds = preds.reshape(1, -1)
            elif preds.ndim > 2:
                # 如果维度过多，取最后一个维度
                preds = preds.reshape(-1, preds.shape[-1])
            
            # 处理 labels
            if not isinstance(labels, np.ndarray):
                labels = np.array(labels)
            
            # 解码预测
            try:
                # 确保 preds 是整数类型
                preds = preds.astype(np.int64)
                
                # 过滤无效的 token IDs（超出词汇表范围）
                vocab_size = len(self.tokenizer)
                decoded_preds = []
                for i in range(len(preds)):
                    sample = preds[i]
                    # 过滤无效的 token IDs
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
            except Exception as e:
                logger.warning(f"Failed to decode predictions: {e}, preds shape: {preds.shape if hasattr(preds, 'shape') else 'unknown'}")
                # 调试：显示第一个样本的原始值
                try:
                    if len(preds) > 0:
                        sample = preds[0]
                        logger.warning(f"Sample pred (first 10 tokens): {sample[:10] if len(sample) > 10 else sample}")
                        # 尝试手动解码第一个
                        try:
                            # 过滤无效 token IDs
                            vocab_size = len(self.tokenizer)
                            valid_tokens = sample[(sample >= 0) & (sample < vocab_size)]
                            if len(valid_tokens) > 0:
                                manual_decode = self.tokenizer.decode(valid_tokens[:20], skip_special_tokens=True)
                                logger.warning(f"Manual decode of first 20 tokens: {manual_decode[:100]}")
                        except:
                            pass
                except:
                    pass
                decoded_preds = [""] * len(preds) if hasattr(preds, '__len__') else [""]
            
            # 处理 labels：将 -100 替换为 pad_token_id
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            try:
                labels = labels.astype(np.int64)
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            except Exception as e:
                logger.warning(f"Failed to decode labels: {e}")
                decoded_labels = [""] * len(labels) if hasattr(labels, '__len__') else [""]

            # 过滤空字符串，但保留所有非空结果
            decoded_preds = [p.strip() for p in decoded_preds if p and p.strip()]
            decoded_labels = [[l.strip()] for l in decoded_labels if l and l.strip()]

            # 确保长度匹配
            min_len = min(len(decoded_preds), len(decoded_labels))
            if min_len == 0:
                logger.warning(
                    f"No valid predictions or labels for BLEU calculation. "
                    f"Preds: {len(decoded_preds)}, Labels: {len(decoded_labels)}"
                )
                # 调试：显示原始解码结果的前几个
                try:
                    raw_preds = self.tokenizer.batch_decode(preds[:3], skip_special_tokens=True) if len(preds) > 0 else []
                    raw_labels = self.tokenizer.batch_decode(labels[:3], skip_special_tokens=True) if len(labels) > 0 else []
                    logger.warning(f"First 3 raw predictions (before filtering): {raw_preds}")
                    logger.warning(f"First 3 raw labels (before filtering): {raw_labels}")
                except:
                    pass
                return {"bleu": 0.0}
            
            decoded_preds = decoded_preds[:min_len]
            decoded_labels = decoded_labels[:min_len]

            # 调试信息：显示前几个样本（仅在 DEBUG 级别）
            if logger.level <= logging.DEBUG and len(decoded_preds) > 0:
                logger.debug(f"Sample prediction: {decoded_preds[0][:100]}")
                logger.debug(f"Sample reference: {decoded_labels[0][0][:100] if decoded_labels[0] else 'empty'}")
                logger.debug(f"Total samples for BLEU: {min_len}")
            elif len(decoded_preds) > 0:
                # 即使不是 DEBUG 级别，也记录前几个样本（用于诊断）
                logger.info(f"Sample prediction: {decoded_preds[0][:100]}")
                logger.info(f"Sample reference: {decoded_labels[0][0][:100] if decoded_labels[0] else 'empty'}")

            bleu_score = self._compute_bleu(decoded_preds, decoded_labels)
            if logger.level <= logging.DEBUG:
                logger.debug(f"Computed BLEU score: {bleu_score}")
            return {"bleu": bleu_score}

        # 创建自定义 Trainer，支持 seq2seq 模型的文本生成
        class Seq2SeqTrainer(Trainer):
            """自定义 Trainer，在评估时生成文本而不是使用 logits"""
            
            def __init__(self, *args, generation_max_length=64, generation_num_beams=1, **kwargs):
                super().__init__(*args, **kwargs)
                self.generation_max_length = generation_max_length
                self.generation_num_beams = generation_num_beams
                # 设置 processing_class 以避免弃用警告
                if hasattr(self, 'tokenizer') and not hasattr(self, 'processing_class'):
                    self.processing_class = self.tokenizer
            
            def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
                """重写预测步骤，对于评估生成文本"""
                has_labels = "labels" in inputs
                inputs = self._prepare_inputs(inputs)
                
                # 计算损失（如果需要）
                loss = None
                if has_labels:
                    with torch.no_grad():
                        outputs = model(**inputs)
                        loss = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else None
                
                # 如果是评估阶段且需要生成文本，使用 model.generate()
                if not prediction_loss_only and has_labels:
                    # 获取实际的模型（处理 DataParallel 包装）
                    actual_model = model.module if hasattr(model, 'module') else model
                    
                    # 设置模型为评估模式
                    actual_model.eval()
                    with torch.no_grad():
                        # 准备生成参数
                        input_ids = inputs["input_ids"]
                        attention_mask = inputs.get("attention_mask")
                        
                        # 生成文本（PEFT 模型需要所有参数都作为关键字参数）
                        generation_kwargs = {
                            "input_ids": input_ids,
                            "max_length": self.generation_max_length,
                            "num_beams": self.generation_num_beams,
                            "pad_token_id": self.tokenizer.pad_token_id,
                            "eos_token_id": self.tokenizer.eos_token_id,
                            "do_sample": False,  # 使用贪心解码
                        }
                        
                        # 如果有 attention_mask，添加到参数中
                        if attention_mask is not None:
                            generation_kwargs["attention_mask"] = attention_mask
                        
                        # 调用 generate（所有参数都使用关键字参数）
                        generated_ids = actual_model.generate(**generation_kwargs)
                    # 对于编码器-解码器模型（如 FLAN-T5），生成的序列就是完整的输出
                    # 不需要去掉输入部分，因为 generate() 已经处理了
                    # 保持为 torch tensor，不要转换为 numpy（Trainer 需要 tensor）
                    preds = generated_ids.cpu()
                else:
                    # 如果只需要损失，使用默认行为
                    if prediction_loss_only:
                        return (loss, None, None) if loss is not None else super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
                    # 否则使用默认行为（训练阶段）
                    return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
                
                # 获取 labels
                labels = inputs.get("labels")
                if labels is not None:
                    labels = labels.detach().cpu()
                else:
                    labels = None
                
                # 返回损失、预测和标签（保持为 torch tensor）
                # Trainer 会在内部处理并转换为 numpy 用于 compute_metrics
                return (loss, preds, labels)
        
        # 使用 processing_class 替代已弃用的 tokenizer 参数
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            processing_class=self.tokenizer,  # 使用 processing_class 替代 tokenizer
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=_compute_metrics,
            generation_max_length=self.generation_max_length,
            generation_num_beams=self.generation_num_beams,
        )

    def _init_bleu_metric(self):
        """初始化 BLEU 计算器，优先使用 pycocoevalcap（与评估模块一致）"""
        self.bleu_metric = None
        self.bleu_method = None

        # 方法1: 优先使用 pycocoevalcap（与项目评估模块一致）
        if _has_pycocoevalcap:
            try:
                self.bleu_scorer = Bleu(4)  # 计算到 BLEU-4
                self.ptb_tokenizer = PTBTokenizer()  # 使用不同的变量名，避免覆盖 self.tokenizer
                self.bleu_method = "pycocoevalcap"
                logger.info("Using pycocoevalcap for BLEU calculation (consistent with evaluation module)")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize pycocoevalcap: {e}")

        # 方法2: 尝试使用 evaluate 库
        if _has_evaluate:
            try:
                # 尝试不同的指标名称
                for metric_name in ["sacrebleu", "bleu"]:
                    try:
                        self.bleu_metric = load_metric(metric_name)
                        self.bleu_method = "evaluate"
                        logger.info(f"Using evaluate library with metric: {metric_name}")
                        return
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(f"Failed to load BLEU from evaluate library: {e}")

        # 方法3: 使用 nltk
        if _has_nltk:
            self.bleu_method = "nltk"
            self.smoothing = SmoothingFunction().method1
            logger.info("Using NLTK for BLEU calculation")
            return

        # 方法4: 简单的 BLEU 实现（fallback）
        self.bleu_method = "simple"
        logger.warning(
            "Using simple BLEU implementation. "
            "For better accuracy consistent with evaluation module, install: pip install pycocoevalcap"
        )

    def _compute_bleu(self, predictions: list, references: list) -> float:
        """计算 BLEU-4 分数（使用 pycocoevalcap，与评估模块一致）"""
        # 方法1: 使用 pycocoevalcap（推荐，与评估模块一致）
        if self.bleu_method == "pycocoevalcap":
            try:
                # pycocoevalcap 需要特定格式：{img_id: [{"caption": "..."}]}
                # 对于训练，我们使用索引作为 img_id
                gts = {}
                res = {}
                for idx, (pred, ref_list) in enumerate(zip(predictions, references)):
                    img_id = idx
                    # ref_list 是列表的列表，取第一个作为参考
                    ref = ref_list[0] if isinstance(ref_list, list) and len(ref_list) > 0 else ref_list
                    gts[img_id] = [{"caption": ref}]
                    res[img_id] = [{"caption": pred}]
                
                # 使用 PTB tokenizer 进行分词
                gts = self.ptb_tokenizer.tokenize(gts)
                res = self.ptb_tokenizer.tokenize(res)
                
                # 计算 BLEU 分数（返回 BLEU-1 到 BLEU-4）
                score, _ = self.bleu_scorer.compute_score(gts, res)
                # score[3] 是 BLEU-4
                bleu4 = float(score[3]) if len(score) > 3 else 0.0
                return bleu4
            except Exception as e:
                logger.warning(f"BLEU computation failed with pycocoevalcap: {e}, falling back")
                self.bleu_method = "evaluate" if _has_evaluate else ("nltk" if _has_nltk else "simple")
        
        if self.bleu_method == "evaluate":
            try:
                if "sacrebleu" in str(type(self.bleu_metric)).lower():
                    # sacrebleu 需要不同的格式
                    result = self.bleu_metric.compute(
                        predictions=predictions,
                        references=references
                    )
                    return result.get("score", 0.0) / 100.0  # sacrebleu 返回 0-100
                else:
                    result = self.bleu_metric.compute(
                        predictions=predictions,
                        references=references
                    )
                    return result.get("bleu", 0.0)
            except Exception as e:
                logger.warning(f"BLEU computation failed with evaluate: {e}, falling back")
                self.bleu_method = "nltk" if _has_nltk else "simple"

        if self.bleu_method == "nltk":
            try:
                scores = []
                for pred, ref_list in zip(predictions, references):
                    # ref_list 是列表的列表，取第一个
                    ref = ref_list[0] if isinstance(ref_list, list) and len(ref_list) > 0 else ref_list
                    pred_tokens = pred.split()
                    ref_tokens = ref.split()
                    score = sentence_bleu(
                        [ref_tokens],
                        pred_tokens,
                        smoothing_function=self.smoothing
                    )
                    scores.append(score)
                return float(np.mean(scores))
            except Exception as e:
                logger.warning(f"BLEU computation failed with nltk: {e}, falling back")
                self.bleu_method = "simple"

        # 简单的 BLEU-4 实现（fallback）
        def simple_bleu(pred_tokens, ref_tokens, n=4):
            """简单的 BLEU-n 实现"""
            if len(pred_tokens) == 0:
                return 0.0

            precisions = []
            for i in range(1, n + 1):
                pred_ngrams = {}
                ref_ngrams = {}

                # 计算预测的 n-grams
                for j in range(len(pred_tokens) - i + 1):
                    ngram = tuple(pred_tokens[j:j + i])
                    pred_ngrams[ngram] = pred_ngrams.get(ngram, 0) + 1

                # 计算参考的 n-grams
                for j in range(len(ref_tokens) - i + 1):
                    ngram = tuple(ref_tokens[j:j + i])
                    ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1

                # 计算精确度
                matches = sum(
                    min(pred_ngrams.get(ngram, 0), ref_ngrams.get(ngram, 0))
                    for ngram in pred_ngrams
                )
                total = sum(pred_ngrams.values())
                precisions.append(matches / total if total > 0 else 0.0)

            # 计算几何平均
            if min(precisions) == 0:
                return 0.0

            # 长度惩罚
            bp = min(1.0, np.exp(1 - len(ref_tokens) / len(pred_tokens))) if len(pred_tokens) > 0 else 0.0

            return bp * (np.prod(precisions) ** (1.0 / n))

        scores = []
        for pred, ref_list in zip(predictions, references):
            ref = ref_list[0] if isinstance(ref_list, list) and len(ref_list) > 0 else ref_list
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            score = simple_bleu(pred_tokens, ref_tokens)
            scores.append(score)

        return float(np.mean(scores))

    def train(self):
        logger.info("Starting LoRA training for %s", self.base_model_path)
        train_result = self.trainer.train()
        self.trainer.save_model()
        self.trainer.log_metrics("train", train_result.metrics)
        self.trainer.save_metrics("train", train_result.metrics)
        self.trainer.save_state()
        return train_result

    def evaluate(self):
        logger.info("Running evaluation on validation split")
        metrics = self.trainer.evaluate()
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        return metrics

