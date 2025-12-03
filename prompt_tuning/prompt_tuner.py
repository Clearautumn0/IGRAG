"""Prompt Tuning核心类 - 管理可学习的prompt embeddings."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class PromptTuner(nn.Module):
    """Prompt Tuning核心类，管理可学习的prompt embeddings。
    
    该类在输入序列前添加可学习的prompt tokens，这些tokens的embedding是可训练的。
    基础模型和LoRA权重都冻结，只训练prompt embeddings。
    
    Attributes:
        prompt_length: Prompt tokens的数量
        embedding_dim: Embedding维度（从模型获取）
        prompt_embeddings: 可学习的prompt embeddings (prompt_length, embedding_dim)
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompt_length: int = 20,
        initialization: str = "random",
        device: Optional[torch.device] = None,
    ):
        """初始化Prompt Tuner。
        
        Args:
            model: 预训练模型（将被冻结）
            tokenizer: 对应的tokenizer
            prompt_length: Prompt tokens的数量
            initialization: 初始化方式 ("random" 或 "text")
            device: 设备（如果为None，使用模型的设备）
        """
        super().__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.initialization = initialization
        
        # 获取设备
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device
        
        # 获取embedding维度
        # 对于seq2seq模型（如FLAN-T5），使用encoder的embedding
        # 对于causal模型（如Qwen），使用模型的embedding
        if hasattr(model, "get_input_embeddings"):
            embedding_layer = model.get_input_embeddings()
            self.embedding_dim = embedding_layer.embedding_dim
        elif hasattr(model, "shared"):  # T5模型
            self.embedding_dim = model.shared.embedding_dim
        else:
            raise ValueError("Cannot find embedding layer in model")
        
        # 创建可学习的prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            self._initialize_embeddings(),
            requires_grad=True
        )
        
        # 冻结基础模型的所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        logger.info(
            f"Initialized PromptTuner: prompt_length={prompt_length}, "
            f"embedding_dim={self.embedding_dim}, initialization={initialization}"
        )
    
    def _initialize_embeddings(self) -> torch.Tensor:
        """初始化prompt embeddings。
        
        Returns:
            初始化的embedding tensor (prompt_length, embedding_dim)
        """
        if self.initialization == "random":
            # 随机初始化（使用正态分布）
            embeddings = torch.randn(
                self.prompt_length,
                self.embedding_dim,
                device=self.device,
                dtype=torch.float32
            )
            # 缩放初始化（与模型embedding的scale一致）
            embeddings = embeddings * 0.02
        elif self.initialization == "text":
            # 使用文本初始化（从tokenizer的vocab中采样）
            # 这里使用一些常见的prompt words
            prompt_text = "Generate a concise and accurate image caption in COCO style."
            prompt_ids = self.tokenizer.encode(
                prompt_text,
                add_special_tokens=False,
                max_length=self.prompt_length,
                truncation=True,
            )
            
            # 获取这些token的embeddings
            if hasattr(self.model, "get_input_embeddings"):
                embedding_layer = self.model.get_input_embeddings()
            elif hasattr(self.model, "shared"):
                embedding_layer = self.model.shared
            else:
                raise ValueError("Cannot find embedding layer")
            
            # 获取token embeddings
            with torch.no_grad():
                token_embeddings = embedding_layer(
                    torch.tensor(prompt_ids, device=self.device)
                )
            
            # 如果prompt_length > len(prompt_ids)，用最后一个token的embedding填充
            if len(token_embeddings) < self.prompt_length:
                padding = token_embeddings[-1:].repeat(
                    self.prompt_length - len(token_embeddings), 1
                )
                embeddings = torch.cat([token_embeddings, padding], dim=0)
            else:
                embeddings = token_embeddings[:self.prompt_length]
        else:
            raise ValueError(f"Unknown initialization: {self.initialization}")
        
        return embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """前向传播：在输入前添加prompt embeddings。
        
        Args:
            input_ids: 输入token IDs (batch_size, seq_len)
            attention_mask: 注意力掩码 (batch_size, seq_len)
            **kwargs: 其他模型参数
        
        Returns:
            模型输出
        """
        batch_size = input_ids.shape[0]
        
        # 扩展prompt embeddings到batch size
        # prompt_embeddings: (prompt_length, embedding_dim)
        # batch_prompt_embeddings: (batch_size, prompt_length, embedding_dim)
        batch_prompt_embeddings = self.prompt_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # 获取输入文本的embeddings
        if hasattr(self.model, "get_input_embeddings"):
            embedding_layer = self.model.get_input_embeddings()
        elif hasattr(self.model, "shared"):
            embedding_layer = self.model.shared
        else:
            raise ValueError("Cannot find embedding layer")
        
        input_embeddings = embedding_layer(input_ids)
        
        # 拼接prompt embeddings和输入embeddings
        # combined_embeddings: (batch_size, prompt_length + seq_len, embedding_dim)
        combined_embeddings = torch.cat(
            [batch_prompt_embeddings, input_embeddings], dim=1
        )
        
        # 更新attention mask以包含prompt tokens
        if attention_mask is not None:
            # 创建prompt部分的attention mask（全1）
            prompt_attention = torch.ones(
                (batch_size, self.prompt_length),
                device=attention_mask.device,
                dtype=attention_mask.dtype
            )
            combined_attention_mask = torch.cat(
                [prompt_attention, attention_mask], dim=1
            )
        else:
            combined_attention_mask = None
        
        # 对于seq2seq模型（如FLAN-T5），需要特殊处理
        if hasattr(self.model, "encoder"):
            # 分离encoder和decoder的参数
            encoder_kwargs = {k: v for k, v in kwargs.items() if k not in ["decoder_input_ids", "labels"]}
            decoder_kwargs = {k: v for k, v in kwargs.items() if k in ["decoder_input_ids", "labels"]}
            
            # 使用encoder处理combined embeddings
            encoder_outputs = self.model.encoder(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention_mask,
                **encoder_kwargs
            )
            
            # 如果有decoder输入，调用完整的模型forward
            if hasattr(self.model, "decoder") and decoder_kwargs:
                return self.model(
                    encoder_outputs=encoder_outputs,
                    **decoder_kwargs
                )
            else:
                return encoder_outputs
        else:
            # 对于causal模型（如Qwen），直接使用inputs_embeds
            return self.model(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention_mask,
                **kwargs
            )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generation_kwargs
    ) -> torch.Tensor:
        """生成文本（包装模型的generate方法）。
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            **generation_kwargs: 生成参数
            
        Returns:
            生成的token IDs
        """
        batch_size = input_ids.shape[0]
        
        # 对于seq2seq模型（如FLAN-T5），需要特殊处理
        # 问题：直接使用inputs_embeds会导致position bias计算错误
        # 解决方案：使用input_ids（包含虚拟prompt tokens）来确保position bias计算正确
        # 但实际embedding使用我们的prompt_embeddings
        if hasattr(self.model, "encoder"):
            # 创建虚拟input_ids用于position bias计算
            # 使用pad_token_id填充prompt部分
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = getattr(self.model.config, "pad_token_id", 0)
            
            # 创建prompt部分的虚拟token IDs（使用pad_token_id）
            prompt_ids = torch.full(
                (batch_size, self.prompt_length),
                pad_token_id,
                device=input_ids.device,
                dtype=input_ids.dtype
            )
            
            # 拼接prompt_ids和input_ids
            combined_input_ids = torch.cat([prompt_ids, input_ids], dim=1)
            
            # 更新attention mask
            if attention_mask is not None:
                prompt_attention = torch.ones(
                    (batch_size, self.prompt_length),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                combined_attention_mask = torch.cat(
                    [prompt_attention, attention_mask], dim=1
                )
            else:
                combined_attention_mask = None
            
            # 获取输入embeddings
            if hasattr(self.model, "get_input_embeddings"):
                embedding_layer = self.model.get_input_embeddings()
            elif hasattr(self.model, "shared"):
                embedding_layer = self.model.shared
            else:
                raise ValueError("Cannot find embedding layer")
            
            # 获取combined_input_ids的embeddings（用于position bias计算）
            combined_token_embeddings = embedding_layer(combined_input_ids)
            
            # 替换prompt部分的embeddings为我们的prompt_embeddings
            batch_prompt_embeddings = self.prompt_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            # 替换前prompt_length个位置的embeddings
            combined_embeddings = torch.cat(
                [batch_prompt_embeddings, combined_token_embeddings[:, self.prompt_length:]], dim=1
            )
            
            # 使用combined_input_ids确保position bias计算正确
            # 但通过hook或直接修改embedding来使用我们的prompt_embeddings
            # 由于transformers库的限制，我们需要使用input_ids，但通过修改embedding层来注入prompt_embeddings
            # 这里我们使用一个技巧：临时修改embedding层
            original_embedding_call = embedding_layer.__call__
            
            def patched_embedding_call(input_ids):
                # 获取原始embeddings
                embeddings = original_embedding_call(input_ids)
                # 如果是combined_input_ids，替换prompt部分的embeddings
                if embeddings.shape[1] == combined_input_ids.shape[1]:
                    # 检查是否是我们的combined_input_ids（通过检查前prompt_length个位置是否都是pad_token_id）
                    if torch.all(input_ids[:, :self.prompt_length] == pad_token_id):
                        # 替换prompt部分的embeddings
                        batch_prompt = self.prompt_embeddings.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
                        embeddings = torch.cat([batch_prompt, embeddings[:, self.prompt_length:]], dim=1)
                return embeddings
            
            embedding_layer.__call__ = patched_embedding_call
            
            try:
                # 现在使用combined_input_ids，但embedding层会自动使用我们的prompt_embeddings
                output = self.model.generate(
                    input_ids=combined_input_ids,
                    attention_mask=combined_attention_mask,
                    **generation_kwargs
                )
            finally:
                # 恢复原始embedding层
                embedding_layer.__call__ = original_embedding_call
        else:
            # 对于causal模型（如Qwen），直接使用inputs_embeds
            if hasattr(self.model, "get_input_embeddings"):
                embedding_layer = self.model.get_input_embeddings()
            elif hasattr(self.model, "shared"):
                embedding_layer = self.model.shared
            else:
                raise ValueError("Cannot find embedding layer")
            
            input_embeddings = embedding_layer(input_ids)
            
            # 添加prompt embeddings
            batch_prompt_embeddings = self.prompt_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            combined_embeddings = torch.cat(
                [batch_prompt_embeddings, input_embeddings], dim=1
            )
            
            # 更新attention mask
            if attention_mask is not None:
                prompt_attention = torch.ones(
                    (batch_size, self.prompt_length),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                combined_attention_mask = torch.cat(
                    [prompt_attention, attention_mask], dim=1
                )
            else:
                combined_attention_mask = None
            
            # 使用inputs_embeds调用generate
            output = self.model.generate(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention_mask,
                **generation_kwargs
            )
        
        return output
    
    def save_prompt_embeddings(self, save_path: Union[str, Path]):
        """保存prompt embeddings到文件。
        
        Args:
            save_path: 保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(
            {
                "prompt_embeddings": self.prompt_embeddings.state_dict() if hasattr(self.prompt_embeddings, "state_dict") else self.prompt_embeddings,
                "prompt_length": self.prompt_length,
                "embedding_dim": self.embedding_dim,
                "initialization": self.initialization,
            },
            save_path
        )
        logger.info(f"Saved prompt embeddings to {save_path}")
    
    def load_prompt_embeddings(self, load_path: Union[str, Path]):
        """从文件加载prompt embeddings。
        
        Args:
            load_path: 加载路径
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Prompt embeddings file not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # 加载embeddings
        if "prompt_embeddings" in checkpoint:
            if isinstance(checkpoint["prompt_embeddings"], dict):
                self.prompt_embeddings.load_state_dict(checkpoint["prompt_embeddings"])
            else:
                self.prompt_embeddings.data = checkpoint["prompt_embeddings"].to(self.device)
        else:
            raise ValueError("Invalid checkpoint format: missing prompt_embeddings")
        
        logger.info(f"Loaded prompt embeddings from {load_path}")

