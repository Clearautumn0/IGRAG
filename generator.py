"""
大语言模型生成器模块
"""
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Union
import logging
import re
from pathlib import Path

from config import MODEL_CONFIG, GENERATION_CONFIG

logger = logging.getLogger(__name__)


class LLMGenerator:
    """大语言模型生成器基类"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        初始化生成器
        
        Args:
            model_name: 模型名称
            device: 计算设备
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
        logger.info(f"初始化LLM生成器: {model_name}, 设备: {device}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            **kwargs: 生成参数
            
        Returns:
            生成的文本
        """
        raise NotImplementedError
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        批量生成文本
        
        Args:
            prompts: 输入提示列表
            **kwargs: 生成参数
            
        Returns:
            生成的文本列表
        """
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"生成失败: {e}")
                results.append("")
        return results


class FLANT5Generator(LLMGenerator):
    """FLAN-T5生成器"""
    
    def __init__(self, model_name: str = "google/flan-t5-large", device: str = "cuda"):
        """
        初始化FLAN-T5生成器
        
        Args:
            model_name: 模型名称
            device: 计算设备
        """
        super().__init__(model_name, device)
        
        # 加载模型和分词器
        logger.info(f"正在加载FLAN-T5模型: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # 移动到指定设备
        self.model = self.model.to(device)
        self.model.eval()
        
        logger.info("FLAN-T5模型加载完成")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        使用FLAN-T5生成文本
        
        Args:
            prompt: 输入提示
            **kwargs: 生成参数
            
        Returns:
            生成的文本
        """
        # 设置默认生成参数
        generation_config = {
            'max_length': GENERATION_CONFIG['max_length'],
            'num_beams': GENERATION_CONFIG['beam_size'],
            'temperature': GENERATION_CONFIG['temperature'],
            'do_sample': GENERATION_CONFIG['do_sample'],
            'top_p': GENERATION_CONFIG['top_p'],
            'repetition_penalty': GENERATION_CONFIG['repetition_penalty'],
            'early_stopping': True,
            'pad_token_id': self.tokenizer.pad_token_id,
            **kwargs
        }
        
        try:
            # 编码输入
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs.to(self.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(inputs, **generation_config)
            
            # 解码输出
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 后处理
            generated_text = self._post_process(generated_text)
            
            return generated_text
            
        except Exception as e:
            logger.error(f"FLAN-T5生成失败: {e}")
            return ""
    
    def _post_process(self, text: str) -> str:
        """
        后处理生成的文本
        
        Args:
            text: 原始生成文本
            
        Returns:
            处理后的文本
        """
        # 移除多余的空格和换行
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 移除重复的句子
        sentences = text.split('.')
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence.lower() not in seen:
                unique_sentences.append(sentence)
                seen.add(sentence.lower())
        
        return '. '.join(unique_sentences) + '.' if unique_sentences else text


class GPT2Generator(LLMGenerator):
    """GPT-2生成器"""
    
    def __init__(self, model_name: str = "gpt2-medium", device: str = "cuda"):
        """
        初始化GPT-2生成器
        
        Args:
            model_name: 模型名称
            device: 计算设备
        """
        super().__init__(model_name, device)
        
        # 加载模型和分词器
        logger.info(f"正在加载GPT-2模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 移动到指定设备
        self.model = self.model.to(device)
        self.model.eval()
        
        logger.info("GPT-2模型加载完成")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        使用GPT-2生成文本
        
        Args:
            prompt: 输入提示
            **kwargs: 生成参数
            
        Returns:
            生成的文本
        """
        # 设置默认生成参数
        generation_config = {
            'max_length': GENERATION_CONFIG['max_length'],
            'num_beams': GENERATION_CONFIG['beam_size'],
            'temperature': GENERATION_CONFIG['temperature'],
            'do_sample': GENERATION_CONFIG['do_sample'],
            'top_p': GENERATION_CONFIG['top_p'],
            'repetition_penalty': GENERATION_CONFIG['repetition_penalty'],
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            **kwargs
        }
        
        try:
            # 编码输入
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs.to(self.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(inputs, **generation_config)
            
            # 解码输出（只取生成的部分）
            generated_tokens = outputs[0][inputs.shape[1]:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # 后处理
            generated_text = self._post_process(generated_text)
            
            return generated_text
            
        except Exception as e:
            logger.error(f"GPT-2生成失败: {e}")
            return ""
    
    def _post_process(self, text: str) -> str:
        """
        后处理生成的文本
        
        Args:
            text: 原始生成文本
            
        Returns:
            处理后的文本
        """
        # 移除多余的空格和换行
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 移除重复的句子
        sentences = text.split('.')
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence.lower() not in seen:
                unique_sentences.append(sentence)
                seen.add(sentence.lower())
        
        return '. '.join(unique_sentences) + '.' if unique_sentences else text


class ImageCaptionGenerator:
    """图像描述生成器 - 整合检索和生成"""
    
    def __init__(self, 
                 retriever,
                 generator: LLMGenerator):
        """
        初始化图像描述生成器
        
        Args:
            retriever: 检索器
            generator: 生成器
        """
        self.retriever = retriever
        self.generator = generator
        
        logger.info("图像描述生成器初始化完成")
    
    def generate_caption(self, image_path: str, **generation_kwargs) -> Dict[str, str]:
        """
        生成图像描述
        
        Args:
            image_path: 图像路径
            **generation_kwargs: 生成参数
            
        Returns:
            包含生成描述和相关信息的字典
        """
        logger.info(f"开始生成图像描述: {image_path}")
        
        try:
            # 执行检索
            retrieval_results = self.retriever.retrieve(image_path)
            
            # 构建提示
            prompt = self.retriever.build_prompt(retrieval_results)
            
            # 生成描述
            caption = self.generator.generate(prompt, **generation_kwargs)
            
            # 构建结果
            result = {
                'caption': caption,
                'prompt': prompt,
                'retrieval_stats': self.retriever.get_retrieval_statistics(retrieval_results),
                'global_results_count': len(retrieval_results['global']),
                'local_results_count': len(retrieval_results['local'])
            }
            
            logger.info(f"图像描述生成完成")
            return result
            
        except Exception as e:
            logger.error(f"图像描述生成失败: {e}")
            return {
                'caption': "",
                'prompt': "",
                'retrieval_stats': "",
                'global_results_count': 0,
                'local_results_count': 0,
                'error': str(e)
            }
    
    def batch_generate_captions(self, image_paths: List[str], **generation_kwargs) -> List[Dict[str, str]]:
        """
        批量生成图像描述
        
        Args:
            image_paths: 图像路径列表
            **generation_kwargs: 生成参数
            
        Returns:
            生成结果列表
        """
        logger.info(f"开始批量生成 {len(image_paths)} 张图像的描述")
        
        results = []
        for i, image_path in enumerate(image_paths):
            logger.info(f"处理第 {i+1}/{len(image_paths)} 张图像: {image_path}")
            result = self.generate_caption(image_path, **generation_kwargs)
            results.append(result)
        
        logger.info("批量生成完成")
        return results
    
    def generate_with_analysis(self, image_path: str, **generation_kwargs) -> Dict:
        """
        生成图像描述并进行分析
        
        Args:
            image_path: 图像路径
            **generation_kwargs: 生成参数
            
        Returns:
            包含生成结果和分析的字典
        """
        logger.info(f"开始生成图像描述并分析: {image_path}")
        
        try:
            # 执行检索
            retrieval_results = self.retriever.retrieve(image_path)
            
            # 分析检索结果
            analysis = self.retriever.analyze_retrieval_results(retrieval_results)
            
            # 构建提示
            prompt = self.retriever.build_prompt(retrieval_results)
            
            # 生成描述
            caption = self.generator.generate(prompt, **generation_kwargs)
            
            # 构建完整结果
            result = {
                'caption': caption,
                'prompt': prompt,
                'retrieval_analysis': analysis,
                'retrieval_stats': self.retriever.get_retrieval_statistics(retrieval_results),
                'global_results': retrieval_results['global'],
                'local_results': retrieval_results['local']
            }
            
            logger.info(f"图像描述生成和分析完成")
            return result
            
        except Exception as e:
            logger.error(f"图像描述生成和分析失败: {e}")
            return {
                'caption': "",
                'prompt': "",
                'retrieval_analysis': {},
                'retrieval_stats': "",
                'global_results': [],
                'local_results': [],
                'error': str(e)
            }


def create_generator(model_type: str = "flan-t5", **kwargs) -> LLMGenerator:
    """
    创建生成器工厂函数
    
    Args:
        model_type: 模型类型 ("flan-t5", "gpt2")
        **kwargs: 额外参数
        
    Returns:
        生成器实例
    """
    if model_type.lower() == "flan-t5":
        return FLANT5Generator(**kwargs)
    elif model_type.lower() == "gpt2":
        return GPT2Generator(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def test_generator():
    """测试生成器"""
    import torch
    
    # 测试FLAN-T5生成器
    print("测试FLAN-T5生成器...")
    try:
        flan_generator = FLANT5Generator(device="cpu")
        test_prompt = "你是一个专业的图像描述生成器。整体相似的图片描述：- A cat sitting on a chair. - A dog running in the park. 在关键局部区域相似的图片描述：- The cat's eyes are green. - The dog has brown fur. 请综合分析以上描述，生成一个全新、准确且详尽的图片描述。"
        
        result = flan_generator.generate(test_prompt, max_length=100)
        print(f"FLAN-T5生成结果: {result}")
        
    except Exception as e:
        print(f"FLAN-T5测试失败: {e}")
    
    # 测试GPT-2生成器
    print("\n测试GPT-2生成器...")
    try:
        gpt2_generator = GPT2Generator(device="cpu")
        test_prompt = "This is a beautiful image showing"
        
        result = gpt2_generator.generate(test_prompt, max_length=50)
        print(f"GPT-2生成结果: {result}")
        
    except Exception as e:
        print(f"GPT-2测试失败: {e}")
    
    print("生成器测试完成")


if __name__ == "__main__":
    test_generator()
