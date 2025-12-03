#!/usr/bin/env python3
"""构建包含密集描述子的知识库。

使用 BLIP-2 模型为每张图像生成多个描述性短句。
输出: image_id_to_dense_captions.pkl

该脚本支持：
- 批处理加速
- 进度条显示
- 错误处理（跳过失败图像）
- 从断点恢复

从 configs/config.yaml 读取配置。
"""
import os
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# 添加项目根目录到Python路径，以便导入utils模块
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import torch
from tqdm import tqdm
import yaml
from PIL import Image
import re

# Transformers imports for BLIP-2
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    _has_transformers = True
except ImportError:
    _has_transformers = False
    logging.error("transformers库未安装，请运行: pip install transformers")

# NLTK imports for noun extraction
try:
    import nltk
    from nltk.tag import pos_tag
    from nltk.tokenize import word_tokenize
    # 尝试下载必要的NLTK数据
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        try:
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            pass
    _has_nltk = True
except ImportError:
    _has_nltk = False
    logging.warning("NLTK未安装，将使用简单的名词提取方法。建议安装: pip install nltk")

from utils.image_utils import load_image


def extract_nouns_from_sentence(sentence: str) -> List[str]:
    """从句子中提取名词。
    
    从生成的完整句子中提取所有名词作为短语。
    例如：
    - "a man on a motorcycle" -> ["man", "motorcycle"]
    - "The woman is cutting the cake" -> ["woman", "cake"]
    - "the boy is holding an umbrella" -> ["boy", "umbrella"]
    
    支持两种方法：
    1. 使用NLTK进行词性标注（如果可用，推荐）
    2. 使用简单的规则提取（fallback）
    
    Args:
        sentence: 输入句子
        
    Returns:
        提取出的名词列表
    """
    if not sentence or not sentence.strip():
        return []
    
    sentence = sentence.strip()
    nouns = []
    
    # 方法1: 使用NLTK进行词性标注（更准确）
    if _has_nltk:
        try:
            # 分词
            tokens = word_tokenize(sentence)
            
            # 词性标注
            tagged = pos_tag(tokens)
            
            # 提取名词（NN, NNS, NNP, NNPS）
            for word, pos_tag_val in tagged:
                # 名词标记: NN (单数), NNS (复数), NNP (专有名词单数), NNPS (专有名词复数)
                if pos_tag_val.startswith('NN'):
                    word_clean = word.lower().strip()
                    # 过滤停用词
                    stop_words = {'a', 'an', 'the'}
                    if word_clean and len(word_clean) > 1 and word_clean not in stop_words:
                        nouns.append(word_clean)
                    
        except Exception as e:
            logging.debug(f"NLTK noun extraction failed: {e}, falling back to simple method")
            # 如果NLTK失败，使用简单方法
            nouns = _extract_nouns_simple(sentence)
    else:
        # 方法2: 使用简单规则提取
        nouns = _extract_nouns_simple(sentence)
    
    # 去重并保持顺序
    seen = set()
    filtered_nouns = []
    for noun in nouns:
        noun = noun.strip().lower()
        # 过滤停用词和太短的词
        if noun and len(noun) > 1 and noun not in seen:
            # 过滤常见的停用词
            stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                         'on', 'in', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'the'}
            if noun not in stop_words:
                seen.add(noun)
                filtered_nouns.append(noun)
    
    return filtered_nouns


def _extract_nouns_simple(sentence: str) -> List[str]:
    """使用简单规则从句子中提取名词（fallback方法）。
    
    当NLTK不可用时使用。提取常见的名词模式，例如：
    - "a man on a motorcycle" -> ["man", "motorcycle"]
    - "The woman is cutting the cake" -> ["woman", "cake"]
    - "the boy is holding an umbrella" -> ["boy", "umbrella"]
    
    Args:
        sentence: 输入句子
        
    Returns:
        提取出的名词列表
    """
    nouns = []
    
    # 转换为小写便于处理
    sentence_lower = sentence.lower()
    
    # 移除标点符号（保留空格）
    sentence_clean = re.sub(r'[^\w\s]', ' ', sentence_lower)
    
    # 常见动词和助动词（需要跳过）
    verbs = {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
             'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should', 'may', 'might',
             'cutting', 'holding', 'running', 'walking', 'sitting', 'standing', 'playing'}
    
    # 常见介词
    prepositions = {'on', 'in', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'about'}
    
    # 常见停用词
    stop_words = {'a', 'an', 'the', 'this', 'that', 'these', 'those'}
    
    # 分词
    words = sentence_clean.split()
    
    # 提取名词：跳过动词、介词、停用词，保留其他有意义的词
    for i, word in enumerate(words):
        word = word.strip()
        
        # 跳过停用词、动词、介词
        if word in stop_words or word in verbs or word in prepositions:
            continue
        
        # 跳过以常见动词后缀结尾的词
        if word.endswith(('ing', 'ed', 'ly')) and word not in {'wedding', 'building', 'thing', 'king', 'ring'}:
            continue
        
        # 跳过太短的词
        if len(word) < 2:
            continue
        
        # 检查是否在介词后面（更可能是名词）
        if i > 0 and words[i-1] in prepositions:
            nouns.append(word)
        # 检查是否在冠词后面（可能是名词）
        elif i > 0 and words[i-1] in {'a', 'an', 'the'}:
            nouns.append(word)
        # 如果不在动词/介词附近，也可能是不错的名词候选
        elif i < len(words) - 1 and words[i+1] not in verbs:
            nouns.append(word)
    
    return nouns


def setup_logging(level_str: str):
    level = getattr(logging, level_str.upper(), logging.ERROR)
    # 如果日志级别为ERROR，至少输出一些基本信息
    if level == logging.ERROR:
        # 使用INFO级别以便看到进度，但保持ERROR作为默认
        level = logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint_file(kb_path: str) -> Dict[int, List[str]]:
    """加载已存在的检查点文件。
    
    Args:
        kb_path: 知识库文件路径
        
    Returns:
        已处理的图像ID到密集描述列表的映射
    """
    if os.path.exists(kb_path):
        try:
            with open(kb_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logging.warning(f"Failed to load checkpoint from {kb_path}: {e}")
            return {}
    return {}


def save_checkpoint(image_id_to_captions: Dict[int, List[str]], kb_path: str):
    """保存检查点文件。
    
    Args:
        image_id_to_captions: 图像ID到密集描述列表的映射
        kb_path: 知识库文件路径
    """
    os.makedirs(os.path.dirname(kb_path) or ".", exist_ok=True)
    with open(kb_path, "wb") as f:
        pickle.dump(image_id_to_captions, f)
    logging.debug(f"Checkpoint saved: {len(image_id_to_captions)} images processed")


def load_coco_image_mapping(annotations_path: str) -> Dict[int, str]:
    """从COCO标注文件中加载image_id到文件名的映射。
    
    Args:
        annotations_path: COCO标注文件路径
        
    Returns:
        image_id到文件名的映射字典
    """
    logging.info(f"Loading COCO image mapping from {annotations_path}")
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    image_id_to_filename = {
        item["id"]: item["file_name"] 
        for item in data.get("images", [])
    }
    
    logging.info(f"Loaded {len(image_id_to_filename)} image mappings")
    return image_id_to_filename


def image_paths_and_ids(
    images_dir: str, 
    image_id_to_filename: Dict[int, str]
) -> Tuple[List[int], List[str]]:
    """构建图像ID和路径列表（仅包含存在的文件）。
    
    Args:
        images_dir: 图像目录路径
        image_id_to_filename: 图像ID到文件名的映射
        
    Returns:
        (图像ID列表, 图像路径列表)
    """
    out_ids = []
    out_paths = []
    for img_id, fname in image_id_to_filename.items():
        full = os.path.join(images_dir, fname)
        if os.path.exists(full):
            out_ids.append(img_id)
            out_paths.append(full)
        else:
            logging.debug(f"Image file missing: {full}")
    return out_ids, out_paths


def init_blip2_model(model_path: str, device: str = "cuda"):
    """初始化BLIP-2模型。
    
    Args:
        model_path: 模型路径
        device: 设备名称
        
    Returns:
        (processor, model) 元组
    """
    if not _has_transformers:
        raise ImportError("transformers库未安装，请运行: pip install transformers")
    
    logging.info(f"Loading BLIP-2 model from {model_path}")
    
    try:
        # 加载processor和model
        processor = Blip2Processor.from_pretrained(model_path)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16  # 使用float16节省显存
        )
        
        # 移动到设备并设置为评估模式
        model = model.to(device)
        model.eval()
        
        logging.info(f"BLIP-2 model loaded successfully on {device}")
        return processor, model
        
    except Exception as e:
        logging.error(f"Failed to load BLIP-2 model from {model_path}: {e}")
        raise


def extract_dense_captions_blip2(
    model,
    processor,
    image_path: str,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 100,
    num_beams: int = 5
) -> List[str]:
    """使用BLIP-2模型为单张图像提取密集描述短语。
    
    Args:
        model: BLIP-2模型
        processor: BLIP-2处理器
        image_path: 图像路径
        prompt: 提示词
        device: 设备名称
        max_new_tokens: 最大生成token数
        num_beams: beam search数量
        
    Returns:
        描述短语列表
    """
    try:
        # 加载图像
        image = load_image(image_path, return_none_on_error=True)
        if image is None:
            return []
        
        # 使用processor处理图像和提示词
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(device)
        
        # 生成描述
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # 解码生成的文本
        generated_text = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        # 提取答案部分（去除提示词）
        # BLIP-2通常会生成包含提示词在内的完整文本
        answer_text = generated_text
        
        # 方法1: 如果包含完整提示词，提取提示词后的内容
        if prompt in generated_text:
            answer_text = generated_text.split(prompt, 1)[-1].strip()
        # 方法2: 如果只包含"Answer:"部分，提取其后的内容
        elif "Answer:" in generated_text:
            answer_text = generated_text.split("Answer:", 1)[-1].strip()
        # 方法3: 如果都没有，直接使用生成文本（可能是纯答案）
        else:
            answer_text = generated_text.strip()
        
        # 按分号分割短语
        phrases = [p.strip() for p in answer_text.split(';')]
        
        # 过滤空字符串
        phrases = [p for p in phrases if p]
        
        # 如果只有一个短语且没有分号分隔，说明可能是一个完整句子
        # 需要从句子中提取名词
        if len(phrases) == 1 and ';' not in answer_text:
            # 这是一个完整句子，需要提取名词
            extracted_nouns = extract_nouns_from_sentence(phrases[0])
            if extracted_nouns:
                return extracted_nouns
            else:
                # 如果提取失败，返回原短语（可能需要进一步处理）
                return phrases
        
        # 如果有多个短语（已用分号分隔），对每个短语提取名词
        processed_phrases = []
        for phrase in phrases:
            # 如果短语已经是一个简短的描述，可能不需要提取
            # 检查是否是一个完整句子（包含动词）
            if any(verb in phrase.lower() for verb in ['is', 'are', 'was', 'were', 'has', 'have', 'doing', 'cutting', 'holding']):
                # 这是一个完整句子，提取名词
                extracted_nouns = extract_nouns_from_sentence(phrase)
                if extracted_nouns:
                    processed_phrases.extend(extracted_nouns)
                else:
                    # 如果提取失败，保留原短语
                    processed_phrases.append(phrase)
            else:
                # 已经是短语形式，直接保留
                processed_phrases.append(phrase)
        
        # 去重并保持顺序
        seen = set()
        unique_phrases = []
        for phrase in processed_phrases:
            phrase_clean = phrase.strip()
            if phrase_clean:
                phrase_lower = phrase_clean.lower()
                if phrase_lower not in seen:
                    seen.add(phrase_lower)
                    unique_phrases.append(phrase_clean)
        
        return unique_phrases if unique_phrases else phrases
        
    except Exception as e:
        logging.warning(f"Failed to extract dense captions for {image_path}: {e}")
        return []


def process_batch(
    model,
    processor,
    image_ids: List[int],
    image_paths: List[str],
    prompt: str,
    device: str,
    max_new_tokens: int = 100,
    num_beams: int = 5
) -> Dict[int, List[str]]:
    """批量处理图像，提取密集描述。
    
    Args:
        model: BLIP-2模型
        processor: BLIP-2处理器
        image_ids: 图像ID列表
        image_paths: 图像路径列表
        prompt: 提示词
        device: 设备名称
        max_new_tokens: 最大生成token数
        num_beams: beam search数量
        
    Returns:
        图像ID到描述列表的映射
    """
    results = {}
    
    for image_id, image_path in zip(image_ids, image_paths):
        phrases = extract_dense_captions_blip2(
            model,
            processor,
            image_path,
            prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams
        )
        if phrases:  # 只保存非空结果
            results[image_id] = phrases
    
    return results


def process_images(
    model_path: str,
    image_ids: List[int],
    image_paths: List[str],
    kb_path: str,
    prompt: str,
    batch_size: int = 8,
    max_new_tokens: int = 100,
    num_beams: int = 5,
    checkpoint_interval: int = 100,
    device: Optional[str] = None
) -> Dict[int, List[str]]:
    """处理所有图像，生成密集描述。
    
    Args:
        model_path: 模型路径
        image_ids: 图像ID列表
        image_paths: 图像路径列表
        kb_path: 知识库输出路径
        prompt: 提示词
        batch_size: 批处理大小（用于进度显示，实际为逐张处理）
        max_new_tokens: 最大生成token数
        num_beams: beam search数量
        checkpoint_interval: 检查点保存间隔
        device: 设备名称（如果为None则自动检测）
        
    Returns:
        完整的图像ID到描述列表的映射
    """
    # 加载已处理的检查点
    image_id_to_captions = load_checkpoint_file(kb_path)
    processed_ids = set(image_id_to_captions.keys())
    
    # 过滤出未处理的图像
    remaining_ids = []
    remaining_paths = []
    for img_id, img_path in zip(image_ids, image_paths):
        if img_id not in processed_ids:
            remaining_ids.append(img_id)
            remaining_paths.append(img_path)
    
    logging.info(f"Found {len(processed_ids)} already processed images")
    logging.info(f"Remaining images to process: {len(remaining_ids)}")
    
    if not remaining_ids:
        logging.info("All images already processed!")
        return image_id_to_captions
    
    # 确定设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logging.info(f"Using device: {device}")
    
    # 加载模型
    processor, model = init_blip2_model(model_path, device=device)
    
    # 处理所有剩余图像
    total_images = len(remaining_ids)
    total_batches = (total_images + batch_size - 1) // batch_size
    
    with tqdm(total=total_images, desc="Processing images") as pbar:
        for i in range(0, len(remaining_ids), batch_size):
            batch_ids = remaining_ids[i:i+batch_size]
            batch_paths = remaining_paths[i:i+batch_size]
            
            # 处理批次
            batch_results = process_batch(
                model,
                processor,
                batch_ids,
                batch_paths,
                prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams
            )
            
            # 更新结果
            image_id_to_captions.update(batch_results)
            pbar.update(len(batch_ids))
            
            # 定期保存检查点
            if (i // batch_size + 1) % (checkpoint_interval // batch_size) == 0 or i + batch_size >= len(remaining_ids):
                save_checkpoint(image_id_to_captions, kb_path)
                logging.info(f"Checkpoint saved: {len(image_id_to_captions)} images processed")
    
    # 最终保存
    save_checkpoint(image_id_to_captions, kb_path)
    logging.info(f"Processing complete: {len(image_id_to_captions)} images processed")
    return image_id_to_captions


def main():
    cfg = load_config()
    setup_logging(cfg.get("log_config", {}).get("log_level", "ERROR"))
    
    # 读取配置
    dense_config = cfg.get("dense_descriptor", {})
    model_path = dense_config.get("model_path", "../models/blip2-opt-2.7b/")
    kb_path = dense_config.get("knowledge_base_path", "./output/image_id_to_dense_captions.pkl")
    prompt = dense_config.get("prompt", "Question: List the objects, scenes, and actions in this image with very short phrases. Answer: ")
    
    data_config = cfg.get("data_config", {})
    images_dir = data_config.get("coco_images_dir")
    annotations_path = data_config.get("coco_annotations_path")
    
    batch_size = dense_config.get("batch_size", 8)
    max_new_tokens = dense_config.get("max_new_tokens", 100)
    num_beams = dense_config.get("num_beams", 5)
    checkpoint_interval = dense_config.get("checkpoint_interval", 100)
    
    if not all([model_path, images_dir, annotations_path]):
        logging.error("Missing configuration values. Please check configs/config.yaml")
        sys.exit(1)
    
    # 检查模型路径
    if not os.path.exists(model_path):
        logging.error(f"Model path does not exist: {model_path}")
        logging.error("Please download the model to this path.")
        sys.exit(1)
    
    logging.info(f"Using prompt: {prompt}")
    
    # 加载COCO图像映射
    image_id_to_filename = load_coco_image_mapping(annotations_path)
    image_ids, image_paths = image_paths_and_ids(images_dir, image_id_to_filename)
    
    if len(image_paths) == 0:
        logging.error(f"No images found in {images_dir}. Aborting.")
        sys.exit(1)
    
    logging.info(f"Found {len(image_paths)} images to process")
    
    # 处理图像
    try:
        image_id_to_captions = process_images(
            model_path=model_path,
            image_ids=image_ids,
            image_paths=image_paths,
            kb_path=kb_path,
            prompt=prompt,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            checkpoint_interval=checkpoint_interval
        )
        
        # 保存最终结果
        save_checkpoint(image_id_to_captions, kb_path)
        
        # 统计信息
        images_with_captions = sum(1 for captions in image_id_to_captions.values() if captions)
        total_phrases = sum(len(captions) for captions in image_id_to_captions.values())
        avg_phrases = total_phrases / len(image_id_to_captions) if image_id_to_captions else 0
        
        logging.info(f"Knowledge base build complete!")
        logging.info(f"Total images processed: {len(image_id_to_captions)}")
        logging.info(f"Images with captions: {images_with_captions}")
        logging.info(f"Total phrases: {total_phrases}")
        logging.info(f"Average phrases per image: {avg_phrases:.2f}")
        logging.info(f"Saved to: {kb_path}")
        
    except Exception as e:
        logging.error(f"Failed during processing: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
