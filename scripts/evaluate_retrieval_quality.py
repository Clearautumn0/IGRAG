#!/usr/bin/env python3
"""检索质量评估脚本

评估IGRAG系统检索环节的性能，分析检索到的描述与真实标注之间的相关性。

评估指标：
1. 语义相似度：检索描述与真实标注在CLIP文本空间中的余弦相似度
2. 词汇重叠度：检索描述与真实标注之间的BLEU-4分数
3. 检索召回率：检索描述中是否包含关键物体/概念
4. 位置一致性：检索图像的视觉内容与查询图像的一致性
"""

import argparse
import json
import logging
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.retriever import ImageRetriever
from utils.image_utils import load_image


def setup_logging(level_str: str = "INFO"):
    """设置日志"""
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_config(path: str = "configs/config.yaml") -> dict:
    """加载配置文件"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_coco_annotations(annotations_path: str) -> Tuple[Dict[int, List[str]], Dict[int, str]]:
    """加载COCO格式的标注文件
    
    Returns:
        (image_id_to_captions, image_id_to_filename)
    """
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    image_id_to_captions = defaultdict(list)
    for ann in data.get("annotations", []):
        image_id_to_captions[ann["image_id"]].append(ann["caption"])
    
    image_id_to_filename = {
        img["id"]: img["file_name"]
        for img in data.get("images", [])
    }
    
    return dict(image_id_to_captions), image_id_to_filename


def extract_text_embeddings(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: List[str],
    device: torch.device,
    batch_size: int = 32
) -> np.ndarray:
    """提取文本的CLIP嵌入向量
    
    Args:
        model: CLIP模型
        processor: CLIP处理器
        texts: 文本列表
        device: 设备
        batch_size: 批处理大小
        
    Returns:
        归一化后的文本嵌入向量 (n_texts, dim)
    """
    model.eval()
    all_embeds = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
            
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            
            text_embeds = model.get_text_features(**inputs)
            text_embeds = text_embeds.cpu().numpy()
            
            # 归一化
            norms = np.linalg.norm(text_embeds, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            text_embeds = text_embeds / norms
            
            all_embeds.append(text_embeds)
    
    if all_embeds:
        return np.concatenate(all_embeds, axis=0)
    else:
        # 返回空数组
        dim = getattr(model.config, "projection_dim", None) or getattr(model.config, "projection_size", None) or 512
        return np.zeros((0, int(dim)), dtype="float32")


def compute_semantic_similarity(
    retrieved_captions: List[str],
    ground_truth_captions: List[str],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
    batch_size: int = 32
) -> Dict[str, float]:
    """计算语义相似度（CLIP文本空间余弦相似度）
    
    Args:
        retrieved_captions: 检索到的描述列表
        ground_truth_captions: 真实标注列表
        clip_model: CLIP模型
        clip_processor: CLIP处理器
        device: 设备
        batch_size: 批处理大小
        
    Returns:
        包含最大相似度、平均相似度等的字典
    """
    if not retrieved_captions or not ground_truth_captions:
        return {
            "max_similarity": 0.0,
            "mean_similarity": 0.0,
            "best_match_similarity": 0.0
        }
    
    # 提取嵌入向量
    all_texts = retrieved_captions + ground_truth_captions
    all_embeds = extract_text_embeddings(clip_model, clip_processor, all_texts, device, batch_size)
    
    n_retrieved = len(retrieved_captions)
    retrieved_embeds = all_embeds[:n_retrieved]
    gt_embeds = all_embeds[n_retrieved:]
    
    # 计算余弦相似度矩阵
    similarity_matrix = np.dot(retrieved_embeds, gt_embeds.T)  # (n_retrieved, n_gt)
    
    # 最大相似度（每个检索描述与所有真实标注的最大相似度）
    max_similarities = np.max(similarity_matrix, axis=1)
    max_similarity = float(np.max(max_similarities))
    mean_similarity = float(np.mean(max_similarities))
    
    # 最佳匹配相似度（每个真实标注与所有检索描述的最大相似度，然后取平均）
    best_match_similarities = np.max(similarity_matrix, axis=0)
    best_match_similarity = float(np.mean(best_match_similarities))
    
    return {
        "max_similarity": max_similarity,
        "mean_similarity": mean_similarity,
        "best_match_similarity": best_match_similarity
    }


def compute_bleu_score(
    retrieved_captions: List[str],
    ground_truth_captions: List[str]
) -> Dict[str, float]:
    """计算BLEU-4分数
    
    Args:
        retrieved_captions: 检索到的描述列表
        ground_truth_captions: 真实标注列表
        
    Returns:
        包含最大BLEU、平均BLEU等的字典
    """
    if not retrieved_captions or not ground_truth_captions:
        return {
            "max_bleu": 0.0,
            "mean_bleu": 0.0,
            "best_match_bleu": 0.0
        }
    
    smoothing = SmoothingFunction().method1
    
    # 将文本转换为词列表
    def tokenize(text: str) -> List[str]:
        return text.lower().split()
    
    gt_tokenized = [tokenize(gt) for gt in ground_truth_captions]
    retrieved_tokenized = [tokenize(ret) for ret in retrieved_captions]
    
    # 计算每个检索描述与所有真实标注的BLEU分数
    bleu_scores = []
    for ret_tokens in retrieved_tokenized:
        scores = [
            sentence_bleu([gt_tokens], ret_tokens, smoothing_function=smoothing)
            for gt_tokens in gt_tokenized
        ]
        bleu_scores.append(max(scores))
    
    max_bleu = float(max(bleu_scores)) if bleu_scores else 0.0
    mean_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0
    
    # 计算每个真实标注与所有检索描述的最佳BLEU分数
    best_match_scores = []
    for gt_tokens in gt_tokenized:
        scores = [
            sentence_bleu([gt_tokens], ret_tokens, smoothing_function=smoothing)
            for ret_tokens in retrieved_tokenized
        ]
        best_match_scores.append(max(scores) if scores else 0.0)
    
    best_match_bleu = float(np.mean(best_match_scores)) if best_match_scores else 0.0
    
    return {
        "max_bleu": max_bleu,
        "mean_bleu": mean_bleu,
        "best_match_bleu": best_match_bleu
    }


def extract_key_concepts(text: str) -> set:
    """提取文本中的关键概念（名词和重要词汇）
    
    简单实现：提取长度>=3的单词（排除常见停用词）
    """
    # 简单的停用词列表
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "can", "this",
        "that", "these", "those", "it", "its", "they", "them", "their"
    }
    
    words = text.lower().split()
    concepts = {w.strip(".,!?;:()[]{}") for w in words if len(w.strip(".,!?;:()[]{}")) >= 3}
    concepts = concepts - stopwords
    return concepts


def compute_recall_rate(
    retrieved_captions: List[str],
    ground_truth_captions: List[str]
) -> Dict[str, float]:
    """计算检索召回率（关键概念覆盖率）
    
    Args:
        retrieved_captions: 检索到的描述列表
        ground_truth_captions: 真实标注列表
        
    Returns:
        包含概念召回率等的字典
    """
    if not retrieved_captions or not ground_truth_captions:
        return {
            "concept_recall": 0.0,
            "concept_coverage": 0.0
        }
    
    # 提取真实标注中的所有关键概念
    gt_concepts = set()
    for gt in ground_truth_captions:
        gt_concepts.update(extract_key_concepts(gt))
    
    if not gt_concepts:
        return {
            "concept_recall": 0.0,
            "concept_coverage": 0.0
        }
    
    # 提取检索描述中的所有关键概念
    retrieved_concepts = set()
    for ret in retrieved_captions:
        retrieved_concepts.update(extract_key_concepts(ret))
    
    # 计算召回率：检索到的概念 / 真实标注中的概念
    covered_concepts = retrieved_concepts & gt_concepts
    concept_recall = len(covered_concepts) / len(gt_concepts) if gt_concepts else 0.0
    
    # 计算覆盖率：检索描述中覆盖的概念数 / 检索描述中的总概念数
    concept_coverage = len(covered_concepts) / len(retrieved_concepts) if retrieved_concepts else 0.0
    
    return {
        "concept_recall": concept_recall,
        "concept_coverage": concept_coverage,
        "gt_concepts_count": len(gt_concepts),
        "retrieved_concepts_count": len(retrieved_concepts),
        "covered_concepts_count": len(covered_concepts)
    }


def compute_visual_consistency(
    query_image_path: str,
    retrieved_image_ids: List[int],
    image_id_to_filename: Dict[int, str],
    images_dir: Path,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device
) -> Dict[str, float]:
    """计算位置一致性（检索图像的视觉内容与查询图像的一致性）
    
    Args:
        query_image_path: 查询图像路径
        retrieved_image_ids: 检索到的图像ID列表
        image_id_to_filename: 图像ID到文件名的映射
        images_dir: 图像目录
        clip_model: CLIP模型
        clip_processor: CLIP处理器
        device: 设备
        
    Returns:
        包含视觉相似度等的字典
    """
    if not retrieved_image_ids:
        return {
            "mean_visual_similarity": 0.0,
            "max_visual_similarity": 0.0
        }
    
    try:
        # 加载查询图像
        query_image = load_image(query_image_path)
        query_inputs = clip_processor(images=[query_image], return_tensors="pt")
        for k, v in query_inputs.items():
            if isinstance(v, torch.Tensor):
                query_inputs[k] = v.to(device)
        
        with torch.no_grad():
            query_embed = clip_model.get_image_features(**query_inputs)
            query_embed = query_embed.cpu().numpy()
            query_embed = query_embed / np.linalg.norm(query_embed, axis=1, keepdims=True)
        
        # 加载检索到的图像并提取特征
        retrieved_embeds = []
        for img_id in retrieved_image_ids:
            filename = image_id_to_filename.get(img_id)
            if not filename:
                continue
            
            img_path = images_dir / filename
            if not img_path.exists():
                continue
            
            try:
                retrieved_image = load_image(str(img_path))
                retrieved_inputs = clip_processor(images=[retrieved_image], return_tensors="pt")
                for k, v in retrieved_inputs.items():
                    if isinstance(v, torch.Tensor):
                        retrieved_inputs[k] = v.to(device)
                
                with torch.no_grad():
                    retrieved_embed = clip_model.get_image_features(**retrieved_inputs)
                    retrieved_embed = retrieved_embed.cpu().numpy()
                    retrieved_embed = retrieved_embed / np.linalg.norm(retrieved_embed, axis=1, keepdims=True)
                    retrieved_embeds.append(retrieved_embed)
            except Exception as e:
                logging.debug(f"Failed to load image {img_path}: {e}")
                continue
        
        if not retrieved_embeds:
            return {
                "mean_visual_similarity": 0.0,
                "max_visual_similarity": 0.0
            }
        
        # 计算余弦相似度
        retrieved_embeds = np.concatenate(retrieved_embeds, axis=0)
        similarities = np.dot(query_embed, retrieved_embeds.T).flatten()
        
        mean_visual_similarity = float(np.mean(similarities))
        max_visual_similarity = float(np.max(similarities))
        
        return {
            "mean_visual_similarity": mean_visual_similarity,
            "max_visual_similarity": max_visual_similarity
        }
    except Exception as e:
        logging.warning(f"Failed to compute visual consistency: {e}")
        return {
            "mean_visual_similarity": 0.0,
            "max_visual_similarity": 0.0
        }


def evaluate_single_image(
    image_id: int,
    image_path: Path,
    ground_truth_captions: List[str],
    retriever: ImageRetriever,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    train_image_id_to_filename: Dict[int, str],
    train_images_dir: Path,
    device: torch.device,
    top_k: int = 5
) -> Optional[Dict]:
    """评估单张图像的检索质量
    
    Returns:
        评估结果字典，如果失败则返回None
    """
    try:
        # 执行检索
        retrieved_data = retriever.get_retrieved_captions(str(image_path), top_k=top_k)
        
        # 收集所有检索到的描述
        retrieved_captions = []
        retrieved_image_ids = []
        hybrid_scores = []  # 混合检索分数（如果使用混合检索）
        object_match_scores = []  # 物体匹配分数（如果使用混合检索）
        for item in retrieved_data:
            captions = item.get("captions", [])
            retrieved_captions.extend(captions)
            retrieved_image_ids.append(item.get("image_id"))
            # 记录混合检索的额外信息（如果存在）
            if "hybrid_score" in item:
                hybrid_scores.append(item.get("hybrid_score"))
            if "object_match_score" in item:
                object_match_scores.append(item.get("object_match_score"))
        
        if not retrieved_captions:
            logging.warning(f"No retrieved captions for image {image_id}")
            return None
        
        # 计算各项指标
        semantic_sim = compute_semantic_similarity(
            retrieved_captions, ground_truth_captions,
            clip_model, clip_processor, device
        )
        
        bleu_scores = compute_bleu_score(retrieved_captions, ground_truth_captions)
        
        recall_metrics = compute_recall_rate(retrieved_captions, ground_truth_captions)
        
        visual_consistency = compute_visual_consistency(
            str(image_path), retrieved_image_ids,
            train_image_id_to_filename, train_images_dir,
            clip_model, clip_processor, device
        )
        
        result = {
            "image_id": image_id,
            "retrieved_count": len(retrieved_data),
            "retrieved_captions_count": len(retrieved_captions),
            "ground_truth_count": len(ground_truth_captions),
            "semantic_similarity": semantic_sim,
            "bleu_score": bleu_scores,
            "recall_rate": recall_metrics,
            "visual_consistency": visual_consistency,
            "retrieved_image_ids": retrieved_image_ids
        }
        
        # 如果使用混合检索，记录额外的混合检索指标
        if hybrid_scores:
            result["hybrid_retrieval_metrics"] = {
                "mean_hybrid_score": float(np.mean(hybrid_scores)) if hybrid_scores else 0.0,
                "max_hybrid_score": float(np.max(hybrid_scores)) if hybrid_scores else 0.0,
                "mean_object_match_score": float(np.mean(object_match_scores)) if object_match_scores else 0.0,
                "max_object_match_score": float(np.max(object_match_scores)) if object_match_scores else 0.0
            }
        
        return result
    except Exception as e:
        logging.error(f"Failed to evaluate image {image_id}: {e}")
        return None


def compute_statistics(results: List[Dict]) -> Dict:
    """计算整体统计指标"""
    if not results:
        return {}
    
    # 收集所有指标值
    metrics = {
        "semantic_similarity_max": [],
        "semantic_similarity_mean": [],
        "semantic_similarity_best_match": [],
        "bleu_max": [],
        "bleu_mean": [],
        "bleu_best_match": [],
        "concept_recall": [],
        "concept_coverage": [],
        "mean_visual_similarity": [],
        "max_visual_similarity": []
    }
    
    for result in results:
        if result is None:
            continue
        
        sem_sim = result.get("semantic_similarity", {})
        metrics["semantic_similarity_max"].append(sem_sim.get("max_similarity", 0.0))
        metrics["semantic_similarity_mean"].append(sem_sim.get("mean_similarity", 0.0))
        metrics["semantic_similarity_best_match"].append(sem_sim.get("best_match_similarity", 0.0))
        
        bleu = result.get("bleu_score", {})
        metrics["bleu_max"].append(bleu.get("max_bleu", 0.0))
        metrics["bleu_mean"].append(bleu.get("mean_bleu", 0.0))
        metrics["bleu_best_match"].append(bleu.get("best_match_bleu", 0.0))
        
        recall = result.get("recall_rate", {})
        metrics["concept_recall"].append(recall.get("concept_recall", 0.0))
        metrics["concept_coverage"].append(recall.get("concept_coverage", 0.0))
        
        visual = result.get("visual_consistency", {})
        metrics["mean_visual_similarity"].append(visual.get("mean_visual_similarity", 0.0))
        metrics["max_visual_similarity"].append(visual.get("max_visual_similarity", 0.0))
    
    # 计算统计量
    statistics = {}
    for key, values in metrics.items():
        if values:
            statistics[key] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
    
    return statistics


def analyze_by_complexity(results: List[Dict], image_id_to_captions: Dict[int, List[str]]) -> Dict:
    """按复杂度分析（基于描述长度）"""
    if not results:
        return {}
    
    # 根据ground truth描述的平均长度分类
    complexity_groups = {
        "simple": [],  # 平均长度 < 10
        "medium": [],  # 10 <= 平均长度 < 20
        "complex": []  # 平均长度 >= 20
    }
    
    for result in results:
        if result is None:
            continue
        
        image_id = result.get("image_id")
        gt_captions = image_id_to_captions.get(image_id, [])
        if not gt_captions:
            continue
        
        avg_length = np.mean([len(caption.split()) for caption in gt_captions])
        
        if avg_length < 10:
            complexity_groups["simple"].append(result)
        elif avg_length < 20:
            complexity_groups["medium"].append(result)
        else:
            complexity_groups["complex"].append(result)
    
    # 计算每个组的统计量
    analysis = {}
    for group_name, group_results in complexity_groups.items():
        if group_results:
            analysis[group_name] = {
                "count": len(group_results),
                "statistics": compute_statistics(group_results)
            }
    
    return analysis


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="评估IGRAG系统检索环节的性能",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径（默认: configs/config.yaml）"
    )
    parser.add_argument(
        "--val-annotations",
        type=str,
        default=None,
        help="COCO验证集标注文件路径（如果未指定，从配置文件读取）"
    )
    parser.add_argument(
        "--val-images-dir",
        type=str,
        default=None,
        help="COCO验证集图像目录（如果未指定，从配置文件读取）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出JSON文件路径（如果未指定，自动生成）"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="评估子集大小（只评估前N张图片，用于快速测试）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批处理大小（默认: 32）"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="检索top-k数量（默认: 5）"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认: INFO）"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 抑制警告
    warnings.filterwarnings("ignore")
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    
    # 加载配置
    config = load_config(args.config)
    
    # 获取数据路径
    if args.val_annotations:
        val_annotations_path = args.val_annotations
    else:
        val_annotations_path = config.get("evaluation", {}).get("val_annotations_path")
        if not val_annotations_path:
            logging.error("未指定验证集标注文件路径")
            return
    
    if args.val_images_dir:
        val_images_dir = Path(args.val_images_dir)
    else:
        val_images_dir = Path(config.get("evaluation", {}).get("val_images_dir", ""))
        if not val_images_dir:
            logging.error("未指定验证集图像目录")
            return
    
    if not Path(val_annotations_path).exists():
        logging.error(f"验证集标注文件不存在: {val_annotations_path}")
        return
    
    if not val_images_dir.exists():
        logging.error(f"验证集图像目录不存在: {val_images_dir}")
        return
    
    # 加载COCO标注
    logging.info("加载COCO验证集标注...")
    image_id_to_captions, image_id_to_filename = load_coco_annotations(val_annotations_path)
    logging.info(f"加载了 {len(image_id_to_captions)} 张图像的标注")
    
    # 加载训练集图像ID到文件名的映射（用于计算视觉一致性）
    logging.info("加载训练集图像映射...")
    train_annotations_path = config.get("data_config", {}).get("coco_annotations_path")
    train_images_dir = Path(config.get("data_config", {}).get("coco_images_dir", ""))
    train_image_id_to_filename = {}
    if train_annotations_path and Path(train_annotations_path).exists():
        _, train_image_id_to_filename = load_coco_annotations(train_annotations_path)
        logging.info(f"加载了 {len(train_image_id_to_filename)} 张训练集图像的映射")
    else:
        logging.warning("无法加载训练集标注，视觉一致性计算可能失败")
    
    # 初始化检索器（仅使用基础检索器，不再支持混合检索或重排序）
    logging.info("初始化检索器...")
    logging.info("使用基础检索器 (ImageRetriever)")
    retriever = ImageRetriever(config)
    
    # 加载CLIP模型用于文本相似度计算
    logging.info("加载CLIP模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model_path = config.get("model_config", {}).get("clip_model_path")
    clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
    clip_model = CLIPModel.from_pretrained(clip_model_path).to(device)
    clip_model.eval()
    
    # 获取要评估的图像ID列表
    image_ids = sorted(image_id_to_captions.keys())
    if args.subset:
        image_ids = image_ids[:args.subset]
        logging.info(f"评估子集: {len(image_ids)} 张图像")
    
    # 执行评估
    logging.info("开始评估检索质量...")
    results = []
    
    try:
        iterator = tqdm(image_ids, desc="评估中", unit="张", ncols=100)
        for image_id in iterator:
            filename = image_id_to_filename.get(image_id)
            if not filename:
                continue
            
            image_path = val_images_dir / filename
            if not image_path.exists():
                continue
            
            ground_truth_captions = image_id_to_captions.get(image_id, [])
            if not ground_truth_captions:
                continue
            
            result = evaluate_single_image(
                image_id, image_path, ground_truth_captions,
                retriever, clip_model, clip_processor,
                train_image_id_to_filename, train_images_dir,
                device, top_k=args.top_k
            )
            
            if result:
                results.append(result)
    except KeyboardInterrupt:
        logging.warning("评估被用户中断")
    
    if not results:
        logging.error("没有有效的评估结果")
        return
    
    logging.info(f"成功评估 {len(results)} 张图像")
    
    # 计算统计指标
    logging.info("计算统计指标...")
    statistics = compute_statistics(results)
    
    # 按复杂度分析
    logging.info("按复杂度分析...")
    complexity_analysis = analyze_by_complexity(results, image_id_to_captions)
    
    # 构建最终结果
    final_result = {
        "evaluation_info": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "num_images": len(results),
            "config_path": args.config,
            "val_annotations_path": str(val_annotations_path),
            "val_images_dir": str(val_images_dir),
            "train_images_dir": str(train_images_dir) if train_images_dir else None,
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "retrieval_mode": "hybrid" if use_hybrid else "base",
            "hybrid_retrieval_config": hybrid_config if use_hybrid else None
        },
        "overall_statistics": statistics,
        "complexity_analysis": complexity_analysis,
        "detailed_results": results
    }
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("evaluation_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"retrieval_quality_{timestamp}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    logging.info(f"评估完成，结果已保存到: {output_path}")
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("检索质量评估摘要")
    print("=" * 60)
    print(f"评估图像数: {len(results)}")
    print("\n整体统计指标:")
    if statistics:
        for key, stats in statistics.items():
            print(f"  {key}:")
            print(f"    平均值: {stats['mean']:.4f}")
            print(f"    中位数: {stats['median']:.4f}")
            print(f"    标准差: {stats['std']:.4f}")
    
    if complexity_analysis:
        print("\n按复杂度分析:")
        for group_name, group_data in complexity_analysis.items():
            print(f"  {group_name} (n={group_data['count']}):")
            group_stats = group_data.get("statistics", {})
            if group_stats:
                # 显示关键指标
                for metric_key in ["semantic_similarity_mean", "bleu_mean", "concept_recall"]:
                    if metric_key in group_stats:
                        stats = group_stats[metric_key]
                        print(f"    {metric_key}: {stats['mean']:.4f} (median: {stats['median']:.4f})")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

