"""辅助函数：用于物体感知检索的分数计算和重排序逻辑。"""
import logging
from typing import List, Set, Dict, Tuple

logger = logging.getLogger(__name__)


def compute_object_match_score(query_objects: Set[str], candidate_objects: Set[str]) -> float:
    """计算两个物体集合的匹配度分数。
    
    使用Jaccard相似度（交集/并集）作为匹配度分数。
    
    Args:
        query_objects: 查询图像的物体类别集合
        candidate_objects: 候选图像的物体类别集合
        
    Returns:
        匹配度分数，范围[0.0, 1.0]
    """
    if not query_objects and not candidate_objects:
        # 两者都为空，返回1.0（完全匹配）
        return 1.0
    
    if not query_objects or not candidate_objects:
        # 一方为空，返回0.0（完全不匹配）
        return 0.0
    
    # 计算交集和并集
    intersection = query_objects & candidate_objects
    union = query_objects | candidate_objects
    
    if not union:
        return 0.0
    
    # Jaccard相似度
    jaccard = len(intersection) / len(union)
    return jaccard


def compute_hybrid_score(
    clip_score: float,
    object_match_score: float,
    object_weight: float = 0.5
) -> float:
    """计算混合分数，融合CLIP相似度和物体匹配度。
    
    Args:
        clip_score: CLIP相似度分数（FAISS内积，对于归一化向量范围在[-1, 1]，实际通常为[0, 1]）
        object_match_score: 物体匹配度分数（范围[0.0, 1.0]）
        object_weight: 物体匹配度的权重（范围[0.0, 1.0]），clip权重为(1 - object_weight)
        
    Returns:
        混合分数
    """
    # 将CLIP分数归一化到[0, 1]范围
    # FAISS内积对于归一化向量范围在[-1, 1]，但实际检索结果通常为[0, 1]
    # 为了安全，我们处理[-1, 1]范围的情况
    if clip_score < 0:
        # 如果分数为负，将其映射到[0, 1]
        normalized_clip_score = (clip_score + 1.0) / 2.0
    else:
        # 如果分数为正，直接使用（但限制在[0, 1]范围内）
        normalized_clip_score = min(1.0, max(0.0, clip_score))
    
    # 确保归一化后的分数在[0, 1]范围内
    normalized_clip_score = max(0.0, min(1.0, normalized_clip_score))
    
    # 线性融合
    hybrid_score = (1.0 - object_weight) * normalized_clip_score + object_weight * object_match_score
    
    return hybrid_score


def reorder_by_objects(
    candidates: List[Dict],
    query_objects: Set[str],
    candidate_objects_map: Dict[int, Set[str]],
    object_weight: float = 0.5
) -> List[Dict]:
    """根据物体匹配度对候选结果进行重排序。
    
    Args:
        candidates: 候选结果列表，每个元素包含至少 {"image_id": int, "score": float}
        query_objects: 查询图像的物体类别集合
        candidate_objects_map: 候选图像ID到物体集合的映射
        object_weight: 物体匹配度的权重
        
    Returns:
        重排序后的候选结果列表，每个元素增加了 "object_match_score" 和 "hybrid_score" 字段
    """
    if not query_objects:
        # 如果查询图像没有检测到物体，直接返回原始结果（按CLIP分数排序）
        logger.debug("Query image has no detected objects, returning original order")
        return candidates
    
    enhanced_candidates = []
    
    for candidate in candidates:
        image_id = candidate.get("image_id")
        clip_score = candidate.get("score", 0.0)
        
        # 获取候选图像的物体集合
        candidate_objects = candidate_objects_map.get(image_id, set())
        
        # 计算物体匹配度
        object_match_score = compute_object_match_score(query_objects, candidate_objects)
        
        # 计算混合分数
        hybrid_score = compute_hybrid_score(clip_score, object_match_score, object_weight)
        
        # 创建增强的候选结果
        enhanced = candidate.copy()
        enhanced["object_match_score"] = object_match_score
        enhanced["hybrid_score"] = hybrid_score
        enhanced["_query_objects"] = list(query_objects)
        enhanced["_candidate_objects"] = list(candidate_objects)
        
        enhanced_candidates.append(enhanced)
    
    # 按混合分数降序排序
    enhanced_candidates.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    
    return enhanced_candidates

