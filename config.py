"""
图像描述生成框架 - 配置文件
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据路径配置
DATA_CONFIG = {
    "coco_root": "/path/to/coco2017",  # 需要用户设置COCO数据集路径
    "train_images": "train2017",
    "val_images": "val2017", 
    "annotations": "annotations",
    "karpathy_split": True,  # 使用Karpathy标准划分
}

# 模型配置
MODEL_CONFIG = {
    "clip_model": "ViT-B/32",  # CLIP模型版本
    "clip_checkpoint": None,   # 可选：自定义CLIP权重路径
    "llm_model": "google/flan-t5-large",  # 大语言模型
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
}

# 特征提取配置
FEATURE_CONFIG = {
    "global_feature_dim": 512,  # CLIP ViT-B/32的全局特征维度
    "patch_feature_dim": 512,  # 每个patch的特征维度
    "num_patches": 49,  # ViT-B/32的patch数量 (7x7)
    "patch_size": 32,   # patch大小
}

# 检索配置
RETRIEVAL_CONFIG = {
    "global_top_k": 5,    # 全局检索Top-K
    "local_top_m": 10,    # 局部检索Top-M
    "key_patch_strategy": "norm",  # 关键块选择策略: "norm", "random", "attention"
    "num_key_patches": 3,  # 选择的关键块数量
    "similarity_threshold": 0.7,  # 相似度阈值
}

# 生成配置
GENERATION_CONFIG = {
    "max_length": 256,     # 生成文本最大长度
    "beam_size": 3,        # beam search大小
    "temperature": 0.7,    # 生成温度
    "do_sample": True,     # 是否采样
    "top_p": 0.9,         # nucleus sampling参数
    "repetition_penalty": 1.1,  # 重复惩罚
}

# FAISS索引配置
INDEX_CONFIG = {
    "index_type": "IVF",  # 索引类型: "Flat", "IVF", "HNSW"
    "nlist": 1000,       # IVF索引的聚类中心数
    "nprobe": 10,        # 搜索时的聚类中心数
    "use_gpu": False,    # 是否使用GPU加速
}

# 评估配置
EVALUATION_CONFIG = {
    "metrics": ["bleu", "meteor", "cider", "spice"],
    "coco_eval_path": "/path/to/coco-caption",  # COCO评估工具路径
}

# 日志和输出配置
OUTPUT_CONFIG = {
    "log_level": "INFO",
    "save_dir": PROJECT_ROOT / "outputs",
    "cache_dir": PROJECT_ROOT / "cache",
    "checkpoint_dir": PROJECT_ROOT / "checkpoints",
    "wandb_project": "image-caption-global-local",  # Weights & Biases项目名
}

# 提示模板配置
PROMPT_TEMPLATE = """
你是一个专业的图像描述生成器。

整体相似的图片描述：
{global_descriptions}

在关键局部区域相似的图片描述：
{local_descriptions}

请综合分析以上描述，生成一个全新、准确且详尽的图片描述。
"""

# 创建必要的目录
for dir_path in [OUTPUT_CONFIG["save_dir"], OUTPUT_CONFIG["cache_dir"], OUTPUT_CONFIG["checkpoint_dir"]]:
    dir_path.mkdir(parents=True, exist_ok=True)
