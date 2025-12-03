#!/usr/bin/env python3
"""Build COCO knowledge base: extract CLIP embeddings for images and build FAISS index.

Outputs:
  - coco_knowledge_base.faiss
  - image_id_to_captions.pkl
  - image_id_to_objects.pkl (new: object labels for object-aware retrieval)

This script reads settings from `configs/config.yaml`.
"""
import os
import sys
import json
import pickle
import logging
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
import faiss
import yaml
from transformers import CLIPModel, CLIPProcessor

from utils.image_utils import load_image


def setup_logging(level_str: str):
    level = getattr(logging, level_str.upper(), logging.ERROR)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")


def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_image_id_to_captions(coco_annotations_path, max_per_image=5):
    logging.info(f"Loading COCO annotations from {coco_annotations_path}")
    with open(coco_annotations_path, "r") as f:
        data = json.load(f)

    image_id_to_captions = {}
    # annotations: list of {image_id, caption}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        caption = ann.get("caption", "").strip()
        if not caption:
            continue
        image_id_to_captions.setdefault(img_id, []).append(caption)

    # truncate to max_per_image
    for k, v in list(image_id_to_captions.items()):
        image_id_to_captions[k] = v[:max_per_image]

    return image_id_to_captions, {img["id"]: img["file_name"] for img in data.get("images", [])}


def image_paths_and_ids(images_dir, image_id_to_filename):
    # build mapping of image_id -> full path if file exists
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


def extract_clip_features(model, processor, image_paths, device, batch_size=32):
    all_feats = []
    model.to(device)
    model.eval()
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i : i + batch_size]
        images = [load_image(p) for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        # move tensors to device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        with torch.no_grad():
            image_embeds = model.get_image_features(**inputs)

        # normalize
        image_embeds = image_embeds.cpu()
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        all_feats.append(image_embeds.numpy())

    if all_feats:
        return np.concatenate(all_feats, axis=0)
    else:
        # attempt to get projection dim from model config, fallback to 512
        dim = getattr(model.config, "projection_dim", None) or getattr(model.config, "projection_size", None) or 512
        return np.zeros((0, int(dim)), dtype="float32")


def build_faiss_index(vectors):
    d = vectors.shape[1]
    # using inner product with normalized vectors -> equivalent to cosine similarity
    index = faiss.IndexFlatIP(d)
    index.add(vectors)
    return index


def save_index(index, path):
    faiss.write_index(index, path)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_coco_object_annotations(instances_annotations_path):
    """加载并解析COCO instances标注文件，提取图像ID到物体类别的映射。
    
    Args:
        instances_annotations_path: instances_train2017.json文件路径
        
    Returns:
        tuple: (image_id_to_category_ids, category_id_to_name)
            - image_id_to_category_ids: dict, {image_id: set(category_id, ...)}
            - category_id_to_name: dict, {category_id: category_name}
    """
    logging.info(f"Loading COCO instances annotations from {instances_annotations_path}")
    
    if not os.path.exists(instances_annotations_path):
        logging.warning(f"Instances annotations file not found: {instances_annotations_path}")
        return {}, {}
    
    try:
        with open(instances_annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load instances annotations: {e}")
        return {}, {}
    
    # 构建category_id -> category_name映射
    category_id_to_name = {}
    for cat in data.get("categories", []):
        cat_id = cat.get("id")
        cat_name = cat.get("name", "").strip()
        if cat_id is not None and cat_name:
            category_id_to_name[cat_id] = cat_name
    
    logging.info(f"Loaded {len(category_id_to_name)} categories")
    
    # 构建image_id -> set(category_id)映射
    image_id_to_category_ids = {}
    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        cat_id = ann.get("category_id")
        
        if img_id is not None and cat_id is not None:
            if img_id not in image_id_to_category_ids:
                image_id_to_category_ids[img_id] = set()
            image_id_to_category_ids[img_id].add(cat_id)
    
    logging.info(f"Found object annotations for {len(image_id_to_category_ids)} images")
    
    return image_id_to_category_ids, category_id_to_name


def integrate_object_tags(image_ids, coco_objects_dict, category_id_to_name):
    """将COCO物体标注与当前处理的图像ID列表对齐，转换为物体名称列表。
    
    Args:
        image_ids: 当前知识库中实际存在的图像ID列表
        coco_objects_dict: image_id -> set(category_id)的映射
        category_id_to_name: category_id -> category_name的映射
        
    Returns:
        dict: {image_id: [object_name1, object_name2, ...]}，物体名称已去重并排序
    """
    image_id_to_objects = {}
    
    for img_id in image_ids:
        category_ids = coco_objects_dict.get(img_id, set())
        # 将category_id转换为category_name
        object_names = []
        for cat_id in category_ids:
            cat_name = category_id_to_name.get(cat_id)
            if cat_name:
                object_names.append(cat_name)
        
        # 去重并排序（保证一致性）
        image_id_to_objects[img_id] = sorted(list(set(object_names)))
    
    # 统计信息
    images_with_objects = sum(1 for objs in image_id_to_objects.values() if objs)
    total_objects = sum(len(objs) for objs in image_id_to_objects.values())
    
    logging.info(
        f"Integrated object tags: {images_with_objects}/{len(image_ids)} images have objects, "
        f"total {total_objects} object instances"
    )
    
    return image_id_to_objects


def main():
    cfg = load_config()
    setup_logging(cfg.get("log_config", {}).get("log_level", "ERROR"))

    clip_model_path = cfg.get("model_config", {}).get("clip_model_path")
    llm_model_path = cfg.get("model_config", {}).get("llm_model_path")
    images_dir = cfg.get("data_config", {}).get("coco_images_dir")
    annotations_path = cfg.get("data_config", {}).get("coco_annotations_path")
    instances_annotations_path = cfg.get("data_config", {}).get("coco_instances_annotations_path", "")
    top_k = cfg.get("retrieval_config", {}).get("top_k", 3)
    kb_path = cfg.get("knowledge_base_config", {}).get("knowledge_base_path")
    map_path = cfg.get("knowledge_base_config", {}).get("image_id_to_captions_path")
    objects_path = cfg.get("knowledge_base_config", {}).get("image_id_to_objects_path", "")

    if not all([clip_model_path, images_dir, annotations_path, kb_path, map_path]):
        logging.error("Missing configuration values. Please check configs/config.yaml")
        sys.exit(1)

    # build mapping
    image_id_to_captions, image_id_to_filename = build_image_id_to_captions(annotations_path)
    img_ids, img_paths = image_paths_and_ids(images_dir, image_id_to_filename)

    if len(img_paths) == 0:
        logging.error(f"No images found in {images_dir}. Aborting.")
        sys.exit(1)

    # load CLIP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        processor = CLIPProcessor.from_pretrained(clip_model_path)
        model = CLIPModel.from_pretrained(clip_model_path)
    except Exception as e:
        logging.error(f"Failed to load CLIP model from {clip_model_path}: {e}")
        sys.exit(1)

    # extract features in batches
    batch_size = 64
    try:
        import numpy as np

        vectors = extract_clip_features(model, processor, img_paths, device, batch_size=batch_size)
    except Exception as e:
        logging.error(f"Failed during feature extraction: {e}")
        raise

    # build faiss
    try:
        index = build_faiss_index(vectors.astype("float32"))
    except Exception as e:
        logging.error(f"Failed to build FAISS index: {e}")
        raise

    # save
    os.makedirs(os.path.dirname(kb_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(map_path) or ".", exist_ok=True)

    try:
        save_index(index, kb_path)
    except Exception as e:
        logging.error(f"Failed to save FAISS index to {kb_path}: {e}")
        raise

    # save mapping but only for images that exist
    image_id_to_captions_filtered = {img_id: image_id_to_captions.get(img_id, []) for img_id in img_ids}
    try:
        save_pickle(image_id_to_captions_filtered, map_path)
    except Exception as e:
        logging.error(f"Failed to save mapping to {map_path}: {e}")
        raise

    # Extract and save object tags if instances annotations are available
    if instances_annotations_path and objects_path:
        try:
            logging.info("Extracting object tags from COCO instances annotations...")
            coco_objects_dict, category_id_to_name = load_coco_object_annotations(instances_annotations_path)
            
            if coco_objects_dict and category_id_to_name:
                image_id_to_objects = integrate_object_tags(img_ids, coco_objects_dict, category_id_to_name)
                
                # Save object tags mapping
                os.makedirs(os.path.dirname(objects_path) or ".", exist_ok=True)
                save_pickle(image_id_to_objects, objects_path)
                logging.info(f"Saved object tags mapping to {objects_path}")
                logging.info(f"Object tags extracted for {len(image_id_to_objects)} images")
            else:
                logging.warning("No object annotations found or loaded, skipping object tags extraction")
        except Exception as e:
            logging.warning(f"Failed to extract object tags: {e}. Continuing without object tags.")
    else:
        logging.info("Object tags extraction skipped (instances_annotations_path or objects_path not configured)")

    print("Knowledge base build complete.")


if __name__ == "__main__":
    main()
