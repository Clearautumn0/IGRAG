#!/usr/bin/env python3
"""Build COCO knowledge base: extract CLIP embeddings for images and build FAISS index.

Outputs:
  - coco_knowledge_base.faiss
  - image_id_to_captions.pkl

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


def main():
    cfg = load_config()
    setup_logging(cfg.get("log_config", {}).get("log_level", "ERROR"))

    clip_model_path = cfg.get("model_config", {}).get("clip_model_path")
    llm_model_path = cfg.get("model_config", {}).get("llm_model_path")
    images_dir = cfg.get("data_config", {}).get("coco_images_dir")
    annotations_path = cfg.get("data_config", {}).get("coco_annotations_path")
    # maintain compatibility: prefer top_k_global, fall back to top_k
    top_k = cfg.get("retrieval_config", {}).get("top_k_global",
                                                  cfg.get("retrieval_config", {}).get("top_k", 3))
    kb_path = cfg.get("knowledge_base_config", {}).get("knowledge_base_path")
    map_path = cfg.get("knowledge_base_config", {}).get("image_id_to_captions_path")

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

    # --- Build text knowledge base (CLIP text encoder) ---
    text_kb_path = cfg.get("knowledge_base_config", {}).get(
        "text_knowledge_base_path", os.path.join(os.path.dirname(kb_path) or ".", "text_knowledge_base.faiss")
    )
    text_map_path = cfg.get("knowledge_base_config", {}).get(
        "text_id_to_captions_path", os.path.join(os.path.dirname(map_path) or ".", "text_descriptions.pkl")
    )

    # flatten all captions into a list (unique)
    all_texts = []
    seen_texts = set()
    for caps in image_id_to_captions.values():
        for c in caps:
            if not c:
                continue
            if c in seen_texts:
                continue
            seen_texts.add(c)
            all_texts.append(c)

    if len(all_texts) == 0:
        logging.warning("No text descriptions found to build text knowledge base.")
        print("Knowledge base build complete.")
        return

    # extract text features using CLIP text encoder
    def extract_text_features(model, processor, texts, device, batch_size=256):
        all_feats = []
        model.to(device)
        model.eval()
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting text features"):
            batch_texts = texts[i : i + batch_size]
            inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
            # move tensors to device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            with torch.no_grad():
                text_embeds = model.get_text_features(**inputs)

            # normalize
            text_embeds = text_embeds.cpu()
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            all_feats.append(text_embeds.numpy())

        if all_feats:
            return np.concatenate(all_feats, axis=0)
        else:
            dim = getattr(model.config, "text_projection_dim", None) or getattr(model.config, "projection_dim", None) or 512
            return np.zeros((0, int(dim)), dtype="float32")

    try:
        text_vectors = extract_text_features(model, processor, all_texts, device, batch_size=256)
    except Exception as e:
        logging.error(f"Failed during text feature extraction: {e}")
        raise

    try:
        text_index = build_faiss_index(text_vectors.astype("float32"))
    except Exception as e:
        logging.error(f"Failed to build text FAISS index: {e}")
        raise

    # save text index and mapping
    os.makedirs(os.path.dirname(text_kb_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(text_map_path) or ".", exist_ok=True)

    try:
        save_index(text_index, text_kb_path)
    except Exception as e:
        logging.error(f"Failed to save text FAISS index to {text_kb_path}: {e}")
        raise

    # text mapping: index -> caption
    try:
        save_pickle(all_texts, text_map_path)
    except Exception as e:
        logging.error(f"Failed to save text mapping to {text_map_path}: {e}")
        raise

    print("Knowledge base build complete.")


if __name__ == "__main__":
    main()
