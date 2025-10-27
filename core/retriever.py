import os
import logging
import pickle
from typing import List, Tuple, Union, Dict

import torch
import faiss
import yaml
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from utils.image_utils import load_image


logger = logging.getLogger(__name__)


class ImageRetriever:
    """Image retriever using CLIP + FAISS.

    Notes:
    - Expects a FAISS index built from normalized CLIP image embeddings.
    - Expects the mapping pickle to be a dict mapping image_id -> [captions],
      and relies on the insertion order of that dict to map FAISS row ids -> image ids.
    """

    def __init__(self, config: Union[dict, str]):
        """Initialize retriever.

        Args:
            config: dict or path to YAML config (same structure as configs/config.yaml)
        """
        if isinstance(config, str):
            with open(config, "r") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = config

        # logging level
        log_level = cfg.get("log_config", {}).get("log_level", "ERROR")
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.ERROR))

        clip_model_path = cfg.get("model_config", {}).get("clip_model_path")
        kb_path = cfg.get("knowledge_base_config", {}).get("knowledge_base_path")
        mapping_path = cfg.get("knowledge_base_config", {}).get("image_id_to_captions_path")

        if not clip_model_path or not kb_path or not mapping_path:
            logger.error("Missing required paths in config for retriever.")
            raise ValueError("Missing required config values for ImageRetriever")

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load CLIP
        try:
            self.processor = CLIPProcessor.from_pretrained(clip_model_path)
            self.model = CLIPModel.from_pretrained(clip_model_path).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load CLIP model from {clip_model_path}: {e}")
            raise

        # load faiss index
        if not os.path.exists(kb_path):
            logger.error(f"FAISS index file not found: {kb_path}")
            raise FileNotFoundError(kb_path)
        try:
            self.index = faiss.read_index(kb_path)
        except Exception as e:
            logger.error(f"Failed to read FAISS index from {kb_path}: {e}")
            raise

        # load mapping (image_id -> captions list)
        if not os.path.exists(mapping_path):
            logger.error(f"Mapping file not found: {mapping_path}")
            raise FileNotFoundError(mapping_path)
        try:
            with open(mapping_path, "rb") as f:
                self.image_id_to_captions: Dict[int, List[str]] = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load mapping from {mapping_path}: {e}")
            raise

        # preserve ordering to map faiss indices -> image ids
        self.index_id_to_image_id = list(self.image_id_to_captions.keys())

    def extract_features(self, image: Union[str, Image.Image]) -> 'np.ndarray':
        """Extract CLIP image features for a single image.

        Args:
            image: PIL.Image or path to image file

        Returns:
            numpy array (1, dim) normalized
        """
        import numpy as np

        if isinstance(image, str):
            img = load_image(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError("image must be a file path or PIL.Image.Image")

        inputs = self.processor(images=[img], return_tensors="pt")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        with torch.no_grad():
            embeds = self.model.get_image_features(**inputs)

        embeds = embeds.cpu().numpy()
        # normalize
        norms = (embeds ** 2).sum(axis=1, keepdims=True) ** 0.5
        norms[norms == 0] = 1.0
        embeds = embeds / norms
        return embeds

    def retrieve_similar_images(self, query_image: Union[str, Image.Image], top_k: int = 3) -> List[Tuple[int, float]]:
        """Retrieve most similar images to query_image.

        Returns a list of (image_id, score) sorted by score desc.
        Scores are FAISS inner-product values (with normalized vectors ~ cosine similarity).
        """
        import numpy as np

        qv = self.extract_features(query_image)
        if qv.shape[0] == 0:
            return []

        # faiss expects float32
        qv = qv.astype("float32")
        D, I = self.index.search(qv, top_k)

        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self.index_id_to_image_id):
                continue
            image_id = self.index_id_to_image_id[idx]
            results.append((image_id, float(score)))
        return results

    def get_retrieved_captions(self, query_image: Union[str, Image.Image], top_k: int = None) -> List[Dict]:
        """Get captions for retrieved images.

        Returns a list of dicts: {"image_id": id, "score": float, "captions": [..]}
        """
        if top_k is None:
            # fall back to config default if available
            top_k = 3

        hits = self.retrieve_similar_images(query_image, top_k=top_k)
        out = []
        for image_id, score in hits:
            captions = self.image_id_to_captions.get(image_id, [])
            out.append({"image_id": image_id, "score": score, "captions": captions})
        return out
