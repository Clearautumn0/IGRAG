import os
import logging
import pickle
from typing import List, Tuple, Union, Dict

import torch
import faiss
import yaml
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import numpy as np

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
        # keep full config for later (patch manager / text index)
        self.cfg = cfg

        # text knowledge base (for patch -> text retrieval)
        text_kb_path = cfg.get("knowledge_base_config", {}).get("text_knowledge_base_path")
        text_map_path = cfg.get("knowledge_base_config", {}).get("text_id_to_captions_path")
        # fallback to defaults in same output dir as image KB if not provided
        if not text_kb_path and kb_path:
            text_kb_path = os.path.join(os.path.dirname(kb_path) or ".", "text_knowledge_base.faiss")
        if not text_map_path and mapping_path:
            text_map_path = os.path.join(os.path.dirname(mapping_path) or ".", "text_descriptions.pkl")
        self.text_index = None
        self.text_id_to_caption = None
        if text_kb_path and os.path.exists(text_kb_path):
            try:
                self.text_index = faiss.read_index(text_kb_path)
            except Exception as e:
                logger.error(f"Failed to load text FAISS index from {text_kb_path}: {e}")
                self.text_index = None
        if text_map_path and os.path.exists(text_map_path):
            try:
                with open(text_map_path, "rb") as f:
                    self.text_id_to_caption = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load text id->caption mapping from {text_map_path}: {e}")
                self.text_id_to_caption = None

        if not clip_model_path or not kb_path or not mapping_path:
            logger.error("Missing required paths in config for retriever.")
            raise ValueError("Missing required config values for ImageRetriever")

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # retrieval defaults
        # support both old `top_k` and new `top_k_global` config keys for compatibility
        self.top_k_global = cfg.get("retrieval_config", {}).get(
            "top_k_global", cfg.get("retrieval_config", {}).get("top_k", 3)
        )

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
                self.image_id_to_captions = pickle.load(f)
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

    def extract_patch_features(self, image_patches: List[Image.Image], batch_size: int = 32) -> 'np.ndarray':
        """Extract CLIP features for a list of PIL image patches.

        Returns a numpy array of shape (N, D) with normalized vectors.
        """
        import numpy as np

        all_feats = []
        self.model.to(self.device)
        self.model.eval()
        for i in range(0, len(image_patches), batch_size):
            batch = image_patches[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            with torch.no_grad():
                image_embeds = self.model.get_image_features(**inputs)
            image_embeds = image_embeds.cpu()
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            all_feats.append(image_embeds.numpy())

        if all_feats:
            return np.concatenate(all_feats, axis=0)
        else:
            # fallback empty
            dim = getattr(self.model.config, "projection_dim", None) or getattr(self.model.config, "projection_size", None) or 512
            return np.zeros((0, int(dim)), dtype="float32")

    def retrieve_similar_texts_for_patches(self, query_patches: List[Image.Image], top_k: int = 3) -> Dict[int, List[str]]:
        """For each patch, retrieve the most similar text descriptions from the text FAISS index.

        Returns a dict mapping patch_index -> list of dicts {"caption": str, "score": float}.
        """
        results = {}
        if self.text_index is None or self.text_id_to_caption is None:
            logger.error("Text FAISS index or text mapping not available for patch retrieval.")
            return results

        # extract patch features in batch
        vectors = self.extract_patch_features(query_patches)
        if vectors.shape[0] == 0:
            return results

        # ensure float32
        vectors = vectors.astype("float32")

        # search text index
        D, I = self.text_index.search(vectors, top_k)

        for p_idx in range(I.shape[0]):
            hits = []
            seen = set()
            for col in range(I.shape[1]):
                tid = int(I[p_idx, col])
                if tid < 0:
                    continue
                score = float(D[p_idx, col]) if D is not None else 0.0
                # mapping may be dict or list
                caption = None
                if isinstance(self.text_id_to_caption, dict):
                    caption = self.text_id_to_caption.get(tid)
                elif isinstance(self.text_id_to_caption, (list, tuple)):
                    if 0 <= tid < len(self.text_id_to_caption):
                        caption = self.text_id_to_caption[tid]
                if caption:
                    # simple duplicate filtering by caption text
                    if caption in seen:
                        continue
                    seen.add(caption)
                    hits.append({"caption": caption, "score": score})
            results[p_idx] = hits

        return results

    def retrieve_similar_images(self, query_image: Union[str, Image.Image], top_k: int = None) -> List[Tuple[int, float]]:
        """Retrieve most similar images to query_image.

        Returns a list of (image_id, score) sorted by score desc.
        Scores are FAISS inner-product values (with normalized vectors ~ cosine similarity).
        """
        import numpy as np

        qv = self.extract_features(query_image)
        if qv.shape[0] == 0:
            return []

        # determine top_k
        if top_k is None:
            top_k = int(self.top_k_global)

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

    def get_retrieved_captions(self, query_image: Union[str, Image.Image], top_k: int = None) -> Dict:
        """Get both global and patch-level captions for a query image.

        Returns a dict with keys:
          - "global": list of dicts {image_id, score, captions}
          - "patches": dict mapping patch_index -> list of captions

        Note: Patch retrieval requires a text FAISS index and mapping to be provided in the config
        (knowledge_base_config.text_knowledge_base_path and text_id_to_captions_path).
        """
        if top_k is None:
            top_k = int(self.top_k_global)

        # global
        hits = self.retrieve_similar_images(query_image, top_k=top_k)
        global_out = []
        for image_id, score in hits:
            captions = self.image_id_to_captions.get(image_id, [])
            global_out.append({"image_id": image_id, "score": score, "captions": captions})

        # patches
        patch_out = {}
        # check if patch retrieval requested in config
        patch_cfg = self.cfg.get("patch_config", {}) if self.cfg else {}
        if patch_cfg.get("enabled", False) and self.text_index is not None and self.text_id_to_caption is not None:
            try:
                from core.patch_manager import PatchManager

                pm = PatchManager(self.cfg)
                patches, coords = pm.split_image(query_image)
                # retrieve top_k_patches per patch
                top_k_patches = int(patch_cfg.get("top_k_patches", patch_cfg.get("top_k_patches", 3)))
                patch_hits = self.retrieve_similar_texts_for_patches(patches, top_k=top_k_patches)
                # map by coords (x,y) for clarity
                for idx, cap_list in patch_hits.items():
                    coord = coords[int(idx)] if idx < len(coords) else {"x": 0, "y": 0}
                    key = f"{coord.get('x')},{coord.get('y')}"
                    patch_out[key] = cap_list
            except Exception as e:
                logger.error(f"Patch retrieval failed: {e}")

        return {"global": global_out, "patches": patch_out}
