from PIL import Image, UnidentifiedImageError
import os
import logging
from typing import Optional, Any


def load_image(image_path: str, return_none_on_error: bool = False) -> Optional[Image.Image]:
    """Load an image from a path and return a PIL.Image in RGB mode.

    Args:
        image_path: Path to image file.
        return_none_on_error: If True, return None on error instead of raising.

    Returns:
        PIL.Image or None
    """
    try:
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except FileNotFoundError:
        logging.error(f"Image not found: {image_path}")
        if return_none_on_error:
            return None
        raise
    except UnidentifiedImageError:
        logging.error(f"Cannot identify image file: {image_path}")
        if return_none_on_error:
            return None
        raise
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        if return_none_on_error:
            return None
        raise


def preprocess_image(image: Image.Image, processor: Any, return_tensors: str = "pt") -> Any:
    """Preprocess a PIL image using a CLIP processor.

    Args:
        image: PIL.Image.Image
        processor: CLIPProcessor (from transformers)
        return_tensors: tensor format (default 'pt')

    Returns:
        Processor outputs (e.g., dict of tensors)
    """
    if processor is None:
        raise ValueError("processor is required for preprocess_image")
    # processor can accept a single image or list; we keep API simple
    return processor(images=[image], return_tensors=return_tensors)


def list_images_in_dir(images_dir, valid_extensions={'.jpg', '.jpeg', '.png'}):
    """List image files in a directory (non-recursive).

    Returns full paths.
    """
    out = []
    if not os.path.isdir(images_dir):
        return out
    for fname in os.listdir(images_dir):
        if os.path.splitext(fname)[1].lower() in valid_extensions:
            out.append(os.path.join(images_dir, fname))
    return out
