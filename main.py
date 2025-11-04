#!/usr/bin/env python3
"""IGRAG main entrypoint: integrate retriever and generator to produce a caption for a test image."""
import sys
import logging
from pathlib import Path
import yaml
import argparse

from core.retriever import ImageRetriever
from core.generator import CaptionGenerator
from utils.image_utils import load_image


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(cfg: dict):
    level = cfg.get("log_config", {}).get("log_level", "ERROR")
    logging.basicConfig(level=getattr(logging, level.upper(), logging.ERROR), format="%(asctime)s %(levelname)s: %(message)s")


def init_components(cfg: dict):
    retriever = ImageRetriever(cfg)
    generator = CaptionGenerator(cfg)
    return retriever, generator


def run_pipeline(cfg: dict, test_image_path: str):
    retriever, generator = init_components(cfg)

    img = load_image(test_image_path)
    # Global retrieval + optional patch retrieval
    try:
        retrieved = retriever.get_retrieved_captions(img)
    except Exception as e:
        logging.error(f"Retrieval failed: {e}")
        print("No similar images found or knowledge base is empty.")
        retrieved = {"global": [], "patches": {}}

    # validate retrieved structure
    if not isinstance(retrieved, dict):
        # backwards compatibility: if retriever returned a list, convert to dict
        global_list = retrieved if isinstance(retrieved, list) else []
        retrieved = {"global": global_list, "patches": {}}

    if not retrieved.get("global") and not retrieved.get("patches"):
        print("No similar images found or knowledge base is empty.")
        return

    # show retrieved summaries (global)
    print("Retrieved images and their captions (global):")
    for item in retrieved.get("global", []):
        try:
            print(f"- Image ID: {item.get('image_id')}  Score: {item.get('score'):.4f}")
            for c in item.get('captions', []):
                print(f"    • {c}")
        except Exception:
            # defensive: skip malformed entries
            continue

    # show patch-level summaries if present
    if retrieved.get("patches"):
        print("\nRetrieved patch-level descriptions:")
        for region_key, caps in retrieved.get("patches", {}).items():
            # region_key formatted as 'x,y' earlier
            print(f"- Region {region_key}:")
            if not caps:
                print("    • (no captions)")
                continue
            # caps expected to be list of dicts {caption, score} or strings
            for entry in caps:
                if isinstance(entry, dict):
                    cap = entry.get("caption") or entry.get("captions") or str(entry)
                    score = entry.get("score")
                    if score is not None:
                        print(f"    • {cap}  (score: {score:.4f})")
                    else:
                        print(f"    • {cap}")
                else:
                    print(f"    • {entry}")

    # build prompt and generate (generator supports both legacy list and new dict)
    prompt = generator.build_prompt(retrieved)
    # debug=True to print prompt and tokenization to help diagnose generation issues
    caption = generator.generate_caption(prompt, debug=True)

    # basic validity check for generated caption
    def _is_valid(s: str) -> bool:
        if not s:
            return False
        t = s.strip()
        # require at least 6 non-punctuation characters
        import re

        core = re.sub(r"[\W_]+", "", t)
        return len(core) >= 6

    if not _is_valid(caption):
        mode = cfg.get("runtime_config", {}).get("mode", "deploy")
        if mode == "deploy":
            # fallback: synthesize from retrieved captions (take first caption per image), use English semicolon
            parts = []
            for item in retrieved:
                caps = item.get("captions", [])
                if caps:
                    parts.append(caps[0].strip().rstrip('.'))
            if parts:
                caption = ";".join(parts)
                if caption and caption[-1] not in list("。！？.!?"):
                    # keep punctuation consistent in English fallback
                    caption = caption
        elif mode == "test":
            # In test mode, report failure clearly and do not fallback
            print("生成失败!")
            return
        else:
            # default to deploy behavior for unknown mode
            parts = []
            for item in retrieved:
                caps = item.get("captions", [])
                if caps:
                    parts.append(caps[0].strip().rstrip('.'))
            if parts:
                caption = ";".join(parts)

    print("\nGenerated caption:")
    print(caption)


def main():
    parser = argparse.ArgumentParser(description="IGRAG: image retrieval-augmented generation")
    parser.add_argument("--mode", type=str, choices=["deploy", "test"], help="Override runtime mode (deploy/test)")
    parser.add_argument("--i", "--input", dest="input", type=str, help="Input image path (default: input/test_image.jpg)")
    args = parser.parse_args()

    try:
        cfg = load_config()
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        sys.exit(1)

    # allow CLI override of runtime mode
    if args.mode:
        cfg.setdefault("runtime_config", {})["mode"] = args.mode

    setup_logging(cfg)

    # determine test image path
    default_image = Path("input/test_image.jpg")
    test_image_path = Path(args.input) if args.input else default_image
    if not test_image_path.exists():
        logging.error(f"Test image not found at {test_image_path}. Please place a test image at this path or pass --i <path>.")
        sys.exit(1)

    try:
        run_pipeline(cfg, str(test_image_path))
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
