#!/usr/bin/env python3
"""IGRAG main entrypoint: integrate retriever and generator to produce a caption for a test image."""
import sys
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
import yaml
import argparse
import os
import shutil
from datetime import datetime
import copy

from core.retriever import ImageRetriever
from core.generator import CaptionGenerator
from utils.image_utils import load_image


@dataclass
class CaptionResult:
    caption: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_config(config: Optional[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
    if config is None:
        return load_config()
    if isinstance(config, str):
        return load_config(config)
    # return a copy to avoid mutating caller's config
    return copy.deepcopy(config)


def setup_logging(cfg: dict):
    level = cfg.get("log_config", {}).get("log_level", "ERROR")
    logging.basicConfig(level=getattr(logging, level.upper(), logging.ERROR), format="%(asctime)s %(levelname)s: %(message)s")


def init_components(cfg: dict):
    retriever = ImageRetriever(cfg)
    generator = CaptionGenerator(cfg)
    return retriever, generator


def save_debug_patches(img, retrieved_data, output_dir: Path):
    """保存检测到的显著区域图像用于调试。"""
    try:
        local_regions = retrieved_data.get('local_regions', [])
        if not local_regions:
            return
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 为每个区域保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for idx, region in enumerate(local_regions):
            bbox = region.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                # 确保坐标在图像范围内
                x1 = max(0, min(x1, img.width))
                y1 = max(0, min(y1, img.height))
                x2 = max(x1, min(x2, img.width))
                y2 = max(y1, min(y2, img.height))
                
                if x2 > x1 and y2 > y1:
                    patch = img.crop((x1, y1, x2, y2))
                    class_label = region.get('class_label', 'unknown').replace('/', '_')
                    patch_path = output_dir / f"{timestamp}_patch_{idx+1}_{class_label}.jpg"
                    patch.save(patch_path)
                    logging.info(f"Saved debug patch: {patch_path}")
    except Exception as e:
        logging.warning(f"Failed to save debug patches: {e}")


def execute_pipeline(
    cfg: dict,
    test_image_path: str,
    *,
    emit_output: bool = False,
    show_prompt: bool = False,
) -> CaptionResult:
    start_time = time.time()
    retriever, generator = init_components(cfg)

    img = load_image(test_image_path)

    runtime_cfg = cfg.get("runtime_config", {})
    verbose = runtime_cfg.get("verbose", False)
    mode = runtime_cfg.get("mode", "deploy")
    use_patch_retrieval = cfg.get("retrieval_config", {}).get("use_patch_retrieval", False)

    metadata: Dict[str, Any] = {
        "source_image": str(test_image_path),
        "runtime_mode": mode,
        "retrieval": {
            "mode": "patch" if use_patch_retrieval else "global",
        },
        "generation": {
            "fallback_used": False,
            "valid": True,
        },
        "timestamps": {
            "started_at": datetime.utcnow().isoformat() + "Z",
        },
    }

    def _emit(text: str = "", *, force: bool = False):
        if force or emit_output or verbose:
            print(text)

    def _print_model_prompt(prompt_text: str):
        if verbose or show_prompt:
            _emit("\n=== MODEL PROMPT INPUT ===\n", force=True)
            _emit(prompt_text, force=True)
            _emit("", force=True)

    retrieval_time: Optional[float] = None
    prompt_time: Optional[float] = None
    gen_time: Optional[float] = None
    retrieved_for_fallback = []

    if use_patch_retrieval:
        if verbose:
            _emit("Using patch retrieval mode...")
        retrieval_start = time.time()

        try:
            retriever.enable_patch_retrieval()
            retrieved_data = retriever.retrieve_with_patches(img)
            retrieval_time = time.time() - retrieval_start

            retrieved_data["_config"] = cfg

            global_descriptions = retrieved_data.get("global_descriptions", [])
            local_regions = retrieved_data.get("local_regions", [])

            metadata["retrieval"].update(
                {
                    "global_candidates": len(global_descriptions),
                    "local_regions_detected": len(local_regions),
                    "retrieval_time_sec": retrieval_time,
                }
            )

            if verbose:
                _emit(f"\nRetrieval completed in {retrieval_time:.2f}s")
                _emit(f"Global descriptions: {len(global_descriptions)}")
                _emit(f"Local regions detected: {len(local_regions)}")

                if global_descriptions:
                    _emit("\nGlobal similar images and their captions:")
                    for item in global_descriptions:
                        score = item.get("score")
                        _emit(f"- Image ID: {item.get('image_id')}  Score: {score:.4f}" if score is not None else f"- Image ID: {item.get('image_id')}")
                        for c in item.get("captions", [])[:2]:
                            _emit(f"    • {c}")

                if local_regions:
                    _emit("\nDetected local regions:")
                    for idx, region in enumerate(local_regions):
                        _emit(
                            f"  Region {idx+1}: {region.get('class_label')} "
                            f"(confidence: {region.get('confidence', 0):.3f})"
                        )
                        descs = region.get("descriptions", [])
                        if descs:
                            _emit(f"    Retrieved descriptions: {len(descs)}")
                            for desc in descs[:2]:
                                _emit(f"      • {desc}")

            patch_config = cfg.get("patch_config", {})
            if patch_config.get("save_debug_patches", False):
                debug_dir = Path("output/debug_patches")
                save_debug_patches(img, retrieved_data, debug_dir)

            prompt_start = time.time()
            prompt = generator.build_prompt(retrieved_data)
            prompt_time = time.time() - prompt_start
            _print_model_prompt(prompt)

            gen_start = time.time()
            caption = generator.generate_caption(prompt, debug=verbose)
            gen_time = time.time() - gen_start

            if verbose:
                _emit(f"\nTiming: Prompt building: {prompt_time:.2f}s, Generation: {gen_time:.2f}s")

            retrieved_for_fallback = global_descriptions

        except Exception as e:
            logging.error(f"Patch retrieval failed: {e}, falling back to global retrieval")
            metadata["retrieval"]["mode"] = "patch_fallback_global"
            metadata["retrieval"]["patch_error"] = str(e)

            retrieved_for_fallback = retriever.get_retrieved_captions(img)
            retrieval_time = time.time() - retrieval_start
            metadata["retrieval"]["retrieval_time_sec"] = retrieval_time
            metadata["retrieval"]["global_candidates"] = len(retrieved_for_fallback)

            retrieved_data = {"global_descriptions": retrieved_for_fallback, "local_regions": []}
            prompt = generator.build_prompt(retrieved_data)
            _print_model_prompt(prompt)
            gen_start = time.time()
            caption = generator.generate_caption(prompt, debug=verbose)
            gen_time = time.time() - gen_start
    else:
        if verbose:
            _emit("Using global retrieval mode...")
        retrieval_start = time.time()
        retrieved_for_fallback = retriever.get_retrieved_captions(img)
        retrieval_time = time.time() - retrieval_start
        metadata["retrieval"].update(
            {
                "global_candidates": len(retrieved_for_fallback),
                "retrieval_time_sec": retrieval_time,
            }
        )

        if not retrieved_for_fallback:
            metadata["generation"]["valid"] = False
            metadata["status"] = "no_similar_images"
            _emit("No similar images found or knowledge base is empty.")
            metadata["timing"] = {
                "total_sec": time.time() - start_time,
                "retrieval_sec": retrieval_time,
                "prompt_sec": None,
                "generation_sec": None,
            }
            return CaptionResult(caption="", metadata=metadata)

        if verbose:
            _emit("Retrieved images and their captions:")
            for item in retrieved_for_fallback:
                score = item.get("score")
                _emit(f"- Image ID: {item.get('image_id')}  Score: {score:.4f}" if score is not None else f"- Image ID: {item.get('image_id')}")
                for c in item.get("captions", []):
                    _emit(f"    • {c}")

        prompt_start = time.time()
        prompt = generator.build_prompt(retrieved_for_fallback)
        prompt_time = time.time() - prompt_start
        _print_model_prompt(prompt)

        gen_start = time.time()
        caption = generator.generate_caption(prompt, debug=verbose)
        gen_time = time.time() - gen_start

    def _is_valid(s: str) -> bool:
        if not s:
            return False
        t = s.strip()
        import re

        core = re.sub(r"[\W_]+", "", t)
        return len(core) >= 6

    if not _is_valid(caption):
        metadata["generation"]["valid"] = False
        if mode == "deploy":
            parts = []
            for item in retrieved_for_fallback:
                caps = item.get("captions", [])
                if caps:
                    parts.append(caps[0].strip().rstrip("."))
            if parts:
                caption = ";".join(parts)
                metadata["generation"]["fallback_used"] = True
        elif mode == "test":
            _emit("生成失败!")
            metadata["status"] = "generation_failed"
            metadata["generation"]["fallback_used"] = False
            metadata["timing"] = {
                "total_sec": time.time() - start_time,
                "retrieval_sec": retrieval_time,
                "prompt_sec": prompt_time,
                "generation_sec": gen_time,
            }
            return CaptionResult(caption="", metadata=metadata)
        else:
            parts = []
            for item in retrieved_for_fallback:
                caps = item.get("captions", [])
                if caps:
                    parts.append(caps[0].strip().rstrip("."))
            if parts:
                caption = ";".join(parts)
                metadata["generation"]["fallback_used"] = True

    total_time = time.time() - start_time
    metadata["timing"] = {
        "total_sec": total_time,
        "retrieval_sec": retrieval_time,
        "prompt_sec": prompt_time,
        "generation_sec": gen_time,
    }

    if verbose:
        _emit("\nGenerated caption:")
        _emit(caption)
        _emit(f"\nTotal pipeline time: {total_time:.2f}s")
    elif emit_output:
        _emit(caption, force=True)

    return CaptionResult(caption=caption, metadata=metadata)


def generate_caption(
    image_path: Union[str, Path],
    config: Optional[Union[str, Dict[str, Any]]] = None,
    *,
    emit_output: bool = False,
    show_prompt: bool = False,
    configure_logging: bool = True,
) -> CaptionResult:
    cfg = _resolve_config(config)
    if configure_logging:
        setup_logging(cfg)
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
    return execute_pipeline(cfg, str(image_path), emit_output=emit_output, show_prompt=show_prompt)


def _clear_debug_directory(folder: Union[str, Path]):
    folder_path = Path(folder)
    if not folder_path.exists():
        return
    for filename in folder_path.iterdir():
        try:
            if filename.is_file() or filename.is_symlink():
                filename.unlink()
            elif filename.is_dir():
                shutil.rmtree(filename)
        except Exception as e:
            print(f"Failed to delete {filename}. Reason: {e}")


def main():
    _clear_debug_directory(Path("./output/debug_patches"))

    parser = argparse.ArgumentParser(description="IGRAG: image retrieval-augmented generation")
    parser.add_argument("--mode", type=str, choices=["deploy", "test"], help="Override runtime mode (deploy/test)")
    parser.add_argument("--i", "--input", dest="input", type=str, help="Input image path (default: input/test_image.jpg)")
    parser.add_argument("--config", dest="config_path", type=str, help="Custom config file path")
    parser.add_argument("--model", type=str, choices=["qwen", "flan-t5"], help="Model type to use (qwen or flan-t5)")
    args = parser.parse_args()

    try:
        base_config = _resolve_config(args.config_path)
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        sys.exit(1)

    if args.mode:
        base_config.setdefault("runtime_config", {})["mode"] = args.mode
    
    # 根据--model参数设置模型路径和类型
    if args.model:
        model_config = base_config.setdefault("model_config", {})
        if args.model == "flan-t5":
            model_config["llm_model_path"] = "../models/flan-t5-base/"
            model_config["model_type"] = "flan-t5"
        elif args.model == "qwen":
            model_config["llm_model_path"] = "../models/Qwen2.5-3B-instruct/"
            model_config["model_type"] = "qwen"

    setup_logging(base_config)

    default_image = Path("input/test_image.jpg")
    test_image_path = Path(args.input) if args.input else default_image
    if not test_image_path.exists():
        logging.error(
            f"Test image not found at {test_image_path}. Please place a test image at this path or pass --i <path>."
        )
        sys.exit(1)

    try:
        result = generate_caption(
            test_image_path,
            base_config,
            emit_output=True,
            show_prompt=True,
            configure_logging=False,
        )
        if base_config.get("runtime_config", {}).get("verbose", False):
            logging.debug("Caption metadata: %s", result.metadata)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
