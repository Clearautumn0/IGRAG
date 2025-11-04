#!/usr/bin/env python3
"""IGRAG main entrypoint: integrate retriever and generator to produce a caption for a test image."""
import sys
import logging
import time
from pathlib import Path
import yaml
import argparse
from datetime import datetime

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


def run_pipeline(cfg: dict, test_image_path: str):
    start_time = time.time()
    retriever, generator = init_components(cfg)

    img = load_image(test_image_path)
    
    # 检查是否启用分块检索
    use_patch_retrieval = cfg.get("retrieval_config", {}).get("use_patch_retrieval", False)
    
    if use_patch_retrieval:
        print("Using patch retrieval mode...")
        retrieval_start = time.time()
        
        try:
            # 启用分块检索
            retriever.enable_patch_retrieval()
            
            # 执行分块检索
            retrieved_data = retriever.retrieve_with_patches(img)
            retrieval_time = time.time() - retrieval_start
            
            # 保存配置用于调试
            retrieved_data['_config'] = cfg
            
            # 显示检索统计信息
            global_descriptions = retrieved_data.get("global_descriptions", [])
            local_regions = retrieved_data.get("local_regions", [])
            
            print(f"\nRetrieval completed in {retrieval_time:.2f}s")
            print(f"Global descriptions: {len(global_descriptions)}")
            print(f"Local regions detected: {len(local_regions)}")
            
            # 显示全局描述
            if global_descriptions:
                print("\nGlobal similar images and their captions:")
                for item in global_descriptions:
                    print(f"- Image ID: {item.get('image_id')}  Score: {item.get('score'):.4f}")
                    for c in item.get('captions', [])[:2]:  # 只显示前2个描述
                        print(f"    • {c}")
            
            # 显示局部区域信息
            if local_regions:
                print("\nDetected local regions:")
                for idx, region in enumerate(local_regions):
                    print(f"  Region {idx+1}: {region.get('class_label')} "
                          f"(confidence: {region.get('confidence', 0):.3f})")
                    descs = region.get('descriptions', [])
                    if descs:
                        print(f"    Retrieved descriptions: {len(descs)}")
                        for desc in descs[:2]:  # 只显示前2个描述
                            print(f"      • {desc}")
            
            # 可选：保存调试图像
            patch_config = cfg.get("patch_config", {})
            if patch_config.get("save_debug_patches", False):
                debug_dir = Path("output/debug_patches")
                save_debug_patches(img, retrieved_data, debug_dir)
            
            # 构建提示词和生成
            prompt_start = time.time()
            prompt = generator.build_prompt(retrieved_data)
            prompt_time = time.time() - prompt_start
            
            gen_start = time.time()
            caption = generator.generate_caption(prompt, debug=True)
            gen_time = time.time() - gen_start
            
            print(f"\nTiming: Prompt building: {prompt_time:.2f}s, Generation: {gen_time:.2f}s")
            
            # 使用全局描述作为fallback数据源
            retrieved_for_fallback = global_descriptions
            
        except Exception as e:
            logging.error(f"Patch retrieval failed: {e}, falling back to global retrieval")
            # 回退到全局检索
            retrieved_for_fallback = retriever.get_retrieved_captions(img)
            retrieved_data = {"global_descriptions": retrieved_for_fallback, "local_regions": []}
            prompt = generator.build_prompt(retrieved_data)
            caption = generator.generate_caption(prompt, debug=True)
    else:
        # 传统全局检索流程
        print("Using global retrieval mode...")
        retrieval_start = time.time()
        retrieved_for_fallback = retriever.get_retrieved_captions(img)
        retrieval_time = time.time() - retrieval_start
        
        if not retrieved_for_fallback:
            print("No similar images found or knowledge base is empty.")
            return

        # show retrieved summaries
        print("Retrieved images and their captions:")
        for item in retrieved_for_fallback:
            print(f"- Image ID: {item.get('image_id')}  Score: {item.get('score'):.4f}")
            for c in item.get('captions', []):
                print(f"    • {c}")

        # build prompt and generate
        prompt = generator.build_prompt(retrieved_for_fallback)
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
            for item in retrieved_for_fallback:
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
            for item in retrieved_for_fallback:
                caps = item.get("captions", [])
                if caps:
                    parts.append(caps[0].strip().rstrip('.'))
            if parts:
                caption = ";".join(parts)

    total_time = time.time() - start_time
    print("\nGenerated caption:")
    print(caption)
    print(f"\nTotal pipeline time: {total_time:.2f}s")


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
