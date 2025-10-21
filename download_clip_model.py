#!/usr/bin/env python3
"""
下载 CLIP 模型文件的脚本
"""
import os
import requests
from pathlib import Path
import hashlib
import json

def download_file(url, filepath, chunk_size=8192):
    """下载文件并显示进度"""
    print(f"正在下载: {url}")
    print(f"保存到: {filepath}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\r进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
    
    print(f"\n下载完成: {filepath}")
    return filepath

def create_huggingface_cache_structure():
    """创建 HuggingFace 缓存目录结构"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / "models--timm--vit_base_patch32_clip_224.openai"
    snapshots_dir = model_dir / "snapshots"
    
    # 创建目录
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建一个唯一的快照ID（模拟HuggingFace的缓存结构）
    snapshot_id = "main"
    snapshot_dir = snapshots_dir / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    return snapshot_dir

def download_clip_model():
    """下载 CLIP 模型文件"""
    print("开始下载 CLIP 模型文件...")
    
    # 创建缓存目录
    snapshot_dir = create_huggingface_cache_structure()
    
    # 模型文件URL列表
    model_files = {
        "open_clip_pytorch_model.bin": "https://huggingface.co/timm/vit_base_patch32_clip_224.openai/resolve/main/open_clip_pytorch_model.bin",
        "open_clip_model.safetensors": "https://huggingface.co/timm/vit_base_patch32_clip_224.openai/resolve/main/open_clip_model.safetensors",
        "config.json": "https://huggingface.co/timm/vit_base_patch32_clip_224.openai/resolve/main/config.json",
        "preprocessor_config.json": "https://huggingface.co/timm/vit_base_patch32_clip_224.openai/resolve/main/preprocessor_config.json"
    }
    
    downloaded_files = []
    
    for filename, url in model_files.items():
        filepath = snapshot_dir / filename
        try:
            if not filepath.exists():
                download_file(url, filepath)
                downloaded_files.append(filepath)
            else:
                print(f"文件已存在，跳过: {filepath}")
        except Exception as e:
            print(f"下载失败 {filename}: {e}")
            continue
    
    # 创建 refs/main 文件
    refs_dir = snapshot_dir.parent.parent / "refs"
    refs_dir.mkdir(exist_ok=True)
    main_ref = refs_dir / "main"
    snapshot_id = snapshot_dir.name
    with open(main_ref, 'w') as f:
        f.write(snapshot_id)
    
    print(f"\n下载完成！文件保存在: {snapshot_dir}")
    print("下载的文件:")
    for file in downloaded_files:
        print(f"  - {file}")
    
    return snapshot_dir

if __name__ == "__main__":
    try:
        download_clip_model()
    except Exception as e:
        print(f"下载过程中出现错误: {e}")
        print("请检查网络连接或尝试使用代理")
