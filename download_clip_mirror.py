#!/usr/bin/env python3
"""
使用镜像源下载 CLIP 模型文件的脚本
"""
import os
import requests
from pathlib import Path
import json

def download_from_mirror():
    """从镜像源下载模型文件"""
    print("尝试从镜像源下载 CLIP 模型...")
    
    # 创建缓存目录
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / "models--timm--vit_base_patch32_clip_224.openai"
    snapshots_dir = model_dir / "snapshots" / "main"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    
    # 尝试不同的镜像源
    mirrors = [
        "https://hf-mirror.com/timm/vit_base_patch32_clip_224.openai/resolve/main/",
        "https://huggingface.co/timm/vit_base_patch32_clip_224.openai/resolve/main/",
        "https://cdn-lfs.huggingface.co/timm/vit_base_patch32_clip_224.openai/resolve/main/"
    ]
    
    model_files = [
        "open_clip_pytorch_model.bin",
        "open_clip_model.safetensors", 
        "config.json",
        "preprocessor_config.json"
    ]
    
    for mirror in mirrors:
        print(f"\n尝试镜像源: {mirror}")
        success = True
        
        for filename in model_files:
            url = mirror + filename
            filepath = snapshots_dir / filename
            
            if filepath.exists():
                print(f"文件已存在，跳过: {filename}")
                continue
                
            try:
                print(f"下载: {filename}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"✓ 下载成功: {filename}")
                
            except Exception as e:
                print(f"✗ 下载失败 {filename}: {e}")
                success = False
                break
        
        if success:
            print(f"\n✓ 所有文件下载成功！")
            break
        else:
            print(f"✗ 镜像源 {mirror} 下载失败，尝试下一个...")
    
    return snapshots_dir

if __name__ == "__main__":
    try:
        download_from_mirror()
    except Exception as e:
        print(f"下载失败: {e}")
