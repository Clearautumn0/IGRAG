#!/usr/bin/env python3
"""检查密集描述知识库的结果"""
import pickle
from pathlib import Path

result_file = Path("output/image_id_to_dense_captions.pkl")
if result_file.exists():
    with open(result_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"总条目数: {len(data)}")
    print(f"\n前10个条目:")
    count = 0
    for img_id, phrases in list(data.items())[:10]:
        print(f"  Image ID {img_id}: {phrases} (数量: {len(phrases)})")
        count += 1
        if count >= 10:
            break
    
    # 统计有短语的图像数量
    images_with_phrases = sum(1 for phrases in data.values() if phrases and len(phrases) > 0)
    print(f"\n有短语的图像数: {images_with_phrases} / {len(data)}")
    
    # 检查所有短语是否为空
    all_empty = all(not phrases or len(phrases) == 0 for phrases in data.values())
    print(f"所有条目是否为空: {all_empty}")
else:
    print(f"结果文件不存在: {result_file}")

