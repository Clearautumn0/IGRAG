#!/usr/bin/env python3
"""测试 LoRA 加载逻辑的脚本"""
import yaml
from pathlib import Path
from core.generator import CaptionGenerator

def test_lora_loading():
    """测试不同配置下的 LoRA 加载行为"""
    
    # 加载配置
    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("测试 LoRA 加载逻辑")
    print("=" * 60)
    print(f"\n当前配置:")
    print(f"  lora_config.enabled: {config.get('lora_config', {}).get('enabled', False)}")
    print(f"  lora_config.weights_path: {config.get('lora_config', {}).get('weights_path', 'null')}")
    print()
    
    # 测试 1: 当前配置
    print("测试 1: 使用当前配置加载生成器...")
    try:
        generator = CaptionGenerator(config)
        print(f"  ✅ 生成器加载成功")
        print(f"  LoRA 使用状态: {generator.use_lora}")
        print()
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        print()
    
    # 测试 2: 禁用 LoRA
    print("测试 2: 禁用 LoRA...")
    config_disabled = config.copy()
    config_disabled['lora_config'] = {'enabled': False}
    try:
        generator2 = CaptionGenerator(config_disabled)
        print(f"  ✅ 生成器加载成功（LoRA 已禁用）")
        print()
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        print()
    
    # 测试 3: 启用但路径不存在
    print("测试 3: 启用 LoRA 但路径不存在...")
    config_fake_path = config.copy()
    config_fake_path['lora_config'] = {
        'enabled': True,
        'weights_path': 'lora_training/checkpoints/nonexistent'
    }
    try:
        generator3 = CaptionGenerator(config_fake_path)
        print(f"  ✅ 生成器加载成功（LoRA 路径不存在，已跳过）")
        print()
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        print()
    
    # 检查是否有实际的 checkpoint
    print("检查 LoRA checkpoint 目录...")
    checkpoint_dirs = [
        "lora_training/checkpoints/best",
        "lora_training/checkpoints",
    ]
    for ckpt_dir in checkpoint_dirs:
        path = Path(ckpt_dir)
        if path.exists():
            print(f"  ✅ 找到: {ckpt_dir}")
            # 列出子目录
            subdirs = [d.name for d in path.iterdir() if d.is_dir()]
            if subdirs:
                print(f"     子目录: {', '.join(subdirs[:5])}")
        else:
            print(f"  ❌ 不存在: {ckpt_dir}")
    
    print("\n" + "=" * 60)
    print("建议:")
    print("=" * 60)
    print("1. 如果 LoRA 未启用，在 configs/config.yaml 中设置:")
    print("   lora_config:")
    print("     enabled: false")
    print()
    print("2. 如果 LoRA 已启用但没有 checkpoint，需要先训练:")
    print("   参考 lora_training/README.md 进行训练")
    print()
    print("3. 训练完成后，在 configs/config.yaml 中设置:")
    print("   lora_config:")
    print("     enabled: true")
    print("     weights_path: \"lora_training/checkpoints/best\"")
    print("=" * 60)

if __name__ == "__main__":
    test_lora_loading()

