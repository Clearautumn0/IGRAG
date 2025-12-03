#!/usr/bin/env python3
"""测试配置文件格式（不需要安装mmdet）。

仅验证配置文件的基本格式和关键字段。
"""
import os
import sys
import re
from pathlib import Path

def check_config_file():
    """检查配置文件格式。"""
    print("=" * 60)
    print("配置文件格式检查")
    print("=" * 60)
    
    model_path = Path("../models/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det")
    config_file = model_path / "config.py"
    
    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    print(f"✅ 找到配置文件: {config_file}")
    
    # 读取配置文件
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"✅ 配置文件大小: {len(content)} 字符")
        print(f"✅ 配置文件行数: {len(content.splitlines())} 行")
        
        # 检查关键字段
        checks = {
            "model定义": re.search(r'^model\s*=\s*dict\(', content, re.MULTILINE),
            "backbone": re.search(r"type\s*=\s*['\"]SwinTransformer", content),
            "bbox_head": re.search(r'bbox_head\s*=\s*dict\(', content),
            "checkpoint路径": re.search(r'\.pth|\.ckpt|model\.safetensors', content),
        }
        
        print("\n关键字段检查:")
        all_passed = True
        for name, match in checks.items():
            if match:
                print(f"  ✅ {name}: 找到")
            else:
                print(f"  ⚠️  {name}: 未找到")
                all_passed = False
        
        # 检查是否有_base_引用
        if '_base_' in content:
            base_match = re.search(r"_base_\s*=\s*['\"]([^'\"]+)['\"]", content)
            if base_match:
                base_file = base_match.group(1)
                print(f"\n⚠️  发现_base_引用: {base_file}")
                base_path = model_path / base_file
                if base_path.exists():
                    print(f"  ✅ 基础配置文件存在: {base_path}")
                else:
                    print(f"  ⚠️  基础配置文件不存在（可能在其他位置）")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        return False


def check_checkpoint_files():
    """检查检查点文件。"""
    print("\n" + "=" * 60)
    print("检查点文件检查")
    print("=" * 60)
    
    model_path = Path("../models/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det")
    
    checkpoint_patterns = ["*.pth", "*.ckpt", "model.safetensors", "pytorch_model.bin"]
    found_files = []
    
    for pattern in checkpoint_patterns:
        files = list(model_path.glob(pattern))
        for f in files:
            if f.is_file() or f.is_symlink():
                size = f.stat().st_size / (1024 * 1024)  # MB
                found_files.append((f.name, size))
    
    if found_files:
        print(f"✅ 找到 {len(found_files)} 个检查点文件:")
        for name, size in found_files:
            print(f"   - {name} ({size:.1f} MB)")
        return True
    else:
        print("❌ 未找到检查点文件")
        return False


def check_dependencies():
    """检查依赖库。"""
    print("\n" + "=" * 60)
    print("依赖库检查")
    print("=" * 60)
    
    dependencies = {
        "mmdet": "mmdetection (用于模型加载和推理)",
        "mmcv": "mmcv-full (mmdetection的依赖)",
        "torch": "PyTorch (深度学习框架)",
    }
    
    all_installed = True
    for module, desc in dependencies.items():
        try:
            if module == "mmcv":
                # 尝试导入mmcv
                try:
                    from mmcv import Config
                    print(f"✅ {module}: 已安装 ({desc})")
                except:
                    try:
                        import mmcv
                        print(f"✅ {module}: 已安装，版本 {mmcv.__version__} ({desc})")
                    except:
                        raise ImportError
            else:
                mod = __import__(module)
                version = getattr(mod, '__version__', '未知版本')
                print(f"✅ {module}: 已安装，版本 {version} ({desc})")
        except ImportError:
            print(f"❌ {module}: 未安装 ({desc})")
            all_installed = False
    
    if not all_installed:
        print("\n💡 安装提示:")
        print("   要安装mmdet和mmcv，请运行:")
        print("   pip install mmdet mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cuda_version}/torch{torch_version}/index.html")
        print("   或者:")
        print("   pip install mmdet")
        print("   pip install mmcv-full")
    
    return all_installed


def main():
    """运行所有检查。"""
    print("开始检查密集描述模型配置...")
    print()
    
    results = []
    
    # 检查配置文件
    results.append(("配置文件格式", check_config_file()))
    
    # 检查检查点文件
    results.append(("检查点文件", check_checkpoint_files()))
    
    # 检查依赖
    results.append(("依赖库", check_dependencies()))
    
    # 总结
    print("\n" + "=" * 60)
    print("检查总结")
    print("=" * 60)
    
    all_passed = True
    for check_name, result in results:
        status = "✅ 通过" if result else "❌ 失败/缺失"
        print(f"{check_name}: {status}")
        if not result and check_name != "依赖库":
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 配置文件和检查点文件检查通过！")
        print("\n⚠️  注意: 依赖库未安装，无法进行实际的模型加载测试。")
        print("   如果已安装依赖库但仍无法加载模型，请检查模型文件完整性。")
    else:
        print("❌ 部分检查失败，请查看上述错误信息。")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

