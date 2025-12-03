#!/usr/bin/env python3
"""调试 DetInferencer 的输出格式"""
import os
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置本地模型路径
bert_local_path = Path(__file__).parent.parent.parent / "models" / "bert-base-uncased"
if bert_local_path.exists():
    os.environ['HF_HOME'] = str(bert_local_path.parent)
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 解决 PyTorch 2.6+ 的 weights_only 限制
import functools
_original_torch_load = torch.load
@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from mmdet.apis import DetInferencer
from mmdet.structures import DetDataSample
import yaml

# 读取配置
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_path = Path(config['dense_descriptor']['model_path'])
config_file = model_path / "config.py"
checkpoint_file = model_path / "pytorch_model.bin"

# 创建 inferencer
print("创建 DetInferencer...")
inferencer = DetInferencer(
    model=str(config_file),
    weights=str(checkpoint_file),
    device='cuda' if torch.cuda.is_available() else 'cpu',
    show_progress=False
)

# 测试一张图像
test_image = "/home/m025/qqw/coco/train2017/000000000009.jpg"
text_prompt = "person . bicycle . car . motorcycle . airplane . bus . train . truck . boat . traffic light . fire hydrant . stop sign . parking meter . bench . bird . cat . dog . horse . sheep . cow . elephant . bear . zebra . giraffe . backpack . umbrella . handbag . tie . suitcase . frisbee . skis . snowboard . sports ball . kite . baseball bat . baseball glove . skateboard . surfboard . tennis racket . bottle . wine glass . cup . fork . knife . spoon . bowl . banana . apple . sandwich . orange . broccoli . carrot . hot dog . pizza . donut . cake . chair . couch . potted plant . bed . dining table . toilet . tv . laptop . mouse . remote . keyboard . cell phone . microwave . oven . toaster . sink . refrigerator . book . clock . vase . scissors . teddy bear . hair drier . toothbrush"

print(f"\n测试图像: {test_image}")
print("运行推理...")
result = inferencer(test_image, texts=text_prompt, return_datasamples=True)

print(f"\n结果类型: {type(result)}")
print(f"结果内容:")
print(result)

if isinstance(result, dict):
    print(f"\n字典键: {result.keys()}")
    if 'predictions' in result:
        pred = result['predictions']
        print(f"predictions 类型: {type(pred)}")
        if isinstance(pred, list) and len(pred) > 0:
            pred = pred[0]
        print(f"pred 类型: {type(pred)}")
        print(f"pred 属性: {dir(pred)}")
        if hasattr(pred, 'pred_instances'):
            instances = pred.pred_instances
            print(f"\npred_instances 类型: {type(instances)}")
            print(f"pred_instances 属性: {dir(instances)}")
            for attr in ['texts', 'label_names', 'labels', 'scores', 'bboxes']:
                if hasattr(instances, attr):
                    val = getattr(instances, attr)
                    print(f"  {attr}: {type(val)} = {val}")
elif isinstance(result, (list, tuple)):
    print(f"\n列表长度: {len(result)}")
    if len(result) > 0:
        print(f"第一个元素类型: {type(result[0])}")
        print(f"第一个元素: {result[0]}")
        if hasattr(result[0], 'pred_instances'):
            instances = result[0].pred_instances
            print(f"\npred_instances 类型: {type(instances)}")
            print(f"pred_instances 属性: {dir(instances)}")
            for attr in ['texts', 'label_names', 'labels', 'scores', 'bboxes']:
                if hasattr(instances, attr):
                    val = getattr(instances, attr)
                    print(f"  {attr}: {type(val)} = {val}")
elif isinstance(result, DetDataSample):
    print(f"\nDetDataSample 属性: {dir(result)}")
    if hasattr(result, 'pred_instances'):
        instances = result.pred_instances
        print(f"\npred_instances 类型: {type(instances)}")
        print(f"pred_instances 属性: {dir(instances)}")
        for attr in ['texts', 'label_names', 'labels', 'scores', 'bboxes']:
            if hasattr(instances, attr):
                val = getattr(instances, attr)
                print(f"  {attr}: {type(val)} = {val}")

