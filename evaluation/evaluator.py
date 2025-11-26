"""简化的评估器，用于评估IGRAG生成的caption"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import warnings
import yaml
from collections import defaultdict
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from evaluation.metrics_calculator import MetricsCalculator
from main import generate_caption, load_config


class FilteredStderr:
    """过滤特定警告消息的stderr包装器"""
    
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.filter_patterns = [
            "The module name",
            "module name",
            "not a valid Python identifier",
            "force_all_finite",
            "ensure_all_finite",
            "FutureWarning",
        ]
    
    def write(self, text):
        # 检查是否包含需要过滤的模式
        if any(pattern.lower() in text.lower() for pattern in self.filter_patterns):
            return  # 过滤掉这些消息
        self.original_stderr.write(text)
    
    def flush(self):
        self.original_stderr.flush()
    
    def __getattr__(self, name):
        return getattr(self.original_stderr, name)


class SimpleEvaluator:
    """简化的评估器，用于评估IGRAG生成的caption"""

    def __init__(
        self,
        eval_config_path: Union[str, Path],
        igrag_config: Optional[Dict] = None
    ) -> None:
        """
        初始化评估器
        
        Args:
            eval_config_path: 评估模块配置文件路径
            igrag_config: IGRAG主配置（可选，如果不提供则从eval_config中读取）
        """
        # 加载评估配置
        eval_config_path = Path(eval_config_path)
        if not eval_config_path.exists():
            raise FileNotFoundError(f"评估配置文件不存在: {eval_config_path}")
        
        with open(eval_config_path, "r", encoding="utf-8") as f:
            self.eval_config = yaml.safe_load(f)
        
        # 加载IGRAG配置
        if igrag_config is None:
            main_config_path = self.eval_config.get("igrag", {}).get("main_config_path", "configs/config.yaml")
            igrag_config = load_config(main_config_path)
        
        # 根据评估配置调整IGRAG配置
        retrieval_mode = self.eval_config.get("igrag", {}).get("retrieval_mode", "global_local")
        if retrieval_mode == "global_only":
            igrag_config.setdefault("retrieval_config", {})["use_patch_retrieval"] = False
        elif retrieval_mode == "global_local":
            igrag_config.setdefault("retrieval_config", {})["use_patch_retrieval"] = True
        
        # 评估时禁用patch保存功能
        igrag_config.setdefault("patch_config", {})["save_debug_patches"] = False
        
        # 评估时设置日志级别为ERROR，减少输出
        igrag_config.setdefault("log_config", {})["log_level"] = "ERROR"
        
        self.igrag_config = igrag_config
        
        # 数据路径
        data_cfg = self.eval_config.get("data", {})
        self.val_images_dir = Path(data_cfg.get("val_images_dir", ""))
        self.annotations_path = Path(data_cfg.get("val_annotations_path", ""))
        
        if not self.val_images_dir.exists():
            raise FileNotFoundError(f"验证集图片目录不存在: {self.val_images_dir}")
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"验证集标注文件不存在: {self.annotations_path}")
        
        # 输出配置
        output_cfg = self.eval_config.get("output", {})
        self.output_dir = Path(output_cfg.get("output_dir", "./evaluation_results/"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载ground truth
        self.references, self.image_id_to_file = self._load_ground_truth()
        
        # 初始化指标计算器
        metrics_cfg = self.eval_config.get("metrics", {})
        self.metrics_calculator = MetricsCalculator(self.annotations_path, metrics_cfg)

    def _load_ground_truth(self) -> Tuple[Dict[int, List[str]], Dict[int, str]]:
        """加载COCO格式的ground truth标注"""
        with open(self.annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        image_id_to_file = {item["id"]: item["file_name"] for item in data.get("images", [])}
        references: Dict[int, List[str]] = defaultdict(list)
        for ann in data.get("annotations", []):
            references[ann["image_id"]].append(ann["caption"])
        
        return references, image_id_to_file

    def _clear_debug_patches(self) -> None:
        """清理debug_patches目录"""
        debug_dir = Path("output/debug_patches")
        if debug_dir.exists():
            try:
                for item in debug_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            except Exception:
                pass  # 忽略清理错误，不影响评估流程
    
    def evaluate_single_image(
        self, 
        image_id: int
    ) -> Dict:
        """
        评估单张图片
        
        Args:
            image_id: 图片ID
            
        Returns:
            包含生成caption、参考caption和指标得分的字典
        """
        # 临时设置日志级别为ERROR，减少输出
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)
        
        # 抑制所有警告和特定stderr输出，保证tqdm进度条连续
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 抑制特定库的警告
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*module name.*")
            warnings.filterwarnings("ignore", message=".*force_all_finite.*")
            warnings.filterwarnings("ignore", message=".*ensure_all_finite.*")
            
            # 设置环境变量抑制transformers警告
            original_transformers_verbosity = os.environ.get("TRANSFORMERS_VERBOSITY")
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            
            # 使用过滤的stderr来抑制特定警告消息，但保留tqdm输出
            original_stderr = sys.stderr
            filtered_stderr = FilteredStderr(original_stderr)
            sys.stderr = filtered_stderr
            
            try:
                # 获取图片路径
                file_name = self.image_id_to_file.get(image_id)
                if not file_name:
                    return None
                
                image_path = self.val_images_dir / file_name
                if not image_path.exists():
                    return None
                
                # 生成caption
                try:
                    result = generate_caption(
                        str(image_path),
                        self.igrag_config,
                        emit_output=False,
                        show_prompt=False,
                        configure_logging=False
                    )
                    generated_caption = result.caption
                except Exception as exc:
                    generated_caption = ""
                
                # 清理debug_patches目录（每评估完一张图片就清理）
                self._clear_debug_patches()
                
                # 获取参考caption
                coco_captions = self.references.get(image_id, [])
                
                return {
                    "image_id": image_id,
                    "file_name": file_name,
                    "generated_caption": generated_caption,
                    "coco_captions": coco_captions,
                }
            finally:
                # 恢复stderr
                sys.stderr = original_stderr
                # 恢复环境变量
                if original_transformers_verbosity is not None:
                    os.environ["TRANSFORMERS_VERBOSITY"] = original_transformers_verbosity
                elif "TRANSFORMERS_VERBOSITY" in os.environ:
                    del os.environ["TRANSFORMERS_VERBOSITY"]
        
        # 恢复原始日志级别
        logging.getLogger().setLevel(original_level)

    def evaluate_dataset(
        self, 
        image_ids: Iterable[int],
        subset_size: Optional[int] = None
    ) -> Dict:
        """
        评估数据集
        
        Args:
            image_ids: 图片ID列表
            subset_size: 子集大小（如果指定，只评估前N张图片）
            
        Returns:
            评估结果字典
        """
        image_ids = list(image_ids)
        if subset_size is not None and subset_size > 0:
            image_ids = image_ids[:subset_size]
        
        if not image_ids:
            logging.error("没有可评估的图片")
            return {}
        
        # 设置日志级别为ERROR，减少输出，保证tqdm进度条连续
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)
        
        # 抑制所有警告，保证tqdm进度条连续
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*module name.*")
        warnings.filterwarnings("ignore", message=".*force_all_finite.*")
        warnings.filterwarnings("ignore", message=".*ensure_all_finite.*")
        
        # 设置环境变量抑制transformers警告
        original_transformers_verbosity = os.environ.get("TRANSFORMERS_VERBOSITY")
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        
        # 使用过滤的stderr来抑制特定警告消息，但保留tqdm输出
        original_stderr = sys.stderr
        filtered_stderr = FilteredStderr(original_stderr)
        sys.stderr = filtered_stderr
        
        # 清理debug_patches目录（评估开始时清理一次）
        self._clear_debug_patches()
        
        # 评估每张图片
        results = []
        predictions = []
        
        try:
            from tqdm import tqdm
            iterator = tqdm(image_ids, desc="评估中", unit="张", ncols=100)
        except ImportError:
            iterator = image_ids
        
        try:
            for image_id in iterator:
                result = self.evaluate_single_image(image_id)
                if result is None:
                    continue
                
                results.append(result)
                predictions.append({
                    "image_id": image_id,
                    "caption": result["generated_caption"]
                })
        finally:
            # 恢复stderr
            sys.stderr = original_stderr
            # 恢复日志级别和警告设置
            logging.getLogger().setLevel(original_level)
            if original_transformers_verbosity is not None:
                os.environ["TRANSFORMERS_VERBOSITY"] = original_transformers_verbosity
            elif "TRANSFORMERS_VERBOSITY" in os.environ:
                del os.environ["TRANSFORMERS_VERBOSITY"]
        
        if not predictions:
            logging.warning("没有有效的预测结果")
            return {}
        
        # 计算指标
        metrics_result = self.metrics_calculator.evaluate(predictions)
        
        # 合并指标到结果中
        per_image_metrics = metrics_result.get("per_image", {})
        for result in results:
            image_id_str = str(result["image_id"])
            result["metrics"] = per_image_metrics.get(image_id_str, {})
        
        # 构建最终结果
        final_result = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "num_images": len(results),
            "aggregate_metrics": metrics_result.get("aggregate", {}),
            "results": results
        }
        
        # 保存结果
        output_file = self.eval_config.get("output", {}).get("output_file")
        if output_file is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation_{timestamp}.json"
        
        output_path = self.output_dir / output_file
        self._save_results(output_path, final_result)
        
        print(f"\n评估完成，结果已保存到: {output_path}")
        
        return final_result

    def _save_results(self, output_path: Path, results: Dict) -> None:
        """保存评估结果到JSON文件"""
        tmp_path = output_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        tmp_path.replace(output_path)
