"""
动作分类与不确定度估计器 —— action_classifier.py
=============================================

职责：
    1. 封装训练好的 ST-GCN 神经网络模型，提供端到端推理接口；
    2. 输入：归一化 3D 骨架序列 (T, V, 3) 或批张量 (N, C, T, V, M)；
    3. 输出：动作预测类别、Softmax 概率分布、Shannon 熵不确定度与置信度余量；
    4. 不确定度计算公式：
       - Shannon 熵: H(p) = - \\sum_{k=1}^K p_k \\ln(p_k + \\epsilon)
       - 归一化熵: U_{norm}(p) = H(p) / \\ln(K) \\in [0, 1]
       - 决策置信余量: M(p) = p_{(1)} - p_{(2)}
    5. 为 Phase 4 Active View 提供下游动作识别评价打分。
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ea_avs_mvp_v11.action_recognition.st_gcn_model import STGCN
from ea_avs_mvp_v11.perception.skeleton_definition import SkeletonDefinition, get_skeleton_definition
from ea_avs_mvp_v11.perception.skeleton_normalizer import SkeletonNormalizer

logger = logging.getLogger(__name__)


@dataclass
class ActionPredictionResult:
    """动作分类与不确定度估计结果实体。"""
    predicted_class: int
    predicted_label: str
    probabilities: Dict[str, float]
    raw_probabilities: np.ndarray
    entropy: float                      # Shannon 熵 H(p)
    normalized_entropy: float           # 归一化熵 [0, 1]
    confidence_margin: float            # Top-1 与 Top-2 概率差
    top1_confidence: float              # Top-1 概率
    is_uncertain: bool = False          # 当归一化熵 > 0.5 时标记为不确定
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "predicted_label": self.predicted_label,
            "top1_confidence": round(float(self.top1_confidence), 4),
            "entropy": round(float(self.entropy), 4),
            "normalized_entropy": round(float(self.normalized_entropy), 4),
            "confidence_margin": round(float(self.confidence_margin), 4),
            "is_uncertain": self.is_uncertain,
            "probabilities": {k: round(float(v), 4) for k, v in self.probabilities.items()},
            "metadata": self.metadata,
        }


class ActionClassifier:
    """ST-GCN 动作分类与不确定度分析引擎。"""

    def __init__(
        self,
        model: Optional[STGCN] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        action_classes: Optional[List[str]] = None,
        device: Optional[str] = None,
        skel_def: Optional[SkeletonDefinition] = None,
    ):
        self.skel_def = skel_def or get_skeleton_definition()
        self.normalizer = SkeletonNormalizer(skel_def=self.skel_def)

        # 默认 6 动作类别
        if action_classes is None:
            self.action_classes = ["standing", "walking", "sitting", "bending", "reaching", "fall_related"]
        else:
            self.action_classes = action_classes

        self.num_classes = len(self.action_classes)

        # 设备选择
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 初始化模型
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = STGCN(
                in_channels=3,
                num_classes=self.num_classes,
                skel_def=self.skel_def,
            ).to(self.device)

        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)

        self.model.eval()

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """加载训练好的模型权重。"""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)
        logger.info("Loaded ST-GCN checkpoint from: %s", checkpoint_path)
        self.model.eval()

    @staticmethod
    def compute_uncertainty_metrics(probs: np.ndarray, num_classes: int) -> Tuple[float, float, float]:
        """
        计算 Shannon 熵、归一化熵与置信余量。
        """
        eps = 1e-8
        entropy = float(-np.sum(probs * np.log(probs + eps)))
        max_entropy = math.log(max(num_classes, 2))
        normalized_entropy = float(np.clip(entropy / max_entropy, 0.0, 1.0))

        sorted_probs = np.sort(probs)[::-1]
        top1 = float(sorted_probs[0])
        top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        confidence_margin = float(top1 - top2)

        return entropy, normalized_entropy, confidence_margin

    def predict_sequence(
        self,
        skeleton_sequence: np.ndarray,
        is_normalized: bool = False,
    ) -> ActionPredictionResult:
        """
        对单段 (T, V, 3) 骨架时序进行动作分类与不确定度计算。
        """
        if not is_normalized:
            norm_seq = self.normalizer.normalize_sequence(skeleton_sequence)
        else:
            norm_seq = skeleton_sequence

        # 转换为 PyTorch (1, C, T, V, 1)
        # norm_seq: (T, V, C) -> (C, T, V)
        tensor_data = np.transpose(norm_seq, (2, 0, 1)) # (3, T, V)
        tensor_input = torch.tensor(tensor_data, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_input) # (1, num_classes)
            probs = F.softmax(logits, dim=-1).cpu().numpy().squeeze(0) # (num_classes,)

        pred_idx = int(np.argmax(probs))
        pred_label = self.action_classes[pred_idx] if pred_idx < len(self.action_classes) else f"class_{pred_idx}"

        entropy, norm_entropy, margin = self.compute_uncertainty_metrics(probs, self.num_classes)

        prob_dict = {
            self.action_classes[i]: float(probs[i])
            for i in range(len(self.action_classes))
        }

        return ActionPredictionResult(
            predicted_class=pred_idx,
            predicted_label=pred_label,
            probabilities=prob_dict,
            raw_probabilities=probs,
            entropy=entropy,
            normalized_entropy=norm_entropy,
            confidence_margin=margin,
            top1_confidence=float(probs[pred_idx]),
            is_uncertain=(norm_entropy > 0.5 or margin < 0.2),
        )

    def predict_batch(
        self,
        tensor_batch: Union[torch.Tensor, np.ndarray],
    ) -> List[ActionPredictionResult]:
        """
        对批量 (N, C, T, V, M) 张量进行动作预测与不确定度估计。
        """
        if isinstance(tensor_batch, np.ndarray):
            tensor_input = torch.tensor(tensor_batch, dtype=torch.float32).to(self.device)
        else:
            tensor_input = tensor_batch.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_input)
            batch_probs = F.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for i in range(batch_probs.shape[0]):
            probs = batch_probs[i]
            pred_idx = int(np.argmax(probs))
            pred_label = self.action_classes[pred_idx] if pred_idx < len(self.action_classes) else f"class_{pred_idx}"
            entropy, norm_entropy, margin = self.compute_uncertainty_metrics(probs, self.num_classes)
            prob_dict = {
                self.action_classes[c]: float(probs[c])
                for c in range(len(self.action_classes))
            }
            results.append(ActionPredictionResult(
                predicted_class=pred_idx,
                predicted_label=pred_label,
                probabilities=prob_dict,
                raw_probabilities=probs,
                entropy=entropy,
                normalized_entropy=norm_entropy,
                confidence_margin=margin,
                top1_confidence=float(probs[pred_idx]),
                is_uncertain=(norm_entropy > 0.5 or margin < 0.2),
            ))
        return results
