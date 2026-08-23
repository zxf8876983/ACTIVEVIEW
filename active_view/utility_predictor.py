"""
视点效用预测器高层服务接口 —— utility_predictor.py (v11.3)
=========================================================

职责：
    1. 封装 ViewpointUtilityPredictorNet 神经网络，提供端到端推理与打分服务；
    2. 支持加载训练好的 PyTorch 权重文件；
    3. 接收当前观察状态与候选视点集合，快速批量预测各视点的预期识别增益 U(v) = H_curr - H_cand。
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ea_avs_mvp_v11.active_view.models.utility_predictor import ViewpointUtilityPredictorNet
from ea_avs_mvp_v11.active_view.viewpoint_types import Viewpoint

logger = logging.getLogger("utility_predictor")


class ViewpointUtilityPredictor:
    """视点识别效用预测器服务封装。"""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        in_dim: int = 11,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or {}
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ViewpointUtilityPredictorNet(in_dim=in_dim).to(self.device)
        self.model_path = model_path

        if model_path and Path(model_path).exists():
            self.load_checkpoint(model_path)
        else:
            self.model.eval()

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """加载训练好的神经网络权重。"""
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Utility predictor checkpoint not found at: {ckpt_p}")

        ckpt = torch.load(ckpt_p, map_location=self.device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)

        self.model.eval()
        self.model_path = str(ckpt_p)
        logger.info("Successfully loaded ViewpointUtilityPredictor checkpoint from: %s", ckpt_p)

    @staticmethod
    def construct_features(
        current_viewpoint: Dict[str, Any],
        candidate_viewpoint: Viewpoint,
    ) -> np.ndarray:
        """从当前视点字典与候选视点对象构建 11 维输入特征。"""
        d_curr = float(current_viewpoint.get("distance_to_human", 4.0)) / 5.0
        ang_curr_rad = math.radians(float(current_viewpoint.get("angle_to_human", 0.0)))
        yaw_curr_rad = math.radians(float(current_viewpoint.get("yaw", 0.0)))
        sin_ang_c, cos_ang_c = math.sin(ang_curr_rad), math.cos(ang_curr_rad)
        sin_yaw_c, cos_yaw_c = math.sin(yaw_curr_rad), math.cos(yaw_curr_rad)

        d_cand = float(candidate_viewpoint.distance) / 5.0
        ang_cand_rad = math.radians(float(candidate_viewpoint.angle))
        yaw_cand_rad = math.radians(float(candidate_viewpoint.yaw))
        sin_ang_v, cos_ang_v = math.sin(ang_cand_rad), math.cos(ang_cand_rad)
        sin_yaw_v, cos_yaw_v = math.sin(yaw_cand_rad), math.cos(yaw_cand_rad)
        nav_cost = float(candidate_viewpoint.navigation_cost) / 10.0

        return np.array([
            d_curr, sin_ang_c, cos_ang_c, sin_yaw_c, cos_yaw_c,
            d_cand, sin_ang_v, cos_ang_v, sin_yaw_v, cos_yaw_v, nav_cost
        ], dtype=np.float32)

    def predict_utility(
        self,
        current_observation: Dict[str, Any],
        candidate_viewpoints: List[Viewpoint],
    ) -> List[float]:
        """
        批量预测每个候选视点的未来动作识别效用得分。

        参数:
            current_observation: 当前状态字典 (需包含 "current_viewpoint" 或机器人位姿)
            candidate_viewpoints: 经过 Habitat 过滤后的可行候选视点列表

        返回:
            scores: 各候选视点的预测效用增益列表 [float, ...]
        """
        if not candidate_viewpoints:
            return []

        curr_vp = current_observation.get("current_viewpoint", current_observation)

        feat_list = [self.construct_features(curr_vp, vp) for vp in candidate_viewpoints]
        tensor_batch = torch.tensor(np.array(feat_list), dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(tensor_batch).cpu().numpy().squeeze(-1)

        if preds.ndim == 0:
            return [float(preds)]
        return [float(p) for p in preds]

    def predict(self, features: Union[List[List[float]], np.ndarray, List[float]]) -> List[float]:
        """批量特征矩阵直接前向预测。"""
        feat_arr = np.array(features, dtype=np.float32)
        if feat_arr.ndim == 1:
            feat_arr = feat_arr.reshape(1, -1)
        tensor_batch = torch.tensor(feat_arr, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(tensor_batch).cpu().numpy().squeeze(-1)
        if preds.ndim == 0:
            return [float(preds)]
        return [float(p) for p in preds]

    def predict_single(self, features: Union[List[float], np.ndarray]) -> float:
        """单样本前向预测。"""
        preds = self.predict(features)
        return float(preds[0])

