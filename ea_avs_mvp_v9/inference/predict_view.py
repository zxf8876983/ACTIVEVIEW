"""
感知质量驱动的主动视角推理预测器 —— predict_view.py
==================================================

职责：
    1. 加载已训练好的 PerceptionAwareViewScorer 权重；
    2. 基于当前不完整观测状态 ObservationState 与候选视点池进行神经网络前向推理；
    3. 预测各候选视点的信息增益 Gain(v | O_curr) 并输出排序；
    4. 严禁使用真实 GT 姿态或 Action Label 作为输入。

# GT is only used for supervision/evaluation.
# It must never enter model forward pass.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v9.core.types import ObservationState, ViewFeature
from ea_avs_mvp_v9.models.observation_encoder import extract_observation_vector
from ea_avs_mvp_v9.models.view_encoder import extract_view_vector
from ea_avs_mvp_v9.models.view_scorer import PerceptionAwareViewScorer

logger = logging.getLogger(__name__)


class ViewPredictor:
    """基于当前感知质量的主动视角选择推理引擎。"""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model: Optional[PerceptionAwareViewScorer] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = PerceptionAwareViewScorer(obs_input_dim=71, view_input_dim=13).to(self.device)
            if checkpoint_path:
                self.load_checkpoint(checkpoint_path)

        self.model.eval()

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """加载权重检查点。"""
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {p}")
        ckpt = torch.load(p, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded PerceptionAwareViewScorer checkpoint from %s (Epoch %d)", p, ckpt.get("epoch", -1))

    def predict_viewpoints(
        self,
        viewpoints: List[CandidateViewpoint],
        features: List[ViewFeature],
        observation_state: Optional[ObservationState] = None,
        human_joints_3d: Optional[Dict[str, List[float]]] = None,
        human_yaw_deg: float = 0.0,
        action_metadata: Optional[str] = None,
        ablate_obs: bool = False,
    ) -> Dict[str, Any]:
        """
        # GT is only used for supervision/evaluation.
        # It must never enter model forward pass.
        
        对所有候选视点进行纯当前观测感知驱动的信息增益预测 Gain(v | O_curr)。
        """
        if not viewpoints:
            raise ValueError("Candidate viewpoints list is empty")

        # 1. 构造当前观测感知特征向量 (71d)
        if observation_state is not None:
            obs_vec = extract_observation_vector(observation_state)
        else:
            # 兼容性包装
            j_map = human_joints_3d or {}
            c_map = {k: 0.9 for k in j_map}
            p_map = {k: 0.9 for k in ["head", "torso", "pelvis", "left_hand", "right_hand", "left_leg", "right_leg"]}
            tmp_obs = ObservationState(
                estimated_joints_3d=j_map,
                joint_confidences=c_map,
                body_part_confidences=p_map,
                mean_confidence=0.9,
                missing_joint_count=0,
            )
            obs_vec = extract_observation_vector(tmp_obs)

        # 2. 提取 13 维候选视角描述子向量
        view_vecs = [extract_view_vector(f) for f in features]

        # 3. 构造 Tensor
        obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)   # (1, 71)
        views_arr = np.array(view_vecs, dtype=np.float32)
        views_t = torch.tensor(views_arr, dtype=torch.float32, device=self.device).unsqueeze(0) # (1, N, 13)

        # 4. 模型前向推理 Gain_hat(v | O_curr)
        self.model.eval()
        with torch.no_grad():
            scores_t = self.model(
                obs_t, views_t,
                ablate_obs=ablate_obs,
            )  # (1, N)
            scores = scores_t.squeeze(0).cpu().numpy()  # (N,)

        # 5. 组装结果与排序
        scored_views = []
        for i, (vp, feat) in enumerate(zip(viewpoints, features)):
            s = float(scores[i])
            if not vp.feasible:
                s = 0.0

            scored_views.append({
                "viewpoint_id": vp.viewpoint_id,
                "predicted_gain": round(s, 3),
                "predicted_score": round(s, 3),
                "position": [round(float(x), 2) for x in vp.position],
                "yaw_deg": round(float(vp.yaw_deg), 1),
                "distance": feat.distance,
                "viewing_angle_deg": feat.viewing_angle_deg,
                "pose_coverage": feat.pose_coverage,
                "body_part_visibilities": feat.body_part_visibilities,
                "feasible": vp.feasible,
            })

        # 按预测得分降序排列
        ranked_views = sorted(scored_views, key=lambda item: (item["feasible"], item["predicted_gain"]), reverse=True)
        best_view = ranked_views[0]

        return {
            "action_metadata": action_metadata,
            "best_viewpoint_id": best_view["viewpoint_id"],
            "best_predicted_gain": best_view["predicted_gain"],
            "best_predicted_score": best_view["predicted_gain"],
            "best_view": best_view,
            "ranked_views": ranked_views,
            "scores_map": {v["viewpoint_id"]: v["predicted_gain"] for v in scored_views},
        }


# 别名兼容
PerceptionAwarePredictor = ViewPredictor
