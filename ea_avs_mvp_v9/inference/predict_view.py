"""
学习型视角推理预测器 —— predict_view.py
======================================

职责：
    1. 加载已训练好的 LearnableViewScorer 权重；
    2. 对输入的任意人体位姿、动作类别与候选视点池进行批量前向推理打分；
    3. 输出排序结果并选定最优主动观察视角。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v9.action.action_encoder import ActionEncoder
from ea_avs_mvp_v9.core.types import ActionClass, ViewFeature
from ea_avs_mvp_v9.models.pose_encoder import extract_pose_vector
from ea_avs_mvp_v9.models.view_encoder import extract_view_vector
from ea_avs_mvp_v9.models.view_scorer import LearnableViewScorer

logger = logging.getLogger(__name__)


class ViewPredictor:
    """视角打分模型推理引擎。"""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model: Optional[LearnableViewScorer] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = LearnableViewScorer().to(self.device)
            if checkpoint_path:
                self.load_checkpoint(checkpoint_path)

        self.model.eval()
        self.action_encoder = ActionEncoder()

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """加载权重检查点。"""
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {p}")
        ckpt = torch.load(p, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded LearnableViewScorer checkpoint from %s (Epoch %d)", p, ckpt.get("epoch", -1))

    def predict_viewpoints(
        self,
        viewpoints: List[CandidateViewpoint],
        features: List[ViewFeature],
        human_joints_3d: Dict[str, List[float]],
        human_yaw_deg: float = 0.0,
        action_label: Union[str, ActionClass] = "standing",
        ablate_action: bool = False,
        ablate_pose: bool = False,
    ) -> Dict[str, Any]:
        """
        对所有候选视点进行推理预测。
        """
        if not viewpoints:
            raise ValueError("Candidate viewpoints list is empty")

        # 1. 提取人体状态姿态向量 (无 Action 编码输入)
        pose_vec = extract_pose_vector(human_joints_3d, human_yaw_deg=human_yaw_deg)
        act_embed = self.action_encoder.encode(action_label) if action_label else None

        # 2. 提取 13 维视角特征向量
        view_vecs = [extract_view_vector(f) for f in features]

        # 3. 构造 Tensor
        pose_t = torch.tensor(pose_vec, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 49)
        views_arr = np.array(view_vecs, dtype=np.float32)
        views_t = torch.tensor(views_arr, dtype=torch.float32, device=self.device).unsqueeze(0)# (1, N, 13)

        # 4. 模型前向推理 Q(v | H)
        self.model.eval()
        with torch.no_grad():
            scores_t = self.model(
                pose_t, views_t,
                ablate_pose=ablate_pose,
            )  # (1, N)
            scores = scores_t.squeeze(0).cpu().numpy()  # (N,)

        # 5. 组装结果与排序
        scored_views = []
        for i, (vp, feat) in enumerate(zip(viewpoints, features)):
            s = float(scores[i])
            if not vp.feasible:
                s = 0.0  # 不可行视点置零

            scored_views.append({
                "viewpoint_id": vp.viewpoint_id,
                "predicted_score": round(s, 3),
                "position": [round(float(x), 2) for x in vp.position],
                "yaw_deg": round(float(vp.yaw_deg), 1),
                "distance": feat.distance,
                "viewing_angle_deg": feat.viewing_angle_deg,
                "feasible": vp.feasible,
            })

        # 按预测得分降序排列
        ranked_views = sorted(scored_views, key=lambda item: (item["feasible"], item["predicted_score"]), reverse=True)
        best_view = ranked_views[0]

        return {
            "action_name": act_embed.action_name,
            "best_viewpoint_id": best_view["viewpoint_id"],
            "best_predicted_score": best_view["predicted_score"],
            "best_view": best_view,
            "ranked_views": ranked_views,
            "scores_map": {v["viewpoint_id"]: v["predicted_score"] for v in scored_views},
        }
