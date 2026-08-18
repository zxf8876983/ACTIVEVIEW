"""
模块化消融策略 —— ablation_policies.py
========================================

功能：
    v4.0 正式引入的模块化消融策略。
    通过"只使用某一种评分项"来验证各个模块的独立贡献。

消融设计（对应论文分析）：
    基础（Fixed / Random / Nearest）见 policies.py。
    本文件提供 6 个消融策略：

    A. VisibilityOnlyPolicy    只使用通用 FOV 可见性        Q = S_kp_pred
    B. ActionPartOnlyPolicy    只使用动作关键部位（无遮挡）  Q = S_action_part_pred
    C. OrientationOnlyPolicy   只使用朝向适配                Q = S_orient_pred
    D. OcclusionOnlyPolicy     只使用遮挡后通用可见性        Q = S_kp_occ_pred
    E. ActionOrientationPolicy 动作+朝向+距离+移动（仿真 v3.0）
    F. FullOursPolicy          完整 v4.0                    Q = Q_pred

核心比较：
    VisibilityOnly vs ActionPartOnly → 动作关键部位建模贡献
    ActionPartOnly vs ActionOrientation → 朝向建模贡献
    ActionOrientation vs FullOurs → 遮挡建模贡献
"""

from typing import List

from .candidate_sampler import CandidateView


class _BaseAblationPolicy:
    """消融策略基类：按某个 pred_score 字段取最大（允许不移动）。

    子类只需设置 name 和 score_key（或重写 _score_of）。
    """

    name = "ablation"
    score_key = "Q_pred"
    # 依赖遮挡评分的策略应过滤 is_occlusion_valid_pred=False 的候选；
    # 不依赖遮挡的（Random/Nearest/Fixed 及纯 FOV/朝向）保持原定义。
    requires_occlusion_valid = False

    def select(
        self,
        current_view: CandidateView,
        candidates: List[CandidateView],
    ) -> CandidateView:
        all_views = []
        if current_view.pred_score and self._score_of(current_view) is not None:
            if (not self.requires_occlusion_valid
                    or current_view.pred_score.get("is_occlusion_valid_pred", False)):
                all_views.append(current_view)
        for c in candidates:
            if not c.is_valid or not c.pred_score:
                continue
            if self._score_of(c) is None:
                continue
            if self.requires_occlusion_valid and not c.pred_score.get(
                    "is_occlusion_valid_pred", False):
                continue
            all_views.append(c)

        if not all_views:
            return current_view
        return max(all_views, key=self._score_of)

    def _score_of(self, view: CandidateView):
        return view.pred_score.get(self.score_key)


class VisibilityOnlyPolicy(_BaseAblationPolicy):
    """仅使用通用关键点 FOV 可见性（v3.0 之前的口径）。"""
    name = "VisibilityOnly"
    score_key = "S_kp_pred"


class ActionPartOnlyPolicy(_BaseAblationPolicy):
    """仅使用动作关键部位可见性（不含朝向 / 遮挡 / 距离）。"""
    name = "ActionPartOnly"
    score_key = "S_action_part_pred"


class OrientationOnlyPolicy(_BaseAblationPolicy):
    """仅使用朝向适配评分。"""
    name = "OrientationOnly"
    score_key = "S_orient_pred"


class OcclusionOnlyPolicy(_BaseAblationPolicy):
    """仅使用遮挡后的通用关键点可见性。

    依赖遮挡评分 → 过滤 invalid-occlusion 候选。
    """
    name = "OcclusionOnly"
    score_key = "S_kp_occ_pred"
    requires_occlusion_valid = True


class ActionOrientationPolicy(_BaseAblationPolicy):
    """动作关键部位 + 朝向 + 距离 + 移动代价（仿真 v3.0，无遮挡）。

    使用与 v3.0 相同的预测权重组合，但显式不使用遮挡感知项。
    用于验证"遮挡建模"的相对贡献（对比 FullOursPolicy）。
    """
    name = "ActionOrientation"

    def __init__(self, config: dict):
        self.pred_cfg = config.get("predictive_score", {})

    def _score_of(self, view: CandidateView):
        ps = view.pred_score
        if not ps or ps.get("S_action_part_pred") is None:
            return None
        pc = self.pred_cfg
        # v4.0 配置中动作部位权重名为 w_action_occ_pred，
        # 这里用它充当"动作部位重要性"权重来模拟 v3.0 组合
        w_action = pc.get("w_action_part_pred", pc.get("w_action_occ_pred", 0.5))
        return (
            w_action * ps["S_action_part_pred"]
            + pc.get("w_orient_pred", 0.20) * ps.get("S_orient_pred", 0.0)
            + pc.get("w_center_pred", 0.15) * ps.get("S_center_pred", 0.0)
            + pc.get("w_dist_pred", 0.10) * ps.get("S_dist_pred", 0.0)
            - pc.get("w_move", 0.15) * ps.get("C_move", 0.0)
        )


class FullOursPolicy(_BaseAblationPolicy):
    """完整 v4.0 策略（= OursPolicy），使用最终 Q_pred。

    注：与 policies.OursPolicy 评分完全相同，为避免 metrics.csv 出现重复行，
    默认运行脚本直接实例化 OursPolicy（即 FullOurs）。本类保留用于
    "FullOurs" 命名明确的独立引用。
    """
    name = "FullOurs"
    score_key = "Q_pred"
    requires_occlusion_valid = True