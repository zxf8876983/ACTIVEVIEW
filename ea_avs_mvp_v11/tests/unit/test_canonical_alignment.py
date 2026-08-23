"""
Unit Tests for Canonical Skeleton Alignment Module (v11.4.1)
=============================================================
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.active_view.action_registry import ActionRegistry
from ea_avs_mvp_v11.active_view.skeleton_canonicalizer import (
    CanonicalSkeletonAligner,
    get_canonical_skeleton_aligner,
)


class TestCanonicalSkeletonAlignment(unittest.TestCase):
    """测试人体骨架坐标系正规化与视角不变性对齐。"""

    def setUp(self):
        self.aligner = CanonicalSkeletonAligner()
        self.registry = ActionRegistry()
        self.sample_skel = self.registry.get_skeleton_sequence(action_id=0, instance_idx=0)  # (30, 33, 3)

    def test_shape_preservation(self):
        """测试输入输出张量形状严格一致。"""
        # 1. (T, V, 3)
        res_seq = self.aligner.align(self.sample_skel)
        self.assertEqual(res_seq.shape, (30, 33, 3))

        # 2. 单帧 (V, 3)
        single_frame = self.sample_skel[0]
        res_frame = self.aligner.align(single_frame)
        self.assertEqual(res_frame.shape, (33, 3))

        # 3. 4D 张量 (C, T, V, M)
        tensor_4d = np.transpose(self.sample_skel, (2, 0, 1))[..., np.newaxis]
        res_tensor = self.aligner.align_tensor_sequence(tensor_4d)
        self.assertEqual(res_tensor.shape, (3, 30, 33, 1))

    def test_no_nan_or_inf(self):
        """测试输出不存在任何 NaN 或 Inf 异常值。"""
        res = self.aligner.align(self.sample_skel)
        self.assertFalse(np.isnan(res).any(), "Found NaN in canonical skeleton output")
        self.assertFalse(np.isinf(res).any(), "Found Inf in canonical skeleton output")

        # 极端零输入测试
        zero_skel = np.zeros((30, 33, 3), dtype=np.float32)
        zero_res = self.aligner.align(zero_skel)
        self.assertFalse(np.isnan(zero_res).any(), "Found NaN on zero input")

    def test_structural_distance_preservation(self):
        """测试旋转正规化前后人体各关节点之间的刚体欧氏距离完全保持一致。"""
        res = self.aligner.align(self.sample_skel)

        # 检查关键骨骼长度 (双肩距离, 脊柱长度)
        # 双肩: 11 与 12
        orig_shoulder_dist = np.linalg.norm(self.sample_skel[0, 12] - self.sample_skel[0, 11])
        canon_shoulder_dist = np.linalg.norm(res[0, 12] - res[0, 11])
        self.assertAlmostEqual(orig_shoulder_dist, canon_shoulder_dist, places=5)

        # 双髋: 23 与 24
        orig_hip_dist = np.linalg.norm(self.sample_skel[0, 24] - self.sample_skel[0, 23])
        canon_hip_dist = np.linalg.norm(res[0, 24] - res[0, 23])
        self.assertAlmostEqual(orig_hip_dist, canon_hip_dist, places=5)

    def test_viewpoint_yaw_invariance(self):
        """测试在任意视角旋转下输入，正规化后的骨架均高度一致。"""
        canonical_ref = self.aligner.align(self.sample_skel)

        angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        for ang in angles:
            rad = math.radians(ang)
            R = np.array(
                [
                    [math.cos(rad), 0.0, -math.sin(rad)],
                    [0.0, 1.0, 0.0],
                    [math.sin(rad), 0.0, math.cos(rad)],
                ],
                dtype=np.float32,
            )

            rot_skel = np.zeros_like(self.sample_skel)
            for t in range(self.sample_skel.shape[0]):
                rot_skel[t] = (R @ self.sample_skel[t].T).T

            aligned = self.aligner.align(rot_skel)
            max_err = float(np.max(np.abs(aligned - canonical_ref)))
            self.assertLess(
                max_err,
                1e-4,
                f"Yaw invariance failed for angle {ang} deg with max diff {max_err}",
            )


if __name__ == "__main__":
    unittest.main()
