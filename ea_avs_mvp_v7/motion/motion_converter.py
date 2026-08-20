"""
AMASS 到 Habitat SMPL-X 动作转换器 —— motion_converter.py
=========================================================

功能：
    1. 基于 Habitat 官方 MotionConverterSMPLX 将 AMASS .npz 动作数据转换为 Habitat 可直接回放的 .pkl 格式；
    2. 同时兼容标准 AMASS Schema B (root_orient 从 poses[:, :3] 派生) 与 Schema A (显式 root_orient)；
    3. 自适应对齐 SMPL-X 54 关节维度 (156 -> 162 dims 零填充)；
    4. 支持按 BABEL 标注时间戳/帧范围 (start_frame:end_frame) 精确切片；
    5. 生成包含完整动作语义与时序元数据的标准 .pkl 结构。
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import magnum as mn
    from habitat.utils.humanoid_utils import MotionConverterSMPLX
except ImportError:
    mn = None
    MotionConverterSMPLX = None

from tools.motion_assets.data_paths import (
    get_assets_dir,
    get_data_root,
    from_relative_data_path,
    to_relative_data_path,
)

logger = logging.getLogger(__name__)

NUM_SMPLX_JOINTS = 54
SMPLX_JOINT_DIM = 162  # 54 joints * 3 dof


class AMASSMotionConverter:
    """AMASS 动作到 Habitat SMPL-X 关节/刚体序列的转换器。"""

    def __init__(self, urdf_path: Union[str, Path]):
        self.urdf_path = Path(urdf_path).resolve()
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"Humanoid URDF not found at {self.urdf_path}")

        if MotionConverterSMPLX is None:
            raise RuntimeError(
                "habitat.utils.humanoid_utils.MotionConverterSMPLX is unavailable. "
                "Please run inside the habitat conda environment."
            )

        self._converter = MotionConverterSMPLX(str(self.urdf_path))

    def convert_motion_data(
        self,
        npz_path: Union[str, Path],
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """转换单个 AMASS .npz 文件或其片段为 Habitat 动作字典。"""
        npz_p = Path(npz_path).resolve()
        if not npz_p.exists():
            raise FileNotFoundError(f"AMASS npz file not found: {npz_p}")

        content = np.load(npz_p, allow_pickle=True)

        if "trans" not in content or "poses" not in content:
            raise ValueError(f"Invalid AMASS file (missing trans or poses): {npz_p}")

        trans = content["trans"]
        poses = content["poses"]
        num_raw_frames = poses.shape[0]

        # 帧率解析 (兼容多种键名)
        fps = 30.0
        for fps_key in ["mocap_framerate", "mocap_frame_rate", "frame_rate", "framerate", "fps"]:
            if fps_key in content:
                val = content[fps_key]
                fps = float(val) if np.ndim(val) == 0 else float(val[0])
                break

        # Root orient 解析 (Schema A vs Schema B)
        if "root_orient" in content and content["root_orient"].shape[0] == num_raw_frames:
            root_orient = content["root_orient"]
            body_pose_raw = poses[:, 3:] if poses.shape[1] > 3 else poses
        else:
            root_orient = poses[:, :3]
            body_pose_raw = poses[:, 3:]

        # 补齐 SMPL-X 关节至 162 维 (54 joints * 3)
        padded_pose = np.zeros((num_raw_frames, SMPLX_JOINT_DIM), dtype=np.float32)
        valid_dims = min(body_pose_raw.shape[1], SMPLX_JOINT_DIM)
        padded_pose[:, :valid_dims] = body_pose_raw[:, :valid_dims]

        # 帧范围切片
        sf = max(0, int(start_frame)) if start_frame is not None else 0
        ef = min(num_raw_frames - 1, int(end_frame)) if end_frame is not None else num_raw_frames - 1
        if sf > ef:
            sf, ef = 0, num_raw_frames - 1

        transform_array = []
        joints_array = []

        for f_idx in range(sf, ef + 1):
            root_t, root_r, pose_quat = self._converter.convert_pose_to_rotation(
                trans[f_idx],
                root_orient[f_idx],
                padded_pose[f_idx],
            )
            transform_mat = np.array(mn.Matrix4.from_(root_r, root_t))
            transform_array.append(transform_mat[None, :])
            joints_array.append(np.array(pose_quat, dtype=np.float32)[None, :])

        transform_array = np.concatenate(transform_array, axis=0)
        joints_array = np.concatenate(joints_array, axis=0)

        meta_out = dict(metadata or {})
        meta_out.update({
            "source_npz_path": to_relative_data_path(npz_p),
            "start_frame": sf,
            "end_frame": ef,
            "num_converted_frames": ef - sf + 1,
            "total_frames_in_source": num_raw_frames,
            "fps": fps,
        })

        return {
            "pose_motion": {
                "joints_array": joints_array,
                "transform_array": transform_array,
                "displacement": None,
                "fps": fps,
            },
            "metadata": meta_out,
        }

    def convert_to_file(
        self,
        npz_path: Union[str, Path],
        output_pkl_path: Union[str, Path],
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """转换并保存为 .pkl 文件。"""
        res = self.convert_motion_data(npz_path, start_frame, end_frame, metadata)
        out_p = Path(output_pkl_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "wb") as f:
            pickle.dump(res, f)
        logger.info("Saved converted motion to %s", out_p)
        return out_p


def convert_single_amass_motion(
    manifest_item: Dict[str, Any],
    urdf_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """依据 manifest 条目转换单条动作。"""
    rel_npz = manifest_item["local_motion_path"]
    npz_path = from_relative_data_path(rel_npz)
    sf = manifest_item.get("start_frame")
    ef = manifest_item.get("end_frame")
    target_class = manifest_item.get("target_class", "action")
    sid = manifest_item.get("babel_sid", "unknown")

    out_base = Path(output_dir) if output_dir else get_assets_dir("motions/converted")
    out_base.mkdir(parents=True, exist_ok=True)
    out_file = out_base / f"{target_class}_{sid}.pkl"

    converter = AMASSMotionConverter(urdf_path)
    return converter.convert_to_file(
        npz_path=npz_path,
        output_pkl_path=out_file,
        start_frame=sf,
        end_frame=ef,
        metadata=manifest_item,
    )


def batch_convert_manifest_motions(
    manifest_path: Optional[Union[str, Path]] = None,
    urdf_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    target_class_filter: Optional[str] = None,
) -> List[Path]:
    """批量转换 motion_asset_manifest.json 中声明的所有动作。"""
    data_root = get_data_root()
    man_path = Path(manifest_path) if manifest_path else data_root / "assets/motions/raw/motion_asset_manifest.json"
    if not man_path.exists():
        raise FileNotFoundError(f"Manifest not found: {man_path}")

    with open(man_path, "r", encoding="utf-8") as f:
        manifest_items = json.load(f)

    if target_class_filter:
        manifest_items = [m for m in manifest_items if m.get("target_class") == target_class_filter]

    out_paths = []
    default_urdf = Path("/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/habitat_humanoids/neutral_0/neutral_0.urdf")
    resolved_urdf = Path(urdf_path) if urdf_path else default_urdf

    converter = AMASSMotionConverter(resolved_urdf)
    out_base = Path(output_dir) if output_dir else get_assets_dir("motions/converted")
    out_base.mkdir(parents=True, exist_ok=True)

    for item in manifest_items:
        rel_npz = item.get("local_motion_path")
        if not rel_npz:
            continue
        npz_p = from_relative_data_path(rel_npz)
        if not npz_p.exists():
            logger.warning("NPZ not found on disk, skipping: %s", npz_p)
            continue

        target_class = item.get("target_class", "action")
        sid = item.get("babel_sid", "0")
        out_pkl = out_base / f"{target_class}_{sid}.pkl"

        p = converter.convert_to_file(
            npz_path=npz_p,
            output_pkl_path=out_pkl,
            start_frame=item.get("start_frame"),
            end_frame=item.get("end_frame"),
            metadata=item,
        )
        out_paths.append(p)

    return out_paths
