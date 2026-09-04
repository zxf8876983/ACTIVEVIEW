"""文件用途：
    处理 AMASS/BABEL 动作资产与 Habitat 转换。

主要输入：
    - 动作标注、NPZ 资产和 URDF。
主要输出：
    - 规范化动作、映射或动作清单。
项目角色：
    - 属于 data.motion 数据模块。
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from activeview.core.paths import get_data_root, get_humanoid_urdf_path
from activeview.data.motion.babel_selected16_manifest import SELECTED_LABELS
from activeview.data.motion.humanoid_grounding import (
    ground_humanoid_to_floor,
    humanoid_geometry_y_bounds,
)
from activeview.perception.pose.ultralytics import UltralyticsPose3DEstimator
from activeview.perception.skeleton import get_skeleton_definition
from activeview.perception.normalization import SkeletonNormalizer
from activeview.data.motion.amass_loader import NormalizedMotion
from activeview.data.motion.motion_converter import MotionConverter

LOGGER = logging.getLogger(__name__)

LABEL_TO_ID = {label: idx for idx, label in enumerate(SELECTED_LABELS)}
SKELETON_PREPROCESSING = "camera_to_gravity+root_center+torso_scale+yaw_only_y_up_v5"
RENDERING_PROTOCOL = "amass_root_rotation+scene_yaw+per_frame_grounding+diagnostic_floor_mesh_v3"
COORDINATE_TRANSFORM = "VideoPose3D_H36M_Ydown_Zdepth_to_Habitat_gravity; Habitat_-Z_to_ActiveView_+Z"
URDF_PATH = get_humanoid_urdf_path("male_0")


def _load_resampled_motion(record: Mapping[str, Any], target_frames: int) -> NormalizedMotion:
    """Load one BABEL interval and temporally resample it to target frames."""
    source_path = Path(str(record["source_path"]))
    with np.load(source_path, allow_pickle=True) as data:
        trans_raw = np.asarray(data["trans"], dtype=np.float32)
        poses_raw = np.asarray(data["poses"], dtype=np.float32)
        total_frames = int(poses_raw.shape[0])
        fps = float(record.get("fps", 30.0))
        if "root_orient" in data and np.asarray(data["root_orient"]).shape[0] == total_frames:
            root_raw = np.asarray(data["root_orient"], dtype=np.float32)
            body_raw = poses_raw[:, 3:] if poses_raw.shape[1] > 3 else poses_raw
        else:
            root_raw = poses_raw[:, :3]
            body_raw = poses_raw[:, 3:]
        body = np.zeros((total_frames, 162), dtype=np.float32)
        body[:, : min(body_raw.shape[1], 162)] = body_raw[:, :162]

        start = max(0, min(int(record["start_frame"]), total_frames - 1))
        end = max(start, min(int(record["end_frame"]), total_frames - 1))
        indices = np.linspace(start, end, target_frames, dtype=np.int64)
        return NormalizedMotion(
            translation=trans_raw[indices],
            root_rotation=root_raw[indices],
            body_pose=body[indices],
            fps=fps,
            num_frames=target_frames,
            metadata=dict(record),
        )


def compose_root_rotation(root_transform: np.ndarray, scene_yaw_deg: float = 0.0) -> np.ndarray:
    """Preserve AMASS roll/pitch/yaw and optionally prepend a scene yaw."""
    transform = np.asarray(root_transform, dtype=np.float32)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("root_transform must be a finite 4x4 matrix")
    yaw = np.radians(float(scene_yaw_deg))
    scene_rotation = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float32,
    )
    composed = np.eye(4, dtype=np.float32)
    composed[:3, :3] = scene_rotation @ transform[:3, :3]
    return composed


def transform_camera_sequence_to_gravity(
    sequence: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    """Map VideoPose3D camera coordinates into the Habitat gravity frame.

    VideoPose3D uses the Human3.6M camera convention (``+X`` right, ``+Y``
    down, ``+Z`` depth). Habitat sensors use an OpenGL-like camera frame with
    ``+X`` right, ``+Y`` up and ``-Z`` forward. The per-frame Habitat camera
    rotation therefore acts after flipping both Y and Z.
    Translation is intentionally omitted because VideoPose3D has no global
    trajectory and the downstream normalizer removes the root position.
    """
    points = np.asarray(sequence, dtype=np.float32)
    transforms = np.asarray(camera_to_world, dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("sequence must have shape (T, V, 3)")
    if transforms.shape != (points.shape[0], 4, 4):
        raise ValueError("camera_to_world must have shape (T, 4, 4)")
    if not np.isfinite(points).all() or not np.isfinite(transforms).all():
        raise ValueError("sequence and camera_to_world must be finite")

    # Convert Human3.6M camera coordinates to Habitat's camera frame before
    # applying the camera's local-to-world rotation.
    habitat_camera = points.copy()
    habitat_camera[..., 1] *= -1.0
    habitat_camera[..., 2] *= -1.0
    rotations = transforms[:, :3, :3]
    return np.einsum("tij,tvj->tvi", rotations, habitat_camera).astype(np.float32)


def precompute_grounding_offsets(
    human: Any,
    joints_sequence: np.ndarray,
    root_transforms: np.ndarray,
    *,
    scene_yaw_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute per-frame floor offsets and visual-body centers once.

    The values depend on the articulated pose and scene yaw, but not on the
    camera viewpoint or horizontal placement.  They can therefore be reused
    across all views of the same action.
    """
    import magnum as mn

    offsets: List[float] = []
    centers: List[float] = []
    for joints, root_transform in zip(joints_sequence, root_transforms):
        human.joint_positions = np.asarray(joints, dtype=np.float32)
        human.transformation = mn.Matrix4(compose_root_rotation(root_transform, scene_yaw_deg))
        human.translation = np.zeros(3, dtype=np.float32)
        min_y, max_y = humanoid_geometry_y_bounds(human, URDF_PATH)
        offsets.append(-float(min_y))
        centers.append((float(min_y) + float(max_y)) * 0.5)
    return np.asarray(offsets, dtype=np.float32), np.asarray(centers, dtype=np.float32)


def apply_humanoid_pose(
    human: Any,
    joints: np.ndarray,
    root_transform: np.ndarray,
    *,
    base_position: Sequence[float],
    scene_yaw_deg: float = 0.0,
    floor_y: float = 0.0,
    grounding_offset: Optional[float] = None,
) -> float:
    """Apply articulated joints and root orientation, then place the body on the floor."""
    import magnum as mn

    base = np.asarray(base_position, dtype=np.float32)
    if base.shape != (3,):
        raise ValueError("base_position must have shape (3,)")
    human.joint_positions = np.asarray(joints, dtype=np.float32)
    human.transformation = mn.Matrix4(compose_root_rotation(root_transform, scene_yaw_deg))
    if grounding_offset is None:
        return ground_humanoid_to_floor(
            human,
            base_position=base,
            floor_y=floor_y,
            urdf_path=URDF_PATH,
        )
    grounded_y = float(floor_y) + float(grounding_offset)
    human.translation = np.asarray([base[0], grounded_y, base[2]], dtype=np.float32)
    return grounded_y


class BabelCleanDatasetGenerator:
    """Resume-able RGB perception dataset generator for one split."""

    def __init__(
        self,
        output_root: Optional[Path] = None,
        *,
        image_size: int = 512,
        target_frames: int = 30,
        camera_height: float = 1.2,
        seed: int = 42,
        device: Optional[str] = None,
        label_to_id: Optional[Mapping[str, int]] = None,
        pose_backend: str = "ultralytics_yolo26n",
        yolo_weights: Optional[Path] = None,
        videopose_weights: Optional[Path] = None,
    ) -> None:
        import habitat_sim
        import quaternion  # noqa: F401 - required by Habitat rotations

        self.habitat_sim = habitat_sim
        self.image_size = int(image_size)
        self.target_frames = int(target_frames)
        self.camera_height = float(camera_height)
        if self.camera_height <= 0.0:
            raise ValueError("camera_height must be positive")
        self.seed = int(seed)
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.output_root = Path(output_root) if output_root else get_data_root() / "datasets" / "stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed"
        self.output_root.mkdir(parents=True, exist_ok=True)
        mapping_path = self.output_root / "label_mapping.json"
        if label_to_id is None and mapping_path.exists():
            label_to_id = json.loads(mapping_path.read_text(encoding="utf-8"))
        self.label_to_id = {str(k): int(v) for k, v in (label_to_id or LABEL_TO_ID).items()}
        self.skel_def = get_skeleton_definition(backend="h36m_17")
        self.normalizer = SkeletonNormalizer(skel_def=self.skel_def)
        self.converter = MotionConverter(URDF_PATH)
        self.pose_backend = str(pose_backend).lower()
        if self.pose_backend not in {"ultralytics_yolo26n", "yolo26n_pose"}:
            raise ValueError("v11 uses only the Ultralytics YOLO26n-Pose backend")
        weights = yolo_weights or get_data_root() / "checkpoints/ultralytics/yolo26n-pose.pt"
        self.estimator = UltralyticsPose3DEstimator(
            device=str(self.device),
            image_size=self.image_size,
            weights=weights,
            skel_def=self.skel_def,
            backend_label="ultralytics_yolo26n_pose",
            videopose_weights=videopose_weights,
        )
        self.perception_chain = "RGB -> Ultralytics YOLO26n-Pose -> VideoPose3D"

    def _make_sim(self) -> Tuple[Any, Any]:
        import habitat_sim
        import magnum as mn

        backend = habitat_sim.SimulatorConfiguration()
        backend.scene_id = "NONE"
        backend.enable_physics = True
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid = "color_sensor"
        sensor.sensor_type = habitat_sim.SensorType.COLOR
        sensor.resolution = [self.image_size, self.image_size]
        sensor.position = [0.0, 0.0, 0.0]
        sensor.hfov = 50.0
        agent = habitat_sim.AgentConfiguration()
        agent.sensor_specifications = [sensor]
        sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
        floor_manager = sim.get_object_template_manager()
        floor_handle = "activeview_diagnostic_floor"
        if floor_handle not in floor_manager.get_template_handles():
            floor_template = floor_manager.get_template_by_handle("cubeSolid")
            floor_template.scale = mn.Vector3(10.0, 0.02, 10.0)
            floor_template_id = floor_manager.register_template(floor_template, floor_handle, True)
            floor_handle = floor_manager.get_template_handle_by_id(floor_template_id)
        floor = sim.get_rigid_object_manager().add_object_by_template_handle(floor_handle)
        floor.translation = np.asarray([0.0, -0.02, 0.0], dtype=np.float32)
        floor.motion_type = habitat_sim.physics.MotionType.STATIC
        human = sim.get_articulated_object_manager().add_articulated_object_from_urdf(str(URDF_PATH))
        return sim, human

    def _render_rgb(
        self,
        sim: Any,
        human: Any,
        joints: np.ndarray,
        root_transforms: np.ndarray,
        record: Mapping[str, Any],
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        import habitat_sim
        import quaternion

        rng = random.Random(self.seed + int(record.get("candidate_index", record.get("babel_sid", 0))))
        angle = np.radians(rng.uniform(-25.0, 25.0))
        distance = rng.uniform(2.4, 2.8)
        frames: List[np.ndarray] = []
        camera_to_world: List[np.ndarray] = []
        grounding_offsets, center_y_values = precompute_grounding_offsets(
            human,
            joints,
            root_transforms,
            scene_yaw_deg=0.0,
        )
        for frame_index, (q_joints, root_transform) in enumerate(zip(joints, root_transforms)):
            apply_humanoid_pose(
                human,
                q_joints,
                root_transform,
                base_position=(0.0, 0.0, 0.0),
                grounding_offset=float(grounding_offsets[frame_index]),
            )
            # ``center_y_values`` is measured before grounding.  The body is
            # translated upward by the same per-frame offset below, so the
            # camera target must be translated as well; otherwise standing
            # bodies are framed too low and their upper half is clipped.
            center_y = float(center_y_values[frame_index] + grounding_offsets[frame_index])
            camera_position = np.array(
                [distance * np.sin(angle), self.camera_height, distance * np.cos(angle)], dtype=np.float32
            )
            target = np.array([0.0, center_y, 0.0], dtype=np.float32)
            direction = target - camera_position
            direction /= np.linalg.norm(direction)
            yaw = np.arctan2(-direction[0], -direction[2])
            pitch = np.arcsin(direction[1])
            rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0]) * quaternion.from_rotation_vector([pitch, 0.0, 0.0])
            state = habitat_sim.AgentState()
            state.position = camera_position
            state.rotation = rotation
            sim.get_agent(0).set_state(state)
            observation = sim.get_sensor_observations()["color_sensor"]
            frames.append(np.asarray(observation[:, :, :3], dtype=np.uint8))
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = quaternion.as_rotation_matrix(rotation).astype(np.float32)
            c2w[:3, 3] = camera_position
            camera_to_world.append(c2w)
        return frames, np.stack(camera_to_world, axis=0)

    def _sample_path(self, split: str, record_id: str) -> Path:
        path = self.output_root / "estimated_skeletons" / split / f"{record_id}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _process_record(self, sim: Any, human: Any, record: Mapping[str, Any]) -> Dict[str, Any]:
        norm_motion = _load_resampled_motion(record, self.target_frames)
        converted = self.converter.convert(norm_motion)
        joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
        root_transforms = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)
        rgb_frames, camera_to_world = self._render_rgb(sim, human, joints, root_transforms, record)
        estimated, confidences = self.estimator.estimate_sequence(rgb_frames)
        gravity_aligned = transform_camera_sequence_to_gravity(estimated, camera_to_world)
        normalized = self.normalizer.normalize_sequence(gravity_aligned, align_canonical=True)
        tensor = np.transpose(normalized, (2, 0, 1))[:, :, :, np.newaxis].astype(np.float32)
        out_path = self._sample_path(str(record["split"]), str(record["record_id"]))
        np.savez_compressed(out_path, skeleton=tensor, confidence=confidences.astype(np.float32))
        return {
            "record_id": record["record_id"],
            "action_label": record["action_label"],
            "label_id": self.label_to_id[str(record["action_label"])],
            "split": record["split"],
            "source_group": record["source_group"],
            "skeleton_path": str(out_path.relative_to(self.output_root)),
            "mean_confidence": float(np.mean(confidences)),
            "num_frames": self.target_frames,
            "source_num_frames": int(record["num_frames"]),
            "skeleton_preprocessing": SKELETON_PREPROCESSING,
            "rendering_protocol": RENDERING_PROTOCOL,
            "coordinate_transform": COORDINATE_TRANSFORM,
            "camera_height_m": self.camera_height,
            "pose_backend": self.pose_backend,
        }

    def generate_split(self, split: str, manifest_path: Path, max_records: Optional[int] = None) -> Dict[str, Any]:
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
        if max_records is not None:
            records = records[: max(0, int(max_records))]
        metadata_path = self.output_root / f"{split}_metadata.json"
        existing: Dict[str, Dict[str, Any]] = {}
        if metadata_path.exists():
            existing = {str(item["record_id"]): item for item in json.loads(metadata_path.read_text(encoding="utf-8"))}
        sim, human = self._make_sim()
        metadata: List[Dict[str, Any]] = []
        try:
            for index, record in enumerate(records, 1):
                record_id = str(record["record_id"])
                sample_path = self._sample_path(split, record_id)
                existing_item = existing.get(record_id)
                # A changed detector backend must invalidate the resumable
                # sample; otherwise a YOLO26 run could silently reuse a
                # YOLO11 skeleton while reporting the new configuration.
                if existing_item is not None and sample_path.exists() and existing_item.get("pose_backend") == self.pose_backend:
                    metadata.append(existing_item)
                    continue
                item = self._process_record(sim, human, record)
                metadata.append(item)
                if index % 25 == 0:
                    LOGGER.info("%s: processed %d/%d", split, index, len(records))
                    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        finally:
            sim.close()
        metadata.sort(key=lambda item: str(item["record_id"]))
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        data = np.stack([np.load(self.output_root / item["skeleton_path"])["skeleton"] for item in metadata])
        labels = np.asarray([item["label_id"] for item in metadata], dtype=np.int64)
        np.save(self.output_root / f"{split}_data.npy", data)
        np.save(self.output_root / f"{split}_labels.npy", labels)
        summary = {
            "split": split,
            "samples": len(metadata),
            "data_shape": list(data.shape),
            "skeleton_preprocessing": SKELETON_PREPROCESSING,
            "rendering_protocol": RENDERING_PROTOCOL,
            "coordinate_transform": COORDINATE_TRANSFORM,
            "perception_chain": self.perception_chain,
        }
        (self.output_root / f"{split}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
