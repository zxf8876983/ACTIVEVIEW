#!/usr/bin/env python3
"""Render a sample ParaHome → Habitat replay video through the real Habitat simulator.

Uses the same clean `scene_id=NONE` replay path as
``activeview/scripts/eval/audit_parahome_feasibility.py`` (diagnostic floor,
scanned rigid objects, male_0 humanoid) but renders a contiguous segment as an
MP4 with three camera views side by side, so the retargeted pose can be
inspected directly.

``--variant fixed`` uses the shipped hierarchical retargeter, ``--variant
prefix`` reproduces the pre-fix world-space solver for an A/B comparison.

Example
-------
PYTHONPATH=. python render_retarget_sample_video.py \
    --parahome-root /home/zxf/MG08/robot/ParaHome --sequence s78 \
    --frame-start 445 --frame-end 700 --variant both
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / "activeview" / "core" / "paths.py").is_file():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from activeview.data.motion.babel_clean_dataset_generator import (  # noqa: E402
    apply_humanoid_pose,
    precompute_grounding_offsets,
)
from activeview.data.motion.parahome_retarget import (  # noqa: E402
    PARAHOME_BODY_INDEX,
    PARAHOME_CHILDREN,
    PARAHOME_TO_HABITAT,
    ParaHomeSkeletonRetargeter,
    _align_vectors,
)
from render_retarget_pose_ab import GLB, SkinnedHumanoid, _fk_all  # noqa: E402
from activeview.scripts.eval import audit_parahome_feasibility as audit  # noqa: E402

FPS = audit.FPS
SCENE_OBJECTS = ("chair", "laptop", "cup", "kettle")
PANEL_SIZE = 512
# Robot-eye camera: the robot stands on the floor and its camera sits
# ROBOT_CAMERA_HEIGHT_M above the ground, ROBOT_STANDOFF_M away from the human,
# rotated ROBOT_SIDE_DEG off the human's facing so the view is a 3/4 view.  The
# aim point is a fixed height above the floor, which keeps the horizon (and the
# floor the human stands on) visible instead of looking down from above.
ROBOT_CAMERA_HEIGHT_M = 1.2
ROBOT_STANDOFF_M = 2.8
ROBOT_SIDE_DEG = 30.0
ROBOT_LOOK_AT_HEIGHT_M = 1.0
ROBOT_BASE_M = (0.36, 0.12, 0.36)
# The robot is a static platform: by default its position *and* its camera
# orientation are fixed for the whole clip (it is placed once, relative to the
# human's first frame).  ``--robot-anchor follow`` makes the platform move with
# the human and ``--robot-aim track`` keeps the base fixed but pans the camera to
# keep the human framed.
ROBOT_ANCHOR_DEFAULT = "fixed"
ROBOT_AIM_DEFAULT = "fixed"

# Legacy three body-relative cameras, kept for comparison renders.  Offsets are
# relative to (human root + LOOK_AT_HEIGHT) in the avatar's own frame: +Z is the
# male_0 rest facing (verified by rendering the rest pose), +Y up.
VIEWS: Dict[str, np.ndarray] = {
    "front": np.asarray([0.15, 0.55, 2.60], dtype=np.float32),
    "side": np.asarray([2.60, 0.55, 0.05], dtype=np.float32),
    "top": np.asarray([0.20, 2.45, 0.55], dtype=np.float32),
}
LOOK_AT_HEIGHT = 0.75


class PreFixRetargeter(ParaHomeSkeletonRetargeter):
    """Pre-fix solver: one independent world-space alignment per joint."""

    def _solve_global_rotations(self, desired: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            parent: _align_vectors(
                [self._rest_offsets[child] for child in children],
                [desired[child] - desired[parent] for child in children],
            )
            for parent, children in PARAHOME_CHILDREN.items()
        }


def _annotations(sequence_path: Path) -> Dict[int, str]:
    raw = json.loads((sequence_path / "text_annotation.json").read_text(encoding="utf-8"))
    values: Dict[int, str] = {}
    for interval, text in raw.items():
        start, end = (int(part) for part in str(interval).split())
        for frame_id in range(start, end):
            values[frame_id] = str(text)
    return values


def _label(panel: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(panel.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for index, text in enumerate(lines):
        draw.text((8, 6 + 14 * index), text, fill=(255, 255, 255))
    return np.asarray(image)


def _add_robot_base(sim: Any, size: float = 0.36, height: float = 0.12) -> Any:
    """A small box marking the robot footprint on the floor."""
    import habitat_sim
    import magnum as mn

    manager = sim.get_object_template_manager()
    template = manager.get_template_by_handle("cubeSolid")
    template.scale = mn.Vector3(size, height, size)
    manager.register_template(template, "retarget_robot_base", True)
    base = sim.get_rigid_object_manager().add_object_by_template_handle("retarget_robot_base")
    base.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    return base


def _mesh_positions(
    retargeter: ParaHomeSkeletonRetargeter,
    humanoid: SkinnedHumanoid,
    joints: np.ndarray,
    root: np.ndarray,
    source_frame: np.ndarray,
) -> np.ndarray:
    """Y coordinates (world) of the skinned mesh for one frame, root at y=0."""
    desired = {
        name: PARAHOME_TO_HABITAT @ source_frame[index]
        for name, index in PARAHOME_BODY_INDEX.items()
    }
    positions, rotations = _fk_all(
        retargeter, joints, root[:3, :3], np.zeros(3), offsets=retargeter._frame_offsets(desired)
    )
    return humanoid.deform(positions, rotations)[:, 1]


def _robot_camera(root: np.ndarray, frame: np.ndarray, *, standoff: float, side_deg: float, height: float, look_at_height: float) -> Tuple[np.ndarray, np.ndarray]:
    """Camera position on the floor and the aim point, both in world coordinates."""
    ground = np.asarray([float(root[0]), 0.0, float(root[2])], dtype=np.float64)
    facing = frame @ np.asarray([0.0, 0.0, 1.0])
    azimuth = math.atan2(float(facing[0]), float(facing[2])) + math.radians(side_deg)
    direction = np.asarray([math.sin(azimuth), 0.0, math.cos(azimuth)], dtype=np.float64)
    position = ground + direction * standoff + np.asarray([0.0, height, 0.0])
    target = ground + np.asarray([0.0, look_at_height, 0.0])
    return position, target


def _scene_simulator(image_size: int) -> Tuple[Any, Any]:
    """Clean floor scene with a sensor placed exactly at the agent origin."""
    import habitat_sim
    import magnum as mn

    from activeview.core.paths import get_humanoid_urdf_path

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = "NONE"
    backend.enable_physics = True
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "color"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [image_size, image_size]
    sensor.position = mn.Vector3(0.0, 0.0, 0.0)
    sensor.hfov = mn.Deg(70.0)
    agent = habitat_sim.AgentConfiguration()
    agent.sensor_specifications = [sensor]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    manager = sim.get_object_template_manager()
    floor_template = manager.get_template_by_handle("cubeSolid")
    floor_template.scale = mn.Vector3(10.0, 0.02, 10.0)
    manager.register_template(floor_template, "retarget_sample_floor", True)
    floor = sim.get_rigid_object_manager().add_object_by_template_handle("retarget_sample_floor")
    floor.translation = mn.Vector3(0.0, -0.02, 0.0)
    floor.motion_type = habitat_sim.physics.MotionType.STATIC
    human = sim.get_articulated_object_manager().add_articulated_object_from_urdf(str(get_humanoid_urdf_path("male_0")))
    return sim, human


def _look_at(sim: Any, position: np.ndarray, target: np.ndarray) -> None:
    import habitat_sim
    import magnum as mn
    import quaternion

    camera_position = np.asarray(position, dtype=np.float32)
    direction = np.asarray(target, dtype=np.float32) - camera_position
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    yaw = math.atan2(-float(direction[0]), -float(direction[2]))
    pitch = math.asin(float(direction[1]))
    state = habitat_sim.AgentState()
    state.position = mn.Vector3(*camera_position)
    state.rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0]) * quaternion.from_rotation_vector([pitch, 0.0, 0.0])
    sim.get_agent(0).set_state(state)


def _object_names(
    parahome_root: Path,
    object_transforms: Sequence[Mapping[str, Any]],
    present: Mapping[str, Any],
) -> List[str]:
    return [
        name for name in SCENE_OBJECTS
        if bool(present.get(name, False))
        and f"{name}_base" in object_transforms[0]
        and (parahome_root / "data" / "scan" / name / "simplified" / "base.obj").is_file()
    ]


def render_variant(
    *,
    parahome_root: Path,
    sequence: str,
    variant: str,
    retargeter: ParaHomeSkeletonRetargeter,
    frame_ids: Sequence[int],
    output_dir: Path,
    panel_size: int = PANEL_SIZE,
    views: Sequence[str] = tuple(VIEWS),
    quality: int = 8,
    camera_mode: str = "robot",
    robot_anchor: str = ROBOT_ANCHOR_DEFAULT,
    robot_aim: str = ROBOT_AIM_DEFAULT,
    standoff: float = ROBOT_STANDOFF_M,
    side_deg: float = ROBOT_SIDE_DEG,
    camera_height: float = ROBOT_CAMERA_HEIGHT_M,
    show_robot_base: bool = True,
) -> Dict[str, Any]:
    import magnum as mn
    from PIL import Image, ImageDraw

    sequence_path = parahome_root / "data" / "seq" / sequence
    with (sequence_path / "joint_positions.pkl").open("rb") as handle:
        source_joints = np.asarray(pickle.load(handle), dtype=np.float32)
    with (sequence_path / "body_global_transform.pkl").open("rb") as handle:
        para_root = np.asarray(pickle.load(handle), dtype=np.float64)[:, :3, 3]
    with (sequence_path / "object_transformations.pkl").open("rb") as handle:
        object_transforms = pickle.load(handle)
    present = json.loads((sequence_path / "object_in_scene.json").read_text(encoding="utf-8"))
    annotations = _annotations(sequence_path)

    converted, diagnostics = retargeter.retarget(source_joints, fps=FPS)
    joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
    roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)
    align, align_rmse = audit._rigid_alignment(para_root, roots[:, :3, 3].astype(np.float64))
    object_names = _object_names(parahome_root, object_transforms, present)

    sim, human = _scene_simulator(panel_size)
    objects: Dict[str, Any] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in object_names:
        objects[name] = audit._register_obj(
            sim, parahome_root / "data" / "scan" / name / "simplified" / "base.obj", f"sample_{name}"
        )
    # Ground on the *skinned mesh* rather than the URDF debug boxes used by
    # precompute_grounding_offsets: the boxes are bone primitives, so a seated
    # pose could keep the visible feet ~15 cm above the floor.  The offline
    # skinning reproduces Habitat's own skinning (verified against rendered
    # frames), so the lowest deformed vertex is the honest floor contact.
    humanoid_mesh = SkinnedHumanoid(GLB, retargeter)
    offsets = np.asarray(
        [
            float(-_mesh_positions(retargeter, humanoid_mesh, joints[i], roots[i], source_joints[frame_id]).min())
            for i, frame_id in enumerate(frame_ids)
        ],
        dtype=np.float64,
    )
    if not np.isfinite(offsets).all():
        raise RuntimeError("non-finite grounding offset from the skinned humanoid mesh")
    base = _add_robot_base(sim) if (camera_mode == "robot" and show_robot_base) else None
    fixed_camera: Tuple[np.ndarray, np.ndarray] | None = None
    if camera_mode == "robot" and robot_anchor == "fixed":
        # Place the static platform once, centred on the recorded trajectory so it
        # keeps the whole clip in view instead of only the first frame.
        ground_center = np.asarray(
            [float(roots[frame_ids, 0, 3].mean()), 0.0, float(roots[frame_ids, 2, 3].mean())],
            dtype=np.float64,
        )
        fixed_camera = _robot_camera(
            ground_center,
            roots[frame_ids[0], :3, :3].astype(np.float64),
            standoff=standoff,
            side_deg=side_deg,
            height=camera_height,
            look_at_height=ROBOT_LOOK_AT_HEIGHT_M,
        )

    video_path = output_dir / f"{sequence}_retarget_{variant}.mp4"
    writer = imageio.get_writer(str(video_path), fps=float(FPS), codec="libx264", quality=quality)
    written_frames = 0
    try:
        for index, frame_id in enumerate(frame_ids):
            apply_humanoid_pose(
                human,
                joints[frame_id],
                roots[frame_id],
                base_position=roots[frame_id, :3, 3],
                grounding_offset=float(offsets[index]),
            )
            mapped_human = np.asarray(human.translation, dtype=np.float32)
            for name, obj in objects.items():
                obj.transformation = mn.Matrix4(
                    (align @ np.asarray(object_transforms[frame_id][f"{name}_base"], dtype=np.float64)).astype(np.float32)
                )
            panels = []
            frame = roots[frame_id, :3, :3].astype(np.float64)
            if camera_mode == "robot":
                position, target = _robot_camera(
                    mapped_human, frame, standoff=standoff, side_deg=side_deg,
                    height=camera_height, look_at_height=ROBOT_LOOK_AT_HEIGHT_M,
                )
                if robot_anchor == "fixed":
                    # Static platform: place it once, from the first rendered frame.
                    if fixed_camera is None:
                        fixed_camera = (position, target)
                    position = fixed_camera[0]
                    if robot_aim == "fixed":
                        target = fixed_camera[1]
                _look_at(sim, position.astype(np.float32), target.astype(np.float32))
                observation = np.asarray(sim.get_sensor_observations()["color"])
                if observation.ndim != 3 or observation.shape[-1] < 3 or not np.isfinite(observation[..., :3]).all():
                    raise RuntimeError(f"invalid RGB observation at frame {frame_id}, camera {camera_mode}")
                panels.append(_label(observation[..., :3], ("robot view",)))
                if base is not None:
                    footprint = np.asarray([position[0], ROBOT_BASE_M[1] * 0.5, position[2]], dtype=np.float64)
                    base.translation = footprint.astype(np.float32)
            else:
                # Body-relative cameras so "front" always means the avatar's front.
                look_at = mapped_human + np.asarray([0.0, LOOK_AT_HEIGHT, 0.0], dtype=np.float32)
                for view_name in views:
                    offset = VIEWS[view_name]
                    world_offset = (frame @ offset.astype(np.float64)).astype(np.float32)
                    _look_at(sim, look_at + world_offset, look_at)
                    observation = np.asarray(sim.get_sensor_observations()["color"])
                    if observation.ndim != 3 or observation.shape[-1] < 3 or not np.isfinite(observation[..., :3]).all():
                        raise RuntimeError(f"invalid RGB observation at frame {frame_id}, view {view_name}")
                    panels.append(_label(observation[..., :3], (view_name,)))
            composite = np.concatenate(panels, axis=1)
            header = np.zeros((34, composite.shape[1], 3), dtype=np.uint8)
            composite = np.concatenate([header, composite], axis=0)
            header_image = Image.fromarray(composite)
            ImageDraw.Draw(header_image).text(
                (8, 10),
                f"ParaHome {sequence} | retarget {variant} | frame {frame_id} | {annotations.get(frame_id, 'unannotated')}",
                fill=(255, 255, 255),
            )
            composite = np.asarray(header_image)
            writer.append_data(composite)
            written_frames += 1
            if frame_id == frame_ids[len(frame_ids) // 2]:
                Image.fromarray(composite).save(output_dir / f"{sequence}_retarget_{variant}_frame_{frame_id:04d}.png")
    finally:
        writer.close()
        sim.close()

    return {
        "variant": variant,
        "sequence": sequence,
        "video": str(video_path),
        "frames": written_frames,
        "frame_ids": [int(frame_ids[0]), int(frame_ids[-1])],
        "fps": FPS,
        "camera_mode": camera_mode,
        "camera_height_m": (camera_height if camera_mode == "robot" else None),
        "robot_standoff_m": (standoff if camera_mode == "robot" else None),
        "robot_anchor": (robot_anchor if camera_mode == "robot" else None),
        "robot_aim": (robot_aim if camera_mode == "robot" else None),
        "views": (["robot"] if camera_mode == "robot" else sorted(views)),
        "objects": object_names,
        "retarget_mean_joint_rmse_m": diagnostics.mean_joint_rmse_m,
        "root_alignment_rmse_m": align_rmse,
        "grounding_offset_range_m": [float(offsets.min()), float(offsets.max())],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parahome-root", type=Path, default=Path("/home/zxf/MG08/robot/ParaHome"))
    parser.add_argument("--sequence", default="s78")
    parser.add_argument("--variant", choices=("fixed", "prefix", "both"), default="fixed")
    parser.add_argument("--frame-start", type=int, default=445)
    parser.add_argument("--frame-end", type=int, default=700)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--camera", choices=("robot", "body3"), default="robot",
                        help="robot: one robot-eye camera standing on the floor (default); body3: legacy three views")
    parser.add_argument("--robot-anchor", choices=("fixed", "follow"), default=ROBOT_ANCHOR_DEFAULT,
                        help="fixed: the robot platform stands still (default); follow: it moves with the human")
    parser.add_argument("--robot-aim", choices=("fixed", "track"), default=ROBOT_AIM_DEFAULT,
                        help="fixed: camera orientation also frozen (default); track: base still, camera pans")
    parser.add_argument("--standoff", type=float, default=ROBOT_STANDOFF_M)
    parser.add_argument("--side-deg", type=float, default=ROBOT_SIDE_DEG)
    parser.add_argument("--camera-height", type=float, default=ROBOT_CAMERA_HEIGHT_M)
    parser.add_argument("--panel-size", type=int, default=PANEL_SIZE)
    parser.add_argument("--views", default="front,side,top", help="comma separated subset of front,side,top")
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "experiments/parahome_feasibility_v1/retarget_pose_fidelity/videos")
    args = parser.parse_args()

    from activeview.core.paths import get_humanoid_urdf_path

    urdf = get_humanoid_urdf_path("male_0")
    with (args.parahome_root.resolve() / "data" / "seq" / args.sequence / "joint_positions.pkl").open("rb") as handle:
        sequence_frames = len(pickle.load(handle))
    frame_end = min(args.frame_end, sequence_frames)
    frame_ids = list(range(max(0, args.frame_start), frame_end, max(1, args.stride)))
    variants: List[Tuple[str, ParaHomeSkeletonRetargeter]] = []
    if args.variant in ("fixed", "both"):
        variants.append(("fixed", ParaHomeSkeletonRetargeter(urdf)))
    if args.variant in ("prefix", "both"):
        variants.append(("prefix", PreFixRetargeter(urdf)))

    report = [
        render_variant(
            parahome_root=args.parahome_root.resolve(),
            sequence=args.sequence,
            variant=variant,
            retargeter=retargeter,
            frame_ids=frame_ids,
            output_dir=args.output_dir.resolve(),
            panel_size=args.panel_size,
            views=tuple(part.strip() for part in args.views.split(",") if part.strip()),
            quality=args.quality,
            camera_mode=args.camera,
            robot_anchor=args.robot_anchor,
            robot_aim=args.robot_aim,
            standoff=args.standoff,
            side_deg=args.side_deg,
            camera_height=args.camera_height,
        )
        for variant, retargeter in variants
    ]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
