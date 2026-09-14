"""Runtime helpers for the reduced12 RGB-D human-state audit.

The helpers deliberately keep raw images/depth transient.  Only compact
keypoints, depth summaries and reconstructed joints are serialized below the
runtime data root.  They are imported by the experiment runner and are not a
new production model.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from activeview.data.motion.babel_clean_dataset_generator import (
    MotionConverter,
    _load_resampled_motion,
    apply_humanoid_pose,
    precompute_grounding_offsets,
)
from activeview.data.motion.habitat_h36m17_fk import H36M17_LINKS
from activeview.scripts.data.generate_hm3d_train_rgb_observations import (
    _load_skeleton_metadata,
    _set_agent_state,
)
from activeview.scripts.eval.analyze_reduced12_scene_joint_visibility_oracle import (
    _ray_visible,
    _scene_sim,
    _world_h36m17,
)

NUM_VIEWS = 32
JOINTS = 17
IMAGE_SIZE = 256
HFOV_DEG = 75.0
SENSOR_HEIGHT = 1.10
DEPTH_NEIGHBORHOOD = 2
MIN_VALID_DEPTH_SAMPLES = 5
YOLO_CONF_THRESHOLD = 0.15
RELIABLE_CONF_THRESHOLD = 0.25
RAY_EPSILON = 0.03


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def rotation_wxyz_to_matrix(values: Sequence[float]) -> np.ndarray:
    q = np.asarray(values, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("camera rotation must be finite WXYZ")
    q = q / max(float(np.linalg.norm(q)), 1.0e-12)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def camera_backproject(
    pixel: Sequence[float], depth: float, position: np.ndarray, rotation_wxyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject one pixel using the fixed Habitat 75-degree camera."""
    x, y = float(pixel[0]), float(pixel[1])
    focal = 0.5 * IMAGE_SIZE / math.tan(math.radians(HFOV_DEG) * 0.5)
    x_norm = (x - 0.5 * IMAGE_SIZE) / focal
    y_norm = -(y - 0.5 * IMAGE_SIZE) / focal
    camera = np.asarray([x_norm * depth, y_norm * depth, -depth], dtype=np.float64)
    sensor = np.asarray(position, dtype=np.float64) + np.asarray([0.0, SENSOR_HEIGHT, 0.0])
    world = rotation_wxyz_to_matrix(rotation_wxyz) @ camera + sensor
    return camera.astype(np.float32), world.astype(np.float32)


def map_yolo_coco_to_h36m(keypoints: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map COCO-17 pose output to the canonical H36M-17 ordering."""
    xy = np.asarray(keypoints, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32)
    if xy.shape != (17, 2) or conf.shape != (17,):
        raise ValueError(f"unexpected YOLO pose shape {xy.shape}/{conf.shape}")

    def midpoint(a: int, b: int) -> tuple[np.ndarray, float]:
        return (xy[a] + xy[b]) * 0.5, float(min(conf[a], conf[b]))

    values: list[np.ndarray] = []
    scores: list[float] = []
    pelvis, pelvis_c = midpoint(11, 12)
    spine1, spine1_c = midpoint(11, 12)
    shoulders, shoulders_c = midpoint(5, 6)
    spine3, spine3_c = shoulders, shoulders_c
    neck, neck_c = shoulders, shoulders_c
    values.extend(
        [
            pelvis, xy[12], xy[14], xy[16], xy[11], xy[13], xy[15], spine1,
            spine3, neck, xy[0], xy[5], xy[7], xy[9], xy[6], xy[8], xy[10],
        ]
    )
    scores.extend(
        [
            pelvis_c, conf[12], conf[14], conf[16], conf[11], conf[13], conf[15],
            spine1_c, spine3_c, neck_c, conf[0], conf[5], conf[7], conf[9],
            conf[6], conf[8], conf[10],
        ]
    )
    return np.asarray(values, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def sample_depth(depth: np.ndarray, pixel: np.ndarray) -> tuple[float, float, int]:
    """Return median/MAD/count from a fixed 5x5 metric-depth neighborhood."""
    x, y = int(round(float(pixel[0]))), int(round(float(pixel[1])))
    lo_x, hi_x = max(0, x - DEPTH_NEIGHBORHOOD), min(IMAGE_SIZE, x + DEPTH_NEIGHBORHOOD + 1)
    lo_y, hi_y = max(0, y - DEPTH_NEIGHBORHOOD), min(IMAGE_SIZE, y + DEPTH_NEIGHBORHOOD + 1)
    values = np.asarray(depth[lo_y:hi_y, lo_x:hi_x], dtype=np.float32)
    valid = values[np.isfinite(values) & (values > 0.01)]
    if valid.size == 0:
        return float("nan"), float("nan"), 0
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    return median, mad, int(valid.size)


def placement_yaw(source_path: Path, region: str) -> float:
    manifest = source_path.parents[1] / "candidate_metadata" / "manifest.json"
    if not manifest.is_file():
        return 0.0
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for item in payload.get("placements_data", []):
        if str(item.get("placement_id")) == str(region):
            return float(item.get("yaw_deg", 0.0))
    return 0.0


def template_from_archives(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, float]:
    """Build a fixed root-relative template and pelvis bbox ratio from Train."""
    samples: list[np.ndarray] = []
    pelvis_ratios: list[float] = []
    # The same motion archive appears once per placement/viewpoint context.
    # Equal-record sampling avoids reopening tens of thousands of identical
    # NPZ files while retaining a representative pose for every Train record.
    seen_records: set[str] = set()
    for row in rows:
        record_id = str(row.get("record_id", row["archive_path"]))
        if record_id in seen_records:
            continue
        seen_records.add(record_id)
        with np.load(Path(str(row["archive_path"])), allow_pickle=False) as archive:
            skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
            view = int(row["current_viewpoint_id"])
            if skeleton.shape != (32, 3, 30, 17):
                raise ValueError("unexpected archive skeleton shape")
            pose = skeleton[view, :, 0, :].T
        root = pose[0].copy()
        rel = pose - root
        scale = float(np.median(np.linalg.norm(rel[1:], axis=1)))
        if np.isfinite(scale) and scale > 1.0e-4:
            samples.append(rel / scale)
    if not samples:
        raise RuntimeError("unable to derive Train pose template")
    template = np.median(np.stack(samples, axis=0), axis=0).astype(np.float32)
    return template, 0.55


def _align_template(
    template: np.ndarray, root_world: np.ndarray, observed: np.ndarray, reliable: np.ndarray
) -> tuple[np.ndarray, float]:
    """Rigidly align the Train template to reliable shoulder/hip evidence."""
    finite = reliable & np.isfinite(observed).all(axis=1)
    scale = 1.0
    pairs = ((1, 4), (11, 14))
    ratios: list[float] = []
    for left, right in pairs:
        if finite[left] and finite[right]:
            observed_len = float(np.linalg.norm(observed[right] - observed[left]))
            template_len = float(np.linalg.norm(template[right] - template[left]))
            if observed_len > 1.0e-4 and template_len > 1.0e-4:
                ratios.append(observed_len / template_len)
    if ratios:
        scale = float(np.clip(np.median(ratios), 0.5, 2.0))
    lateral_values: list[np.ndarray] = []
    for left, right in pairs:
        if finite[left] and finite[right]:
            delta = observed[right] - observed[left]
            delta[1] = 0.0
            if np.linalg.norm(delta) > 1.0e-4:
                lateral_values.append(delta / np.linalg.norm(delta))
    target = np.mean(lateral_values, axis=0) if lateral_values else np.asarray([1.0, 0.0, 0.0])
    target = target / max(float(np.linalg.norm(target)), 1.0e-8)
    template_lateral = template[14] - template[11]
    template_lateral[1] = 0.0
    template_lateral = template_lateral / max(float(np.linalg.norm(template_lateral)), 1.0e-8)
    angle = math.atan2(float(np.cross(template_lateral, target)[1]), float(np.dot(template_lateral, target)))
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    completed = root_world[None] + (scale * (template @ rotation.T)).astype(np.float32)
    completed[finite] = observed[finite]
    return completed, float(angle)


def complete_joints(
    observed: np.ndarray, reliable: np.ndarray, root_world: np.ndarray, template: np.ndarray
) -> tuple[np.ndarray, float]:
    if not np.isfinite(root_world).all():
        raise ValueError("root must be finite before template completion")
    values, yaw = _align_template(template, root_world, observed, reliable)
    return values.astype(np.float32), yaw


def depth_simulator(scene_root: Path, scene_id: str) -> tuple[Any, Any]:
    import habitat_sim
    import magnum as mn

    scene_dir = scene_root / scene_id
    glb = next(scene_dir.glob("*.basis.glb"))
    navmesh = next(scene_dir.glob("*.basis.navmesh"))
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(glb)
    backend.enable_physics = True
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "depth_0"
    sensor.sensor_type = habitat_sim.SensorType.DEPTH
    sensor.resolution = [IMAGE_SIZE, IMAGE_SIZE]
    sensor.position = mn.Vector3(0.0, SENSOR_HEIGHT, 0.0)
    sensor.hfov = HFOV_DEG
    sensor.near = 0.01
    sensor.far = 100.0
    agent = habitat_sim.AgentConfiguration()
    agent.sensor_specifications = [sensor]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    if not sim.pathfinder.load_nav_mesh(str(navmesh)):
        sim.close()
        raise RuntimeError(f"failed to load navmesh for {scene_id}")
    human = sim.get_articulated_object_manager().add_articulated_object_from_urdf(
        str(Path(os.environ.get("ACTIVEVIEW_HUMANOID_URDF", "")) or "")
    ) if os.environ.get("ACTIVEVIEW_HUMANOID_URDF") else None
    return sim, human


def render_scene(payload: Mapping[str, Any]) -> str:
    """Render one scene's transient depth and write a compact scene shard."""
    import habitat_sim  # noqa: F401

    data_root = Path(str(payload["data_root"]))
    scene_root = Path(str(payload["scene_root"]))
    output = Path(str(payload["output"]))
    rows = payload["rows"]
    yolo = np.load(Path(str(payload["yolo_path"])), allow_pickle=False)
    sim, _ = depth_simulator(scene_root, str(payload["scene_id"]))
    # Add the humanoid after sensor construction; its geometry is needed only
    # for depth rendering and is never persisted.
    from activeview.core.paths import get_humanoid_urdf_path

    human = sim.get_articulated_object_manager().add_articulated_object_from_urdf(
        str(get_humanoid_urdf_path("male_0"))
    )
    converter = MotionConverter(get_humanoid_urdf_path("male_0"))
    motions = json.loads(
        (data_root / "datasets/reduced12_no_kneel_clean_babel_diversity_v1/raw-val/official_val.json").read_text()
    )
    motion_by_id = {str(item["record_id"]): item for item in motions}
    converted_cache: dict[str, Mapping[str, Any]] = {}
    out_indices: list[int] = []
    roots_world: list[np.ndarray] = []
    roots_camera: list[np.ndarray] = []
    joints_world: list[np.ndarray] = []
    reliable_values: list[np.ndarray] = []
    depths_values: list[np.ndarray] = []
    depth_valid_values: list[np.ndarray] = []
    depth_mad_values: list[np.ndarray] = []
    root_pixels: list[np.ndarray] = []
    person_conf: list[float] = []
    bbox_values: list[np.ndarray] = []
    torso_depths: list[float] = []
    torso_mads: list[float] = []
    for item in rows:
        index = int(item["index"])
        source = Path(str(item["archive_path"]))
        metadata = _load_skeleton_metadata(source)
        record_id = str(item["record_id"])
        converted = converted_cache.get(record_id)
        if converted is None:
            motion = motion_by_id[record_id]
            converted = converter.convert(_load_resampled_motion(motion, 30))
            converted_cache[record_id] = converted
        joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
        roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)
        yaw = placement_yaw(source, str(item["region"]))
        offsets, _ = precompute_grounding_offsets(human, joints, roots, scene_yaw_deg=yaw)
        base = np.asarray(metadata["placement_position"], dtype=np.float32)
        apply_humanoid_pose(human, joints[0], roots[0], base_position=base, scene_yaw_deg=yaw, floor_y=float(base[1]), grounding_offset=float(offsets[0]))
        view = int(item["current_viewpoint_id"])
        positions = np.asarray(metadata["viewpoint_agent_positions"], dtype=np.float32)
        rotations = np.asarray(metadata["viewpoint_rotations_wxyz"], dtype=np.float32)
        _set_agent_state(sim.get_agent(0), positions[view], rotations[view])
        depth = np.asarray(sim.get_sensor_observations([0])[0]["depth_0"], dtype=np.float32)
        pixels = np.asarray(yolo["keypoints_xy"][index], dtype=np.float32)
        conf = np.asarray(yolo["keypoint_conf"][index], dtype=np.float32)
        bbox = np.asarray(yolo["bbox_xyxy"][index], dtype=np.float32)
        obs = np.full((JOINTS, 3), np.nan, dtype=np.float32)
        depths = np.full(JOINTS, np.nan, dtype=np.float32)
        mads = np.full(JOINTS, np.nan, dtype=np.float32)
        valid_mask = np.zeros(JOINTS, dtype=bool)
        for joint in range(JOINTS):
            median, mad, count = sample_depth(depth, pixels[joint])
            depths[joint], mads[joint] = median, mad
            valid_mask[joint] = bool(conf[joint] >= RELIABLE_CONF_THRESHOLD and count >= MIN_VALID_DEPTH_SAMPLES)
            if valid_mask[joint] and np.isfinite(median):
                _, obs[joint] = camera_backproject(pixels[joint], median, positions[view], rotations[view])
        torso_idx = np.asarray([1, 4, 11, 14], dtype=np.int64)
        torso_good = valid_mask[torso_idx] & np.isfinite(depths[torso_idx])
        torso_depth = float(np.median(depths[torso_idx][torso_good])) if np.any(torso_good) else float(np.nanmedian(depth[np.isfinite(depth) & (depth > 0.01)]))
        torso_mad = float(np.median(mads[torso_idx][torso_good])) if np.any(torso_good) else float("nan")
        hips = valid_mask[[1, 4]] & np.isfinite(pixels[[1, 4]]).all(axis=1)
        root_pixel = np.mean(pixels[[1, 4]], axis=0) if np.all(hips) else np.asarray([(bbox[0] + bbox[2]) * 0.5, bbox[1] + 0.55 * max(1.0, bbox[3] - bbox[1])], dtype=np.float32)
        root_depth, _, root_count = sample_depth(depth, root_pixel)
        if not np.isfinite(root_depth) or root_count < MIN_VALID_DEPTH_SAMPLES:
            root_depth = torso_depth
        if np.isfinite(root_depth):
            root_camera, root_world = camera_backproject(root_pixel, root_depth, positions[view], rotations[view])
        else:
            root_camera, root_world = np.full(3, np.nan, dtype=np.float32), np.full(3, np.nan, dtype=np.float32)
        out_indices.append(index)
        roots_world.append(root_world)
        roots_camera.append(root_camera)
        joints_world.append(obs)
        reliable_values.append(valid_mask)
        depths_values.append(depths)
        depth_valid_values.append(np.isfinite(depths) & (depths > 0.0))
        depth_mad_values.append(mads)
        root_pixels.append(root_pixel)
        person_conf.append(float(yolo["person_conf"][index]))
        bbox_values.append(bbox)
        torso_depths.append(torso_depth)
        torso_mads.append(torso_mad)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        indices=np.asarray(out_indices, dtype=np.int64),
        root_world=np.asarray(roots_world, dtype=np.float32),
        root_camera=np.asarray(roots_camera, dtype=np.float32),
        observed_joints_world=np.asarray(joints_world, dtype=np.float32),
        reliable=np.asarray(reliable_values, dtype=bool),
        joint_depth=np.asarray(depths_values, dtype=np.float32),
        joint_depth_valid=np.asarray(depth_valid_values, dtype=bool),
        joint_depth_mad=np.asarray(depth_mad_values, dtype=np.float32),
        root_pixel=np.asarray(root_pixels, dtype=np.float32),
        person_conf=np.asarray(person_conf, dtype=np.float32),
        bbox_xyxy=np.asarray(bbox_values, dtype=np.float32),
        torso_depth=np.asarray(torso_depths, dtype=np.float32),
        torso_depth_mad=np.asarray(torso_mads, dtype=np.float32),
    )
    sim.close()
    return str(output)


def raycast_scene(payload: Mapping[str, Any]) -> str:
    """Compute D1/D2 per-joint environmental visibility for one scene shard."""
    data_root = Path(str(payload["data_root"]))
    scene_root = Path(str(payload["scene_root"]))
    output = Path(str(payload["output"]))
    rows = payload["rows"]
    template = np.asarray(payload["template"], dtype=np.float32)
    # Materialize each compact array once.  Re-indexing an ``NpzFile`` inside
    # the context loop would reopen/decompress the whole member for every
    # row, turning an otherwise linear audit into an unexpectedly long run.
    with np.load(Path(str(payload["depth_path"])), allow_pickle=False) as depth_archive:
        depth = {key: np.asarray(depth_archive[key]) for key in depth_archive.files}
    sim = _scene_sim(scene_root, str(payload["scene_id"]), physics=True)
    from activeview.core.paths import get_humanoid_urdf_path

    human_sim = _scene_sim(scene_root, str(payload["scene_id"]), physics=True)
    human = human_sim.get_articulated_object_manager().add_articulated_object_from_urdf(str(get_humanoid_urdf_path("male_0")))
    converter = MotionConverter(get_humanoid_urdf_path("male_0"))
    motions = json.loads((data_root / "datasets/reduced12_no_kneel_clean_babel_diversity_v1/raw-val/official_val.json").read_text())
    motion_by_id = {str(item["record_id"]): item for item in motions}
    converted_cache: dict[str, Mapping[str, Any]] = {}
    gt_world_cache: dict[tuple[str, str], np.ndarray] = {}
    # Raycast diagnostics only need frame 0.  Avoid constructing all six
    # diagnostic frames for every context; this keeps the privileged D1/D2
    # audit tractable while preserving the exact frame-0 pose definition.
    def frame0_world(converted_motion: Mapping[str, Any], placement: np.ndarray, yaw_deg: float) -> np.ndarray:
        joints = np.asarray(converted_motion["pose_motion"]["joints_array"], dtype=np.float32)
        roots = np.asarray(converted_motion["pose_motion"]["transform_array"], dtype=np.float32)
        offsets, _ = precompute_grounding_offsets(human, joints, roots, scene_yaw_deg=yaw_deg)
        apply_humanoid_pose(
            human,
            joints[0],
            roots[0],
            base_position=placement,
            scene_yaw_deg=yaw_deg,
            floor_y=float(placement[1]),
            grounding_offset=float(offsets[0]),
        )
        names = {str(human.get_link_name(index)): index for index in range(int(human.num_links))}
        values: list[np.ndarray] = [np.asarray(human.translation, dtype=np.float32)]
        for name in H36M17_LINKS[1:]:
            node = human.get_link_scene_node(names[name])
            values.append(np.asarray(node.absolute_translation, dtype=np.float32))
        return np.asarray(values, dtype=np.float32)
    result_indices: list[int] = []
    # Candidate pools can have different legal cardinalities.  Persist a
    # rectangular shard (padded with NaN) so ``np.savez`` never falls back to
    # an object array, which would be unreadable with ``allow_pickle=False``.
    max_candidates = max((len(item["candidate_ids"]) for item in rows), default=0)
    d1_vis = np.full((len(rows), max_candidates, JOINTS), np.nan, dtype=np.float32)
    d2_vis = np.full((len(rows), max_candidates, JOINTS), np.nan, dtype=np.float32)
    d1_joints_out: list[np.ndarray] = []
    d2_joints_out: list[np.ndarray] = []
    d2_yaw_out: list[float] = []
    for item in rows:
        index = int(item["index"])
        source = Path(str(item["archive_path"]))
        metadata = _load_skeleton_metadata(source)
        record_id = str(item["record_id"])
        converted = converted_cache.get(record_id)
        if converted is None:
            converted = converter.convert(_load_resampled_motion(motion_by_id[record_id], 30))
            converted_cache[record_id] = converted
        yaw = placement_yaw(source, str(item["region"]))
        gt_key = (str(source), str(item["region"]))
        gt_world = gt_world_cache.get(gt_key)
        if gt_world is None:
            gt_world = frame0_world(converted, np.asarray(metadata["placement_position"], dtype=np.float32), yaw)
            gt_world_cache[gt_key] = gt_world.copy()
        root = np.asarray(depth["root_world"][index], dtype=np.float32)
        observed = np.asarray(depth["observed_joints_world"][index], dtype=np.float32)
        reliable = np.asarray(depth["reliable"][index], dtype=bool)
        estimated, estimated_yaw = complete_joints(observed, reliable, root, template)
        d1 = gt_world + (root - gt_world[0])
        positions = np.asarray(metadata["viewpoint_agent_positions"], dtype=np.float32)
        candidate_ids = [int(v) for v in item["candidate_ids"]]
        d1_row = np.zeros((len(candidate_ids), JOINTS), dtype=np.float32)
        d2_row = np.zeros((len(candidate_ids), JOINTS), dtype=np.float32)
        for slot, viewpoint in enumerate(candidate_ids):
            camera = positions[viewpoint] + np.asarray([0.0, SENSOR_HEIGHT, 0.0], dtype=np.float32)
            d1_row[slot] = [float(_ray_visible(sim, camera, endpoint)) for endpoint in d1]
            d2_row[slot] = [float(_ray_visible(sim, camera, endpoint)) for endpoint in estimated]
        local_index = len(result_indices)
        result_indices.append(index)
        d1_vis[local_index, : len(candidate_ids)] = d1_row
        d2_vis[local_index, : len(candidate_ids)] = d2_row
        d1_joints_out.append(d1)
        d2_joints_out.append(estimated)
        d2_yaw_out.append(float(estimated_yaw))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, indices=np.asarray(result_indices, dtype=np.int64), d1_visibility=d1_vis, d2_visibility=d2_vis, d1_joints=np.asarray(d1_joints_out, dtype=np.float32), d2_joints=np.asarray(d2_joints_out, dtype=np.float32), d2_yaw=np.asarray(d2_yaw_out, dtype=np.float32))
    human_sim.close()
    sim.close()
    return str(output)
