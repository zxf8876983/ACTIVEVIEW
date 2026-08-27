import json

import numpy as np

from ea_avs_mvp_v11.active_view.policy_episode_builder import (
    audit_episode_files,
    build_dynamic_candidate_pool,
    build_navigation_geometry_pool,
    choose_current_view,
    materialize_candidate_pool,
    stable_episode_seed,
)


class _Pathfinder:
    pass


def _cost(_pathfinder, start, end):
    distance = float(np.linalg.norm(np.asarray(start) - np.asarray(end)))
    return distance if distance > 0 else 0.0


def test_current_view_seed_is_reproducible():
    seed = stable_episode_seed(42, "record", "scene", "bedroom")
    assert choose_current_view([0, 1, 2, 3], seed) == choose_current_view([0, 1, 2, 3], seed)


def test_current_is_excluded_and_pool_uses_dynamic_cost():
    views = {
        i: {
            "position": np.array([float(i), 0.0, 0.0]),
            "snapped_position": np.array([float(i), 0.0, 0.0]),
            "agent_position": np.array([float(i), 0.0, 0.0]),
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "azimuth_deg": float(i * 45),
            "skeleton_source_path": f"pack_{i}.npz",
            "pose_confidence_available": True,
        }
        for i in range(4)
    }
    pool = build_dynamic_candidate_pool(
        current_viewpoint_id=1, views=views, valid_skeleton_ids=[0, 1, 2, 3],
        pathfinder=_Pathfinder(), path_cost_fn=_cost,
    )
    assert [item["viewpoint_id"] for item in pool] == [0, 2, 3]
    assert all(item["viewpoint_id"] != 1 for item in pool)


def test_geometry_cache_is_independent_of_record_validity_and_paths(tmp_path):
    views = {
        i: {
            "position": np.array([float(i), 0.0, 0.0]),
            "snapped_position": np.array([float(i), 0.0, 0.0]),
            "agent_position": np.array([float(i), 0.0, 0.0]),
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "azimuth_deg": float(i * 45),
        }
        for i in range(4)
    }
    geometry = build_navigation_geometry_pool(
        current_viewpoint_id=1, views=views,
        pathfinder=_Pathfinder(), path_cost_fn=_cost,
    )
    assert [item["viewpoint_id"] for item in geometry] == [0, 2, 3]
    assert all("skeleton_source_path" not in item for item in geometry)

    archive_a = tmp_path / "record_a.npz"
    archive_b = tmp_path / "record_b.npz"
    record_a = materialize_candidate_pool(
        geometry, valid_skeleton_ids=[0, 1, 2],
        archive_path=archive_a, pose_confidence_available=True,
    )
    record_b = materialize_candidate_pool(
        geometry, valid_skeleton_ids=[1, 3],
        archive_path=archive_b, pose_confidence_available=False,
    )
    assert [item["viewpoint_id"] for item in record_a] == [0, 2]
    assert [item["viewpoint_id"] for item in record_b] == [3]
    assert {item["skeleton_source_path"] for item in record_a} == {str(archive_a.resolve())}
    assert {item["skeleton_source_path"] for item in record_b} == {str(archive_b.resolve())}
    assert all("valid_skeleton_ids" not in item for item in geometry)


def test_audit_episode_files_derives_integrity_flags_from_serialized_records(tmp_path):
    archive = tmp_path / "record.npz"
    archive.touch()
    valid = {
        "episode_id": "valid",
        "policy_split": "train",
        "record_id": "record_a",
        "current_view": {"viewpoint_id": 0, "skeleton_source_path": str(archive)},
        "candidate_pool": [{
            "viewpoint_id": 1,
            "position": [0.0, 0.0, 0.0],
            "snapped_position": [0.0, 0.0, 0.0],
            "relative_position": [1.0, 0.0, 0.0],
            "euclidean_distance_m": 1.0,
            "geodesic_distance_m": 1.0,
            "relative_azimuth_deg": 0.0,
            "skeleton_source_path": str(archive),
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        }],
        "candidate_count": 1,
    }
    invalid = {
        **valid,
        "episode_id": "invalid",
        "policy_split": "val",
        "record_id": "record_a",
        "current_view": {"viewpoint_id": 1, "skeleton_source_path": str(archive)},
        "candidate_pool": [{
            **valid["candidate_pool"][0],
            "viewpoint_id": 1,
            "geodesic_distance_m": float("nan"),
            "skeleton_source_path": str(tmp_path / "other.npz"),
        }],
    }
    train_path = tmp_path / "train_episodes.jsonl"
    val_path = tmp_path / "val_episodes.jsonl"
    train_path.write_text(json.dumps(valid) + "\n", encoding="utf-8")
    val_path.write_text(json.dumps(invalid, allow_nan=True) + "\n", encoding="utf-8")
    audit = audit_episode_files(
        {"train": train_path, "val": val_path},
        expected_record_splits={"record_a": "train"},
    )
    assert audit["counts"]["episodes"] == 2
    assert audit["counts"]["split_mismatch"] == 1
    assert audit["counts"]["same_record_split_violations"] == 1
    assert audit["counts"]["current_in_pool_violations"] == 1
    assert audit["counts"]["nonfinite_candidate_costs"] == 1
    assert audit["counts"]["candidate_path_mismatch"] == 1
    assert not audit["integrity_checks"]["current_not_in_candidate_pool"]
    assert not audit["integrity_checks"]["same_record_same_split_across_scenes"]


def test_audit_validates_cached_skeleton_and_nested_leakage_schema(tmp_path):
    archive = tmp_path / "record.npz"
    np.savez(
        archive,
        skeleton=np.zeros((32, 3, 30, 17), dtype=np.float32),
        viewpoint_ids=np.arange(32, dtype=np.int64),
        viewpoint_positions=np.zeros((32, 3), dtype=np.float32),
        viewpoint_snapped_positions=np.zeros((32, 3), dtype=np.float32),
        viewpoint_agent_positions=np.zeros((32, 3), dtype=np.float32),
        viewpoint_rotations_wxyz=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (32, 1)),
    )
    episode = {
        "episode_id": "episode",
        "policy_split": "train",
        "record_id": "record",
        "scene_id": "scene_a",
        "current_view": {
            "viewpoint_id": 0,
            "position": [0.0, 0.0, 0.0],
            "snapped_position": [0.0, 0.0, 0.0],
            "agent_position": [0.0, 0.0, 0.0],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "skeleton_source_path": str(archive),
        },
        "candidate_pool": [{
            "viewpoint_id": 1,
            "position": [1.0, 0.0, 0.0],
            "snapped_position": [1.0, 0.0, 0.0],
            "relative_position": [1.0, 0.0, 0.0],
            "euclidean_distance_m": 1.0,
            "geodesic_distance_m": 1.0,
            "relative_azimuth_deg": 0.0,
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "skeleton_source_path": str(archive),
        }],
        "candidate_count": 1,
        "nested_metadata": {"future_rgb": "must be rejected"},
    }
    path = tmp_path / "train_episodes.jsonl"
    path.write_text(json.dumps(episode) + "\n", encoding="utf-8")
    audit = audit_episode_files(
        {"train": path},
        expected_record_splits={"record": "train"},
        validate_cached_skeletons=True,
    )
    assert audit["counts"]["cached_skeleton_file_errors"] == 0
    assert audit["counts"]["current_view_data_violations"] == 0
    assert audit["counts"]["candidate_skeleton_data_violations"] == 0
    assert audit["counts"]["forbidden_information_violations"] == 1
    assert audit["integrity_checks"]["all_cached_skeletons_complete"]
    assert audit["integrity_checks"]["all_current_view_data_valid"]
    assert audit["integrity_checks"]["all_candidate_skeleton_data_valid"]
    assert not audit["integrity_checks"]["no_forbidden_future_perception_fields"]


def _write_archive(path, *, nonfinite_viewpoint=None):
    skeleton = np.zeros((32, 3, 30, 17), dtype=np.float32)
    if nonfinite_viewpoint is not None:
        skeleton[int(nonfinite_viewpoint), 0, 0, 0] = np.nan
    np.savez(
        path,
        skeleton=skeleton,
        viewpoint_ids=np.arange(32, dtype=np.int64),
        viewpoint_positions=np.stack([np.array([float(i), 0.0, 0.0], dtype=np.float32) for i in range(32)]),
        viewpoint_snapped_positions=np.stack([np.array([float(i), 0.0, 0.0], dtype=np.float32) for i in range(32)]),
        viewpoint_agent_positions=np.stack([np.array([float(i), 0.0, 0.0], dtype=np.float32) for i in range(32)]),
        viewpoint_rotations_wxyz=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (32, 1)),
    )


def _episode_for_archive(archive, *, episode_id="episode", current_id=0, candidate_id=1):
    def view(viewpoint_id):
        return {
            "viewpoint_id": viewpoint_id,
            "position": [float(viewpoint_id), 0.0, 0.0],
            "snapped_position": [float(viewpoint_id), 0.0, 0.0],
            "agent_position": [float(viewpoint_id), 0.0, 0.0],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "skeleton_source_path": str(archive),
        }
    candidate = view(candidate_id)
    candidate.update({
        "relative_position": [float(candidate_id - current_id), 0.0, 0.0],
        "euclidean_distance_m": 1.0,
        "geodesic_distance_m": 1.0,
        "relative_azimuth_deg": 0.0,
    })
    return {
        "episode_id": episode_id,
        "policy_split": "train",
        "record_id": "record",
        "scene_id": "scene",
        "region": "bedroom",
        "current_view": view(current_id),
        "candidate_pool": [candidate],
        "candidate_count": 1,
    }


def test_partial_nonfinite_archive_is_allowed_when_unused_view_is_filtered(tmp_path):
    archive = tmp_path / "partial.npz"
    _write_archive(archive, nonfinite_viewpoint=31)
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(_episode_for_archive(archive)) + "\n", encoding="utf-8")
    audit = audit_episode_files({"train": path}, validate_cached_skeletons=True)
    assert audit["counts"]["nonfinite_cached_skeleton_viewpoints"] == 1
    assert audit["counts"]["current_view_data_violations"] == 0
    assert audit["counts"]["candidate_skeleton_data_violations"] == 0
    assert audit["integrity_checks"]["all_cached_skeletons_complete"]


def test_audit_checks_episode_uniqueness_and_npz_geometry(tmp_path):
    archive = tmp_path / "geometry.npz"
    _write_archive(archive)
    first = _episode_for_archive(archive, episode_id="duplicate")
    second = _episode_for_archive(archive, episode_id="duplicate")
    second["current_view"]["position"] = [99.0, 0.0, 0.0]
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in (first, second)) + "\n", encoding="utf-8")
    audit = audit_episode_files({"train": path}, validate_cached_skeletons=True)
    assert audit["counts"]["duplicate_episode_keys"] == 1
    assert audit["counts"]["duplicate_episode_ids"] == 1
    assert audit["counts"]["npz_geometry_mismatches"] == 1
    assert not audit["integrity_checks"]["unique_record_scene_region"]
    assert not audit["integrity_checks"]["unique_episode_ids"]
    assert not audit["integrity_checks"]["episode_geometry_matches_npz"]
