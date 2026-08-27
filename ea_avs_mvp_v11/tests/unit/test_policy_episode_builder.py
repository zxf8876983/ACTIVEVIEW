import numpy as np

from ea_avs_mvp_v11.active_view.policy_episode_builder import (
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
