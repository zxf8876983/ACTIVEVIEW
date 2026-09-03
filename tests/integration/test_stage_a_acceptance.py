"""Acceptance checks for the serialized v11.5 Stage A Episode manifests.

The JSONL audit is intentionally independent of Habitat. The Pathfinder test
is a small real-HM3D integration check and is skipped when Habitat assets are
not installed in the active environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from activeview.data.generation.policy_episodes import (
    REGIONS,
    audit_episode_coverage,
    audit_episode_files,
    audit_scene_coverage,
)
from activeview.core.paths import get_data_root, get_habitat_data_root


SPLITS = ("train", "val", "test")
LEGACY_SCENE_IDS = {"00800-TEEsavR23oF"}


def _dataset_context() -> tuple[Path, dict[str, Path], dict[str, str]]:
    root = get_data_root() / "datasets/policy_v11_5"
    summary_path = root / "stage_a_summary.json"
    if not summary_path.exists():
        pytest.skip("Stage A output is not available")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split_summary = json.loads((root / "splits" / "summary.json").read_text(encoding="utf-8"))
    if summary.get("policy_split", {}).get("counts") != split_summary.get("split_counts"):
        pytest.fail("Stage A Episodes are stale relative to the current policy split files")
    files = {split: Path(summary["episode_files"][split]) for split in SPLITS}
    expected: dict[str, str] = {}
    for split in SPLITS:
        split_path = root / "splits" / f"{split}.json"
        records = json.loads(split_path.read_text(encoding="utf-8"))
        expected.update({str(item["record_id"]): split for item in records})
    return root, files, expected


def test_final_jsonl_integrity_audit_passes():
    root, files, expected = _dataset_context()
    audit = audit_episode_files(files, expected_record_splits=expected)
    summary = json.loads((root / "stage_a_summary.json").read_text(encoding="utf-8"))
    records = [
        item
        for split in SPLITS
        for item in json.loads((root / "splits" / f"{split}.json").read_text(encoding="utf-8"))
    ]
    coverage = audit_episode_coverage(
        files,
        Path(summary["exclusions_file"]),
        policy_records=records,
        complete_scene_ids=summary["scene_ids_used"],
        regions=summary.get("regions", REGIONS),
    )
    assert audit["counts"]["episodes"] > 0
    assert all(
        value == 0
        for key, value in audit["counts"].items()
        if key not in {"episodes", "split_overlap", "nonfinite_cached_skeleton_viewpoints"}
    )
    assert audit["counts"]["split_overlap"] is False
    assert all(
        value
        for key, value in audit["integrity_checks"].items()
        if key != "split_overlap"
    )
    assert coverage["counts"]["missing_tuple_count"] == 0
    assert coverage["counts"]["duplicate_accounted_tuple_count"] == 0
    assert coverage["counts"]["episode_and_exclusion_overlap"] == 0
    assert coverage["integrity_checks"]["complete_tuple_coverage"]
    # The persisted summary predates the current protocol and still lists the
    # legacy minival scene.  It is intentionally excluded from the 21-scene
    # HM3D-train evaluation set; audit only the canonical target set here.
    target_scene_ids = [
        scene_id
        for scene_id in summary["target_scene_ids"]
        if scene_id not in LEGACY_SCENE_IDS
    ]
    scene_audit = audit_scene_coverage(target_scene_ids, summary["scene_ids_used"])
    assert scene_audit["integrity_checks"]["all_target_scenes_used"]


def test_cached_skeleton_archives_are_complete_when_full_audit_requested():
    """Run the archive-level check explicitly for a full Stage A acceptance."""
    if not bool(int(os.environ.get("ACTIVEVIEW_STAGE_A_FULL_AUDIT", "0"))):
        pytest.skip("Set ACTIVEVIEW_STAGE_A_FULL_AUDIT=1 for the full archive audit")
    _root, files, expected = _dataset_context()
    audit = audit_episode_files(
        files,
        expected_record_splits=expected,
        validate_cached_skeletons=True,
    )
    assert audit["counts"]["cached_skeleton_file_errors"] == 0
    assert audit["counts"]["cached_skeleton_shape_violations"] == 0
    assert audit["counts"]["current_view_data_violations"] == 0
    assert audit["counts"]["candidate_skeleton_data_violations"] == 0
    assert audit["counts"]["duplicate_episode_keys"] == 0
    assert audit["counts"]["duplicate_episode_ids"] == 0
    assert audit["counts"]["npz_geometry_mismatches"] == 0
    assert audit["integrity_checks"]["all_cached_skeletons_complete"]
    assert audit["integrity_checks"]["episode_geometry_matches_npz"]


def test_real_habitat_shortest_path_for_final_episode():
    habitat_sim = pytest.importorskip("habitat_sim")
    _root, files, _expected = _dataset_context()
    first_episode = None
    for path in files.values():
        with path.open(encoding="utf-8") as handle:
            first_episode = json.loads(next(handle))
        break
    assert first_episode is not None
    scene_id = str(first_episode["scene_id"])
    habitat_root = get_habitat_data_root()
    scene_root = None
    for scene_set in ("hm3d-minival", "hm3d-train"):
        candidate = habitat_root / scene_set / scene_id
        if candidate.exists():
            scene_root = habitat_root / scene_set
            break
    if scene_root is None:
        pytest.skip(f"Habitat scene assets are unavailable for {scene_id}")

    from activeview.scripts.data.build_policy_episodes import _make_sim
    from activeview.scripts.eval.evaluate_hm3d_train_dynamic_reachability import _path_cost

    sim = _make_sim(scene_root, scene_id)
    try:
        current = first_episode["current_view"]
        candidate = first_episode["candidate_pool"][0]
        cost = _path_cost(
            sim.pathfinder,
            current["agent_position"],
            candidate["snapped_position"],
        )
        assert cost is not None and cost >= 0.0
        assert cost == pytest.approx(float(candidate["geodesic_distance_m"]), rel=1e-5, abs=1e-5)
    finally:
        sim.close()
