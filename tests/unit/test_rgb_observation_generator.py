from pathlib import Path

import numpy as np

from activeview.scripts.data.generate_hm3d_train_rgb_observations import (
    FRAME_INDEX,
    IMAGE_SIZE,
    VERSION,
    VIEWS_PER_RECORD,
    WORKERS,
    _load_skeleton_metadata,
    _output_path,
    _sha256,
    _validate_rgb_file,
)


def _write_skeleton(path: Path) -> dict[str, object]:
    positions = np.arange(VIEWS_PER_RECORD * 3, dtype=np.float32).reshape(VIEWS_PER_RECORD, 3)
    rotations = np.zeros((VIEWS_PER_RECORD, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    np.savez(
        path,
        viewpoint_ids=np.arange(VIEWS_PER_RECORD, dtype=np.int32),
        scene_id=np.asarray("scene"),
        region=np.asarray("bedroom"),
        placement_id=np.asarray("placement"),
        placement_position=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        viewpoint_agent_positions=positions,
        viewpoint_rotations_wxyz=rotations,
    )
    return {
        "viewpoint_ids": np.arange(VIEWS_PER_RECORD, dtype=np.int32),
        "viewpoint_agent_positions": positions,
        "viewpoint_rotations_wxyz": rotations,
    }


def _write_rgb(path: Path, source_relative: str, source_hash: str, meta: dict[str, object]) -> None:
    np.savez_compressed(
        path,
        rgb_observation_version=np.asarray(VERSION),
        rgb=np.ones((VIEWS_PER_RECORD, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        viewpoint_ids=meta["viewpoint_ids"],
        scene_id=np.asarray("scene"),
        region=np.asarray("bedroom"),
        record_id=np.asarray("record"),
        placement_id=np.asarray("placement"),
        frame_index=np.asarray(FRAME_INDEX, dtype=np.int32),
        image_size=np.asarray(IMAGE_SIZE, dtype=np.int32),
        source_skeleton_relative_path=np.asarray(source_relative),
        source_skeleton_sha256=np.asarray(source_hash),
        viewpoint_agent_positions=meta["viewpoint_agent_positions"],
        viewpoint_rotations_wxyz=meta["viewpoint_rotations_wxyz"],
    )


def test_rgb_record_validation_accepts_canonical_schema(tmp_path: Path) -> None:
    source = tmp_path / "scene" / "bedroom" / "record.npz"
    source.parent.mkdir(parents=True)
    _write_skeleton(source)
    source_meta = _load_skeleton_metadata(source)
    output = tmp_path / "rgb.npz"
    relative = "scene/bedroom/record.npz"
    _write_rgb(output, relative, _sha256(source), {**source_meta})

    assert _validate_rgb_file(output, Path(relative), source_meta, _sha256(source))


def test_rgb_record_validation_rejects_source_hash_mutation(tmp_path: Path) -> None:
    source = tmp_path / "record.npz"
    _write_skeleton(source)
    source_meta = _load_skeleton_metadata(source)
    output = tmp_path / "rgb.npz"
    _write_rgb(output, "record.npz", "0" * 64, {**source_meta})

    assert not _validate_rgb_file(output, Path("record.npz"), source_meta, _sha256(source))


def test_output_path_preserves_skeleton_relative_path(tmp_path: Path) -> None:
    source_root = tmp_path / "skeleton"
    source_path = source_root / "scene" / "region" / "record.npz"
    output_root = tmp_path / "rgb"

    assert _output_path(output_root, source_root, source_path) == output_root / "scene/region/record.npz"


def test_full_generation_uses_authorized_worker_count() -> None:
    assert WORKERS == 16
