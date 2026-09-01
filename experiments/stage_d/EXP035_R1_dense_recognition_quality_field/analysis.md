# EXP035-R1

R1 context-safe dense field; old record_id-only results are invalid.

```json
{
  "experiment_id": "EXP035-R1",
  "status": "COMPLETED",
  "split": [
    "train",
    "val"
  ],
  "test_used": false,
  "training_performed": false,
  "perception_regenerated": false,
  "habitat_rendering_performed": false,
  "stgcn_retrained": false,
  "train": {
    "context_count": 29133,
    "view_count": 932256,
    "skeleton_archives": 29133
  },
  "val": {
    "context_count": 9742,
    "view_count": 311744,
    "skeleton_archives": 9742
  },
  "field_structure_audit": {
    "ce_mean": 6.196847438812256,
    "ce_std": 6.171261310577393,
    "within_record_range_mean": 15.626724243164062,
    "neighbor_azimuth_abs_diff": 4.62072229385376,
    "neighbor_radius_abs_diff": 2.553574562072754,
    "opposite_view_abs_diff": 3.743682622909546,
    "azimuth_neighbor_correlation": {
      "pearson": 0.3880786912097157,
      "spearman": 0.3984846171308308
    },
    "radius_neighbor_correlation": {
      "pearson": 0.6830753821460746,
      "spearman": 0.6913965981845926
    }
  },
  "stage_b_reproduction_audit": {
    "matched_episode_count": 29133,
    "matched_candidate_count": 217475,
    "identity_mismatch_count": 0,
    "scene_mismatch_count": 0,
    "region_mismatch_count": 0,
    "source_path_mismatch_count": 0,
    "candidate_missing_count": 0,
    "max_abs_logp_error": 0.007377147674560547,
    "mean_abs_logp_error": 0.0007388058380255981,
    "p99_abs_logp_error": 0.0043125152587890625,
    "status": "PASS",
    "val": {
      "matched_episode_count": 9742,
      "matched_candidate_count": 72784,
      "identity_mismatch_count": 0,
      "scene_mismatch_count": 0,
      "region_mismatch_count": 0,
      "source_path_mismatch_count": 0,
      "candidate_missing_count": 0,
      "max_abs_logp_error": 0.007377147674560547,
      "mean_abs_logp_error": 0.0007152688989941897,
      "p99_abs_logp_error": 0.00433349609375,
      "status": "PASS"
    }
  },
  "identity_audit": {
    "train": {
      "split": "train",
      "episode_rows": 41819,
      "unique_record_id": 589,
      "unique_context_key": 41819,
      "record_ids_with_multiple_contexts": 589,
      "max_contexts_per_record_id": 71,
      "collision_examples": [
        {
          "record_id": "babel_val_00047_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00047_000.npz"
        },
        {
          "record_id": "babel_val_00047_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00047_000.npz"
        },
        {
          "record_id": "babel_val_00047_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00047_000.npz"
        },
        {
          "record_id": "babel_val_00127_011",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00127_011.npz"
        },
        {
          "record_id": "babel_val_00127_011",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00127_011.npz"
        },
        {
          "record_id": "babel_val_00127_011",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00127_011.npz"
        },
        {
          "record_id": "babel_val_00163_007",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00163_007.npz"
        },
        {
          "record_id": "babel_val_00163_007",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00163_007.npz"
        },
        {
          "record_id": "babel_val_00163_007",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00163_007.npz"
        },
        {
          "record_id": "babel_val_00183_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00183_000.npz"
        },
        {
          "record_id": "babel_val_00183_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00183_000.npz"
        },
        {
          "record_id": "babel_val_00183_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00183_000.npz"
        },
        {
          "record_id": "babel_val_00214_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00214_004.npz"
        },
        {
          "record_id": "babel_val_00214_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00214_004.npz"
        },
        {
          "record_id": "babel_val_00214_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00214_004.npz"
        },
        {
          "record_id": "babel_val_00215_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00215_000.npz"
        },
        {
          "record_id": "babel_val_00215_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00215_000.npz"
        },
        {
          "record_id": "babel_val_00215_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00215_000.npz"
        },
        {
          "record_id": "babel_val_00221_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00221_004.npz"
        },
        {
          "record_id": "babel_val_00221_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00221_004.npz"
        },
        {
          "record_id": "babel_val_00221_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00221_004.npz"
        },
        {
          "record_id": "babel_val_00255_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00255_004.npz"
        },
        {
          "record_id": "babel_val_00255_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00255_004.npz"
        },
        {
          "record_id": "babel_val_00255_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00255_004.npz"
        },
        {
          "record_id": "babel_val_00268_002",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00268_002.npz"
        },
        {
          "record_id": "babel_val_00268_002",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00268_002.npz"
        },
        {
          "record_id": "babel_val_00268_002",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00268_002.npz"
        },
        {
          "record_id": "babel_val_00272_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00272_000.npz"
        },
        {
          "record_id": "babel_val_00272_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00272_000.npz"
        },
        {
          "record_id": "babel_val_00272_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00272_000.npz"
        },
        {
          "record_id": "babel_val_00409_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00409_004.npz"
        },
        {
          "record_id": "babel_val_00409_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00409_004.npz"
        },
        {
          "record_id": "babel_val_00409_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00409_004.npz"
        },
        {
          "record_id": "babel_val_00415_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00415_004.npz"
        },
        {
          "record_id": "babel_val_00415_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00415_004.npz"
        },
        {
          "record_id": "babel_val_00415_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00415_004.npz"
        },
        {
          "record_id": "babel_val_00442_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00442_000.npz"
        },
        {
          "record_id": "babel_val_00442_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00442_000.npz"
        },
        {
          "record_id": "babel_val_00442_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00442_000.npz"
        },
        {
          "record_id": "babel_val_00443_005",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00443_005.npz"
        },
        {
          "record_id": "babel_val_00443_005",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00443_005.npz"
        },
        {
          "record_id": "babel_val_00443_005",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00443_005.npz"
        },
        {
          "record_id": "babel_val_00447_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00447_004.npz"
        },
        {
          "record_id": "babel_val_00447_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00447_004.npz"
        },
        {
          "record_id": "babel_val_00447_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00447_004.npz"
        },
        {
          "record_id": "babel_val_00456_003",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00456_003.npz"
        },
        {
          "record_id": "babel_val_00456_003",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00456_003.npz"
        },
        {
          "record_id": "babel_val_00456_003",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00456_003.npz"
        },
        {
          "record_id": "babel_val_00501_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00501_000.npz"
        },
        {
          "record_id": "babel_val_00501_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00501_000.npz"
        },
        {
          "record_id": "babel_val_00501_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00501_000.npz"
        },
        {
          "record_id": "babel_val_00501_003",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00501_003.npz"
        },
        {
          "record_id": "babel_val_00501_003",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00501_003.npz"
        },
        {
          "record_id": "babel_val_00501_003",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00501_003.npz"
        },
        {
          "record_id": "babel_val_00504_074",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00504_074.npz"
        },
        {
          "record_id": "babel_val_00504_074",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00504_074.npz"
        },
        {
          "record_id": "babel_val_00504_074",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00504_074.npz"
        },
        {
          "record_id": "babel_val_00524_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00524_000.npz"
        },
        {
          "record_id": "babel_val_00524_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00524_000.npz"
        },
        {
          "record_id": "babel_val_00524_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00524_000.npz"
        }
      ]
    },
    "val": {
      "split": "val",
      "episode_rows": 13987,
      "unique_record_id": 197,
      "unique_context_key": 13987,
      "record_ids_with_multiple_contexts": 197,
      "max_contexts_per_record_id": 71,
      "collision_examples": [
        {
          "record_id": "babel_val_00214_006",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00214_006.npz"
        },
        {
          "record_id": "babel_val_00214_006",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00214_006.npz"
        },
        {
          "record_id": "babel_val_00214_006",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00214_006.npz"
        },
        {
          "record_id": "babel_val_00317_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00317_000.npz"
        },
        {
          "record_id": "babel_val_00317_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00317_000.npz"
        },
        {
          "record_id": "babel_val_00317_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00317_000.npz"
        },
        {
          "record_id": "babel_val_00380_006",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00380_006.npz"
        },
        {
          "record_id": "babel_val_00380_006",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00380_006.npz"
        },
        {
          "record_id": "babel_val_00380_006",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00380_006.npz"
        },
        {
          "record_id": "babel_val_00537_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00537_000.npz"
        },
        {
          "record_id": "babel_val_00537_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00537_000.npz"
        },
        {
          "record_id": "babel_val_00537_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00537_000.npz"
        },
        {
          "record_id": "babel_val_00612_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00612_000.npz"
        },
        {
          "record_id": "babel_val_00612_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00612_000.npz"
        },
        {
          "record_id": "babel_val_00612_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00612_000.npz"
        },
        {
          "record_id": "babel_val_00674_025",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00674_025.npz"
        },
        {
          "record_id": "babel_val_00674_025",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00674_025.npz"
        },
        {
          "record_id": "babel_val_00674_025",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00674_025.npz"
        },
        {
          "record_id": "babel_val_00674_026",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00674_026.npz"
        },
        {
          "record_id": "babel_val_00674_026",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00674_026.npz"
        },
        {
          "record_id": "babel_val_00674_026",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00674_026.npz"
        },
        {
          "record_id": "babel_val_00674_027",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00674_027.npz"
        },
        {
          "record_id": "babel_val_00674_027",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00674_027.npz"
        },
        {
          "record_id": "babel_val_00674_027",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00674_027.npz"
        },
        {
          "record_id": "babel_val_00743_009",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00743_009.npz"
        },
        {
          "record_id": "babel_val_00743_009",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00743_009.npz"
        },
        {
          "record_id": "babel_val_00743_009",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00743_009.npz"
        },
        {
          "record_id": "babel_val_00799_006",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00799_006.npz"
        },
        {
          "record_id": "babel_val_00799_006",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00799_006.npz"
        },
        {
          "record_id": "babel_val_00799_006",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00799_006.npz"
        },
        {
          "record_id": "babel_val_00799_007",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00799_007.npz"
        },
        {
          "record_id": "babel_val_00799_007",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00799_007.npz"
        },
        {
          "record_id": "babel_val_00799_007",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00799_007.npz"
        },
        {
          "record_id": "babel_val_00866_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00866_000.npz"
        },
        {
          "record_id": "babel_val_00866_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00866_000.npz"
        },
        {
          "record_id": "babel_val_00866_000",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00866_000.npz"
        },
        {
          "record_id": "babel_val_00985_007",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_00985_007.npz"
        },
        {
          "record_id": "babel_val_00985_007",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_00985_007.npz"
        },
        {
          "record_id": "babel_val_00985_007",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_00985_007.npz"
        },
        {
          "record_id": "babel_val_01067_008",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_01067_008.npz"
        },
        {
          "record_id": "babel_val_01067_008",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_01067_008.npz"
        },
        {
          "record_id": "babel_val_01067_008",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_01067_008.npz"
        },
        {
          "record_id": "babel_val_01205_001",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_01205_001.npz"
        },
        {
          "record_id": "babel_val_01205_001",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_01205_001.npz"
        },
        {
          "record_id": "babel_val_01205_001",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_01205_001.npz"
        },
        {
          "record_id": "babel_val_01257_001",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_01257_001.npz"
        },
        {
          "record_id": "babel_val_01257_001",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_01257_001.npz"
        },
        {
          "record_id": "babel_val_01257_001",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_01257_001.npz"
        },
        {
          "record_id": "babel_val_01295_001",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_01295_001.npz"
        },
        {
          "record_id": "babel_val_01295_001",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_01295_001.npz"
        },
        {
          "record_id": "babel_val_01295_001",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_01295_001.npz"
        },
        {
          "record_id": "babel_val_01321_005",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_01321_005.npz"
        },
        {
          "record_id": "babel_val_01321_005",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_01321_005.npz"
        },
        {
          "record_id": "babel_val_01321_005",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_01321_005.npz"
        },
        {
          "record_id": "babel_val_01344_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_01344_004.npz"
        },
        {
          "record_id": "babel_val_01344_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_01344_004.npz"
        },
        {
          "record_id": "babel_val_01344_004",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_01344_004.npz"
        },
        {
          "record_id": "babel_val_01403_005",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "bedroom",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/bedroom/babel_val_01403_005.npz"
        },
        {
          "record_id": "babel_val_01403_005",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "dining_area",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/dining_area/babel_val_01403_005.npz"
        },
        {
          "record_id": "babel_val_01403_005",
          "scene_id": "00006-HkseAnWCgqk",
          "region": "kitchen",
          "source_path": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/offline/hm3d-train/00006-HkseAnWCgqk/kitchen/babel_val_01403_005.npz"
        }
      ]
    }
  },
  "split_audit": {
    "episode_id_overlap": 0,
    "context_key_overlap": 0,
    "record_id_overlap_allowed_by_protocol": 0,
    "train_episode_count": 29133,
    "val_episode_count": 9742
  },
  "old_experiment_status": "INVALID_PENDING_R1",
  "provenance": {
    "source_commit": "78e2fc414dc03ec8ec9b782bbf43a4bfe07df3c8",
    "stage_b_summary_sha256": "3cb52e01e1a36de6ec580c075d39345d01737820fe7313a4e6fe8312136b295f"
  }
}
```
