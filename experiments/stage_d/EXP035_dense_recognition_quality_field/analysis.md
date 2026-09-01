# EXP035

Frozen 32-view ST-GCN quality field and topology audit.

```json
{
  "experiment_id": "EXP035",
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
    "record_count": 589,
    "view_count": 18848,
    "smoke_max_abs_logp_error": 0.0027561187744140625,
    "source_sha256": "35bd127c8df20d61050d83a67b8291fffebf42aec8f1d77e3982c79afb008181"
  },
  "val": {
    "record_count": 197,
    "view_count": 6304,
    "smoke_max_abs_logp_error": 0.00021503865718841553,
    "source_sha256": "7d29f20026be72a497cf9a73eba3840343ab6bce8e57fb01a2ffe1d36973daac"
  },
  "field_structure_audit": {
    "ce_mean": 4.906191349029541,
    "ce_std": 5.746397972106934,
    "within_record_range_mean": 14.973085403442383,
    "best_worst_gap": 14.973085403442383,
    "neighbor_azimuth_abs_diff": 4.097837924957275,
    "neighbor_radius_abs_diff": 2.714878559112549,
    "opposite_view_abs_diff": 4.344564914703369,
    "azimuth_neighbor_correlation": {
      "pearson": 0.28370576577568435,
      "spearman": 0.31997009722158043
    },
    "radius_neighbor_correlation": {
      "pearson": 0.4411623781195674,
      "spearman": 0.42682751580626477
    }
  },
  "privileged_evaluation_only": true,
  "smoke_status": "PASS",
  "smoke_tolerance": 0.01,
  "provenance": {
    "source_commit": "46e0da21e9ced1004be50b9bb509d02f30450bc0",
    "stage_b_summary_sha256": "3cb52e01e1a36de6ec580c075d39345d01737820fe7313a4e6fe8312136b295f"
  }
}
```
