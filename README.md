# ACTIVEVIEW v11.5

ACTIVEVIEW studies active viewpoint selection for robust indoor human action
recognition, with an emphasis on elderly daily activities under occlusion and
limited robot viewpoints.

## Canonical scientific protocol

- 16 selected BABEL action classes (`lie` and `stumble` included; `fall`
  excluded);
- record split `train/val/test = 589/197/194`;
- RGB-only `256×256` observations, uniformly sampled to 30 frames;
- `male_0` Habitat rendering → YOLO26n-Pose → VideoPose3D → H36M-17
  normalization → frozen ST-GCN;
- 21 HM3D-train scenes, four furniture regions, and 32 candidate viewpoints;
- candidate decisions never use future RGB/depth or future perception outputs.

See [`docs/V11_5_SELECTED16_DATASET_PROTOCOL.md`](docs/V11_5_SELECTED16_DATASET_PROTOCOL.md)
for the protocol and [`docs/V11_5_DEVELOPMENT.md`](docs/V11_5_DEVELOPMENT.md)
for implementation and data history.

## Runtime data

Runtime data lives under `ACTIVEVIEW_DATA_ROOT` (default:
`../../data/ActiveView/`) and is not committed to Git. Habitat scenes and
semantic annotations are read only from `ACTIVEVIEW_HABITAT_DATA_ROOT` (the
configured `robot/DATA/` root); no other disk is scanned.

- ST-GCN data/checkpoint: `datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/` and `checkpoints/`;
- formal offline strategy data: `datasets/offline/hm3d-train/`;
- accepted Stage A/B/C runtime artifacts: `datasets/policy_v11_5/`;
- historical reference-only minival data: `datasets/offline/hm3d-minival/00800-TEEsavR23oF/`;
- baseline strategy results: `results/semantic_region_offline_baselines.json`.

## Main commands

```bash
python -m activeview.scripts.prepare_selected16_manifests
python -m activeview.scripts.generate_selected16_habitat_dataset --split train
python -m activeview.scripts.generate_selected16_habitat_dataset --split val
python -m activeview.scripts.train_selected16_habitat_stgcn
python -m activeview.scripts.generate_semantic_region_candidate_metadata
python -m activeview.scripts.generate_semantic_region_offline_views --workers 4
python -m activeview.scripts.evaluate_semantic_region_offline
```

## Research experiments

Each Stage C-v1 experiment is a small research record under
`experiments/stage_c_v1/`, containing a README, executable configuration and
run script. Large checkpoints, predictions and logs stay under
`ACTIVEVIEW_DATA_ROOT/experiments/`. Development uses Train and Val only;
Test is run only after the final method is explicitly selected.

## Repository layout

- `activeview/`: sole production and research source package;
- `tests/`: scientific unit and integration tests;
- `docs/`: active protocol/development documents and historical archive;
- `.ai/`: concise AI/human research state;
- `experiments/`: lightweight experiment records.

Historical v1–v10 documents are archived under `docs/archive/legacy/` and are
not part of the default agent context.

## Frozen final method

The final frozen method is **WM-E + Multi-Positive Joint Revision + Closed-Loop
H2**. Final Test has been completed on the locked protocol:

- Full: Accuracy `0.684841`, Macro-F1 `0.627749`;
- FrozenStageCv0: Accuracy `0.625381`, Macro-F1 `0.563705`;
- Multi-positive vs FrozenStageCv0: `+5.946` percentage points Accuracy and
  `+6.404` percentage points Macro-F1.

The formal package surface for the final pipeline is under
`activeview/active_view/` (`geometry`, `data`, `world_model`,
`joint_revision`, `rollout`, `baselines`, `evaluation`, `rgb_features`, and
`rgb_cache`). The executable final evaluator is
`activeview/scripts/run_final_test.py`; the frozen EXP055 training entry point
is `activeview/scripts/train_exp055_multi_positive.py`. Runtime checkpoints and
caches stay under `ACTIVEVIEW_DATA_ROOT` and are not committed. Historical
Stage-D research implementations are removed from the active package; Git
history and the experiment result directories remain available for audit.
