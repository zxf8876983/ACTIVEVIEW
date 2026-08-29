# ACTIVEVIEW Scientific State

Updated: 2026-08-30

## Research goal

Improve indoor elderly action recognition by actively selecting robot viewpoints
that reduce perception uncertainty under occlusion and self-occlusion.

## Canonical pipeline

```text
AMASS/BABEL → male_0 Habitat RGB → YOLO26n-Pose → VideoPose3D
→ H36M-17 camera/gravity conversion and normalization
→ frozen ST-GCN → Stage B utility → Stage C viewpoint policy
```

The ST-GCN receives estimated skeletons only. RGB is 256×256 and sequences are
uniformly sampled to 30 frames. Normalization centers the root, normalizes
torso scale and applies yaw-only alignment so gravity-related roll/pitch remain.

## Frozen protocol

- 16 selected action classes (`lie` and `stumble` included; `fall` excluded);
- 21 HM3D-train scenes and four furniture regions;
- 32 candidate viewpoints per placement;
- motion-record split: train 589, val 197, test 194 (980 records total);
- no future candidate RGB/depth/perception in policy inputs;
- Stage A, Stage B and Stage C-v0 artifacts are accepted and frozen.

## Stage A/B/C-v0

Stage A episodes, Stage B utility labels and Stage C feature cache are stored
under `ACTIVEVIEW_DATA_ROOT/datasets/policy_v11_5/`. Their validators passed;
the accepted artifacts must not be regenerated during code maintenance.

Stage C-v0 baselines (offline Test diagnostic):

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| NoMove | 41.27% | 38.18% |
| Pairwise | 61.45% | 55.33% |
| Set Ranker | 62.54% | 56.37% |
| SafeOracle | 84.49% | 81.11% |

Failure analysis found C2 wrong-candidate errors and long-tail hard records;
body-orientation evidence was weak/inconclusive. These are diagnostics, not a
reason to alter the frozen protocol.

## Current experiments

`EXP001_gap_aware_ranking` is **REJECTED**. Its single proposed change was a
stay-inclusive utility-gap-weighted pairwise ranking term:

```text
lambda_gap=1.0, tau_gap=1.0, max_gap_weight=10.0
```

Its compact baseline is in
`experiments/stage_c_v1/EXP001_gap_aware_ranking/baseline.json`. The experiment
Val-only evaluation completed on commit `5b57417`; no Test evaluation was used.

`EXP002_hard_record_sampling` is **REJECTED**. Its Val-only run used the
Train-only hard-record-aware sampler (118 hard and 471 normal records), but
P90/mean regret, headroom and C2 did not improve. Test was not used.

`EXP003_relative_geometry` completed its authorized Val-only run and is
**REJECTED** because the preregistered 5% mean-regret improvement target was
not met. Its positive geometry trend and near-radius bias remain diagnostic
evidence; no Test evaluation was run.

## Stage C-v1 diagnostic experiments

`EXP004_radius_ablation`, `EXP005_direction_geometry`, `EXP006_move_stay_decoupled`
and `EXP007_candidate_relations` completed independent Train→Val runs. Their
Val results did not meet the preregistered improvement target and are recorded
as rejected diagnostic directions; no Test evaluation was used.

## Research queue

Stage C-v2 experiments completed their authorized Train-to-Val runs on the
shared frozen-cache protocol. Test remains locked and no Test evaluation was
performed. The current skeleton representation is body-yaw canonicalized and
therefore does not preserve explicit body-to-candidate directional alignment;
this limitation was preregistered before the runs.

Val results are recorded in `experiments/stage_c_v2/` and the corresponding
runtime roots under `ACTIVEVIEW_DATA_ROOT/experiments/stage_c_v2/`:

| Experiment | Accuracy | Macro-F1 | Mean regret | P90 regret | Headroom | C2 |
|---|---:|---:|---:|---:|---:|---:|
| EXP008 | 0.644813 | 0.598766 | 1.474656 | 5.633397 | 0.770660 | 0.332952 |
| EXP009 | 0.647530 | 0.597071 | 1.502271 | 5.819350 | 0.759083 | 0.308572 |
| EXP010 | 0.651677 | 0.598782 | 1.458965 | 5.660032 | 0.784780 | 0.326875 |

All three are recorded as rejected diagnostic directions: none improved the
primary regret criteria over frozen v0. Their negative results are retained;
no Test evaluation was performed.

The next authorized research task is:

1. prepare (but do not execute) Stage C-v3 predictability diagnostics
   EXP011–EXP013;
2. any separately authorized follow-up experiment after human review;
3. Stage D Habitat closed loop only after a final method is selected.

The v2 experiments test current representation and candidate-conditioned
reasoning, not new losses, samplers, utility targets or perception data.

## Runtime roots

- Source: repository `activeview/` (the only source package).
- Data: `ACTIVEVIEW_DATA_ROOT` or `../../data/ActiveView/`.
- Habitat: `ACTIVEVIEW_HABITAT_DATA_ROOT` or configured `robot/DATA/`.
- Historical documents: `docs/archive/legacy/`; not default context.
