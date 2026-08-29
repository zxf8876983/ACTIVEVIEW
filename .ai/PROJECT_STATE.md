# ACTIVEVIEW Scientific State

Updated: 2026-08-29

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

`EXP003_relative_geometry` has completed its authorized Val-only run and is
**PENDING USER REVIEW**. Its single change was five candidate-set-relative
geometry features appended to the frozen 11-D geometry (16-D total). The
feature cache and Set Ranker checkpoint are stored under the external runtime
root; no Test evaluation was run. Compact results and a geometry-bias
diagnostic are recorded in the EXP003 source experiment directory.

## Research queue

1. User review of EXP003 relative geometry (Val-only result);
2. richer current temporal representation;
3. perceived body orientation only if later evidence supports it;
4. Stage D Habitat closed loop after a final method is selected.

## Runtime roots

- Source: repository `activeview/` (the only source package).
- Data: `ACTIVEVIEW_DATA_ROOT` or `../../data/ActiveView/`.
- Habitat: `ACTIVEVIEW_HABITAT_DATA_ROOT` or configured `robot/DATA/`.
- Historical documents: `docs/archive/legacy/`; not default context.
