# ACTIVEVIEW Handoff

Status: EXP001 PLANNED — awaiting explicit start authorization
Updated: 2026-08-29

## EXP001 — Utility-Gap-Aware Ranking Objective

Created `experiments/stage_c_v1/EXP001_gap_aware_ranking/` and its external
runtime directory. The falsifiable hypothesis, single core change, planned
parameters (`lambda_gap=1.0`, `tau_gap=1.0`, `max_weight=10.0`), frozen items,
Val-only acceptance/rejection criteria, and non-executable planned command are
recorded in that directory. The implementation adds an optional
stay-inclusive, utility-gap-weighted pairwise ranking term while
`lambda_gap=0` preserves Stage C-v0 behavior.

No `start_experiment` call, model training, data regeneration, Habitat/RGB/
YOLO/VideoPose3D processing, Test evaluation, or EXP002 creation occurred.
Test remains locked and `test_used=false`. The next action requires explicit
human authorization; do not run `command.sh` automatically.

## Phase 0 research infrastructure — completed, review pending

Implemented and hardened `activeview/research/` and the lifecycle CLIs:
`create_experiment`, `start_experiment`, `validate_experiment`,
`finalize_experiment`, `authorize_final_test`, and
`validate_research_infrastructure`. Start now freezes the actual run commit,
run config, hypothesis and command hashes; validators re-hash every frozen
artifact; and the final Test gate accepts only canonical nested manifests. The
Final candidate freezing is a tracked `COMPLETED → FINAL_FROZEN` transition;
after committing that transition, final authorization writes only an external
runtime artifact and never mutates tracked manifest/registry files. Experiment
paths are portable repository/data-root-relative values. The Stage C-v1 registry at
`experiments/stage_c_v1/EXPERIMENT_REGISTRY.csv` contains the single PLANNED
`EXP001_gap_aware_ranking` record. Test is fail-closed until a COMPLETED+ACCEPT experiment is
explicitly FINAL_FROZEN, committed, and given an external authorization with
matching frozen Git commit and config hash. Runtime artifacts belong under
`ACTIVEVIEW_DATA_ROOT/experiments/`.
No Stage C-v1 training, data regeneration, Test evaluation, or Stage D work
was performed. The next step requires human authorization.

## Repository consolidation (2026-08-29) — completed

The repository-only consolidation completed on top of pre-consolidation
commit `a2935fd177bee15eca4a40b896db5907d0e937d1`, protected by tag
`pre-activeview-consolidation`. `activeview/` is the sole production package;
v1–v10 source trees are removed from the working tree and recoverable from Git
history. The Phase 0 research-state files are now present under `.ai/`. No accepted runtime artifact is being
rewritten, and no training, data regeneration, or Test evaluation was part of
this task. The audit and validation record is
`docs/repository_consolidation_audit.md`.

## Canonical v11.5

The active implementation is `activeview/`. The protocol is selected16: audited official-150 14 classes plus `lie` and `stumble`; `fall` is excluded. BABEL `train.json` and `val.json` are used directly after single-label filtering, strict `num_frames > 30`, conflict removal, official Train/Val caps 400/100, and seed 42. Stage A policy records use `train/val/test = 6:2:2`, read from persisted split `summary.json`.

The perception chain is:

```text
AMASS/SMPL → male_0 pure-color Habitat → RGB-only 256×256
→ Ultralytics YOLO26n-Pose → VideoPose3D
→ Human3.6M Y/Z conversion + Habitat camera-to-gravity
→ root center + torso scale + yaw-only → H36M-17 ST-GCN
```

ST-GCN never receives AMASS/SMPL GT joints. Grounding uses URDF visual geometry and supporting-floor raycast; grounding offsets are cached per action. Yaw-only alignment preserves gravity-related roll/pitch.

## Canonical data and model

- Train/Val: `/home/zxf/WorkSpace/code/data/ActiveView/datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/` (3,240/980, `(N,3,30,17,1)`).
- Frozen ST-GCN: `/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled/`. The canonical checkpoint is the final stopped-epoch model from Train-only convergence; the former Val-Macro-F1 legacy weight was removed.
- YOLO: `/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/ultralytics/yolo26n-pose.pt`.
- Habitat/semantic root: `/home/zxf/WorkSpace/code/code/robot/DATA/` only.
- Humanoid: `/home/zxf/WorkSpace/code/data/ActiveView/assets/habitat_humanoids/male_0/`.

## Offline active-view data

Data is under `datasets/offline/<scene-set>/<original-scene-folder>/`. Four furniture-based regions are used: `bedroom`, `living_room`, `kitchen`, `dining_area`. Each placement has 32 views (1.5/2.0/2.5/3.0 m × 8 azimuths). `semantic-region-v2` candidate manifests and `semantic-region-offline-v2` records persist all geometry, camera, static navigation and 32 skeleton/confidence fields; RGB/Depth are not saved. The old minival scene `00800-TEEsavR23oF` is retained only for historical compatibility and is excluded from the current 21-scene HM3D-train evaluation protocol.

Static placement reachability is not sufficient for a trajectory. Evaluation recomputes `ShortestPath(P_current, P_candidate)` from the robot's actual current position before choosing a next view. Current evaluators report `NoMove`, `Fixed`, `Random`, `Nearest` and hindsight candidate-pool `Oracle`; v11.3 Utility Predictor code/checkpoints are not active.

All 21 requested HM3D-train scene manifests are complete. Each scene contains
980 records × 4 regions × 32 viewpoints. `00592-CthA7sQNTPK` and
`00643-ggNAcMh8JPT` were rotation-audited and matched the exact offline render
state; `00643` required only metadata refresh (`npz_changed=0`). No scene
generation process remains active.

The Stage A policy split was regenerated with canonical 6:2:2 ratio
(589/197/194 records; 980 unique records). Episodes were rebuilt from
existing offline caches: 41,819/13,987/13,774 train/val/test Episodes (69,580
total) for the 21 current HM3D-train scenes. The historical minival scene
`00800-TEEsavR23oF` is intentionally not part of this protocol.

## Historical boundary

The historical v10 tree contained candidate generation and per-view entropy
analysis, not a learned active-view selector. The old v11.3 Utility Predictor
modules are deleted from the working tree but remain recoverable from Git
history.

## Validation and next step

The ST-GCN trainer now uses full-Train loss for scheduling and early stopping, saves the final stopped-epoch weights, and evaluates Val once post-training. The old Val-Macro-F1 checkpoint was removed; v10 historical ST-GCN modules remain read-only reference code. Before future code changes, run a lightweight `compileall`/focused test set, then update this handoff with the new commit and process state.

## Stage A audit hardening (2026-08-28)

`policy_episode_builder.audit_episode_files()` now derives integrity flags
from serialized Episode JSONL rather than hard-coded values. It checks split
isolation, current/candidate IDs and geometry, record-local skeleton paths,
candidate costs, and recursive future-perception leakage fields. With
`validate_cached_skeletons=True`, it validates every referenced NPZ shape,
navigation arrays, viewpoint IDs, and finite skeleton frames. The read-only
acceptance entry point is `activeview/scripts/validate_stage_a.py`;
`--verify-habitat` recomputes real HM3D `ShortestPath` for final Episodes.
The 6:2:2 static + cached-NPZ audit passes for 69,580 Episodes (no duplicate
episode keys/IDs and no integrity failures). The full all-Episode Habitat path
replay was executed and reported `path_failures=[]` across 21 current scenes.
The historical summary target list still contains the excluded legacy-v1
minival directory, so the raw scene audit reports it as missing; this does
not affect the current 21-scene evaluation set.

The Stage A audit now allows partial non-finite skeleton viewpoints. It records
`nonfinite_cached_skeleton_viewpoints` for diagnosis, while current/candidate
validity is checked against the per-archive finite ID set. It also checks
`record_id × scene_id × region` and `episode_id` uniqueness, NPZ/Episode
geometry correspondence, expanded future-perception field leakage, and
label/label_id consistency in policy splits.

The latest audit hardening also validates split summary metadata against the
actual `train.json`/`val.json`/`test.json` files (counts, unique IDs,
per-class counts, and canonical 6:2:2 ratios). Coverage now uses the target
scene list before generation, separately audits scene-level failures, and
requires `all_target_scenes_used=true`; a failed scene cannot be silently
removed from the expected tuple set.

## Stage B completion (2026-08-28)

Implemented and executed the v11.5 Stage B offline utility-label pipeline.
`build_stage_b_utility_labels.py` consumes only accepted Stage A Episode JSONL
and cached estimated-skeleton NPZs, runs the frozen ST-GCN with direct
`log_softmax`, and caches predictions by archive path. It emits compact
current/candidate diagnostics, utility values, CandidateOracle and SafeOracle
labels, aggregate metrics and headroom statistics without future RGB/depth,
skeleton, logits or probability arrays. `validate_stage_b.py` is an
independent read-only integrity and leakage validator.

Artifacts:

- Stage B root: `/home/zxf/WorkSpace/code/data/ActiveView/datasets/policy_v11_5/stage_b/`
- Summary: `stage_b_summary.json`
- Validator report: `validation_report.json` (`passed=true`)
- Validator log: `/home/zxf/WorkSpace/code/data/ActiveView/results/stage_b_validate.log`

Full build counts: 41,819/13,987/13,774 Episodes and
306,869/102,637/101,074 candidate pairs (train/val/test). The validator
reported zero duplicate IDs/pairs, zero missing/unexpected Episodes and zero
record errors. Focused tests pass (`18 passed`). Stage C was subsequently
implemented from these frozen artifacts; see the Stage C completion section
below.

After the initial build, the Stage B audit was hardened. The validator now
recomputes the complete metrics tree (including per-class, per-region,
headroom and rescue/degradation fields), cross-checks candidate geodesic
distances against Stage A, verifies the Stage A summary hash plus all three
Episode-file hashes, and rejects metric/provenance corruption. Near-zero
utility bins are mutually exclusive from positive/negative bins; degradation
also reports its conditional rate among current-correct Episodes. The full
Stage B labels were regenerated with these rules and the validator again
passed. Additional regression tests cover metrics corruption, geodesic
mismatch, stale provenance and the real saved artifact. The validator also
independently loads `datasets/policy_v11_5/splits/` through the canonical
policy-split loader and requires frozen 589/197/194 counts in the split JSON,
Stage A summary, and Stage B summary. A regression test confirms that
consistently altered counts such as 600/190/190 still fail; the canonical full
Stage B artifact was revalidated successfully.

## Stage C completion (2026-08-28)

Implemented current-conditioned utility prediction without changing Stage A/B.
The feature cache contains 41,819/13,987/13,774 Episodes for train/val/test;
current input is 275-D frozen ST-GCN state and candidate input is 11-D geometry
(snapped displacement in the current agent yaw frame plus distance/azimuth/path
features and snapped placement radii).
Both models use record-balanced training, SmoothL1 plus stay-inclusive listwise
ranking, and Val recognition Macro-F1 checkpoint selection; Test is final-only.

Artifacts:

- Feature root and diagnostics: `/home/zxf/WorkSpace/code/data/ActiveView/datasets/policy_v11_5/stage_c/`
- Pairwise checkpoint: `checkpoints/stage_c/pairwise_mlp_best.pth`, epoch 29, 142,785 parameters.
- Set ranker checkpoint: `checkpoints/stage_c/set_ranker_best.pth`, epoch 46, 407,745 parameters.
- Combined summary: `stage_c/stage_c_summary.json`.
- Independent validator result: `/home/zxf/WorkSpace/code/data/ActiveView/datasets/policy_v11_5/stage_c/validation_report.json`, `passed=true`.

The Stage C validator independently reselects each candidate/Stay action from
predicted utilities, obtains recognition outcomes and regret from Stage B,
then recomputes all saved Val/Test metrics from prediction JSONL. It also
verifies canonical split counts, Stage A/B/feature provenance hashes, finite
feature schemas and counts, and confirms Stage D has not started.
No Habitat re-rendering or learned-policy online evaluation was performed.

Recorded Stage C metrics: Test NoMove 41.27% Accuracy / 38.18% Macro-F1,
Pairwise 61.45% / 55.33%, Set Ranker 62.54% / 56.37%, and SafeOracle
84.49% / 81.11%. Set Ranker Test regret is 1.614 mean (0.0075 median,
6.143 p90) with 74.93% aggregate positive-headroom capture; Pairwise is
1.802 (0.0108, 6.582) with 70.83% capture. These are offline results only;
Stage D Habitat learned-policy evaluation remains pending.

## Stage C-v0 failure analysis (2026-08-28)

Failure analysis was completed read-only from the frozen Test artifacts. The
entry point is `activeview/scripts/analyze_stage_c_failures.py`; unit tests
are in `tests/unit/test_stage_c_failure_analysis.py` and the frozen-artifact
coverage check is in `tests/integration/test_stage_c_failure_analysis.py`.
The external runtime output is
`datasets/policy_v11_5/stage_c/failure_analysis/`; a review snapshot is under
`docs/results/stage_c_failure_analysis/`.

The analysis confirms 13,774 Test Episodes and 194 independent motion records.
Set Ranker regret is right-skewed (mean 1.614, median 0.00746, p90 6.143).
Decision failures are primarily wrong-candidate high-loss (32.44%), missed
move (21.95%), and unnecessary move (5.34%); 29.85% match SafeOracle. Exact
candidate hit is 33.93% versus 74.93% aggregate headroom capture. The most
difficult classes are `lie`, `play instrument`, `stumble`, and `knock`; region
effects are not strong enough to support a scene-generalization claim. Entropy
and margin have weak correlations with regret and pose confidence is nearly
uncorrelated. Mirror-like geometry with large utility asymmetry is only
modestly enriched among high-regret cases, giving weak/inconclusive evidence
for adding perceived body orientation. The report recommends that any future
Stage C-v1 discussion start with hard-example/long-tail and representation
diagnostics. The corrected report uses mutually exclusive candidate-set gap
bins, separates CandidateOracle geometry from SafeOracle move geometry, matches
symmetric pairs by candidate radius, reports an explicit enrichment ratio
(1.21x), and reports per-class
and per-region high-regret rates. No Stage A/B/C accepted artifact, model, or
evaluation protocol was changed, no Test-based tuning was performed, and Stage
D was not started.
