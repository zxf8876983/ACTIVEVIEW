# ACTIVEVIEW Scientific State

Updated: 2026-09-11

## Research goal

Improve indoor elderly action recognition by actively selecting robot viewpoints
that reduce perception uncertainty under occlusion and self-occlusion.  The
current scientific question is narrower: determine which parts of candidate
HAR utility are due to motion, scene/placement, viewpoint geometry and
sequential reachability, and use that evidence to decide whether to pursue a
sequential information-acquisition policy rather than another one-shot scalar
utility predictor.

## Current canonical pipeline (reduced12 research protocol)

```text
AMASS/BABEL → male_0 Habitat RGB → YOLO26n-Pose → VideoPose3D
→ H36M-17 camera/gravity conversion and normalization
→ frozen ST-GCN → Stage B utility → Stage C viewpoint policy
```

The ST-GCN receives estimated skeletons only. RGB is 256×256 and sequences are
uniformly sampled to 30 frames. Normalization centers the root, normalizes
torso scale and applies yaw-only alignment so gravity-related roll/pitch remain.
The current ActiveView policy cache uses the reduced12 eight-placement protocol;
the historical 16-class final-test protocol below is retained only for audit
traceability.

## Frozen protocol for current work

- 12 action classes: `walk`, `sit`, `stand up`, `bend`, `crawl`, `stumble`,
  `clap`, `throw`, `kick`, `knock`, `punch`, `touching face`;
- 20 HM3D-train scenes selected by the first five-digit prefixes in
  `hm3d-train-semantic-annots`;
- eight furniture-anchored placements per scene with fixed random yaw;
- 32 candidate viewpoints per placement;
- Official Train cap 300/class and Official Val cap 50/class (seed 42);
- current ActiveView records-only split: 313 train / 105 val / 0 test;
- current Stage-D context counts: 46,324 train / 15,540 val, with 10,080
  Val moving contexts in the latest utility audits;
- no future candidate RGB/depth/perception in policy inputs;
- frozen reduced12 ST-GCN, Stage-B utility, Stage-C-v0 and existing caches;
- policy Test is not read for current research diagnostics.

The old 16-class, 21-scene, four-region and 589/197/194 split are historical
v11.5/official-Final-Test records only; they are not the current default.

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

Stage C-v3 EXP011–EXP013 predictability diagnostics have completed their
authorized Train→Val / read-only Val runs. EXP011's corrected 17-D teacher did
not improve regret/headroom over v0; EXP012 showed only modest utility
predictability gains from legal current state; EXP013 showed strong offline
Top-K coverage by K=5. Results are recorded in `experiments/stage_c_v3/` and
the external runtime root, pending human scientific review.

The next authorized research task is a separately approved follow-up after
review; Stage D Habitat closed loop remains out of scope until a final method
is selected.

The v2 experiments test current representation and candidate-conditioned
reasoning, not new losses, samplers, utility targets or perception data.

## Stage D results (corrected Val-only rerun)

The one-shot Stage C-v0 ranking phase is frozen. EXP011–EXP013 showed moderate
online utility predictability but strong Top-K proposal coverage. Execution
records for the approved sequential study are in `experiments/stage_d/`:

- The pre-fix cache used a displacement-bearing implementation for the Stage A
  relative-azimuth fields and is archived for traceability.
- The corrected cache reads existing semantic-region-v2 radial `azimuth_deg`
  metadata and computes `candidate_azimuth - s1_azimuth` with Stage A's
  wrapping rule. EXP014 and EXP015 were rerun Val-only against this cache.
- Corrected EXP014: Accuracy 0.658254, Macro-F1 0.610153, mean regret
  1.422463, P90 regret 5.515663, headroom 0.783313; decision REJECT under the
  recorded thresholds. EXP015 remains an analysis-only INCONCLUSIVE diagnostic.

Both used only Train/Val. Test, Habitat/perception regeneration and ST-GCN
retraining remain prohibited until separately authorized.

## Final closed-loop method

The canonical final method is **WM-E + Multi-Positive Joint Revision +
Closed-Loop H2**. Its frozen protocol is horizon 2 with `ALL_LEGAL` candidate
sets, frozen Stage-C-v0 initial action, frozen WM-E/JR/ST-GCN, current-viewpoint
centered second-step geometry, visited-view exclusion, and final HAR from the
real archived terminal skeleton passed through frozen ST-GCN. EXP051-R2 fixed
the terminal HAR and fused only real visited observations.

EXP051-R2 was a Val-only paired evaluation. On the 9,742 moving contexts,
H1_REAL reached Accuracy/F1 0.661568/0.632699 and H2_REAL reached
0.675529/0.642612 (paired Δ +0.013960/+0.009913; rescued 369, harmful 233,
McNemar p=3.30e-08). On the full 13,987 population, H1_REAL was
0.685780/0.643640 and H2_REAL was 0.695503/0.649220. Real-view fused H1/H2
moving Accuracy/F1 were 0.599364/0.545336 and 0.607370/0.552557. WM-E
history fidelity shifted from h0 agreement/Pearson 0.596064/0.738312 to h1
0.498571/0.645801, while real terminal H2 still improved.

EXP055 multi-positive JR used 29,133 Train contexts (27,077 with at least one
correct action; 25,362 multi-positive; 2,056 no-positive fallback), seed 42,
and frozen checkpoint SHA
`8a6ef93ded8df94154f2045d6cf7d297c23e587ac8cf2601a83fcf3c82f1383c`. It
reached moving/full Val Accuracy/F1 0.683022/0.647338 and 0.700722/0.650955,
respectively, improving EXP051-R2 in both populations (CASE A).

EXP056 tested seeds 42/43/44 without Test access. Multi-positive JR won all
three seeds on Accuracy and F1; mean moving Accuracy was
0.685828±0.003968 versus Original JR 0.665914±0.000640, and mean full
Accuracy was 0.702676 versus 0.688806. The objective stability result is a
stable CASE-A pass; seed 42 remains the recommended frozen run.

## Official Final Test and audit

The explicitly authorized Final Test is complete and recorded in
`experiments/stage_d/FINAL_TEST/result.json` with `test_used=true`. The FULL
population has 13,774 episodes and MOVING has 9,409. MULTI_POSITIVE_JR_H2
achieved FULL Accuracy/F1 0.684841/0.627749 and MOVING 0.661388/0.622984.
The corresponding H1_REAL and ORIGINAL_JR_H2 results were FULL
0.673515/0.623050 and 0.680558/0.629111, and MOVING 0.644808/0.612497 and
0.655117/0.621852. Relative to FrozenStageCv0, Multi improved FULL by
+5.946pp Accuracy/+6.404pp F1 and MOVING by +8.704pp/+9.248pp. Relative to
H1_REAL, Multi improved FULL by +1.133pp/+0.470pp and MOVING by
+1.658pp/+1.049pp; relative to Original JR H2, the Multi deltas were FULL
+0.428pp/-0.136pp and MOVING +0.627pp/+0.113pp.

The post-refactor equivalence audit is recorded in
`experiments/refactor_regression/result.json` with status PASS. Using the
formal modules and existing Test artifacts read-only, all seven frozen
methods (NoMove, Random seed 42, FrozenStageCv0, SafeOracle, H1_REAL,
ORIGINAL_JR_H2 and MULTI_POSITIVE_JR_H2) matched their golden Accuracy/F1
values within 1e-8 on the same FULL/MOVING populations. This was an audit, not
a new method experiment. Frozen artifact hashes include WM-E
`db2573a013ed9a7fab87561ad26800334556894b96e69dd3d498464794d9b5e6`, Original
JR `332b3127747f67d954d7c80f530ee1cc5a9ca30c6472fd13a3a010a080c413ac`, Multi
JR `8a6ef93ded8df94154f2045d6cf7d297c23e587ac8cf2601a83fcf3c82f1383c`, and
ST-GCN `362ac23195688988d637244eb2a13fa0e7b563b21d143846c671a5cec6b0d0ce`.

The final source consolidation is committed at `c941380` and the current
documentation/equivalence commits follow it. No further experiment is
authorized automatically; Test remains locked for any future changes to the
method.

## Reduced14 eight-placement ActiveView retraining (2026-09-06)

The canonical current ActiveView runtime uses `reduced14_kneel` with eight
furniture-anchored placements per scene and the existing raw-val record split
Train/Val/Test = 357/120/120 (57,834/19,440/19,440 contexts), without a scene
split. The 14-class ST-GCN checkpoint remains frozen. A new recognition-aware
WM-E was trained on Train only (12 epochs, seed 42, final loss 0.147612), and
Multi-Positive Joint Revision was trained on its Train counterfactual cache
(20 epochs, seed 42, final loss 1.409037). Frozen ST-GCN terminal recognition
was used for all policy metrics.

On this new population, Test Full Accuracy/Macro-F1 were NoMove
0.313272/0.302466, FrozenStageCv0 0.426698/0.418769, Random
0.311728/0.306334, SafeOracle 0.782870/0.777816 and Multi-positive H2
0.470319/0.468083. Moving-subset values were 0.278551/0.238991,
0.427558/0.411638, 0.276524/0.252854, 0.895459/0.893287 and
0.484863/0.481803 respectively. The learned Multi-positive selector is above
FrozenStageCv0 but far below SafeOracle; this is a new-data retraining result,
not a change to the frozen ST-GCN protocol. Compact results are stored in
`experiments/reduced14_eight_placement_v1/active_view_retraining/`.

## Runtime roots

- Source: repository `activeview/` (the only source package).
- Data: `ACTIVEVIEW_DATA_ROOT` or `../../data/ActiveView/`.
- Habitat: `ACTIVEVIEW_HABITAT_DATA_ROOT` or configured `robot/DATA/`.
- Historical documents: `docs/archive/legacy/`; not default context.

## Latest reduced12 sampling adjustment (2026-09-08)

The active 12-class protocol keeps the Official Train cap at 300 per class
and now uses an Official Val cap of 50 per class.  The isolated raw-val
manifest was regenerated accordingly: 523 selected Official Val records,
with raw-val records-only manifests of 313/105/105; the current policy build
materializes only Train/Val (313/105/0).  The ST-GCN raw-train tensors and
checkpoint were not changed; no Test data was read for evaluation.

## Legacy generated-data cleanup (2026-09-08)

After the reduced12 protocol was finalized, legacy generated datasets and
policy-evaluation artifacts were removed from the external ActiveView data
root. This includes the old four-region/offline and reduced14/15/16 motion
outputs, old RGB/feature/policy caches, old ActiveView/ST-GCN checkpoints,
Stage-C/Stage-D runtime caches, and generated visualizations/results. The
current reduced12 dataset and checkpoint remain intact. Original BABEL and
AMASS datasets, Habitat humanoid assets, YOLO/VideoPose3D pretrained weights,
and the AMASS download index were preserved.

## Current reduced12 runtime assets (2026-09-11)

The frozen reduced12 recognizer is the CUDA/seed-42 checkpoint produced from
the raw-train 300-per-class cap.  Its development-Val Accuracy/Macro-F1 are
0.767347/0.781069.  Official Val uses the 50-per-class cap (523 selected
records); the records-only policy split is 313/105/0 for Train/Val/Test.  The
current external policy root is
`$ACTIVEVIEW_DATA_ROOT/datasets/policy_reduced12_eight_placement_v1/` and the
20-scene offline Habitat output is under
`$ACTIVEVIEW_DATA_ROOT/datasets/offline/habitat-train/00006-00087/`.
Stage A/B/C-v0 and Stage-D artifacts are present for Train/Val; no generation
process is active, and no current diagnostic has read policy Test.

## Latest Val-only utility and reachability diagnostics (2026-09-11)

The latest audits use 10,080 reduced12 Val moving contexts and 68,702 legal
candidate samples (143 scene/placement groups).  They read cached true
candidate ST-GCN log-probabilities only; no model, RGB, skeleton or DINO data
was regenerated.

### Utility source decomposition

- Body-relative azimuth convention was numerically confirmed: world azimuth is
  `atan2(+X,+Z)`, positive placement yaw rotates local +Z toward +X, and
  `body_relative = wrap(world_candidate_azimuth - placement_yaw)`.
- Same-motion utility maps across scene/placement have mean/median Spearman
  0.268/0.400, versus 0.147/0.188 for matched different-motion maps.
- Same scene/placement maps are weak when actions are mixed (0.148), but rise
  to 0.281 for the same action; scene is therefore not an action-independent
  template.
- Leave-one-sample-out explained variance is 0.479 motion-only, 0.104
  scene/placement-only and 0.559 additive motion+scene, leaving 0.441
  interaction residual.  Motion+scene explained variance falls from 0.728 at
  1.5 m to 0.313 at 3.0 m.
- Class structure is heterogeneous: `throw` has the largest additive
  explained variance (0.643), while `bend` and `touching face` retain about
  0.74 interaction residual; `knock` has the strongest same-scene same-action
  consistency (Spearman 0.669).

### Sequential reachability structure

Candidate-only privileged K-hop Accuracy/Macro-F1 are:

| Hop | Accuracy | Macro-F1 |
|---|---:|---:|
| H1/K0 | 0.454266 | 0.444782 |
| K1 | 0.538790 | 0.529863 |
| K2 | 0.597222 | 0.591727 |
| K3 | 0.654762 | 0.650562 |
| K4 | 0.688393 | 0.683594 |
| K5 | 0.701984 | 0.697100 |
| K6 | 0.708234 | 0.702974 |
| Full | 0.709623 | 0.704598 |

The K-hop curves are reachability ceilings, not deployable policies.  The
privileged greedy local oracle reaches only 0.538790 at one step and 0.558234
after four steps, well below the K4 ceiling.  Correct candidates form small
but nontrivial basins (mean largest component 2.23 nodes; 76.3% of correct
nodes in the largest component), with 29.04% of contexts having no reachable
correct candidate in the lattice.

## Current scientific conclusion and next decision

The strongest evidence is **strong motion×scene×view interaction**, with a
secondary motion contribution.  A sequential information-acquisition protocol
is better motivated than another one-shot scalar predictor, but greedy local
search is insufficient because correct basins are fragmented and global jumps
remain common.  The latest work is diagnostic only; no new method, training,
Test evaluation or automatic follow-up is authorized.  The next experiment,
if approved by the researcher, should explicitly address non-greedy sequential
exploration or multimodal motion/scene interaction rather than silently
changing the frozen protocol.

Latest reports:

- `experiments/reduced12_eight_placement_v1/utility_source_decomposition/`
- `experiments/reduced12_eight_placement_v1/khop_oracle_curve/`
- `experiments/reduced12_eight_placement_v1/overnight_nbv_diagnosis/`
