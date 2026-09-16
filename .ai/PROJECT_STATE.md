# ACTIVEVIEW Scientific State

Updated: 2026-09-15

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

## Adaptive head × strict Frame0 NBV audit (2026-09-14)

The latest Train/Moving-Val-only audit evaluated whether adaptive frozen-ST-GCN
heads combine with strict single-step Frame0 NBV. The exact action set was
`current/Stay + Stage-A legal candidate_pool` over 46,324 Train and 10,080
Moving-Val contexts. Selector inputs were current frame-0 DINO context, legal
geometry and the existing visibility auxiliary; selected real O1 alone was
used for terminal HAR. Policy Test, new perception and ST-GCN modifications
were not used.

The historical selector reproduction was exact (shared-head Accuracy/F1
0.506151/0.503722). The old adaptive head reached s1 0.530952/0.555852 but
legal-candidate 0.369800/0.392751, so it is distribution-specialized. The new
record-balanced legal-candidate head reached s1 0.488393/0.493137 and legal
0.364633/0.367584. Strict Frame0 Accuracy/F1: random+shared
0.365079/0.364567; historical selector+shared 0.506151/0.503722; the same
selector+old adaptive 0.523115/0.542017; retrained old-adaptive selector
0.526984/0.547002; retrained balanced-adaptive selector 0.493452/0.497652.

Candidate-only AnyCorrect coverage was 77.173% (shared), 78.542% (old) and
74.712% (balanced). Shared/old and shared/balanced candidate utility rankings
had Spearman 0.908 and 0.954. The balanced branch was -1.270pp below the
historical strict Frame0 baseline; the preregistered decision is **KILL
ADAPTIVE-HEAD × NBV COMBINATION**. The old head remains a matched s1 diagnostic,
not a general NBV recognizer. Report and script are under
`experiments/reduced12_eight_placement_v1/adaptive_head_frame0_nbv_audit/` and
`activeview/scripts/experiments/run_reduced12_adaptive_head_frame0_nbv_audit.py`.

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

### Privileged information ladder / oracle gap decomposition (2026-09-13)

The latest Train/Val-only frame-0 audit used 46,324 Policy Train contexts and
10,080 Moving-Val contexts with the exact current/Stay + Stage-A legal action
set. Frozen reduced12 ST-GCN and shared head were unchanged; Policy Test and
new perception artifacts were not read or generated. A common 3-layer utility
MLP was trained for 12 epochs with 313 records × 16 contexts per epoch.

Moving-Val Accuracy/Macro-F1: Stay 0.302579/0.292976, Random 0.365079/0.364567,
GeometryOnly 0.487302/0.488664, RealVisibility+Geometry 0.520139/0.520467,
GTAction+Geometry 0.488889/0.489240,
GTAction+RealVisibility+Geometry 0.526091/0.522288, and GT-TrueLogP Oracle
0.753175/0.755358. The residual Oracle→GTAction+RealVisibility+Geometry gap is
22.708pp Accuracy. Geometry plus visibility contributes +3.284pp over the
unified GeometryOnly rerun, while action plus geometry contributes only
+0.159pp. The preregistered interpretation is conclusion D: substantial
pre-action uncertainty remains even with privileged action and visibility
cues. No follow-up method was started automatically.

Report: `experiments/reduced12_eight_placement_v1/privileged_information_ladder/`.

## Latest prefix-length causal mixed-view oracle sweep (2026-09-13)

The latest Val-only audit used 10,080 reduced12 Moving-Val contexts, the
exact action set `Stay/current + Stage-A legal candidate_pool`, and the frozen
reduced12 ST-GCN plus frozen shared head.  No model was trained, no perception
artifact was regenerated, and Policy Test was not read.  This was explicitly a
discrete-time view-switch approximation: for prefix length `L`, Stay is
`current[0:30]` and a candidate is `current[0:L] + candidate[L:30]`.

| Prefix | Random Acc/F1 | GT-TrueLogP Acc/F1 | AnyCorrect/Margin Acc/F1 | FullView Acc drop |
|---:|---:|---:|---:|---:|
| L=5 | 0.342262/0.329111 | 0.687599/0.671197 | 0.720040/0.709917 | 0.065575 |
| L=8 | 0.329861/0.315272 | 0.653571/0.633511 | 0.684325/0.671529 | 0.099603 |
| L=10 | 0.328770/0.308258 | 0.633234/0.608946 | 0.664087/0.645445 | 0.119940 |
| L=12 | 0.321429/0.299336 | 0.614087/0.585399 | 0.646825/0.621947 | 0.139087 |
| L=15 | 0.315278/0.295595 | 0.585317/0.554597 | 0.613591/0.587340 | 0.167857 |

FullView GT-TrueLogP reference is 0.753175/0.755358.  L=8 is the longest
prefix meeting the preregistered viability rule (TrueLogP Accuracy >= 0.65
and FullView drop < 0.10); L=10 is below the threshold.  High-occlusion
contexts also favor L=5 (0.534087), followed by L=8 (0.485478), L=10
(0.464995), L=12 (0.444512) and L=15 (0.412412).  Per-class F1 prefers L=5
for every class except `crawl`, which prefers L=8.

Candidate ranking stability decreases as L grows (candidate-only utility
Spearman vs FullView: 0.8711 at L=5, 0.6666 at L=15), while adjacent-prefix
rankings remain high.  Boundary displacement amplification is approximately
8–9x across prefixes, so results should not be interpreted as continuous
navigation.  The longer-prefix family is retained diagnostically with L=8 as
the next candidate; no selector training or automatic follow-up has started.

Report and implementation:
`experiments/reduced12_eight_placement_v1/short_prefix_length_oracle_sweep/`
and
`activeview/scripts/eval/run_reduced12_prefix_length_causal_sweep.py`.

## Structured observability privileged audit (2026-09-13)

Completed the final Train/Val-only reduced12 frame-0 structured-observability
audit over 46,324 Policy Train and 10,080 Moving-Val contexts. The exact action
set remained current/Stay plus the Stage-A legal candidate pool. Frozen
reduced12 ST-GCN/shared head were unchanged; Policy Test was not read and no
new perception artifacts were generated.

Exact frame-0, 17-joint, scene-only Habitat raycast caches were generated for
Train and Val. Their per-candidate mean matches the existing scalar frame-0
visibility cache exactly (max absolute difference 0). Five record-balanced
utility MLP branches used 313 records × 16 contexts/epoch, seed 42, AdamW,
and SmoothL1 + 0.5 listwise loss for 12 epochs.

Moving-Val Accuracy/Macro-F1: ScalarVisibility+Geometry 0.520139/0.520467;
StructuredVisibility17+Geometry 0.512401/0.510476; CurrentPose+Geometry
0.500000/0.499949; CurrentPose+StructuredVisibility17+Geometry
0.522421/0.520503; GTAction+CurrentPose+StructuredVisibility17+Geometry
0.520833/0.518689; GT-TrueLogP Oracle 0.753175/0.755358. Structured visibility
 geometry is -0.774pp versus scalar; adding current pose is +1.002pp; the
GT-action branch does not improve further. The 23.234pp Oracle residual
triggers the preregistered kill rule for pre-action structured-observability
selectors. Do not continue per-limb visibility, RGB visibility, larger
encoders, or task-aware visibility fusion; prioritize future-recognizer
evidence and sequential information acquisition instead.

Report: `experiments/reduced12_eight_placement_v1/structured_observability_final_audit/`.

## Latest pair complementarity generalization audit (2026-09-14)

The latest reduced12 Train/Moving-Val-only audit tested whether PairMeanGreedy
gains reflect stable pair complementarity rather than single-view quality or
absolute viewpoint IDs. It used 46,324 Train contexts to derive single-view
quality/pair priors and 10,080 Moving-Val contexts for evaluation, with the
exact `current/Stay + Stage-A legal candidate_pool` action set, frozen
reduced12 ST-GCN/shared heads and normalized MeanLogP fusion. Policy Test was
not read; no model, recognizer, RGB, skeleton or DINO artifact was changed.

Protocol reproduction passed: Random B2/B3 = 0.429762/0.424594 and
0.487401/0.477211; PairMeanGreedy B2/B3 = 0.508234/0.497486 and
0.557639/0.546970 (Accuracy/Macro-F1). AdditiveQuality reached
0.511310/0.500984 and 0.555456/0.545044, so PairMean minus AdditiveQuality
was -0.308/+0.218pp. ResidualPairGreedy reached only 0.373413/0.363775 and
0.418452/0.403702; its B3 result was 6.895pp below Random B3. Pair matrix vs
Q(i)+Q(j) Spearman/Pearson was 0.773338/0.771953, with residual mean/std
0.394839/0.200308.

Cyclic prior shifts of +45/+90/+135/+180 degrees produced B2 accuracies
0.487599/0.485119/0.485714/0.480754 and B3 accuracies
0.541270/0.544643/0.544841/0.544742; maximum drops were 2.748pp (B2) and
1.637pp (B3), below the strong absolute-viewpoint shortcut threshold.
Original-vs-Shared Train pair matrices correlated at Spearman/Pearson
0.935007/0.928057 (top-10 overlap 0.40), so complementarity was not strongly
recognizer-specific by the preregistered correlation criterion.

Scientific decision: PairMean B=3 remains a useful empirical baseline
(+7.024pp over Random B3), but its gain is mostly single-view quality and
dataset prior; stable residual pair complementarity is not established and no
strong absolute-ID shortcut is detected. Do not automatically start another
experiment. Report and script:
`experiments/reduced12_eight_placement_v1/pair_complementarity_generalization_audit/`
and
`activeview/scripts/experiments/run_reduced12_pair_complementarity_generalization_audit.py`.

## Policy–recognizer coupling audit (2026-09-14)

Completed a strict Frame0 single-step Train/Moving-Val-only audit on 10,080
Moving contexts. The action set was current/Stay plus the Stage-A legal
candidate pool; terminal HAR used the selected real O1 alone. Existing
historical/retrained selectors, shared/old-adaptive heads, option caches and
frame-0 DINO/geometry were reused. No model or perception artifact was
created, and Policy Test was not read.

Adaptive-aware selector + old adaptive reached 0.526984/0.547002
(Accuracy/Macro-F1); static and view-pair priors reached 0.522917/0.546040 and
0.519940/0.543270. RGB shuffling reduced Accuracy by 4.256pp and geometry
shuffling by 15.159pp, but adaptive-aware minus the best fixed prior was only
0.407pp. On identical adaptive-selected views, old adaptive versus shared
gained 1.865pp, below the 2pp criterion. Selector distributions were closely
aligned with historical s1 (JSD 0.014136, top-5 overlap 0.8).

Decision: **MOSTLY FIXED VIEWPOINT PRIOR**, with measurable but insufficient
instance-conditioned RGB signal. Preserve 52.6984% as a strict Frame0
diagnostic and do not automatically promote this as a standalone NBV line.

## Static prior + Frame0 residual NBV audit (2026-09-14)

The Train/Moving-Val-only strict Frame0 audit used the same current/Stay plus
Stage-A legal candidate pool, frozen ST-GCN and old adaptive head. Train-only
old-adaptive GT-Margin means formed `Q(v)`; residual supervision was
`U(x,v)-Q(v)`. The RGB residual branch reused the historical Frame0 DINO
global + geometry + VisibilityAux scorer, while a geometry-only residual was a
small control. Policy Test and future candidate observations were not used as
selector inputs.

On 10,080 Moving-Val contexts, StaticViewPrior scored 0.522917/0.546040
Accuracy/Macro-F1. Prior+RGBResidual scored 0.524008/0.547073 at λ=0.5 and
0.525496/0.548474 at λ=1.0; Prior+GeometryResidual scored
0.518948/0.543900. Instance-only Adaptive-aware remained stronger at
0.526984/0.547002. RGB residual candidate Spearman was 0.161243 (within
context mean/median 0.121620/0.142857), and RGB shuffling reduced λ=1
Accuracy by 1.111pp. Residual-zero reproduced prior actions and metrics
exactly. High-occlusion λ=1 Accuracy was 0.470498 versus StaticPrior 0.459798
and GT-Margin Oracle 0.710486.

Scientific decision: **KILL PRIOR+RESIDUAL FRAME0 NBV**. Frame0 RGB contains a
measurable residual signal, but the best residual branch gains only +0.258pp
Accuracy over the fixed prior and remains -0.149pp below the instance-only
selector; this is insufficient evidence for a standalone prior+residual NBV
route. Preserve the diagnostic and return focus to the established NBV route.

## Yaw8Fair strict Frame0 NBV full re-baseline (2026-09-15)

The Yaw8Fair recognizer is now the frozen Yaw8 ST-GCN encoder plus the matched
Policy-balanced shared head. The strict Frame0 protocol was re-baselined on
the identical candidate-only Stage-A legal action set: 46,324 Policy-Train
contexts (313 records) for selector fitting and 10,080 Moving-Val contexts
(105 records) for evaluation; mean/min/max legal candidate counts were
6.8157/2/21. Current/Stay is reported only as a separate diagnostic and is
not mixed into the main next-view action results. No Policy Test, new RGB,
skeleton, DINO or recognizer modification was used.

Moving-Val candidate-only results (Accuracy/Macro-F1): Random legal
0.426786/0.450423; StaticViewPrior 0.549901/0.571990;
Frame0SceneVisibility 0.583929/0.598955; GeometryOnly-Visibility
0.540377/0.561212; RGBGlobal-Visibility 0.567361/0.582567;
RGBSpatial-Visibility 0.555456/0.571285; GeometryOnly-TrueLogP
0.531448/0.555499; RGBGlobal-TrueLogP 0.545933/0.565164;
RGBGlobal-Margin 0.549504/0.568928; RGBGlobal-TrueLogP+VisibilityAux
0.548512/0.566333; Prior+GeometryResidual 0.541270/0.564201;
Prior+RGBResidual λ=.5 0.550794/0.573851; Prior+RGBResidual λ=1
0.547520/0.569810; GT-TrueLogP Oracle 0.760714/0.775961; GT-Margin Oracle
0.776190/0.796334. Stay is 0.350496/0.364253. Candidate-only AnyCorrect
Coverage is 0.776190 and passes the registered oracle gate.

The best strict Frame0 method is Frame0SceneVisibility (+3.403pp Accuracy
over StaticViewPrior and +15.714pp over Random). The best learned
utility/residual branch is Prior+RGBResidual λ=.5, only +0.089pp over
StaticViewPrior. Thus the current evidence supports keeping an
instance-conditioned Frame0 NBV baseline, but the measurable gain is mainly
from the scene-visibility rule rather than learned utility ranking. For the
best learned branch, candidate-level/within-context utility Spearman was
0.137724/0.083313 (median 0.100000), oracle top-1 agreement 0.246627 and
top-3 hit 0.573710.

The learned branch matched StaticViewPrior on 89.514% of contexts; the
remaining 1,057 switches (10.486%) improved switch-subset Accuracy from
0.572375 to 0.580889 (+0.851pp), with 184 rescue and 175 harm cases. On the
fixed bottom-tertile high-occlusion subset (3,271 contexts),
Frame0SceneVisibility reached 0.520024/0.539481 versus StaticViewPrior
0.481810/0.512912; the best learned branch reached 0.485173/0.516800.
The historical U4 result (0.506151/0.503722 under the old recognizer) became
0.548512/0.566333 after Yaw8Fair retraining. Direct transfer of old
selectors scored 0.572222/0.587019 and 0.555060/0.576933, retained only as
diagnostics.

Decision: **KEEP INSTANCE-CONDITIONED FRAME0 NBV**; do not automatically
start another method family. Complete artifacts are under
`experiments/reduced12_eight_placement_v1/yaw8_strict_frame0_full_rebaseline/`
with runner
`activeview/scripts/experiments/run_reduced12_yaw8_strict_frame0_rebaseline.py`.

## ParaHome feasibility and humanoid retarget (2026-09-17)

ParaHome was audited as a candidate continuous Active HAR dataset and its
23-joint SMPL-X fits were retargeted into the Habitat `male_0` humanoid.  The
dataset audit is read-only and unchanged: 207 sequences, 38 subjects, 486.33
minutes, 5,476 annotations, 16 classes meeting the 30/15/8 and 50/20/10 coverage
thresholds, verdict **CONDITIONAL PROMOTE** (taxonomy duration/coverage and a
coordinate/gender/scene adapter are still required).  No reduced12/BABEL
artifact, recognizer, frozen checkpoint or Policy Test data was touched.

The replay path is a ParaHome-specific hierarchical segment-direction retarget
(`activeview/data/motion/parahome_retarget.py`) that leaves the frozen
AMASS/BABEL `MotionConverter` untouched.  Five defects were found and fixed,
each verified in the real Habitat simulator (CUDA, RTX 4090):

1. Habitat's articulated link order differs from the URDF declaration order
   (`HABITAT_MALE_0_JOINT_ORDER`).
2. `PARAHOME_TO_HABITAT` was a reflection (`det = -1`); because solved joint
   rotations are proper rotations, the `pelvis`/`spine3` fit flipped the body
   frame ~180 deg and the avatar faced away from its own arms.  It is now the
   proper rotation `[[0,-1,0],[0,0,1],[-1,0,0]]` (`det = +1`).
3. The solver discarded the parent twist for single-child joints; it now solves
   top-down and inherits the parent frame.
4. The two rigs use different anatomical joints under the same names (ParaHome
   hip centre vs the `male_0` pelvis link; ParaHome L5/L4 vs the rig's lordotic
   spine), which pushed the abdomen mesh forward; the pelvis is now solved from
   the hip line plus the spine axis and `spine1`/`spine2` keep the rig's rest
   curvature.
5. Joint spacing was never matched (rig hips 12.5 cm vs recorded 18.8-21.9 cm)
   and the recorded stance is narrower than the rig's own rest stance, so the
   legs interpenetrated; `match_hip_width` corrects the stance width without
   dropping below the rig's rest knee separation.

Verified render conventions: `male_0` faces +Z at rest, all 54 URDF joint origins
have `rpy = 0`, grounding must use the skinned mesh (not the URDF debug boxes),
and sample renders use a single **static** robot-eye camera (platform fixed,
camera 1.2 m above the floor, aimed at 1.0 m) placed from the centre of the
recorded trajectory.  Evidence, validators, videos and comparison stills are in
`experiments/parahome_feasibility_v1/retarget_pose_fidelity/`; unit regressions
are in `tests/unit/test_parahome_retarget.py` (81 unit/integration tests pass).
Still open: the bone twist is unobservable from positions (no independent head
yaw / palm roll), rig proportions differ from the recorded subjects, and only the
`male_0` body shape is available locally.
