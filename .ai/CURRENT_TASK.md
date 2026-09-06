# Current Task

## Persist verified ActiveView generation scheduling — 2026-09-06

The fixed end-to-end benchmark established that eight Habitat/perception
workers with one OMP/MKL/OpenBLAS/NumExpr thread per worker are stable and
about 1.53x faster than the former four-worker default. The current task is to
make those runtime defaults canonical in the two offline-generation entry
points without modifying any scientific data-generation semantics. The active
full reduced14 eight-placement generation remains running and must not be
interrupted. No Test access or model training is authorized.

Status: completed. The canonical orchestrator defaults to eight workers, while
the leaf generator injects one-thread native-library limits only into spawned
worker processes. Import behavior and the legacy leaf default remain
backward-compatible. Focused validation and independent code review passed.

Runtime update: at the user's request, the eight-worker run was stopped and
only its incomplete `00475-g7hUFVNac26` output was removed. Canonical
generation has restarted with an explicit four-worker override. All 14
completed scenes remain intact, and each of the four worker environments was
verified to use one OMP/MKL/OpenBLAS/NumExpr thread.

Completion update: the four-worker run completed all 21 scenes successfully;
each scene contains 4,776 NPZ files and a final manifest. The canonical
orchestrator is now named `generate_hm3d_train_offline.py`; imports and current
documentation use this placement-neutral name. This was a naming-only source
change.

## AI context synchronized — 2026-09-04

The project context is synchronized to the frozen final state: WM-E +
Multi-Positive Joint Revision + Closed-Loop H2, with the official Final Test
already completed once (`test_used=true`). The final Multi-positive method
reached FULL Accuracy/F1 0.684841/0.627749 and MOVING 0.661388/0.622984 on
13,774/9,409 episodes. EXP051-R2, EXP055 and EXP056 remain documented as
Train/Val research results with `test_used=false`.

The post-refactor equivalence audit is PASS: all seven frozen methods matched
their golden Accuracy/Macro-F1 values within 1e-8 using the same Test
populations. No algorithm, checkpoint, official result, or runtime dataset was
changed during this context synchronization. No new experiment is authorized
automatically.

## Reduced-12 BABEL diversity protocol

2026-09-04: built an independent 12-class BABEL protocol without modifying
the frozen selected-16 dataset. Official Train was capped at 300/class and
split 90/10 into ST-GCN development (2,600/289); Official Val was capped at
100/class and split 60/20/20 into ActiveView motion (591/197/197). Selection
used seed 42 and prioritized unique source, subject, AMASS dataset and
duration-bin diversity. Generated tensors use the existing frozen perception
chain. A new 12-class ST-GCN was trained on CUDA (RTX 4090) for 200 epochs;
final train Accuracy/Macro-F1 were 0.973077/0.973572 and post-hoc development
Val Accuracy/Macro-F1 were 0.484429/0.474513. No policy Test evaluation was
performed for this protocol.

## Temporal segment jitter follow-up

The reduced12 ST-GCN Train split now has a separate temporal-jitter runtime
variant: 30 segments with m=3 or m=5 ordered candidates, sampled one per
segment during training. The reduced12 development Val array is unchanged and
byte-identical to the base protocol. The independent checkpoint and metrics
are recorded under `experiments/stgcn_reduced12_temporal_segment_jitter_v1/`;
no policy Test evaluation was performed.

## Reduced-15 BABEL diversity protocol

2026-09-04: built an independent 15-class protocol using cap-300/100
diversity-first selection and the official Train 90/10 ST-GCN development
split. The new development data contain 2,926 Train and 325 Val records,
with fixed 30-frame resampling and no temporal segment jitter. A separate
ST-GCN was trained with the Conda `habitat` CUDA runtime on RTX 4090 and
train-only early stopping at epoch 164: final Train Accuracy/Macro-F1
0.997949/0.998383 and post-hoc Val Accuracy/Macro-F1 0.630769/0.665723.
Existing selected16 and reduced12 artifacts were not modified; no Test data
were generated or read.

## Revised reduced-15 replacement

2026-09-04: removed the old reduced15 runtime dataset/checkpoint (52M and 12M)
and regenerated a new independent 15-class protocol: walk, sit, stand up, bend,
crawl, stumble, wave, clap, throw, clean something, jump, kick, knock, punch,
touching face. Cap=300/100 and diversity-first sampling were preserved. Using
16 CUDA Habitat workers, fixed-30-frame Train/Val skeletons contain 2,796/310
records. The new ST-GCN (seed 42, train-only early stopping at epoch 172)
reached Train Accuracy/Macro-F1 0.995708/0.996167 and posthoc Val
Accuracy/Macro-F1 0.651613/0.672486. Test was not generated, read or evaluated.

## Reduced-16 BABEL replacement protocol

2026-09-04: replaced `stretch` and `take/pick something up` from the reduced15
set and added `bend`, `eat`, and `telephone call`, yielding an independent
16-class protocol. Cap=300/100 and diversity-first sampling were preserved;
Official Train was split 90/10 for ST-GCN development. Generated fixed
30-frame skeletons with 16 CUDA Habitat workers: Train=2,716, Val=301. The
new ST-GCN trained with seed 42 and train-only early stopping at epoch 184;
Train Accuracy/Macro-F1=0.997054/0.997775 and posthoc Val
Accuracy/Macro-F1=0.664452/0.620068. No Test data were generated/read.
## Reduced-15 wave-to-kneel replacement

2026-09-04: deleted only the prior revised reduced15 runtime dataset/checkpoint
(about 99M/12M; raw BABEL/AMASS untouched) and created an independent protocol
replacing `wave` with `kneel`. Cap=300/100, diversity-first selection, fixed
30-frame resampling, no temporal segment jitter, and 16 CUDA Habitat workers
were used. New ST-GCN development data contain 2,700 Train and 300 Val records.
Seed-42 CUDA training completed at epoch 200: Train Accuracy/Macro-F1
0.992963/0.991418 and posthoc Val Accuracy/Macro-F1 0.700000/0.710511.
No Test skeleton was generated, read, or evaluated.

## ActiveView Official Val correction

The initial active-sensing generation mistakenly materialized only the 20%
`activeview/val` subset (209 records); those generated artifacts were removed.
The complete cap-100 Official Val selection was rebuilt without reading Test
and generated under `activeview_official_val/` with 16 CUDA Habitat workers:
1,036 records, shape `[1036, 3, 30, 17, 1]`.

## Reduced-15 jump-to-wave replacement (current)

2026-09-04: ST-GCN was trained only on the Official Train selection (90/10
development split). Official Val is reserved for ActiveView and was not used
for ST-GCN optimization. The independent `reduced15_kneel_wave_babel_diversity_v1`
protocol contains ST-GCN Train/Val=2613/290 and ActiveView records Train/Val=620/209
plus complete Official-Val records=1036. Only ST-GCN skeleton generation used
16 CUDA Habitat workers; no ActiveView skeleton or Test skeleton was generated
or read. The
ST-GCN checkpoint `stgcn_reduced15_kneel_wave_best.pth` (SHA256
`47e17af04a24dfa0e671e737c17fb07aeaaa676e8c476c1cfeb8f9ca29dae272`) was
trained exclusively on `stgcn_development`.

No model was trained on the ActiveView dataset in this task, so there are no
ActiveView-trained weights to remove. The ActiveView Train/Val skeleton data
are retained as inputs for future active-perception training; frozen Stage-C
artifacts remain untouched.

The previously created ActiveView pure-color skeleton artifacts were removed;
the ActiveView directories now contain records/manifests only. The complete
Official-Val manifest is ordered by `record_id` for deterministic alignment.

## Reduced-15 wave-to-shake replacement (current)

2026-09-04: deleted only the generated `reduced15_kneel_wave_babel_diversity_v1`
runtime dataset and its ST-GCN checkpoint; raw BABEL/AMASS sources were left
untouched. The replacement protocol keeps 14 labels and replaces `wave` with
`shake`. Official Train/Val caps are 300/100 with diversity-first selection.
`raw-train` contains 2521/280 fixed-30-frame skeleton samples for ST-GCN
development; `raw-val` contains 1001 Official-Val records split 599/202/200
for ActiveView manifests only. Sixteen CUDA Habitat workers generated only
raw-train skeletons. Seed-42 train-only convergence stopped at epoch 160;
final Train Accuracy/Macro-F1=0.990877/0.989584 and posthoc development Val
Accuracy/Macro-F1=0.678571/0.674353. Test was not generated, read or evaluated.

## Reduced-14 shake removal (current)

2026-09-04: removed the generated `reduced15_kneel_shake_babel_diversity_v1`
runtime dataset and checkpoint, leaving raw BABEL/AMASS untouched. The new
14-class protocol removes `shake` and retains walk, sit, stand up, bend, crawl,
stumble, kneel, clap, throw, clean something, kick, knock, punch and touching
face. Cap=300/100 diversity-first selection produced raw-train 2430/270 fixed-
30-frame skeleton samples (Official Train 90/10) and raw-val 936 records-only
Official-Val entries split 560/189/187. Sixteen CUDA Habitat workers generated
raw-train skeletons; ST-GCN seed-42 train-only convergence stopped at epoch 169,
with Train Accuracy/Macro-F1=0.992593/0.991072 and posthoc development Val
Accuracy/Macro-F1=0.688889/0.684317. Test was not generated, read or evaluated.

## Reduced-14 raw-val cap-50 resampling (current)

2026-09-04: only the 14-class Official-Val records were resampled with the
per-class cap reduced from 100 to 50. raw-val now contains 597 selected
Official-Val records split 357/120/120 as records-only ActiveView manifests.
raw-train skeletons, ST-GCN checkpoint and all training metrics are unchanged.
The raw-val `test.json` is an Official-Val record partition, not policy Test;
policy Test was not read or evaluated.

## Current task: reduced14 eight-placement ActiveView retraining (completed)

The latest reduced14 + eight-placement dataset was used with the existing
raw-val Train/Val/Test=357/120/120 record split and no scene partition. WM-E
and Multi-Positive Joint Revision were retrained on Train only; frozen 14-class
ST-GCN supplied terminal recognition. Val and the explicitly requested final
Test evaluations, including NoMove, FrozenStageCv0, Random, SafeOracle,
CandidateOracle and Multi-positive H2, are recorded in
`experiments/reduced14_eight_placement_v1/active_view_retraining/`.

## Current positions-only preparation

The new furniture-anchored placement sampler is implemented at
`activeview/scripts/data/sample_hm3d_train_placements.py`. It generated and
validated eight placements for each of the 21 frozen HM3D-train scenes under
`datasets/offline/hm3d-train_reduced14_kneel/placement_sampling_v2/` (168 total),
using the 14-class raw-val manifest only as provenance. Skeleton generation is
intentionally not started; this task stops after coordinate generation.
