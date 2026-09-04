# Current Task

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
