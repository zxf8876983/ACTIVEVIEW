# Notes: O0-conditioned complementary second-view sweep

## Protocol
- Moving Val has 10,080 contexts; Policy Test must remain unopened.
- Action set is current/Stay plus Stage-A legal candidate pool. B2 selects exactly one legal candidate (Stay is not selectable).
- Pair fusion is normalized `0.5 * (logp0 + logpc)` followed by log-softmax; terminal prediction is the frozen shared-head argmax.

## Existing assets
- Frozen reduced12 ST-GCN cache and provenance are under `diagnostics/reduced12_dual_route_overnight/`.
- Shared multi-view adapted head is under the reduced12 policy checkpoint directory.
- Existing sweep scripts/loaders expose row signatures, candidate geometry, cache validation, and record-balanced sampling.

## Findings
- Cache provenance passed for 46,324 Train and 10,080 Moving-Val rows; formal B2 candidate pool is current + Stage-A legal candidates, while `val_all32` is provenance-only.
- Row geometry is the existing 11-D Stage-C descriptor. The runner augments it with current/candidate 3-D lattice features and a zero Stay flag, yielding the existing 18-D option geometry.
- CUDA preflight passed on NVIDIA GeForce RTX 4090 (torch 2.6.0+cu124).
- First B2 sweep completed: best learned branch was Feature+Posterior+Geometry-PairLogP at 0.517361 accuracy / 0.504616 Macro-F1; Random-B2 was 0.429762 / 0.424594; PairMargin Oracle was 0.706647 / 0.701819. Since the best branch passed the 0.50 conditional gate, B3 must be run before finalizing.
- Final B3 completed: best-B2 plus random third view is 0.533730 Acc / 0.521124 Macro-F1; learned B2 plus learned B3 is 0.558036 / 0.548492; privileged B3 margin oracle is 0.669544 / 0.665012. The learned third-view increment over the random third-view control is +2.431 pp.
- The final rerun validates the frame-0 visibility cache ids and masks against the Val option cache before stratification; no stale action-slot alignment is accepted.
