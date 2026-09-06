# Current Task

## Ranking-aware Recognition WM-E — completed 2026-09-07

Implemented and ran the reduced14 Train/Val-only ranking-aware WM-E
experiment. The existing skeleton world model was augmented with a frozen
ST-GCN-supervised 14-way candidate recognition head and fixed KL/ranking terms;
the checkpoint was selected by Val candidate-ranking Spearman. Train and Val
counterfactual caches were rebuilt from that checkpoint, and the unchanged
Pretrained-Frozen History-aware JR was retrained on the new imagined
recognition cache.

Results are recorded in
`experiments/reduced14_eight_placement_v1/ranking_aware_wm_e/`.
Train contexts: 44,248. Val contexts: 14,809. Test was not read. The formal
WM-E, ST-GCN, taxonomy, split and prior checkpoints remain unchanged. Runtime
checkpoints and caches are stored outside Git under
`ACTIVEVIEW_DATA_ROOT/checkpoints/` and
`ACTIVEVIEW_DATA_ROOT/datasets/.../counterfactual_cache/ranking_aware_wm_e/`.

Status: CLEAN. No follow-up experiment is authorized automatically.
