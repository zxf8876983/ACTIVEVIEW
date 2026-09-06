# Current Task

## Pretrained History Identity -> Multi-positive JR — completed 2026-09-07

Implemented and ran the reduced14 Train/Val-only experiment. A standalone
540->256->128 history identity encoder was trained for 20 epochs with Val
identity Macro-F1 checkpoint selection, then initialized two independent JR
variants for 20 epochs: `PretrainedFrozenJR` and `PretrainedFinetuneJR`.

Results are recorded in
`experiments/reduced14_eight_placement_v1/pretrained_history_jr/`.
Train contexts: 44,248. Val moving contexts: 14,809. Test was not read.
The formal WM-E, ST-GCN, taxonomy, split and prior JR checkpoints remain
unchanged. Runtime checkpoints are stored outside Git under
`ACTIVEVIEW_DATA_ROOT/checkpoints/activeview_reduced14_eight_placement_v1/`.

Status: CLEAN. No follow-up experiment is authorized automatically.
