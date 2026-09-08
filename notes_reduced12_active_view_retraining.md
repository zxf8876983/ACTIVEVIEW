# Notes: Reduced12 ActiveView retraining

## Known assets

- Motion source: `datasets/reduced12_no_kneel_clean_babel_diversity_v1/raw-val/`
- Train/Val source manifests: `train.json` and `val.json` only; policy Test is out of scope.
- Offline skeleton output: `datasets/offline/habitat-train/00006-00087/`
- Frozen 12-class ST-GCN: `checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/stgcn_reduced12_no_kneel_clean_best.pth`
- Labels: walk, sit, stand up, bend, crawl, stumble, clap, throw, kick, knock, punch, touching face.

## Decisions

- Derived policy artifacts and checkpoints will live under independent reduced12 roots.
- Any compatibility change must be a minimal dynamic-path/class-count change; no algorithmic changes.

## Completed run

- Stage A: 46,324 Train and 15,540 Val episodes; 30,580/10,080 v0-moving Stage-D contexts.
- Stage B: 305,175 Train and 102,375 Val candidate utility rows.
- Stage C: frozen 12-class ST-GCN features; Test file remained empty and unused.
- Stage D: 20-scene pairwise navigation cache and Train/Val feature rows generated.
- WM-E: 12 epochs, best Val loss 0.1572399 at epoch 11; checkpoint is independent under the reduced12 checkpoint root.
- Multi-positive JR: 20 epochs, final Train loss 1.554696; independent reduced12 checkpoint.
- Val Full: FrozenStageCv0 0.459331/0.449009, Multi-positive H2 0.487259/0.477739, H0 SafeOracle 0.730824/0.728527.
- Val Moving: FrozenStageCv0 0.454266/0.444782, Multi-positive H2 0.497321/0.492134, H0 SafeOracle 0.727976/0.721495.
- WM diagnostics: agreement 0.371491, Pearson 0.360785, Spearman 0.403005, Top-1/Top-3 positive hit 0.498301/0.739231.
- No policy Test was read. The current eight-placement assets have no DINO cache; this run records `use_rgb=false` rather than silently fabricating RGB inputs.
