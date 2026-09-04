# Analysis

The regenerated Train pool contains 150 frame slots (30 segments × 5), with
per-record counts tracked separately. Current records use `m=3` for 1,748
motions and `m=5` for 852 motions; no interval was at most 30 frames. The
loader selects one candidate in every segment using PyTorch randomness on each
Train access and uses the middle candidate for train-only convergence
diagnostics. This preserves the required 30-frame ST-GCN input shape while
varying temporal locations across epochs.

The development Val array was copied byte-for-byte from the base protocol
(SHA256 `53260e2be72ba93ee3e18aaf20669a580d6db44fd52d4b72be00fa1972351510`)
and remains fixed uniform sampling. Candidate pool
generation required 150 rendered/perceived frames for `m=5` records, so its
generation cost is approximately proportional to `m` rather than the full raw
motion length.

The temporal-jitter checkpoint was trained on CUDA/RTX 4090 for the same fixed
200-epoch train-only protocol. Final Train metrics were loss `0.044569`,
Accuracy `0.972692`, Macro-F1 `0.972863`. Post-hoc fixed Val metrics were
Accuracy `0.480969`, Macro-F1 `0.455983`; Val was not used for training or
selection. No policy Test evaluation was performed.
