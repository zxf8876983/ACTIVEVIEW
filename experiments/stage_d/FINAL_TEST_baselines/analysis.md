# Final Test Baselines

| Method | Full Accuracy | Full Macro-F1 | Moving Accuracy | Moving Macro-F1 |
|---|---:|---:|---:|---:|
| NoMove | 0.412662 | 0.381786 | 0.262940 | 0.255476 |
| Random | 0.408741 | 0.382058 | 0.362632 | 0.349461 |
| FrozenStageCv0 | 0.625381 | 0.563705 | 0.574344 | 0.530501 |
| SafeOracle | 0.844925 | 0.811112 | 0.825486 | 0.800462 |
| Multi-positive H2 | 0.684841 | 0.627749 | 0.661388 | 0.622984 |

Moving subset is exactly the frozen Stage-C-v0 `predicted_stays=false` episode set.

- Multi-positive H2 − FrozenStageCv0 (full): ΔAccuracy +0.059460, ΔMacro-F1 +0.064044.
- Multi-positive H2 − FrozenStageCv0 (moving): ΔAccuracy +0.087044, ΔMacro-F1 +0.092483.
- SafeOracle − FrozenStageCv0 (full): ΔAccuracy +0.219544, ΔMacro-F1 +0.247407.
- SafeOracle − FrozenStageCv0 (moving): ΔAccuracy +0.251143, ΔMacro-F1 +0.269961.

No Test data were regenerated; this is an offline evaluation of frozen artifacts.
