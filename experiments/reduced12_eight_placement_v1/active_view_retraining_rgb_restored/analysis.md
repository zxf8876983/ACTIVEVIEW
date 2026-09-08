# Reduced12 RGB-restored ActiveView retraining

## Protocol

This run restores the pre-later-method formal protocol on the reduced12,
eight-placement Train/Val assets:

`Recognition-aware WM-E (belief + visited DINO spatial RGB) + formal
Multi-positive Joint Revision + closed-loop H2`.

The frozen 12-class ST-GCN teacher was not retrained or modified. WM-E uses
`use_belief=true`, `use_rgb=true`, `residual=false`, `target_scope=all`, and
the loss `SmoothL1 + 0.25 velocity + 0.10 frozen ST-GCN recognition KL`.
JR uses the original multi-positive objective with `0.25 BCE + 0.05 posterior
CE`. All reported numbers are Val-only; no policy Test file was read.

## Val benchmark

| Method | Full Accuracy | Full Macro-F1 | Moving Accuracy | Moving Macro-F1 |
|---|---:|---:|---:|---:|
| NoMove | 0.329601 | 0.329175 | 0.254266 | 0.235500 |
| Random | 0.341570 | 0.341245 | 0.330060 | 0.327350 |
| FrozenStageCv0 | 0.459331 | 0.449009 | 0.454266 | 0.444782 |
| H0 CandidateOracle | 0.699936 | 0.698727 | 0.709325 | 0.704330 |
| H0 SafeOracle | 0.730824 | 0.728526 | 0.727976 | 0.721495 |
| Multi-positive H2 (RGB-restored) | **0.512098** | **0.506333** | **0.535615** | **0.536320** |
| FixedH1-H2 SafeOracle | 0.755727 | 0.751001 | 0.911210 | 0.908047 |

Relative to FrozenStageCv0, Multi-positive H2 improves Full Accuracy by
`+5.277 pp` and Full Macro-F1 by `+5.732 pp`; on Moving it improves by
`+8.135 pp` and `+9.154 pp`, respectively.

The H0 diagnostics are separate from FixedH1-H2: H0 SafeOracle is selected
from s0 over stay plus dynamically legal candidates, while FixedH1-H2 uses
the fixed H1/H2 oracle protocol.

## WM-E diagnostics

| Metric | Previous reduced12 no-RGB | RGB-restored | Delta |
|---|---:|---:|---:|
| Recognition agreement | 0.371491 | 0.542629 | +0.171138 |
| True-class Pearson | 0.360785 | 0.522995 | +0.162210 |
| True-class Spearman | 0.403005 | 0.634493 | +0.231488 |
| Top-1 positive hit | 0.498301 | 0.660090 | +0.161789 |
| Top-3 positive hit | 0.739231 | 0.827579 | +0.088348 |

The visited RGB restoration substantially recovers candidate-level WM fidelity
and ranking. The resulting H2 also improves over the no-RGB frozen baseline
used in the earlier reduced12 run (that earlier result is retained and was
not overwritten).

## Data and training

- Stage-D contexts: Train 30,580; Val 10,080; Full Val 15,540.
- Visited RGB: 40,660 records / 81,320 s0+s1 viewpoints; no missing or extra
  records in the path audit; `future_candidate_rgb_rendered=false`.
- DINO cache: 81,320 unique observations, Train 61,160 / Val 20,160,
  `[16,768]` float16, `facebook/dinov2-base`, extraction 392.58 s.
- WM-E: 12 epochs, seed 42, best Val loss 0.135367.
- JR: 20 epochs, seed 42, final loss 1.311041.
- H0 Val diagnostics: AnyCorrect 0.731338; mean correct-view ratio 0.335912;
  mean correct legal views 2.251737; mean legal-view count 6.587838.

## Integrity and leakage

The reduced12 Stage-B frozen ST-GCN checkpoint SHA256 is
`078cff9490fb51fe52ebaa7d3fa4ac2802ff6c3eebba44de71c8276cd5e86a71`.
A random Val archive audit (24 rows / 149 viewpoints) matched every stored
Stage-B predicted label; maximum stored true-class log-probability deviation
was `0.000950` (serialization precision). The reduced12 scene population is
20 scenes; it differs from the historical reduced14 population and was not
silently mixed. H0 candidate counts are min/mean/median/max =
1 / 6.587838 / 6 / 21, and remaining H2 candidates are
1 / 1.919841 / 2 / 2.

`test_used=false`, `future_candidate_rgb_used=false`, and no Test artifact was
read. No skeleton/perception data were regenerated; only visited RGB and its
DINO cache were added.

## Scientific conclusion

Restoring the formally specified visited s0/s1 RGB context clearly improves
WM-E recognition fidelity and candidate ranking, and yields a stable Val gain
over FrozenStageCv0. The remaining gap to H0 SafeOracle (0.730824 Full
Accuracy) is therefore not explained by the reduced12 recognizer alone; it is
shared between candidate-future modeling and selector/objective quality.
No later ranking-aware, action-discriminative, history-identity, privileged,
or GT-conditioned method was introduced in this run.
