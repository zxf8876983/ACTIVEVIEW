# Reduced14 + Eight-Placement ActiveView Retraining

The raw-val record split was preserved exactly as Train/Val/Test = 357/120/120
(57,834 / 19,440 / 19,440 contexts); no scene split was introduced. The
14-class ST-GCN recognizer remained frozen. WM-E and Multi-Positive Joint
Revision were retrained on Train only, with seed 42 and final-epoch weights.

| Split / population | NoMove | FrozenStageCv0 | Random | SafeOracle | Multi-positive H2 |
|---|---:|---:|---:|---:|---:|
| Val Full Accuracy | 0.299640 | 0.418879 | 0.305864 | 0.769907 | 0.467850 |
| Val Full Macro-F1 | 0.283813 | 0.405564 | 0.298433 | 0.763109 | 0.459003 |
| Val Moving Accuracy | 0.258559 | 0.415085 | 0.266730 | 0.875886 | 0.479371 |
| Val Moving Macro-F1 | 0.213871 | 0.396831 | 0.245809 | 0.873737 | 0.472555 |
| Test Full Accuracy | 0.313272 | 0.426698 | 0.311728 | 0.782870 | 0.470319 |
| Test Full Macro-F1 | 0.302466 | 0.418769 | 0.306334 | 0.777816 | 0.468083 |
| Test Moving Accuracy | 0.278551 | 0.427558 | 0.276524 | 0.895459 | 0.484863 |
| Test Moving Macro-F1 | 0.238991 | 0.411638 | 0.252854 | 0.893287 | 0.481803 |

The latest method is above FrozenStageCv0 on both Val and Test, but remains
well below SafeOracle. On Test Full, Multi-positive H2 − FrozenStageCv0 is
+0.043621 Accuracy and +0.049314
Macro-F1; on the common moving subset the differences are +0.057305 and
+0.070165. SafeOracle remains the action-selection ceiling, indicating that
the retrained learned selector does not yet recover the available utility.

NoMove, Random, FrozenStageCv0, SafeOracle, CandidateOracle and Multi-positive
H2 all use the same frozen initial policy and the same 14-class ST-GCN
recognizer for terminal recognition. Test was read only after training and
checkpoint freeze for the requested final evaluation.
