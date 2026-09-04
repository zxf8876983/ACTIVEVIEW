# Reduced-12 protocol and training record

The protocol was built from BABEL official Train and Val annotations using seed
42. Selection is diversity-first: a candidate receives priority for an unseen
source, subject, AMASS dataset, and duration bin, with deterministic repeat
penalties. Rare classes are retained below the cap when readable duration-valid
records are unavailable.

| subset/split | records | unique sources | subjects | AMASS datasets | duration (s) | mean (s) |
|---|---:|---:|---:|---:|---:|---:|
| ST-GCN train | 2600 | 1361 | 370 | 17 | 9580.093 | 3.6847 |
| ST-GCN val | 289 | 252 | 146 | 13 | 1229.116 | 4.2530 |
| ActiveView train | 591 | 388 | 198 | 14 | 2522.670 | 4.2685 |
| ActiveView val | 197 | 167 | 108 | 14 | 725.616 | 3.6833 |
| ActiveView test | 197 | 172 | 114 | 14 | 702.683 | 3.5669 |

The existing generator produced tensors with shape `[N, 3, 30, 17, 1]`. A
single RTX 4090 (Conda `habitat` environment) was used. ST-GCN was trained for
the fixed 200 epochs with the repository's train-only convergence protocol;
the reported Val numbers are post-hoc diagnostics and were not used for model
selection.

Final training metrics: loss `0.044019`, Accuracy `0.973077`, Macro-F1
`0.973572`. Post-hoc development Val: Accuracy `0.484429`, Macro-F1 `0.474513`.
The independent 12-class checkpoint and generated arrays are runtime artifacts
outside Git. The ActiveView motion `test` partition is generated for the
requested split only; no policy/Test evaluation was performed.
