# Reduced12 selector ceiling decomposition

Train/Val only; policy Test was not read. All terminal predictions use the selected real archived observation and frozen reduced12 ST-GCN.

## Moving Val comparison

| Method | Accuracy | Macro-F1 | Positive-action hit | Stay rate |
|---|---:|---:|---:|---:|
| Frozen H1 | 0.454266 | 0.444782 | 0.454266 | 1.000000 |
| Imagined+Inferred JR | 0.535615 | 0.536320 | 0.535615 | 0.372520 |
| Real+Inferred JR | 0.598313 | 0.606791 | 0.598313 | 0.121925 |
| Imagined+GT JR | 0.712103 | 0.696786 | 0.712103 | 0.446925 |
| Real+GT JR | 0.908532 | 0.904704 | 0.908532 | 0.092956 |
| CurrentLabel-Real | 0.454167 | 0.444633 | 0.454167 | 0.256250 |
| GTLabel-Imagined | 0.678274 | 0.663142 | 0.678274 | 0.401687 |
| FixedH1-H2 Oracle | 0.911210 | 0.908047 | 0.911210 | 0.082242 |

## Full Val comparison

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Frozen H1 | 0.459331 | 0.449009 |
| Imagined+Inferred JR | 0.512098 | 0.506333 |
| Real+Inferred JR | 0.552767 | 0.554133 |
| Imagined+GT JR | 0.626577 | 0.611871 |
| Real+GT JR | 0.753990 | 0.748774 |
| CurrentLabel-Real | 0.459266 | 0.448915 |
| GTLabel-Imagined | 0.604633 | 0.589634 |
| FixedH1-H2 Oracle | 0.755727 | 0.751001 |

## Gap decomposition (percentage points)

- WM gap (Real+Inferred − Imagined+Inferred): **6.270 pp**.
- Identity gap (Real+GT − Real+Inferred): **31.022 pp**.
- Imagined identity gain (Imagined+GT − Imagined+Inferred): **17.649 pp**.
- Architecture residual (FixedH1-H2 Oracle − Real+GT): **0.268 pp**.
- Current H2 − Frozen H1: **8.135 pp Accuracy / 9.154 pp Macro-F1**.
- Real+Inferred − Frozen H1: **14.405 pp Accuracy / 16.201 pp Macro-F1**.
- Real+GT − Frozen H1: **45.427 pp Accuracy / 45.992 pp Macro-F1**.

## Interpretation

Case B is primary: Real+GT is only 0.268 pp below the fixed oracle, while Real+Inferred loses 31.022 pp; action identity/belief inference is the largest bottleneck. Case A also applies as a secondary effect because Real+Inferred exceeds Imagined+Inferred by 6.270 pp, so WM candidate fidelity remains material.

## Protocol and leakage

- taxonomy: reduced12 (walk, sit, stand up, bend, crawl, stumble, clap, throw, kick, knock, punch, touching face)
- candidate budget: ALL_LEGAL; visited viewpoints excluded
- `test_used=false`; no Test path or Test artifact was read
- privileged Real/GT branches are diagnostics only and are not deployable methods
