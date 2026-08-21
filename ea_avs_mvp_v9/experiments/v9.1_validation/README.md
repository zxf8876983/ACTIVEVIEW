# ACTIVEVIEW v9.1 Validation Experiments

This directory contains the complete scientific validation experiment artifacts for **ACTIVEVIEW v9.1: Human-state-aware Learnable Active View Selection**.

## Directory Structure
```text
v9.1_validation/
├── README.md                           # Overview of validation suite
├── V91_EXPERIMENT_REPORT.md             # Full scientific experimental report
├── training/
│   └── training_result.json             # 40-epoch loss and top-1 accuracy logs
├── baseline/
│   └── comparison_report.json           # 5-baseline quantitative comparison
├── ablation/
│   └── ablation_report.json             # 4-way feature ablation evaluation
├── analysis/
│   └── human_state_view_dependency.json # Viewpoint dependency across 5 human states
└── visualization/
    ├── training_curve.png               # Training convergence curves
    ├── viewpoint_ranking.png            # Viewpoint quality ranking and polar layout
    ├── best_view_examples.png           # Multi-state optimal viewpoint angles
    └── body_visibility_analysis.png     # 7-part anatomical visibility comparison
```
