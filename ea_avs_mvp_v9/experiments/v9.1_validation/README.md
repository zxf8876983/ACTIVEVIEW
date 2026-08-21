# ACTIVEVIEW v9.1: Perception-Aware Active View Selection

This directory contains the scientific validation experiment suite for **ACTIVEVIEW v9.1: Perception-Aware Active View Selection**.

## Core Scientific Problem
Under realistic incomplete observations (caused by environment occlusions, self-occlusions, and viewpoint limitations), the robot cannot directly access ground truth human states. The robot actively selects the next viewpoint to maximize future human state estimation quality and **Information Gain**.

## Directory Structure
```text
v9.1_validation/
├── README.md                                # Overview of validation suite
├── V91_EXPERIMENT_REPORT.md                 # Full scientific experimental report
├── training/
│   └── training_result.json                 # 40-epoch loss and top-1 accuracy logs
├── baseline/
│   └── comparison_report.json               # 5-baseline quantitative comparison
├── perception_degradation/
│   └── degradation_report.json              # 4 perception degradation benchmarks
├── information_gain/
│   └── gain_report.json                     # Missing joints recovery and info gain
├── ablation/
│   └── ablation_report.json                 # 4-way feature ablation evaluation
├── analysis/
│   └── human_state_view_dependency.json     # Viewpoint dependency across human states
└── visualization/
    ├── training_curve.png                   # Training convergence curves
    ├── viewpoint_ranking.png                # Information gain ranking & polar layout
    ├── best_view_examples.png               # Confidence gains across degradation cases
    └── body_visibility_analysis.png         # 7-part anatomical confidence improvement
```
