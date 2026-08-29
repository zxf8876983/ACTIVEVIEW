# EXP012 — Candidate Utility Predictability Audit

## Status

COMPLETED — read-only Train-reference/Val-query analysis; Test was not used.

## Scientific question

How deterministic is future utility given the information available to an
online policy? The audit compares geometry-only input with the existing
observable 19-D current semantic context plus 11-D candidate geometry.

## Protocol

For every Val candidate, the k=5 nearest Train candidates are used. All
normalization statistics come from Train only; Val candidates never query Val
utilities. High disagreement is defined once as the interquartile range (IQR)
of the Train utility distribution. A same-neighborhood sign conflict means the
five Train neighbours contain both positive (`utility > 1e-6`) and non-positive
utility.

The report includes neighbour utility standard deviation, absolute disagreement,
sign agreement, Pearson/Spearman correlation, MAE and compact groupings by Val
regret group, action class, candidate-radius bin and absolute azimuth bin.

## Run

`run.sh` performs only the Train-reference/Val-query analysis and writes
`predictability_audit.json` under the external runtime root. It never reads
Test.

## Val result

Using `306,869` Train candidate references and `102,637` Val queries (`k=5`):

| Input | Sign agreement | Sign-conflict rate | High-disagreement rate | MAE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Geometry-only (11-D) | 63.62% | 72.86% | 76.29% | 4.27625 | 0.55544 | 0.46967 |
| Observable state + geometry (30-D) | 67.40% | 68.51% | 74.09% | 4.17351 | 0.57298 | 0.50326 |

Adding the legal current observable state gives a modest predictability gain,
but substantial neighbourhood disagreement remains, especially in the
high-regret group (sign agreement `57.74%`, high-disagreement rate `79.77%`).
The runtime report is
`ACTIVEVIEW_DATA_ROOT/experiments/stage_c_v3/EXP012_predictability_audit/predictability_audit.json`.
