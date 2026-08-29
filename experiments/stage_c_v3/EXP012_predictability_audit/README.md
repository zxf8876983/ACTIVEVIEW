# EXP012 — Candidate Utility Predictability Audit

## Status

PLANNED — read-only Train-reference/Val-query analysis prepared; it has not
been executed.

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
`predictability_audit.json` under the external runtime root. It has not been
executed and never reads Test.
