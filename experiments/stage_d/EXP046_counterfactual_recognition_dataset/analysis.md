# EXP046

Frozen WM-E (variant E, residual=false) generated 32-view recognition states
for 29,133 Train and 9,742 Val contexts.  The EXP014 evaluator gate passed
(Accuracy 0.6582540931, Macro-F1 0.6101526052).  Val imagined→true ST-GCN
agreement was 0.605901; true-label log-probability Pearson/Spearman were
0.745240/0.775903.  Current-s1-correct contexts had agreement 0.657467,
whereas current-s1-wrong contexts had 0.525615.  Train entropy quartile
thresholds were [0.016225, 0.193103, 0.693773]; Val agreement decreased from
0.688045 (Q1) to 0.524185 (Q4).  Full per-class and conditional values are in
`conditional_fidelity.json`; large caches remain outside Git.

Leakage audit: WM-E/ST-GCN frozen, true future recognition evaluator/target
only, future RGB unused, Test unused.

The Train/Val caches are the only large artifacts; their SHA256 values and
checkpoint provenance are recorded in `result.json` and `cache_audit.json`.
