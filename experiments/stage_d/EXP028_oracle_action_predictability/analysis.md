# EXP028 — Oracle Action Predictability / Representation Sufficiency Audit

Val-only diagnostic; no policy was trained and Test was not read.

## Table A — Margin bins

| margin | count | EXP027 3-way acc | binary acc | candidate hit | NN25 entropy |
|---|---:|---:|---:|---:|---:|
| [0.0, 0.05) | 3522 | 0.4730266893810335 | 0.6433844406587166 | 0.3852924474730267 | 0.9224027767979616 |
| [0.05, 0.1) | 458 | 0.4606986899563319 | 0.6331877729257642 | 0.3864628820960699 | 0.9037793312180777 |
| [0.1, 0.25) | 653 | 0.44869831546707506 | 0.6416539050535988 | 0.3568147013782542 | 0.9119314202449483 |
| [0.25, 0.5) | 580 | 0.44482758620689655 | 0.596551724137931 | 0.39482758620689656 | 0.8974258666168521 |
| [0.5, 1.0) | 708 | 0.4646892655367232 | 0.6242937853107344 | 0.4265536723163842 | 0.8991272162816127 |
| [1.0, 2.0) | 995 | 0.46733668341708545 | 0.6150753768844222 | 0.4663316582914573 | 0.8788969722971072 |
| [2.0, +inf) | 2826 | 0.562278839348903 | 0.6748053786270347 | 0.5552016985138004 | 0.7766045023583765 |

## Table B — NN agreement

| k | 3-way accuracy | binary accuracy |
|---:|---:|---:|
| 1 | 0.454527 | 0.607473 |
| 5 | 0.449497 | 0.569698 |
| 10 | 0.451139 | 0.583761 |
| 25 | 0.444570 | 0.573599 |

## Table C — High-margin subsets

| subset | count | 3-way acc | binary acc | candidate hit |
|---|---:|---:|---:|---:|
| S1 (≥0.25) | 5109 | 0.5169309062438834 | 0.647289097670777 | 0.6156953260242354 |
| S2 (≥0.5) | 4529 | 0.526164716272908 | 0.6537867078825348 | 0.6138944555778223 |
| S3 (≥1.0) | 3821 | 0.5375556137136875 | 0.6592515048416645 | 0.6141078838174274 |

## Local consistency

{
  "three_way_mean": 0.5013795935126258,
  "three_way_median": 0.48,
  "binary_mean": 0.6592650379798809,
  "binary_median": 0.64,
  "three_way_ge_0.8": 0.054300964894272226,
  "three_way_ge_0.9": 0.026277971669061793
}

## Action-class breakdown

{
  "oracle_stay": {
    "count": 4265,
    "correct_stay": 2608,
    "false_move_rate": 0.38851113716295427,
    "false_move_mean_regret": 3.3483659644750206
  },
  "oracle_move": {
    "count": 5477,
    "move_recall": 0.6708051853204309,
    "false_stay_rate": 0.3291948146795691
  },
  "oracle_p2": {
    "count": 3185,
    "candidate_hit": 0.4087912087912088,
    "false_stay_rate": 0.3469387755102041
  },
  "oracle_p3": {
    "count": 2292,
    "candidate_hit": 0.3931064572425829,
    "false_stay_rate": 0.3045375218150087
  }
}

## Scientific interpretation

EXP028 is an analysis-only audit. The registered Case A/B/C label remains INCONCLUSIVE until the observed margin-stratified accuracy, NN agreement and entropy are reviewed together. These results assess predictability under the frozen legal representation and held-out-motion protocol; they do not establish that future viewpoints are intrinsically unpredictable.

## Leakage flags

- future_candidate_rgb_used=false
- future_candidate_depth_used=false
- future_candidate_skeleton_used=false
- true_utility_used_as_model_input=false
- val_used_for_feature_normalization=false
- val_used_for_neighbor_index=false
- test_used=false
