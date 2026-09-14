# Budgeted Complementary Multi-View Active HAR

Experiment: Budgeted Complementary Multi-View Active HAR
Moving Val: 10080 contexts
Policy Test: false
Recognizer: frozen ST-GCN + frozen shared head
Task: select a complementary set of additional observation viewpoints under a finite observation budget
Observation budget: B = 2,3,4
O0: already acquired current full observation
Unvisited candidate observation used before selection: false
GT action: false for deterministic selection; oracle only for privileged diagnostics
Main selection information: viewpoint geometry, Train-derived viewpoint complementarity prior, navigation geometry
O0 semantic feature: not required by main methods
Fusion: fixed normalized MeanLogP
Continuous human/navigation synchronization: not modeled

## Reproduction gate
Random B2/B3/B4: 0.429762/0.487401/0.521230 Accuracy; StaticViewPairPrior B2: 0.508234; gate=PASS.

## Main findings
- Stay: Acc=0.302579, Macro-F1=0.292976, path=0.000m
- Random B2: Acc=0.429762, Macro-F1=0.424594, path=3.095m
- Random B3: Acc=0.487401, Macro-F1=0.477211, path=6.141m
- Random B4: Acc=0.521230, Macro-F1=0.505236, path=9.123m
- StaticViewPairPrior B2: Acc=0.508234, Macro-F1=0.497486, path=2.985m
- PairMeanGreedy B2: Acc=0.508234, Macro-F1=0.497486, path=2.985m
- PairMeanGreedy B3: Acc=0.557639, Macro-F1=0.546970, path=5.843m
- PairMeanGreedy B4: Acc=0.569544, Macro-F1=0.556675, path=8.587m
- Pair+Angular beta=0.5 B2: Acc=0.485714, Macro-F1=0.472266, path=4.017m
- Pair+Angular beta=0.5 B3: Acc=0.532837, Macro-F1=0.520427, path=7.699m
- Pair+Angular beta=0.5 B4: Acc=0.551488, Macro-F1=0.538083, path=10.952m
- Pair+GeoCoverage B3: Acc=0.536409, Macro-F1=0.523036, path=6.902m
- PairDiversityNav B3: Acc=0.510615, Macro-F1=0.500302, path=4.058m
- RelativeGeometryPrior B2: Acc=0.434226, Macro-F1=0.423513, path=3.824m
- RelativeGeometryPrior B3: Acc=0.493948, Macro-F1=0.482568, path=7.067m
- SmoothedSet3Prior B3: Acc=0.550298, Macro-F1=0.540837, path=5.856m
- Exact GT-Margin Oracle B2: Acc=0.706647, Macro-F1=0.701819, path=3.066m
- Exact GT-Margin Oracle B3: Acc=0.738591, Macro-F1=0.737738, path=6.078m
- Exact GT-Margin Oracle B4: Acc=0.721032, Macro-F1=0.719061, path=8.986m

Best deterministic policy: PairMeanGreedy B4.
PairMeanGreedy marginal Acc gains: ΔB2=+20.565 pp, ΔB3=+4.940 pp, ΔB4=+1.190 pp.
B3 versus B2: +4.940 pp; B4 versus B3: +1.190 pp.

Pair complementarity is reported as a Train-derived prior; no Val labels or candidate evidence are used to choose formal deterministic actions. Exact oracle sets are privileged diagnostics only.

## Additional registered diagnostics
- MaxMinAngular: B2=0.461706/0.447082; B3=0.526488/0.510574; B4=0.553770/0.536711
- OppositeAzimuth: B2=0.461706/0.447082; B3=0.526488/0.510574; B4=0.553770/0.536711
- GeometricCoverage: B2=0.460020/0.447390; B3=0.510813/0.496662; B4=0.535813/0.518908
- PairMinGreedy: B2=0.508234/0.497486; B3=0.551885/0.541838; B4=0.564583/0.552544
- PairMaxGreedy: B2=0.508234/0.497486; B3=0.553075/0.542520; B4=0.564087/0.552408
- Pair+GeoCoverage: B2=0.468155/0.454706; B3=0.536409/0.523036; B4=0.559325/0.543757
- PairDiversityNav: B2=0.464087/0.456470; B3=0.510615/0.500302; B4=0.539583/0.526424
- PairNav lambda=0.1: B2=0.460119, path=1.634m; B3=0.504663, path=3.547m; B4=0.529762, path=5.814m
- PairNav lambda=0.25: B2=0.417460, path=1.187m; B3=0.472321, path=3.054m; B4=0.506746, path=5.411m
- PairNav lambda=0.5: B2=0.396429, path=1.065m; B3=0.461111, path=2.972m; B4=0.499306, path=5.349m
- Pair+Angular betas: β=0.25: B2=0.500000, B3=0.544048, B4=0.560813, β=0.5: B2=0.485714, B3=0.532837, B4=0.551488, β=1.0: B2=0.468155, B3=0.520734, B4=0.538393
- Set3 versus PairMeanGreedy B3: -0.734 pp.
- RelativeGeometryPrior versus StaticViewPairPrior: B2=-7.401 pp; B3=0.493948.
- Pair structure: angular Spearman=0.016523, radius Spearman=0.008759, matrix symmetry Spearman=0.815975, mean abs asymmetry=0.077040.
- AnyCorrect-fusion coverage: B2=0.706647 (7123), B3=0.738591 (7445).
- Rescue decomposition Random B2: all-single-wrong→correct=208, single-correct→fusion-wrong=1032.
- Rescue decomposition Random B3: all-single-wrong→correct=179, single-correct→fusion-wrong=1529.
- Rescue decomposition PairMeanGreedy B2: all-single-wrong→correct=221, single-correct→fusion-wrong=1035.
- Rescue decomposition PairMeanGreedy B3: all-single-wrong→correct=147, single-correct→fusion-wrong=1468.
- Rescue decomposition PairMeanGreedy B4: all-single-wrong→correct=136, single-correct→fusion-wrong=1747.
- High-occlusion and clutter subsets, per-class budget curves, full navigation costs, Pareto records, and all policy metrics are stored in the corresponding JSON artifacts.

## Final decision: **KEEP COMPLEMENTARY SET-SELECTION**

The report separates exact set oracles from incremental greedy policies and records any B4 dilution. The next step, if any, should be chosen explicitly; no follow-up experiment is launched automatically.
