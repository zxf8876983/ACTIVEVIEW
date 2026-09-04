# Handoff

Status: CLEAN

## Canonical method

The frozen final method is **WM-E + Multi-Positive Joint Revision + Closed-Loop
H2** with horizon 2, `ALL_LEGAL` candidates, frozen Stage-C-v0 initial policy,
frozen WM-E/JR/ST-GCN, visited-view exclusion, current-viewpoint-centered
geometry, and real archived terminal skeleton HAR.

## Final Test

The explicitly authorized Final Test is complete. `experiments/stage_d/FINAL_TEST/`
records `test_used=true`, FULL=13,774 and MOVING=9,409. The final Multi-positive
method reached FULL Accuracy/F1 0.684841/0.627749 and MOVING 0.661388/0.622984.
H1_REAL reached 0.673515/0.623050 FULL and 0.644808/0.612497 MOVING; Original
JR H2 reached 0.680558/0.629111 FULL and 0.655117/0.621852 MOVING. Relative to
FrozenStageCv0, Multi gained +5.946pp/+6.404pp FULL and +8.704pp/+9.248pp
MOVING.

For the same Test population, FrozenStageCv0 was FULL Accuracy/F1
0.625381/0.563705 and MOVING 0.574344/0.530501. The NoMove baseline was
0.412662/0.381786 FULL and 0.262940/0.255476 MOVING; SafeOracle was
0.844925/0.811112 FULL and 0.825486/0.800462 MOVING.

## Latest verification

The source consolidation was followed by a read-only equivalence audit in
`experiments/refactor_regression/result.json`: status PASS, with NoMove,
Random seed 42, FrozenStageCv0, SafeOracle, H1_REAL, Original JR H2 and
Multi-positive JR H2 matching frozen golden Accuracy/F1 values within 1e-8 on
the same Test populations. This audit did not alter official results.

The latest frozen artifact hashes are WM-E
`db2573a013ed9a7fab87561ad26800334556894b96e69dd3d498464794d9b5e6`, Original
JR `332b3127747f67d954d7c80f530ee1cc5a9ca30c6472fd13a3a010a080c413ac`, Multi
JR `8a6ef93ded8df94154f2045d6cf7d297c23e587ac8cf2601a83fcf3c82f1383c`, and
ST-GCN `362ac23195688988d637244eb2a13fa0e7b563b21d143846c671a5cec6b0d0ce`.

## Runtime and research boundaries

Runtime datasets, checkpoints and caches remain external under
`ACTIVEVIEW_DATA_ROOT`; source code is under `activeview/`. Test artifacts are
frozen and no new training, Val/Test evaluation, perception regeneration or
Habitat rendering should start automatically. `test_used=false` applies to the
research experiments (EXP051-R2, EXP055 and EXP056); `test_used=true` applies
only to the explicitly authorized official Final Test. No next experiment is
currently authorized.
