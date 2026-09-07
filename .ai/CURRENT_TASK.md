# Current Task

## WM-imagined H1 Disambiguation — completed 2026-09-07

Implemented and ran a Val-only diagnostic using the frozen old WM-E and
ranking-aware WM-E. For every one of 14,809 reduced14 moving contexts, all
441,283 legal H1 candidates were imagined from archived s0, passed through
frozen ST-GCN and the frozen pretrained History Identity encoder, and then
evaluated with the selected candidate's real archived observation.

Old WM-E imagined min-entropy/max-margin H1 reached Accuracy/F1
0.345668/0.365998 and 0.327166/0.346189. Ranking-aware WM-E reached
0.395840/0.404250 and 0.394220/0.402601. Frozen H1 was
0.482815/0.496501; privileged real min-entropy/max-margin were
0.512594/0.504617 and 0.515970/0.510083. IdentityOracle reached
0.899588/0.900413.

Imagined-versus-real belief alignment was weak at candidate selection level:
old entropy Pearson/Spearman 0.559986/0.477783 and min-entropy overlap
0.048281; ranking-aware 0.531035/0.528160 and overlap 0.052941. The
diagnostic used CUDA, only Val archives/caches, trained no model, modified no
formal checkpoint and did not read Test. Results are in
`experiments/reduced14_eight_placement_v1/wm_imagined_h1_disambiguation/`.

Status: CLEAN. No follow-up experiment is authorized automatically.
