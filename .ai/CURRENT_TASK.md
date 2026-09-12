# Short-prefix5 mixed-view protocol feasibility audit — completed 2026-09-13

Completed a read-only reduced12 eight-placement audit on Moving Val (10,080
contexts), using the exact action set Stay/current plus the Stage-A legal
candidate pool. No training, Policy Test, RGB/skeleton/DINO generation or
recognizer changes were performed. The recognizer was frozen reduced12 ST-GCN
plus the frozen shared adapted head.

Protocol: Stay observes current frames 0:30. A candidate observes current
frames 0:5 followed by that candidate's frames 5:30. This is explicitly a
discrete-time view-switch approximation, not continuous navigation.

Key results (Accuracy/Macro-F1): Stay 0.302579/0.292976; Random-ShortPrefix5
0.342262/0.329111; Mixed5 GT-TrueLogP 0.687599/0.671197; Mixed5 GT-Margin
0.720040/0.709917; FullView GT-TrueLogP 0.753175/0.755358; FullView GT-Margin
0.791964/0.800900. Mixed5 AnyCorrect coverage is 0.720040 and FullView
AnyCorrect coverage is 0.791964. The GT-TrueLogP FullView→Mixed5 ceiling drop
is 0.065575 Accuracy points (0.084161 F1).

Within-context candidate ranking Spearman is 0.794074 mean / 0.890909 median,
with 0.678770 top-1 viewpoint agreement and 0.832573 top-3 overlap. Mixed
candidate switches have 9.395924× the mean-joint-displacement of normal same-
view 4→5 transitions; full-correct→mixed-wrong is 0.226582 conditional on a
full-correct candidate (0.085063 of all candidate samples). The pre-registered
decision is **KEEP** because Mixed5 oracle Accuracy is 0.687599 and the
FullView→Mixed5 drop is below 0.10, while the large boundary discontinuity
remains a protocol concern.

Report:
`experiments/reduced12_eight_placement_v1/short_prefix5_protocol_feasibility/`

Task status: **CLEAN**. Next human decision: whether to authorize a selector
experiment after reviewing the protocol ceiling; no follow-up training was
started automatically.
