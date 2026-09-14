# Notes: Policy–Recognizer Coupling Audit

The attachment requires a Train/Moving-Val-only audit over the reduced12
strict Frame0 protocol. Reuse the historical Frame0 selector checkpoint,
retrained old-adaptive-aware selector checkpoint, shared head, old adaptive
head, existing option/visibility/DINO caches and Stage-A legal action sets.
No new training or perception generation is permitted. Final selected views are
evaluated from the real archived O1 observation with the nominated head.

Required comparisons include selected viewpoint distributions, Train-derived
old-adaptive static and current/view-pair priors, RGB/geometry shuffles,
recognizer comparisons on identical selected views, selector agreement,
per-context switch transitions, and per-class metrics. The primary conclusion
must select among fixed viewpoint prior, recognizer-policy distribution
matching, and meaningful instance-conditioned Frame0 NBV.

## Completed result (2026-09-14)

CUDA Val-only run completed on 10,080 Moving contexts (46,324 Train contexts
used only for the static and view-pair priors). Accuracy/Macro-F1 were
0.526984/0.547002 for the adaptive-aware selector with the old adaptive head,
0.522917/0.546040 for the static viewpoint prior, and 0.519940/0.543270 for
the view-pair prior. RGB shuffling reduced Accuracy by 4.256 pp and geometry
shuffling by 15.159 pp. The adaptive-vs-best-prior gap was only 0.407 pp; the
strict 1.5 pp instance-conditioned criterion was not met. On identical
adaptive-selected views, old adaptive vs shared improved Accuracy by 1.865 pp,
below the 2 pp distribution-matching threshold. The primary interpretation is
therefore **MOSTLY FIXED VIEWPOINT PRIOR**, with measurable but insufficient
instance-conditioned RGB signal. Policy Test was not read.
