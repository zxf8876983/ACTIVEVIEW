# EXP031 — Human Viewpoint Sensitivity / Perception-Quality Audit

Val-only offline diagnostic comparing human-centric candidate geometry with
future pose-quality and recognition-quality surrogates. Method A/D use only
frame-15 human geometry, candidate metadata and frozen Stage-D state. Method B
uses existing future skeleton confidence as a privileged upper bound and is
not deployable. Method C is recorded unavailable when recognition outputs are
absent. Test, perception regeneration, rollout and upstream model changes are
forbidden.
