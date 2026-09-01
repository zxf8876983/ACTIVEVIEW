# EXP035-R1 — Context-safe dense recognition quality field

This rerun replaces the invalid record-id-only association in EXP035 with the
canonical `(scene_id, region, record_id)` context identity.  It uses only the
frozen Stage-B/ST-GCN Train and Val artifacts; no Test or perception generation
is performed.

The full Stage-B candidate reproduction audit is a hard gate before EXP036-R1
and EXP037-R1.  The previous EXP035 result remains preserved as an invalid
archive and is not overwritten.
