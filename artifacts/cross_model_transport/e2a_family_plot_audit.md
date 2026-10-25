# E2A family plot audit

- Families: `12/12`.
- Bases: `24/24`; paired N/R/V/F cells: `96/96`.
- Archived records used: `192` (uncapped + full p25 only).
- Missing / invalid / truncated: `0 / 0 / 0`.
- Recomputed overall `delta m_general`: `3.130208333333` logits.
- Recomputed 95% family-cluster CI: `[1.834190538194, 4.486121961806]`.
- Frozen target: `+3.130208333333` and `[1.834190538194, 4.486121961806]`; exact numerical match: `true`.
- Bootstrap: `20000` draws, seed `20260725`, resampling 12 families with replacement.

No row was removed according to model output. Missing/truncated/invalid states are counted before metrics are evaluated; this frozen slice contains none. Figure 0 must use the 12 observed family means in `e2a_family_geometry.csv`. The lower V/F geometry uses `delta_m_target` in logits and is not combined on an axis with target-follow percentage points.
