# Isolated implementation changes

Source: D:/CEDAR/arr_c1_20260928/scripts. Original v1 and v2 records unchanged.

- Correct negative random dose sign and retain deterministic BF16 actual-norm matching.
- Replace component/half-dose matrices with the approved smoke52/formal3840 schedule.
- Reuse exact validated P1 directions, preprocessing and existing S1 coefficients; no new fit or scientific selection.
- Freeze common-response and V/F-interaction endpoints (broad/specific), plus separate sensitivity reports.
- Remove analysis dependency on the Torch runner; meaningful sign, cleanup and estimator tests.
- Preserve orphan arrays/failure gates; cumulative persisted runtime budget, single-controller/single-runner locks, 30min evidence snapshots and bounded stall checks.

Scientific difference from old v2: signed +/-1 is now the sole primary intervention; original component transfer and +/-0.5 are not run. This is explicitly a new protocol, not a post-hoc repair of scientific outcomes. Technical changes never create independent samples. Jev and new model/direction/layer searches are excluded.
