# Changeset

Created isolated C1 v2 math, preparation, runner, campaign, preflight and analysis code. Main/dose runners and the scientific analyzer have not passed real-model v2 validation. Added timestamped read-only auditing of the blocked campaign and a separate prospective signed-control CPU diagnostic. No original S1/P1 implementation or frozen scientific inputs changed in this work.

The original C1 v1 hook-count gate and a later limited re-audit are preserved. See STATUS_AND_FINDINGS.md for their limitations; neither is a v2 release gate. CPU engineering tests passed. All 5612 planned v2 cells received a saved-state norm check; 14 main cells failed. Zero v2 GPU forwards ran.
