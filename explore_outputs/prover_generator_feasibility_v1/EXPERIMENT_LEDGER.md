# Experiment Ledger

| Date | Stage | Action | Result |
|---|---|---|---|
| 2026-07-23 | Intake | Created isolated branch/worktree from the frozen anchor | Passed |
| 2026-07-23 | Upstream | Inspected ProverGen at `1d8abd2` | Method reference only; no license declared |
| 2026-07-23 | Upstream | Inspected ReviseQA at `9a57786` | MIT; core code accessible via Git objects |
| 2026-07-23 | Solver | Tested external Prover9 Dec-2007 on an official sample | `THEOREM PROVED` |
| 2026-07-23 | Safety | Checked prohibited compute/actions | No Qwen, GPU, capping, S, or behavior selection |
| 2026-07-23 | Sentinel | Ran two toy edits through both routes | 4/4 valid; formal generation remained off |
| 2026-07-23 | Build 1 | Began formal build | Stopped at first Mace4 countermodel marker mismatch after 2.312 s |
| 2026-07-23 | Build 2 | Rebuilt after Mace4 compatibility patch | 14/16 per route after 7.509 s; PGF-06 exposed a Prover9 quantifier adapter error |
| 2026-07-23 | Build 3 | Rebuilt both routes after adapter hardening | 16/16 per route after 6.633 s |
| 2026-07-23 | Validation | Reran both solvers and two complete builds | Passed in 17.992 s; 159 canonical files identical |
| 2026-07-23 | Final freeze | Rebuilt after adding independent manifest audit | 159 core files identical; aggregate `078c53d3...f7a356` |
| 2026-07-23 | Manifest | Recomputed every raw and canonical hash independently | 176/176 entries passed |

The two failed engineering builds are retained in `ENGINEERING_FAILURE_LOG.md`; neither was used
to alter a family card, transition, target, or difficulty contract.
