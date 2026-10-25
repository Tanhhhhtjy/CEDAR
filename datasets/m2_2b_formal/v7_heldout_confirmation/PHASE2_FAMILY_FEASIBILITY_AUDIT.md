# C1/V7 Phase A0 family feasibility audit

Status: `passed_author_static_audit`

This audit binds one constructive witness to every frozen family card before any of the remaining 44 base units are authored. It is a zero-model, solution-first check: no Qwen, GPU, capping, behavioral output, or item selection was used.

## Result

- Families checked: 24/24
- Current family-card hashes matched: 24/24
- Unique original answers: 24/24
- Unique updated answers: 24/24
- Updated answer differs from original: 24/24
- V changes exactly one record node: 24/24
- Feasibility decisions: 24 feasible, 0 infeasible

The machine-readable evidence is in `family_feasibility_audit24.jsonl`. Every row records the canonical SHA-256 of the current card, a concrete original state, a one-node V replacement, both answers, and the verification method.

## Arithmetic and accounting families

| Family | Constructive operator | Original -> updated | Verification |
|---|---|---|---|
| F01 | Odd-roster median after one record correction | 25 -> 18 | independent calculation |
| F02 | Minimum effective stage capacity | 80 -> 70 | independent calculation |
| F03 | Whole-container ceiling cost | 35 -> 21 | independent calculation |
| F04 | Serial yield product | 400 -> 600 | independent calculation |
| F05 | Signed ordinary/contra balance | 140 -> 40 | independent calculation |
| F06 | Inclusive-gross inversion | 100 -> 96 | independent calculation |
| F07 | Distance-weighted average speed | 40 -> 48 | independent calculation |
| F08 | Finite rework recurrence | 75 -> 87.5 | independent calculation |

## Unit and label rule-application families

| Family | Constructive operator | Original -> updated | Verification |
|---|---|---|---|
| F09 | Longest active prefix | Green -> Blue | formal derivation |
| F10 | Explicit prerelease comparison | beta -> rc | formal derivation |
| F11 | Exhaustive interval relation | contained -> partial overlap | formal derivation |
| F12 | Escaped-delimiter parse | green -> blue | formal derivation |
| F13 | Quantity-selected logarithmic multiplier | 20 dB -> 40 dB | independent calculation |
| F14 | Exact protocol-port tuple lookup | Gamma -> Delta | formal derivation |
| F15 | Requested bits after active mask | 101 -> 100 | independent calculation |
| F16 | Nth weekday from month anchor | day 18 -> day 16 | formal derivation |

## Multistep-logic families

| Family | Constructive operator | Original -> updated | Verification |
|---|---|---|---|
| F17 | Unique perfect matching after one complete allowance-row replacement | (1,2,3) -> (3,1,2) | exhaustive enumeration |
| F18 | Directed interval relation | contains -> before | formal derivation |
| F19 | Weighted threshold conjoined with veto | fail -> pass | formal derivation |
| F20 | DAG earliest completion | 7 -> 8 | independent calculation |
| F21 | Condorcet winner after one edge reversal | X -> Y | exhaustive enumeration |
| F22 | Unique minimum hitting set | {B} -> {A} | exhaustive enumeration |
| F23 | Unique sequence under adjacency constraints | ABCD -> ACBD | exhaustive enumeration |
| F24 | Minimal key by FD closure | {R} -> {Q} | exhaustive enumeration |

## Ambiguity and regression checks

No witness requires outside knowledge. All rules needed for the witness can be stated inside a closed-world item. None relies on answer-option ordering. F17 uses the approved atomic replacement of one agent's **complete** allowed-item record; it does not regress to adding or deleting a single graph edge. F18 uses directed relations (`contains`, `before`, `overlaps`) and was checked independently rather than through F11's containment classifier.

These witnesses prove structural realizability only. The full base units must still pass the option, difficulty, migration-balance, one-node-update, duplicate-build, leakage, and author-construction-review gates in the runbook.
