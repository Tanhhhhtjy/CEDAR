# Risks and Boundaries

## Current freeze boundary

1. Gate A is authoritative in the local worktree, but the worktree is not committed or pushed.
2. This is a development bank, not a held-out confirmation set. Model outputs may tune structural parameters only during Step 5 and may not be used to filter a later confirmation set.
3. The v1 route ranking remains retired as `inconclusive`; the hybrid is an engineering selection, not a measured route-performance winner.
4. Naturalness is 4/5 rather than 5/5 throughout: the language is grammatical and clear but formal, repetitive, and sometimes long. Calibration must reveal whether this creates behavioral floor or ceiling effects.
5. `pytest` is unavailable in the current shell. The changed Python files compile and the official independent validator passes, but the repository test suite was not executed through pytest here.

## Step 5 controls

- Start uncapped only; do not start p25/capped in the same decision step.
- Keep thinking and non-thinking as separate strata.
- Record both target-minus-initial margin and the four-state discrete outcome.
- Store option logits/mass, output entropy, reasoning-token count, parse status, and synchronized other-error indicators.
- Verify that V is not a ceiling, F is not a floor, S has dynamic range, and N/R remain stable.
- Do not delete or rewrite items merely because their model outputs do not fit the expected story.

## Still prohibited

- Held-out confirmation generation before development calibration is frozen.
- Capped/p25 confirmation claims.
- LoFa or J-Lens mechanism work before a usable core assay is demonstrated.
- Mixing the deferred longitudinal-update bank into this phase.
