# Phase 1.1 Results

## Outcome

The symbolic generator passed every frozen hard gate. The final state is Gate B because the blank
naturalization workbook has not yet been returned by a real reviewer.

## Symbolic stress test

- Attempts: 48
- Valid / failed: 48 / 0
- Gold errors: 0
- Unique atomic V rate: 100.0%
- Semantic F validation rate: 100.0%
- Six F classes: eight records each, split four easy and four medium
- Candidate pool: min 20, median 32.0, max 90
- Successful edit rank: min 2, median 18.0, max 78
- Formal difficulty gradients: 8/8 families
- Deterministic rebuild: True across 331 generated files per build

These are formal proof-depth results, not claims about model behavior.

## Naturalization

- Frozen sample: 12 records, covering six F classes at easy and medium difficulty
- Automatic checks: True
- Human review: `awaiting_human_naturalization_review`
- Workbook fields: blank; no gold, F class, or symbolic internals are displayed

## Verification

- Targeted Phase 1.1 tests: 24 passed
- Full repository test run: 240 passed, 20 failed (pre-existing asset/environment-bound failures)
- Ruff: passed

The full-repository failures are outside Phase 1.1 and are dominated by unavailable frozen model
assets, absent external commands, and pre-existing hash-bound artifacts. They are not counted as
Phase 1.1 passes; the exact outcome is disclosed rather than suppressed.

## Boundary

No Qwen, GPU, capping, S arm, model-response selection, development bank, or held-out construction
was run. The 48 stress records are not a confirmation set.
