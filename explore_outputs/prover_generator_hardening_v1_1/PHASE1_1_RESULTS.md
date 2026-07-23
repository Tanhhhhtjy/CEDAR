# Phase 1.1 Results

## Outcome

The symbolic generator remains passed. A real reviewer completed the frozen v2
packet, and the item-level all-criteria rule produced Gate C: 0/12 rows passed. This is a
naturalization failure, not a symbolic-gold, atomic-edit, or F-certificate failure.

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

- Frozen sample: 12 v2 records, covering six F classes at easy and medium difficulty
- Automatic checks: True
- Human review: `failed` (Gate C)
- Passed / failed rows: 0 / 12
- Failed IDs: NAT-01, NAT-02, NAT-03, NAT-04, NAT-05, NAT-06, NAT-07, NAT-08, NAT-09, NAT-10, NAT-11, NAT-12
- Naturalness distribution: {3: 12}
- Condition-identity leakage = yes: 12/12
- Clear-reference failures: NAT-11, NAT-12
- Completed workbook: `naturalization_review12_bilingual_v2_completed.xlsx`
- Workbook: `naturalization_review12_bilingual_v2.xlsx`
- Workbook fields: blank; no gold, F class, or symbolic internals are displayed
- Item-level rule: all six criteria must pass on all 12 rows

## Verification

- Targeted Phase 1.1 tests: 35 passed
- Full repository test run: collection stopped: 32 environment/package dependency errors; no test assertions ran
- Ruff: passed

The full-repository failures are outside Phase 1.1 and are dominated by unavailable frozen model
assets, absent external commands, and pre-existing hash-bound artifacts. They are not counted as
Phase 1.1 passes; the exact outcome is disclosed rather than suppressed.

## Boundary

No Qwen, GPU, capping, S arm, model-response selection, development bank, or held-out construction
was run. The 48 stress records are not a confirmation set.
