# Phase 1.2 Sentinel Results

## Outcome

The paired-justification automatic gate passed. The phase stops at Gate B,
`awaiting_human_naturalization_review`; no agent-generated rating is treated as human evidence.

## Formal and pairing checks

- Sentinels: 12/12, covering six F classes at easy and medium depth.
- V: 12/12 Python and Prover9 proofs of `revision_warranted(edit_id)`.
- V proof count: exactly one for all 12.
- F: 12/12 Python/Prover9 non-entailment and Mace4 countermodels.
- F: 12/12 one fatal error, no second unsupported step, and derived class equals contract.
- Easy V proof depth: 2--3; medium: 4--5; gradient: 6/6 classes.
- Reviewer, source type, authentication status, speech act, slot skeleton, and proposed edit:
  exactly equal within every pair.
- V/F English word-count ratio: 1.00 for all 12.
- Global V-only or F-only fixed 2--6 grams: none.
- Encoding, forbidden-language, and coreference checks: 12/12 passed.
- Two clean builds: canonical sentinel, automatic-report, and workbook-content hashes all equal.

## Verification

- Independent validator: 12/12 passed.
- Targeted tests: 15/15 passed.
- Ruff: passed.
- Full repository collection: stopped with 32 environment/package dependency errors before test
  assertions ran. This is disclosed and is not counted as a v1.2 pass; the isolated v1.2 suite
  imports and passes under the existing research worktree environment.

## Boundary

The workbook is blank and awaits a real reviewer under the six item-level criteria. Version 1.2
did not build a development bank or held-out set and did not run Qwen, GPU, capping, S, or any
model-response screening.
