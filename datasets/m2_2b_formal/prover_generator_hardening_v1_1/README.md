# Prover Generator Hardening V1.1

Candidate-only Phase 1.1 stress test for the single architecture
`proof_graph_plus_atomic_edit_certification`, anchored at
`4712b0d555ae2b07f73781c6a11b7caa660ef96a`.

This directory never overwrites v1. It does not contain a development bank, held-out set, model
output, S arm, or capping result. See `docs/PROVER_GENERATOR_HARDENING_V1_1_PROTOCOL.md`.

## Frozen outcome

- Amendment 01 replacement stress run: 48/48 valid.
- Python/Prover9 gold errors: 0.
- Unique atomic V and semantic F validation: 100%.
- Six F classes: eight each, with four easy and four medium.
- Family formal proof-depth gradients: 8/8.
- Amendment 02 automatic naturalization checks: 12/12.
- Final state: Gate B, awaiting a real human review of the v2 packet.

The reviewer should receive only `naturalization_review12_bilingual_v2.xlsx` and
`NATURALIZATION_REVIEW_GUIDE_ZH_v2.md`. The original packet remains preserved at commit
`06fd188`. The v2 workbook intentionally omits gold, F class, and symbolic internals. Do not mark
Phase 1.1 complete or prepare the development bank protocol until the real review has been returned
and finalized.

## Reproduction commands

```powershell
python build_prover_generator_hardening_v1_1.py
python validate_prover_generator_hardening_v1_1.py --verify-rebuild
python naturalize_prover_generator_hardening_v1_1_v2.py --verify-determinism
python analyze_prover_generator_hardening_v1_1.py
```

After a real reviewer returns a completed copy, import it without modifying the workbook:

```powershell
python finalize_naturalization_review_v2.py completed_review.xlsx
```

The finalizer rejects incomplete or altered packets. All 12 rows must individually satisfy all six
frozen criteria; otherwise the result is Gate C and only renderer revision is permitted.

The formal builder must not be rerun with changed scientific settings unless a new amendment is
written and all 48 attempts are rebuilt.
