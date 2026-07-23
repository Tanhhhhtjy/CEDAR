# Engineering Failure Log

## Build attempt 1: legacy Mace4 success marker

- Elapsed: 2.312 seconds.
- Stop point: first `unknown_to_true` F countermodel.
- Symptom: Mace4 returned a concrete `interpretation(...)` and `Exiting with 1 model`, but the
  adapter expected the newer literal string `MODEL FOUND`.
- Classification: solver-version compatibility bug, not a countermodel or family failure.
- Repair: accept only the legacy model interpretation plus successful one-model exit; retain the
  deterministic INPUT/MODEL sections and reject any output without a concrete model.

## Build attempt 2: multi-variable Prover9 quantifier encoding

- Elapsed: 7.509 seconds.
- Observed formal records: 14/16 valid in each route.
- Preserved failed specifications: `PGF-06-E` and `PGF-06-M` in both routes.
- Stored failure condition:
  `dual_initial_agreement=true`, `dual_updated_agreement=false`, while the Python evaluator gave
  the requested `unknown_to_false` transition.
- Root cause: the direct Dec-2007 Prover9 input used `all x y z (...)`; that parser requires
  `all x all y all z (...)`. The solver emitted a fatal parse error. The first adapter also
  synthesized `SEARCH FAILED` when neither recognized terminal marker was present.
- Classification: solver adapter bug and gold-safety bug, not a common-family contradiction.
- Repair: repeat every quantifier explicitly; accept only solver-emitted `THEOREM PROVED` or
  `SEARCH FAILED`; raise on all other outcomes. A regression test now checks multi-variable
  quantifier serialization.

## Rebuild policy

Both routes were rebuilt from all sixteen unchanged specifications after each adapter repair.
No failed specification was dropped, no family was modified, and no route-specific exception was
introduced. Build attempt 3 produced 16/16 valid records for both routes and the full validator
subsequently reproduced identical canonical hashes across two further complete builds.
