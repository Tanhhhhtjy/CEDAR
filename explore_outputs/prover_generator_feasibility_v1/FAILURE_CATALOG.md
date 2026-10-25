# Failure Catalog

## Formal attempt failures

- `provergen_style`: 0/16 failed.
- `reviseqa_style`: 0/16 failed.

No valid record contains a solver disagreement, invalid world, non-atomic V, missing F
countermodel/opposite proof, or second F failure point.

## Rejected search neighbors

Rejected `reviseqa_style` neighbors are retained inside each record's `construction_trace`.
They are candidate edits that left the initial label unchanged or moved to the wrong label; they
are not silently deleted formal attempts. Counts: `{'false': 16, 'true': 16, 'unknown': 11}`.

## Engineering deviations

- Prover9 Dec-2007 was used instead of LADR-2009-11A because the Windows host lacked WSL and a C
  toolchain. The same solver binary was used by both routes.
- Exact timing is not used as data. Stable per-record wall-clock buckets are reported, while
  observed command duration belongs in the experiment ledger.
