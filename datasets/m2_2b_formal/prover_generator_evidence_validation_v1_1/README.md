# CEDAR Evidence Validation v1.1

This is a scope amendment over the frozen development-v1.1 result commit
`5f00a89edc1beb7ad5af3ea5e070e6ab6dcbe26d`.

It does not change any symbolic world, language shell, option mapping, proof,
gold label, or countermodel. It selects the untouched `development_validation`
split and evaluates only N/R/V/F under paired `uncapped` and official `p25`
conditions.

The formal matrix is:

```text
12 families x 2 variants x 4 arms x 2 conditions = 192 generations
```

The parent AO/S calibration remains a preserved negative result and is outside
this formal prover-based evaluation. This internal validation is not a new-salt
or cross-structure confirmation set.

Current state: CPU tokenizer/capping-asset preflight pending. GPU execution is
not authorized until `EVIDENCE_VALIDATION_GATE.json` says
`may_run_paired_validation=true`.
