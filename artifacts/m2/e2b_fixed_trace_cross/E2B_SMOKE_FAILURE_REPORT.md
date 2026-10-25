# E2B fixed-trace smoke failure report

## Outcome

The frozen 32-cell E2B smoke completed, but the engineering replay Gate failed. Formal execution and scientific analysis were therefore not started.

- Branch: `agent/e2b-fixed-trace-cross-v1`
- CPU Gate commit: `d1473782f40f3e468905a8ebaa51e01f526b775f`
- E2A anchor: `0c8517bb5fb4640f7d5e4d51356d0cf167046643`
- Smoke generation: 32/32 records
- Forward conditions: 16 uncapped, 16 full_p25
- Trace sources: 16 e2a_uncapped, 16 e2a_full_p25
- Nonfinite logits: 0
- Hook leaks or residual hooks: 0
- Results-file hash validation: passed
- Replay tolerance: `1e-4`
- Diagonal replay: 2/16 passed, 14/16 failed
- Largest absolute A/B/C/D logit difference: `0.625`
- Formal run: not started
- Analyzer: not run
- Infrastructure resumes: none

## Replay results

The two passing diagonal cells were:

- `CEDAR-DEV-043-N::e2a_full_p25::full_p25`
- `CEDAR-DEV-048-V::e2a_uncapped::uncapped`

The other 14 diagonal cells exceeded the frozen numerical tolerance. Their exact per-option deltas are retained in `SMOKE_AUDIT.json`; observed maximum absolute differences ranged from `0.125` to `0.625` logits.

## Diagnostic inspection

Static inspection found a material execution-path difference that may explain the mismatch: the E2A anchor answer logits were captured during autoregressive `model.generate` decoding, whereas the frozen E2B runner evaluates the complete prompt plus fixed trace prefix in one teacher-forced model forward. In BF16, cached decode and a single full-sequence forward are not guaranteed to be numerically identical at `1e-4` tolerance.

This is a candidate explanation, not a proven root cause. Testing or correcting it would require a changed execution path and a new protocol/CPU Gate. No such change was made in this run.

## Frozen stopping decision

The protocol states that replay Gate failure stops E2B and that the failed traces and artifacts must be retained. Accordingly:

- no tolerance was relaxed;
- no trace was replaced or retokenized;
- no model, p25, layer, axis, or cap setting was changed;
- no cell was deleted or selectively rerun;
- the 384-cell formal matrix was not started;
- no scientific estimates were produced.

This is an engineering replay failure, not a scientific result about the E2B estimands.
