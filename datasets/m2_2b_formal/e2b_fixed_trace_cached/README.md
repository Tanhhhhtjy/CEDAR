# E2B fixed-trace cached replay v1.1

This release reuses the immutable 192 traces and 384-cell matrix from
`e2b_fixed_trace_cross`. It changes only the numerical replay path from one
whole-prefix forward to forced incremental `model.generate()` with KV cache.

The source trace and matrix hashes, v1 failure evidence, amendment, runner,
auditor, analyzer wrapper, validator, tests, model revision, axis, and cap
configuration are bound by `manifest.json` and `CPU_GATE.json` before GPU use.

Outputs are isolated under:

`artifacts/m2/e2b_fixed_trace_cached_v1_1/`

The v1 artifacts under `artifacts/m2/e2b_fixed_trace_cross/` are retained
unchanged.
