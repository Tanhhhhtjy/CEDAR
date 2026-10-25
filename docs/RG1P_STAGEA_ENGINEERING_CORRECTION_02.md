# RG1-P Stage A Engineering Correction 02

Status: GPU-smoke implementation correction after a failed random-reference
engineering audit. Scientific inputs, random directions, seeds, signs,
schedules, target norms, matrices, estimands, and frozen tolerances are
unchanged.

## Preserved failed smoke

The first complete 72-row smoke is retained unchanged under
`smoke-stagea-seed0-failed-random-solver-v1`. Its audit passed replay,
native-lens, no-op, hash, hook, finite-value, schedule-position, and sign-pair
checks. It failed before formal because three random-reference cells exceeded
the frozen BF16 realized-norm tolerance. Formal remained at 0/864.

The 32 failed-version random rows are never used in analysis. The corrected
smoke imports the 32 official and eight no-op rows byte-for-byte and reruns
only the 32 invalid random-reference controls in a separate checkpoint fork.

## BF16 norm solver correction

The previous implementation used four multiplicative ratio updates. Dense
random directions combined with BF16 residual storage make the visible norm a
discrete function of the commanded scalar; the ratio update could oscillate
around small target norms.

Correction 02 brackets each frozen target dose and performs 12 monotone
bisection iterations, retaining the representable BF16 update with minimum
absolute norm error. This changes neither the target norm nor the random
direction. The original frozen pass rule remains exactly:

```text
max absolute error <= 0.03125 OR max relative error <= 0.002
```

The disjunction is applied per layer-token position. A record is valid when
every position satisfies at least one of the two frozen tolerances. Taking a
global maximum for each error type before applying `OR` is invalid because the
two maxima can come from different positions; Correction 02 explicitly tests
the elementwise violation mask.

The ideal requested vector norm is evaluated in float64 to avoid treating a
float32 reduction error at large doses as a changed request. The model-visible
direction cosine is recorded for every cell and must remain strictly positive.

## Official correction alignment audit

The failed audit also exposed that a raw cosine threshold of 0.995 is not a
valid engineering test for small dense corrections stored in BF16. For the 11
flagged layer/cell pairs, every observed off-axis residual was contained in
the exact L2 envelope of one half-ULP of the stored BF16 output.

Correction 02 therefore does not lower a cosine cutoff. It replaces that
non-representability-blind test with a deterministic precision-aware test:

1. the realized correction must have positive projection on the official
   Assistant Axis; and
2. its off-axis residual must not exceed the componentwise BF16 half-ULP L2
   envelope.

Raw realized cosine, correction norm, and envelope ratio remain reported.
The official capping implementation, axis, thresholds, layers, model dtype,
and replay logits are unchanged.

## Provenance constraints

Static AST hashes must prove that `run_one`, official schedule capture, and
terminal hidden-state capture are unchanged from CPU Gate `6f5f5dd`. The only
intentional GPU-forward change is the random injection norm solver. Formal
starts fresh from the Correction 02 CPU Gate commit and may begin only after
the corrected composite smoke passes every original scientific gate plus the
precision-aware checks above.
