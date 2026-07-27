# RG1-P Stage A Engineering Correction 01

Status: GPU-smoke implementation correction. The scientific design, frozen
inputs, estimands, thresholds, random controls, and stopping rules are
unchanged.

## Observed failure

The first smoke attempt completed 32 official replay cells and eight no-op
cells, then stopped at the pre-random native-lens gate. The no-op pairs were
exact at every layer (`KL_opt = 0` and `delta_h_l2 = 0`), but the final-layer
native-lens comparison showed BF16-quantized candidate-logit discrepancies of
0.125 or 0.25 for most pairs.

The failed readouts remain preserved in the smoke artifact namespace
`lens_records/`.

## Root cause

The runner applied `final_norm -> lm_head` to all 64 captured layer states in
one `[64, 5120]` matrix. The archived terminal answer logits were produced by
the cache-faithful decode path with a singleton `[1, 1, 5120]` state. In BF16,
these shapes select different GEMM numerical paths.

On the initially reported failing pair
`CEDAR-DEV-008-N::e2a_uncapped::official_p25`, the batched path had maximum
candidate-logit error 0.25, while the singleton decode-shaped path reproduced
all four archived candidate logits exactly (maximum error 0.0).

## Correction

Each captured layer is now passed independently through the same frozen final
norm and full checkpoint LM head with shape `[1, 1, 5120]`. No translator is
trained or fitted. Shape-faithful derived records use the separate cache
namespace `lens_records_shape_faithful_v2/`, so the failed v1 evidence is not
deleted or overwritten.

The original strict final-layer tolerance (`< 1e-3`) remains unchanged. The
runner may continue to the preregistered random references only if every
shape-faithful smoke pair passes that original gate and every all-layer no-op
threshold also passes.

The 40 completed replay records are not regenerated. Their original run
fingerprint remains bound to CPU Gate commit `6f5f5dd643e774862a7185357b1d6027d5685c4f`.
The corrected runner may resume that smoke directory only when all non-Git
configuration fields are byte-equivalent, the new execution commit descends
from the initial Gate, and AST hashes confirm that `run_one` plus all capture,
official-schedule, and random-injection forward operators are unchanged. The
resume event and both commits are recorded in `runtime.json`. Formal execution
starts fresh and is wholly bound to the corrected CPU Gate commit.
