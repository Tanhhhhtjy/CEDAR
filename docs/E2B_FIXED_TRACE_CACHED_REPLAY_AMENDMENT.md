# E2B Fixed-Trace Cached Replay Amendment v1.1

## Status and provenance

This amendment is the single prespecified engineering revision of E2B after the
v1 whole-prefix teacher-forcing smoke failed its frozen replay Gate. It starts
from result commit `e2366b2b33011f6d795d7fbe7484be9e9752799b` and permanently
retains the v1 failure evidence. It does not reinterpret that failure as a
scientific result.

The E2A result anchor remains
`0c8517bb5fb4640f7d5e4d51356d0cf167046643`.

## Reason for the amendment

E2A captured answer-boundary logits during greedy `model.generate()` with an
initial prompt prefill followed by single-token KV-cache decode calls. E2B v1
instead evaluated the prompt and complete frozen trace prefix in one model
forward. Its 32-cell smoke completed without nonfinite logits or hook leakage,
but only 2 of 16 diagonal replays met the frozen `1e-4` raw-logit tolerance.

A zero-GPU decomposition of those records found that the discrepancy was not
only a common shift of all four option logits. The largest centered-option
error was `0.28125`, and the largest `m_general` replay error was `0.375`.
Therefore neither centering nor relaxing the raw-logit Gate is an acceptable
repair.

## Sole engineering change

The scientific matrix and estimands remain unchanged. Only the replay engine
changes from a whole-prefix forward to cache-faithful forced replay:

1. Render and tokenize the original E2A prompt exactly as before.
2. Invoke the same greedy `model.generate()` KV-cache path used by E2A.
3. At each generation step, first capture the unmodified model scores.
4. After capture, force the next frozen trace-prefix token.
5. At the answer boundary, capture the raw A/B/C/D logits and then force the
   trace's original answer label only as a terminal probe token.
6. Exclude that terminal probe label from the fixed trace and all estimands.

The forced-token processor cannot select content based on scores. It receives
the already frozen token sequence, and the runner verifies its emitted token
hash against the v1 frozen trace. Capture must precede force at every step.

## Unchanged scientific contract

- 96 prompts, two frozen trace sources, and two forward conditions.
- 32-cell smoke followed, only on success, by 384 formal cells.
- Qwen/Qwen3-32B revision
  `9216db5781bf21249d130ec9da846c4624c16137`.
- BF16, SDPA, batch size one, exactly four visible GPUs, and 18 GiB per GPU.
- Official p25 experiment `layers_46:54-p0.25`, actual layers 46--53.
- `delta_general_U` and `delta_general_P` are separate primary estimands.
- Family is the independent cluster: 12 families, 20,000 bootstrap draws,
  seed `20260726`, two-sided percentile 95% intervals.
- No content retry, answer repair, trace replacement, sample deletion, or
  outcome-dependent rerun.

The original E2B protocol and analysis plan continue to govern every item not
explicitly amended here.

## Cached replay smoke Gate

The fixed v1 smoke selection is reused without change. All 32 cells must finish,
and all 16 diagonal cells must reproduce the E2A raw A/B/C/D logits within the
original `1e-4` absolute/relative tolerance. The smoke also requires:

- exact prompt and frozen trace-token hashes;
- capture-before-force ordering at every step;
- one prompt prefill and token-by-token decode telemetry;
- zero nonfinite logits, missing captures, hook leaks, or residual hooks;
- exact model, tokenizer, axis, capping-config, layers, and threshold hashes;
- no naturally selected response token and no generated-text analysis.

Scientific direction is not a smoke criterion. Formal is blocked unless the
cached replay smoke audit has `status="passed"`.

## One-chance stopping rule

This is the only replay-engine amendment authorized for E2B. If any diagonal
cached replay still exceeds `1e-4`, formal remains blocked. A final diagnostic
may rerun the original natural E2A generation path on the fixed smoke prompts
to distinguish replay implementation failure from environment/kernel-level
non-reproducibility. That diagnostic cannot authorize formal, change anchors,
or relax tolerance. After such a failure E2B stops permanently.

## Claim limits

Allowed wording remains “conditional on the frozen generated trace” and
“controlled fixed-trace diagnostic.” This amendment does not authorize claims
of a natural direct effect, complete mediation, an answer-layer-only mechanism,
a final-layer circuit, or independent confirmation.
