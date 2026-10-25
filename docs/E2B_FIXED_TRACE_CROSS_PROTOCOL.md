# E2B: Fixed-Trace Crossed Teacher-Forcing Diagnostic

## Status, anchor, and scope

Status: **frozen before implementation and before any E2B model output**.

- E2A result anchor: `0c8517bb5fb4640f7d5e4d51356d0cf167046643`
- E2A scope: exploratory mechanism assay
- E2B scope: controlled fixed-trace diagnostic, not manuscript confirmation
- model: `Qwen/Qwen3-32B`
- revision: `9216db5781bf21249d130ec9da846c4624c16137`
- intervention: official `layers_46:54-p0.25` (layers 46--53)
- dtype/attention: BF16/SDPA
- batch size: 1

E2B may use only the 96 frozen Answer prompts and the paired `uncapped` and
`full_p25` Answer responses already present in E2A. It does not authorize new
generation, Validity prompts, `prefill_only`, `decode_only`, p1, item selection,
activation patching, J-Lens, MUSE, SAE, gradients, or a new controller.

## Question and design

E2B asks whether p25 changes final option geometry when the visible generated
trace is held fixed. Each prompt contributes two source traces and each trace is
evaluated under two forward conditions:

| frozen source trace | uncapped forward | full-p25 forward |
|---|---:|---:|
| E2A uncapped trace | yes | yes |
| E2A full-p25 trace | yes | yes |

```text
96 prompts x 2 source traces x 2 forward conditions = 384 forwards
```

All 384 cells are fixed before execution. Scientific direction, whether the
original answer flipped, trace length, family, arm, and E2A margin are forbidden
selection variables.

## Frozen trace definition

For each E2A Answer response, reconstruct the exact generated token sequence
with the pinned tokenizer and verify its stored response-token SHA-256. The
teacher-forced prefix consists of:

1. the original chat-formatted prompt, byte-for-byte;
2. every generated response token before the selected final A/B/C/D label token,
   including the complete `<think>...</think>` content and the final JSON
   scaffold;
3. no final answer label token and no token after it.

The cut point is the stored `selected_answer_logit_index`, cross-checked against
`answer_boundary.answer_choice_index`, the generated label, and tokenizer
round-trip offsets. The next-token logits at that cut point are the E2B readout.
If a trace cannot be reconstructed uniquely, its token hash differs, more than
one answer boundary survives, or the label has ceased to be a single contextual
token, the entire E2B run stops. A failed trace is retained and is never replaced.

`full_p25` is applied at layers 46--53 to every token in the causal
teacher-forced forward. Thus E2B conditions on a post-treatment trace and does
not localize the effect exclusively to the answer token or final layer.

## Engineering replay Gate

Before the 384-cell matrix, run a 32-forward smoke using one prompt from every
arm-by-variant cell. Within each cell choose the prompt minimizing
`SHA256("cedar-e2b-smoke-v1|" + id)`, then cross it with both trace sources and
both forward conditions. This fixed rule spans all four arms, both variants,
and both trace sources. The smoke must establish:

- prompt, response, trace-prefix, tokenizer, candidate-token, axis, cap-config,
  model, and revision hashes;
- zero hook leakage and finite A/B/C/D logits;
- causal attention and batch-size-one execution;
- `uncapped` forward on the E2A uncapped trace reproduces its stored final-choice
  logits within the frozen E2 numerical tolerance;
- `full_p25` forward on the E2A p25 trace reproduces its stored final-choice
  logits within the same tolerance;
- no-hook and an uncapped registered path are equivalent.

The resulting smoke IDs and hashes are recorded before loading model weights.
Engineering failure after 12 wall-clock hours aborts E2B. Scientific direction
is not a smoke criterion.

## Frozen estimands

For each cell, with initial option `i`, induced target `t`, and the two remaining
options `o1,o2`, compute:

```text
m_target     = logit(t) - logit(i)
m_background = mean(logit(o1), logit(o2)) - logit(i)
m_general    = mean(logit(t), logit(o1), logit(o2)) - logit(i)
m_specific   = logit(t) - mean(logit(o1), logit(o2))
```

The two primary controlled contrasts are:

```text
delta_general_U = m_general(p25 | uncapped trace)
                  - m_general(uncapped | uncapped trace)

delta_general_P = m_general(p25 | p25 trace)
                  - m_general(uncapped | p25 trace)
```

The independent resampling unit is the 12-family cluster. Report the paired
estimate and deterministic 20,000-draw family-bootstrap percentile 95% interval
for each contrast using seed `20260726`. The two contrasts form one declared
primary family; no other E2B statistic may replace them.

Secondary, non-confirmatory readouts are:

- the average of `delta_general_U` and `delta_general_P`;
- the trace-source-by-cap interaction
  `delta_general_P - delta_general_U`;
- `m_target`, `m_background`, and `m_specific` counterparts;
- N/R/V/F arm breakdowns;
- V/F target-minus-initial and V-minus-F selectivity margins;
- option mass, full-vocabulary entropy, option argmax, and top-token integrity;
- leave-one-family-out estimates.

No generated-choice rate is a primary E2B endpoint because no token is sampled.

## Interpretation contract

- If both primary intervals lie wholly above zero, E2B supports the statement
  that visible trace content and length changes are not necessary for the
  positive general-margin shift under these fixed traces.
- If both intervals include zero and estimates are small, E2B does not establish
  a fixed-trace effect; trajectory generation, termination, or their interaction
  remains a plausible mediator.
- If the two trace sources disagree, report trace-by-cap interaction without
  averaging it away.
- Any direction is a completed result. No trace replacement, arm restriction,
  layer search, added condition, or extra sample is allowed after inspection.

Allowed wording: "conditional on the frozen generated trace" and "controlled
fixed-trace diagnostic." Forbidden wording: "natural direct effect," "complete
mediation," "the mechanism occurs only at the final answer," "final-layer
circuit," or "independent confirmation."

## Completion and stopping

Completion requires 384/384 finite records, passed replay and hook audits, a
clean result commit, and reporting of both primary contrasts regardless of
direction. Stop and retain artifacts on any hash drift, incomplete matrix,
hardware/configuration deviation, trace reconstruction ambiguity, or replay
Gate failure. E2B does not automatically authorize patching or another mechanism
experiment.
