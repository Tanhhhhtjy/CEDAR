# E2-Phase: Validity–Choice Dissociation and Stage Localization

## Status and anchor

E2 is an exploratory causal-localization assay. It does not revise the frozen
Evidence result and must not be described as a held-out confirmation.

- Evidence result anchor: `a0ad950ef160b3aec1900d9c005e9b83853b291e`
- Evidence CPU Gate anchor: `4814946fa3a725249a8e6741ce910ceac0f602fa`
- split: `development_validation`
- model: `Qwen/Qwen3-32B`
- revision: `9216db5781bf21249d130ec9da846c4624c16137`
- official p25: `layers_46:54-p0.25` (layers 46–53)
- thinking, greedy, seed 0, batch size 1, `max_new_tokens=4096`
- no content retry, answer repair, item selection, or manual answer credit

The 24 bases and all N/R/V/F texts remain unchanged. New files are derived
views only.

## Research questions

E2 distinguishes five non-exclusive explanations:

1. lower final update/decision threshold;
2. impaired V/F sufficiency discrimination;
3. a changed or shortened thinking path;
4. general compression of the initial option's advantage;
5. target-specific amplification.

## Independent assays

### Answer assay

The existing 96 N/R/V/F prompts are reused byte-for-byte as structured JSONL
objects. Output remains `{"answer":"A|B|C|D"}`. The formal matrix is:

```text
96 inputs × 4 conditions = 384 generations
```

### Validity assay

The 48 V/F prompts are independently rendered as quoted evidence. The model is
asked whether the candidate reasoning logically suffices both to warrant the
proposed edit and to make the proposed target uniquely supported. Output is:

```json
{"judgment":"A|B"}
```

A/B always appear in display order. Their sufficient/insufficient meanings are
swapped by a frozen base-level mapping. V is sufficient and F is insufficient.
Validity and Answer are separate model calls.

## Mapping feasibility amendment

The frozen 24-base graph makes exact simultaneous balance within family,
easy/medium variant, and all 12 position transitions impossible when V/F share
one base-level mapping. An exhaustive search over all `2^12` family-balanced
assignments proved the attainable optimum:

- overall sufficient label: A=12, B=12;
- each family: one A mapping and one B mapping;
- easy/medium: each A=6, B=6;
- position transitions: 10/12 exactly balanced;
- `B→D` has zero A mappings and `D→C` has two A mappings.

This amendment is frozen before GPU execution. The statistical model includes
label mapping and position transition. No result-dependent remapping is
allowed.

## E2A phase intervention

The four formal conditions are a 2×2 design:

| condition | prefill cap | decode cap |
|---|---:|---:|
| uncapped | off | off |
| prefill_only | on | off |
| decode_only | off | on |
| full_p25 | on | on |

`full_p25` retains the historical `ActivationCapper` intervention path. A
read-only phase monitor is registered before it to collect phase telemetry.
`prefill_only` and `decode_only` use `PhaseActivationCapper`. Batch size is one;
sequence length greater than one is prefill and sequence length one is decode.

```text
Answer:   96 × 4 = 384
Validity: 48 × 4 = 192
Total:             576
```

## Anchor equivalence Gate

Before phase interpretation, the new runner's 96 Answer `uncapped` and 96
Answer `full_p25` records must match the frozen anchor on:

- prompt and prompt-token hashes;
- generated-token hashes;
- parsed answer and four-state outcome;
- truncation, parse, and answer-boundary indices;
- A/B/C/D logits within the frozen numerical tolerance;
- hook cleanup and p25 asset identity.

Failure stops E2A interpretation. Directional similarity is insufficient.

## Metrics and analysis unit

The independent unit is the 12-family cluster; 24 bases and repeated arms/
conditions are nested measurements. Primary readouts are Validity
discrimination, Answer selectivity, and common V/F update tendency. Secondary
readouts include all option-relative margins, entropy, option mass, reasoning
length, completion status, and phase/layer dose telemetry.

Absolute logit shifts are not used to infer target specificity. The required
relative geometry is:

```text
m_target     = target - initial
m_background = mean(other two wrong) - initial
m_general    = mean(all three noninitial) - initial
m_specific   = target - mean(other two wrong)
```

## CPU Gate and GPU boundary

GPU use is forbidden until all are true:

1. 96 Answer and 48 Validity inputs rebuild deterministically;
2. single-reviewer Validity audit passes 48/48;
3. pinned tokenizer boundaries and p25 assets pass CPU preflight;
4. phase hooks pass fake-model timing, telemetry, no-op, and cleanup tests;
5. runner, audit, and analysis entry points pass tests and lint;
6. release hashes are frozen in `manifest.json`;
7. `E2_CPU_GATE.json` authorizes only E2A;
8. the Gate commit is pushed and local/upstream/remote SHA agree.

E2B teacher forcing, activation patching, gradient/JVP, J-Lens, p1, tuning,
new capping methods, and item selection are not authorized by this Gate.

## Stop rules

Stop and retain the failed artifact if hashes drift, the worktree is dirty,
HEAD differs from upstream, GPU topology changes, hooks leak, a phase applies
outside its scope, the Anchor equivalence check fails, or the matrix is
incomplete. An undesirable scientific direction is not a retry reason.
