# RG1-P Stage A: Fixed-Trace Assistant-Axis Propagation

## Status and precedence

This protocol implements **Amendment 01 (2026-07-27)** and the GPU-preflight
**Correction A-01 (2026-07-27)**. Where an earlier Stage A description
conflicts with either amendment, this document controls.
The assay is a controlled fixed-trace representation audit. It does not
generate natural answers and does not establish causal mediation.

Frozen base result: `6a7e286283cfc763b0dafec23d84fc91b031a99d`.
The model, tokenizer, chat template, forced KV-cache replay engine, terminal
probe, p25 asset hashes, thresholds, and layers are inherited byte-for-byte
from E2B cached replay v1.1.

## Frozen model configuration

- `Qwen/Qwen3-32B`, revision
  `9216db5781bf21249d130ec9da846c4624c16137`
- BF16, SDPA, batch size 1, eval/inference mode
- `model.generate_kv_cache_forced_tokens`; capture precedes token forcing
- official `layers_46:54-p0.25`, decoder layers 46--53
- no natural generation, retry, answer repair, p1, new data, or new seed

## Coordinate definitions

Layer 45 has no authorized official axis. Only `||delta h_45||` and relative
displacement are reported; it has no parallel/off-axis decomposition. It is a
preregistered reporting layer, not an engineering negative control. Under
cache-faithful replay, answer-boundary states below layer 46 may already carry
effects propagated through the earlier-token KV cache.

At layers 46--53, `a_l` is the normalized official Assistant Axis at layer
`l`. At layers 54--63, all official-p25 differences use the fixed exit
direction `a_exit = a_53`. Downstream quantities are named
`exit-axis-aligned component` and `off-exit-axis component`. They are not
called layer-specific downstream Assistant Axes.

The primary propagation estimand on treatment-naive traces is

`G_off = off_exit_axis_share_60 - off_exit_axis_share_53`.

This estimand measures departure from the injected direction, not entry into
an answer-decision subspace.

## Schedule- and norm-matched random-direction references

The reference battery is run only on `trace_source=e2a_uncapped`. Seeds are
`20260727` and `20260728`. For each seed and capped layer, one Gaussian vector
is sampled once, orthogonalized against `a_l`, normalized, frozen, and shared
by every item. Both signs are run, producing four conditions.

For each item, the official p25 replay is run first. At every layer and causal
token position it records the actual BF16 correction L2 norm. A random
reference injects only at those same layers and positions and uses that same
per-position norm. It never recomputes a trigger from the random trajectory.
The commanded FP32 vector norm must match the official norm to numerical
precision; the model-visible BF16 realized norm is separately audited at the
predeclared BF16 tolerance in `rg1p_common.py`. Any mismatch fails smoke.

Random downstream projections use that condition's signed layer-53 random
direction as its fixed exit direction. These four controls are a
`random-direction reference battery`, not a sampled null distribution and not
random capping.

## Frozen matrices

Smoke:

- 8 prompts x 2 trace sources x 2 official conditions = 32
- 8 uncapped-source prompts x 4 random references = 32
- 8 uncapped-source prompts x 1 independent uncapped repeat = 8 engineering
  no-op controls
- scientific matrix = 64; executed total = 72 replays

Formal:

- 96 prompts x 2 trace sources x 2 official conditions = 384
- 96 uncapped-source prompts x 4 random references = 384
- 96 uncapped-source prompts x 1 independent uncapped repeat = 96 engineering
  no-op controls
- scientific matrix = 768; executed total = 864 replays

The official cells are executed before random cells so that the immutable p25
schedule exists before use. Checkpoint resume may skip completed cells but may
not delete or rerun content records.

## State capture

At the E2B answer-boundary forward call, capture the residual block output for
every decoder layer, exactly `0..model.config.num_hidden_layers-1` (Qwen3-32B:
0--63). Layers 45, 53, 54, 60, and 63 remain the preregistered reporting layers
and the Stage-B patch locations remain frozen at 53 and 60; the full-depth
curve cannot be used to change either set. At capped layers, official p25
additionally captures cap input, cap output, immediate correction, and the
complete flattened causal schedule of per-position correction norms. Tensor
payloads are stored as BF16 safetensors and converted to float32 for analysis;
JSON contains only mappings, hashes, telemetry, logits, and audit summaries.
The 384 official rows are delivered as four condition/trace-source safetensor
shards (96 rows each, below the GitHub single-file limit) plus one unified
`hidden_states.index.json`. Random-reference and no-op states remain in their
immutable per-cell checkpoint safetensors and are indexed by `records.jsonl`.

For each paired condition and layer:

- `delta h_l = h_l(treatment) - h_l(uncapped)`
- signed aligned projection and its norm
- off-direction norm
- aligned/off-direction squared-energy shares
- `relative_delta = ||delta h_l|| / ||h_l(uncapped)||`

Shares must sum to one within numerical tolerance whenever the displacement is
nonzero.

## Native layer-wise output divergence

For each official uncapped/full-p25 pair and every captured layer `l`, apply
the frozen checkpoint's native `final_norm -> lm_head` without training or
fitting a translator. Let `z_l^c` be the resulting full-vocabulary logits.
The preregistered descriptive readouts are:

- `KL_opt(l) = KL(q_l_uncapped || q_l_full_p25)`, where each `q` is the
  softmax restricted and renormalized over token IDs A/B/C/D = 32/33/34/35;
- `KL_vocab(l)` over the full vocabulary, appendix only;
- `m_general(l) = mean(noninitial option logits) - initial option logit` and
  its paired full-p25-minus-uncapped difference;
- paired state L2 displacement, state cosine, and both state L2 norms.

The final-layer native-lens candidate logits must match the logits captured by
the E2B answer processor with max absolute error strictly below `1e-3`.
`KL_opt` and layer-wise `m_general` are the main plotted curves; `KL_vocab` is
appendix-only. Early or middle layer values are constrained by native-logit-
lens readability. No tuned lens is permitted, and a curve transition cannot
be called the origin layer of the mechanism.

The engineering negative control is a real second uncapped replay for each
treatment-naive trace. It is excluded from every scientific estimand. Across
all 64 layers it must satisfy `KL_opt < 1e-6` and `||delta h||_2 < 1e-4`.
Failure at any layer invalidates Stage A and blocks formal execution.

## Statistics

- primary source: `e2a_uncapped`; p25-source is sensitivity only
- independent unit: family (12 families)
- percentile family bootstrap: 20,000 draws, seed 20260727, two-sided 95% CI
- full-depth descriptive curves: 10,000 family-cluster draws with the same
  seed; no row-level or per-layer hypothesis tests
- report family values, majority direction, and leave-one-family-out range
- N/R/V/F are reported separately at layers 53, 54, 60, and 63
- p25 is compared pairwise with each of the four frozen random references for
  `G_off`, final margin shift, option mass, entropy, and top-token integrity

Panel C is descriptive. It may show twelve labeled family points and a
descriptive Spearman coefficient, but no regression line or row-level p-value.
It is not a Route A hard gate.

V/F probes are auxiliary: L2 logistic regression, family-held-out folds,
training-fold-only scaling/PCA (at most 10 PCs), and 4096 family label-flip
permutations. A family-held-out TF-IDF word/character baseline is mandatory.
The permitted claim is condition decodability, not understanding or causal
availability of validity.

## Engineering gates

The CPU Gate verifies the E2B hashes and matrix, model/assets, terminal-probe
contract, module mapping, axis orientation, deterministic random directions,
family bootstrap, family-blocked probes, tests, lint, and clean synchronized
GPU commit.

Smoke requires 72/72 rows, unchanged official E2B logits within `1e-4`, exact
forced-trace hashes, complete tensors, no nonfinite value or hook leak,
official correction alignment, exact random schedule positions
with per-position norm matching, complete 0--63 module mapping, final-layer
native-lens agreement below `1e-3`, and the all-layer no-op thresholds above.
Any schedule mismatch blocks formal; no tolerance, trace,
dtype, GPU, or model substitution is permitted after seeing smoke.

Formal audit requires 864/864 unique cells: 384 official, 384 random-reference,
and 96 engineering no-op rows. It requires all expected condition counts,
complete tensors, unchanged official/no-op replay logits, complete full-depth
layer metrics, no schedule mismatch, no hash failure, and no hook leak.

## Result routing

Route A requires all of: engineered official alignment; positive `G_off` with
CI above zero; at least 7/12 families positive; at least three of four arms
positive; p25 exceeds all four random references on `G_off`, or differs from
all four on margin with consistent signed intervals; and no random-reference
output collapse. It supports only a fixed-trace, direction-referenced
redistribution statement.

Route B applies when p25 propagation and margin effects overlap the random
battery: downstream rotation is consistent with generic residual-stream
perturbation.

Route C applies when off-direction rotation is generic but the p25 margin shift
has the opposite sign from the random references: geometry rotation is generic
while its answer-space consequence is direction-dependent.

Route D applies when family, arm, or random-seed stability is insufficient.
Negative, zero, mixed, and collapsed-control outcomes are retained unchanged.

## Figures and claims

Panel A contains the official p25 curve, all four light-gray random curves and
their range, with layers 53 and 60 preregistered. Its caption states that p25
uses fixed `a_53` downstream and random conditions use their corresponding
`r_53`; four references are not a random null distribution. Panel B is the arm
breakdown. Panel C is the descriptive family association without a fitted
trend line.

Panel D is the descriptive layer-resolved emergence profile over layers 0--63.
Its upper subpanel shows layer-wise delta mean-noninitial `m_general`; its
lower subpanel shows `KL_opt`. U and P trace sources are separate (U solid, P
dashed), and the cap window 46--53 is shaded. An optional N/R/V/F U-source
curve is allowed. The caption must state: "cache-faithful replay; divergence
below layer 46 reflects cache-propagated effects, not direct rectification.
Logit-lens readout; descriptive." It must not localize an origin layer.

Stage A stops after audit, analysis, figures, and delivery. It does not launch
activation patching. Stage B may only be discussed after the separate
2026-07-28 12:00 Asia/Shanghai paper-completeness gate is fully satisfied.

The layer profile compares the same frozen visible token trajectory with only
the internal intervention condition changed. This removes prompt-text and
trajectory-length confounding from the paired profile, but the profile itself
remains descriptive: it is not a mediation test and cannot localize a causal
origin layer. Claims that capping has no effect before layer 46, that the curve
is causal evidence, or that the result generalizes across models, axes, or
doses are forbidden.
