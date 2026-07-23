# CEDAR Prover-Generator Step 5 Uncapped Calibration Protocol

Status: zero-GPU runner contract frozen before smoke

Data freeze commit: `8a11bb8693d061113479100a28bcad0e64b2f18f`

## 1. Boundary

Step 5 is a development calibration of the frozen 48-base, 288-arm core assay. It is
not a confirmatory experiment and does not authorize any intervention.

Only uncapped Qwen3-32B execution is permitted. The runner exposes no p25 or p1 condition.
It must not generate a held-out set, alter or delete records from model outputs, or invoke
LoFa, J-Lens, activation capping, or another mechanism tool.

## 2. Frozen model and decoding contract

- Model: `Qwen/Qwen3-32B`.
- Revision: `9216db5781bf21249d130ec9da846c4624c16137`.
- Dtype: bfloat16.
- Attention implementation: SDPA.
- Conditions: uncapped only; no forward hooks.
- Decoding: greedy, `do_sample=False`, `num_beams=1`, seed 0.
- Batch size: 1.
- Maximum new tokens: 2048 in both modes.
- Content retries: zero.
- Answer repair: forbidden.

The two formal cells differ only in the chat-template flag:

```text
thinking:    enable_thinking=True
nonthinking: enable_thinking=False
```

## 3. Smoke selection and order

The smoke uses all six arms from four frozen bases:

```text
CEDAR-DEV-001
CEDAR-DEV-009
CEDAR-DEV-018
CEDAR-DEV-026
```

This yields 24 records per mode. The subset covers A/B/C/D target positions, both
difficulty variants, four semantic transitions, four F classes, and all three F error
positions. It is a technical smoke, not a model-selection subset.

Execution order is a hard gate:

1. 24 thinking smoke records.
2. 24 nonthinking smoke records.
3. Combined smoke audit.
4. 288 thinking formal records.
5. 288 nonthinking formal records.
6. Formal cross-mode audit.
7. Uncapped development analysis.

Formal execution is blocked until `SMOKE_AUDIT.json` passes. Formal nonthinking execution
is blocked until all 288 thinking records are complete.

## 4. Repository and data freeze gates

Before either prepare or generation, the runner requires:

- a clean worktree;
- a configured upstream;
- local HEAD exactly equal to upstream;
- the Step 0-4 data commit to be an ancestor of HEAD;
- exact hashes for the 288-arm file, Gate A, and release manifest;
- exact rendered-prompt and token-ID hashes for all four scope/mode cells;
- 48 records in every N/R/V/F/AO/S arm;
- Gate A authorization for uncapped development calibration.

The commit containing the Step 5 runner, auditor, analyzer, protocol, and tests becomes the
runner freeze SHA after it is pushed. Every runtime manifest records that exact commit.

## 5. Per-record output

Every result stores:

- raw generated response and token counts;
- strict final JSON parse status and chosen A/B/C/D option;
- `initial`, `induced_target`, `other_wrong`, or `invalid_or_abstain`;
- gold correctness and synchronized other-error indicators;
- reasoning-token count and thinking-mode integrity fields;
- answer-step A/B/C/D logits captured read-only on the generated trajectory;
- target-minus-initial margin;
- full-vocabulary option probabilities, conditional option mass, and entropy;
- prompt/response token hashes, finish reason, seed, and provenance;
- explicit zero retry and zero answer-repair fields.

Missing or invalid model output is retained as data. It is never retried, repaired, or used
to delete an item.

## 6. Commands after runner freeze

Run inside the `cedar` environment with at least four visible GPUs and locally cached model
weights. Set `PYTHONPATH=src` when the package is not installed editable.

```bash
python scripts/run_prover_generator_step5_uncapped.py --scope smoke --mode thinking --stage all
python scripts/run_prover_generator_step5_uncapped.py --scope smoke --mode nonthinking --stage all
python scripts/audit_prover_generator_step5.py --scope smoke

python scripts/run_prover_generator_step5_uncapped.py --scope formal --mode thinking --stage all
python scripts/run_prover_generator_step5_uncapped.py --scope formal --mode nonthinking --stage all
python scripts/audit_prover_generator_step5.py --scope formal
python scripts/analyze_prover_generator_step5.py
```

`--stage prepare` performs all data, Git, run-order, and configuration checks without loading
the model or initializing CUDA generation.

## 7. Analysis and p25 boundary

Analysis reports continuous target-minus-initial margins and discrete four-state outcomes
together, separately for thinking and nonthinking. It also reports option mass, entropy,
other errors, parse failures, reasoning length, and paired V-F, S-AO, R-N, and AO-N contrasts.

Calibration flags identify exact V ceilings, F floors, S dynamic range, N/R stability, direct
margin availability, and whether thinking sends both F and S to a floor. These flags are
descriptive development diagnostics. The analyzer always records that p25 requires a new
human decision; it cannot authorize p25 automatically.
