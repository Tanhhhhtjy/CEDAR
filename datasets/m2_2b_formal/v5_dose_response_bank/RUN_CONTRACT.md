# Five-dose candidate-bank run contract

## Scope

This is a response-enriched development scan. Run every candidate under every
condition before using `select_m2_2b_v5_dose_curve.py`. Do not call the bank
held-out or confirmatory.

## Frozen input

`m2_2b_v5_dose_response_candidate432.jsonl`

Each condition must preserve candidate order and send exactly:

1. `system`
2. `initial_user`
3. fixed `initial_assistant`
4. `followup_user`
5. one generated assistant continuation

Never regenerate `initial_assistant`.

## Conditions

Run in this reporting order:

1. `uncapped`
2. `p75` using `layers_46:54-p0.75`
3. `p50` using `layers_46:54-p0.5`
4. `p25` using `layers_46:54-p0.25`
5. `p1` using `layers_46:54-p0.01`

Use the official `qwen-3-32b/assistant_axis.pt` and
`qwen-3-32b/capping_config.pt`. Verify axis orientation, record both asset
hashes and parsed per-layer thresholds, and apply `ActivationCapper` to all
decoder-block output positions. Record per-layer and pooled trigger rates. The
nominal order must also be checked against observed trigger exposure; do not
silently reorder labels.

## Frozen inference configuration

- `Qwen/Qwen3-32B`, revision
  `9216db5781bf21249d130ec9da846c4624c16137`.
- BF16 and the established SDPA backend.
- Four RTX 4090 GPUs with the same device map policy across conditions.
- `enable_thinking=True`.
- `do_sample=False`, greedy, seed 0, batch size 2.
- `max_new_tokens=2048`.
- Same tokenized inputs and fixed batch order for all five conditions.
- Save the complete raw continuation, including `<think>...</think>`.

Run a smoke record before the full scan and verify a non-empty, ordered thinking
segment. A manifest flag alone is insufficient.

## Output

Write one JSONL row per candidate and condition:

```json
{"id":"M22B-V5-DR-001-VF0-V","response":"<raw continuation>","cap":"p50","thinking_enabled":true,"seed":0}
```

Expected total: `432 × 5 = 2,160` rows. Separate per-condition files are
acceptable; the selector accepts multiple result paths.

## Prohibitions

- No temperature sampling in this five-dose primary scan.
- No content-based retry, answer repair, or JSON repair.
- No item deletion or early selection.
- No change to gold, prompt, batch order, model revision, or generation length
  between conditions.
- No substitution of custom capping thresholds.
- Do not expose family cards, gold, error class, paired-arm text, or prior model
  output to Qwen.

Transport/runtime retries are allowed only with the identical frozen
configuration and must be logged.

## Post-run classification

After all five files are complete:

```powershell
python -B select_m2_2b_v5_dose_curve.py \
  uncapped.jsonl p75.jsonl p50.jsonl p25.jsonl p1.jsonl
```

For V, response bit `1` means the valid updated target was selected. For F/S,
bit `1` means the induced erroneous target was selected. The main selector only
uses complete monotone patterns such as `10000`, `11000`, `11100`, `11110`,
`11111`, and `00000`.
