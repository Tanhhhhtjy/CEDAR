# Experiment handoff

## Input

`m2_2b_v5_dose_response_candidate432.jsonl`

Validate before loading Qwen:

```bash
python -B datasets/m2_2b_formal/v5_dose_response_bank/validate_m2_2b_v5_dose_response_bank.py
```

## Required scan

Run all 432 candidates under each condition:

`uncapped`, `p75`, `p50`, `p25`, `p1`

Total expected generations: 2,160. Use Qwen3-32B revision
`9216db5781bf21249d130ec9da846c4624c16137`, BF16, thinking enabled,
greedy decoding, seed 0, batch 2, and 2,048 new tokens. Preserve complete raw
thinking output.

Official experiment IDs:

- p75: `layers_46:54-p0.75`
- p50: `layers_46:54-p0.5`
- p25: `layers_46:54-p0.25`
- p1: `layers_46:54-p0.01`

The full requirements, asset checks, and prohibitions are in `RUN_CONTRACT.md`.

Run each condition with the supplied resumable runner, for example:

```bash
python scripts/run_m2_2b_v5_dose_bank.py --condition uncapped --stage all
python scripts/run_m2_2b_v5_dose_bank.py --condition p75 --stage all
python scripts/run_m2_2b_v5_dose_bank.py --condition p50 --stage all
python scripts/run_m2_2b_v5_dose_bank.py --condition p25 --stage all
python scripts/run_m2_2b_v5_dose_bank.py --condition p1 --stage all
```

## Return artifacts

For every condition return:

- raw result JSONL with exactly 432 IDs;
- manifest and runtime configuration;
- rendered-prompt and tokenized-input hashes;
- per-layer and pooled cap trigger statistics;
- missing, duplicate, invalid-JSON, missing-thinking, and truncation counts.

Do not select candidates on the experiment machine. After all five result
files are pushed, the data owner runs:

```bash
python -B datasets/m2_2b_formal/v5_dose_response_bank/select_m2_2b_v5_dose_curve.py \
  uncapped.jsonl p75.jsonl p50.jsonl p25.jsonl p1.jsonl
```

The default selector stops if the exact family-grouped curve is unavailable.
