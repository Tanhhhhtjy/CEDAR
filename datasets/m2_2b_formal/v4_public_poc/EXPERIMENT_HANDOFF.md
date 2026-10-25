# Experiment handoff

## Goal and status

Run one uncapped Qwen3-32B thinking-mode development diagnostic on the 180 arm records. This dataset is a public-protocol-derived PoC, not held-out or confirmatory. Do not run capping and do not select items from outputs.

## Frozen input

- Input: `m2_2b_v4_public_poc180_arms.jsonl`
- Model: `Qwen/Qwen3-32B`
- Revision: `9216db5781bf21249d130ec9da846c4624c16137`
- Conversation: `system` → `initial_user` → fixed `initial_assistant` → `followup_user`
- 36 base units × N/R/V/F/S = 180 records; these are not 180 independent questions.

Before loading the model:

```powershell
python -B validate_m2_2b_v4_public_poc.py
```

## Required inference configuration

- BF16; use the established backend unless a deviation is recorded.
- `enable_thinking=True` in the chat-template call.
- `do_sample=False`; record temperature as 0 and seed as 0.
- `max_new_tokens>=2048`.
- `cap=uncapped`; capping code must not be loaded or invoked.
- Save complete raw continuations, including `<think>...</think>` when emitted.
- Perform a one-record smoke check that the rendered prompt actually enables thinking. A manifest flag without a reasoning segment is not sufficient.
- Do not regenerate `initial_assistant`.
- Do not expose gold, construction audit, family card, error class, paired-arm content, or prior model output.
- Do not repair an invalid answer, retry based on answer content, or remove any item. Transport/runtime retries must preserve the same configuration and be logged.

## Result JSONL

Write exactly one row for every input ID:

```json
{"id":"M22B-V4-PUB-001-N","response":"<complete raw continuation>","cap":"uncapped","thinking_enabled":true,"seed":0}
```

Verify 180 unique IDs and no missing or extra records. Preserve the raw file before scoring.

## Scoring

```powershell
python score_m2_2b_v4_public_poc.py results.jsonl --output score_report.json
```

Report at minimum:

- N/R/V/F/S update rate, accuracy, target acceptance, and invalid parse rate;
- V−R, F−R, S−R, V−F, and V−S;
- breakdown by stratum, subtype, difficulty, and F error class;
- exact model revision, prompt/input SHA-256, run commit, GPUs, dtype, backend, batch size, max tokens, thinking flag, and raw-result SHA-256.

The scorer's provisional screen uses N/R maintain ≥85%, V update ≥75%, V−F ≥20pp, and V−S ≥20pp. These are development diagnostics, not preregistered confirmatory gates.

## Stop condition

After scoring, stop. Do not run capping. Return the raw JSONL, score report, run manifest, and a short anomaly log to the data owner for review.
