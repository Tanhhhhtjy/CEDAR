# Uncapped thinking-mode diagnostic contract

This run is a development diagnostic. It is not held-out or confirmatory and does not authorize capping.

## Required conversation

For each row in `m2_2b_v4_public_poc180_arms.jsonl`, send exactly:

1. `system`
2. `initial_user`
3. the fixed `initial_assistant` from the row
4. `followup_user`
5. generate one new assistant response

Do not regenerate the initial assistant message. Do not pass gold, audits, family IDs, error classes, or other arms to the model.

## Required Qwen configuration

- Same Qwen checkpoint and revision as the existing M2.2b diagnostic unless a new run name explicitly records a different model.
- `enable_thinking=True` in the Qwen chat-template call.
- `do_sample=False` and temperature recorded as 0.
- `max_new_tokens>=2048`; the old 192-token limit is not valid for this PoC.
- Save the complete raw continuation, including any `<think>...</think>` content.
- Do not parse, repair, retry, or filter an item based on its answer. A transport failure may be retried under the normal atomic-checkpoint policy.
- `cap=uncapped`; do not load or invoke activation capping.

Illustrative Transformers call:

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
```

The exact argument name must be verified against the pinned tokenizer version. Record the rendered prompt hash and confirm on a smoke item that a reasoning segment is actually produced; merely setting a manifest flag is insufficient.

## Output format

Store one JSONL row per record:

```json
{"id":"M22B-V4-PUB-001-N","response":"<complete raw continuation>","cap":"uncapped","thinking_enabled":true}
```

Score with:

```powershell
python score_m2_2b_v4_public_poc.py results.jsonl --output score_report.json
```

The scorer ignores text before the final `</think>` marker and extracts the final JSON answer. It does not use the reasoning trace as a judge.

## Interpretation

This PoC deliberately changes both the data generator and the deliberation contract relative to v3. A better result would show that Qwen can discriminate when formal evidence is machine-checkable, F has a local one-step flaw, and thinking is enabled. It would not establish that the old held-out result was wrong, nor may the two versions be pooled.
