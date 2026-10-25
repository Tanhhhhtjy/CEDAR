# Step 5 Amendment 01: Robust Read-Only Answer-Logit Alignment

Status: zero-result engineering amendment before a successful smoke
Date: 2026-07-24
Original frozen runner: `2cecc8dc63e4c038c8a7b86400d440fe091f9167`

## Trigger

The first thinking-smoke generation (`CEDAR-DEV-001-N`) reached a strictly parseable answer but
the original runner raised `strict answer lacks aligned on-trajectory logit capture` before it
could write a checkpoint. The pipeline stopped immediately. Both `results.jsonl` and
`record_runtime.jsonl` remained empty; nonthinking smoke, formal generation, audit, and analysis
did not run. No behavioral aggregate or retained model response was available when this amendment
was defined.

Static inspection identified two instrumentation mismatches. The strict parser accepts ordinary
JSON whitespace, while the read-only logit processor watched only the compact token sequence
`{"answer":"`. Separately, the boundary locator required a full decode/re-encode round trip even
when the logit processor had already captured an exact on-trajectory candidate token matching the
strictly parsed final answer.

## Amendment

- The read-only processor still returns the score tensor unchanged, but recognizes semantically
  identical JSON whitespace variants. A bounded decoded-tail regex is used only when compact token
  variants do not match.
- If the global tokenizer round-trip locator is unavailable, the runner may align to the last
  captured on-trajectory A/B/C/D token that matches the strictly parsed answer. The original
  locator error remains recorded as diagnostic metadata.
- Any future unresolved mismatch remains a hard failure, but the exception now includes the raw
  response tail, token-ID tail, boundary report, and capture candidates.
- The run manifest records
  `step5-amendment-01-robust-answer-logit-alignment`, and the auditor requires that identity.

Tests cover compact capture, JSON whitespace and newlines, fallback alignment, and audit identity.
The failed zero-record attempt is archived rather than deleted before restarting smoke.

## Scientific invariants

This amendment does not change the 288 frozen records, their order, prompts, initial/target/gold
mapping, model or tokenizer revision, BF16/SDPA configuration, thinking toggle, greedy decoding,
seed, maximum tokens, answer parser, selected answer, logits, or any activation. It adds no p25,
p1, confirmation data, LoFa, J-Lens, or activation intervention. It changes only read-only
measurement alignment and failure observability, before any usable Step 5 result existed.
