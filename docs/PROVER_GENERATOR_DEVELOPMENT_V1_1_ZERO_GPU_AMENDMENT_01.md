# Prover-Generator Development v1.1 Zero-GPU Amendment 01

Status: engineering amendment before Gate A1 and before any GPU execution
Date: 2026-07-24

## Trigger

The frozen Gate A1 command failed before producing a tokenizer result for two
platform-layout reasons:

1. `--cache-dir "$HF_HOME"` was passed to Transformers as the cache directory,
   while the existing pinned snapshot is stored under the standard
   `$HF_HOME/hub` layout.
2. Several frozen parent text hashes were computed from Windows CRLF bytes.
   Git's Linux checkout contains the same UTF-8 text with LF endings, causing
   raw-byte identity checks to fail.

No Qwen generation, GPU use, capping, behavioral inspection, or usable Gate A1
output occurred before this amendment.

## Amendment

- The tokenizer preflight accepts either an HF home directory containing
  `hub/` or the hub cache directory itself. It remains offline and continues to
  require the pinned Qwen revision.
- Parent `.json` and `.jsonl` identity uses `utf8_lf_canonical_v1`: decode as
  UTF-8, normalize CRLF and CR to LF, then compute SHA-256. Raw hashes remain
  available elsewhere for diagnostics but are not a cross-platform hard gate.
- Regression tests cover HF-home resolution, LF/CRLF equivalence, and
  sensitivity to an actual character change.

## Scientific invariants

This amendment does not change any symbolic world, question, option order,
gold/initial/target label, arm, V/F content, AO/S1/S2 template, split, runner,
threshold, seed, generation setting, model, tokenizer, or model revision. It
does not authorize p25, p1, development validation, the full 48-base run, a
confirmation set, LoFa, J-Lens, or any mechanism experiment.
