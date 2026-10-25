# C2/V8 natural social confirmation dataset

This directory contains the frozen zero-GPU construction for C2/V8. It is derived from the
official SycoBench-600 `data/questions.json` at commit
`3b2d3680b0a6d456e618206e6281904c87cb49dd`.

The upstream dataset and this adapted dataset are distributed under CC BY 4.0. Retain attribution
to Debu Sinha and cite:

> Sinha, Debu. SycoBench-600: Measuring Sycophancy and Correction Selectivity in LLM Assistants.
> Findings of ACL 2026. DOI: 10.18653/v1/2026.findings-acl.1759.

## Files

- `source_syco600.official.json`: pinned upstream JSON content; local line endings are not part of
  its scientific identity.
- `source_syco600.jsonl`: canonical JSONL conversion of the same 600 objects.
- `c2_v8_base.jsonl`: 600 frozen bases.
- `c2_v8_ao_s_arms.jsonl`: 1,200 paired AO/S records.
- `c2_v8_free_generation_subset.jsonl`: preselected 240-base generation subset.
- `c2_v8_smoke12.jsonl`: engineering-smoke base set.
- `review/review_packet_a.csv` and `review_packet_b.csv`: independently ordered 120-pair
  human-review packets with blank judgment columns.
- `review/review_metadata_template.json` and `adjudication_template.csv`: templates for the
  original two-reviewer path.
- `review/single_review_attestation.json`: the disclosed data-lead review waiver used by this run.
- `manifest.json`, `source_manifest.json`, and audit JSON files: provenance and hashes.

## Rebuild and validate

```bash
conda run -n cedar python \
  datasets/c2_v8_natural_social_confirmation/build_c2_v8_natural_social_confirmation.py \
  --offline
conda run -n cedar python \
  datasets/c2_v8_natural_social_confirmation/validate_c2_v8_natural_social_confirmation.py \
  --stage source
conda run -n cedar python \
  datasets/c2_v8_natural_social_confirmation/validate_c2_v8_natural_social_confirmation.py \
  --stage construction
```

## Human review gate

`docs/C2_V8_PROTOCOL_AMENDMENT_01_SINGLE_REVIEW_WAIVER.md` changes this execution to a disclosed
single-data-lead full review. It does not claim two independent reviewers, raw agreement, or Cohen's
kappa. The attestation binds the current 120-pair packet, pair IDs, 600 bases, 1,200 arms, and the
amendment by portable content hashes. The validator retains the original dual-review path for a
future execution, but the paths cannot be mixed.

Then run:

```bash
conda run -n cedar python \
  datasets/c2_v8_natural_social_confirmation/validate_c2_v8_natural_social_confirmation.py \
  --stage review
```

Until `HUMAN_REVIEW_GATE.json` is generated with `status: passed`, no C2/V8 GPU run is authorized.
After it passes, follow `docs/C2_V8_NATURAL_SOCIAL_CONFIRMATION_RUN_LOG.md`; the runner enforces
the frozen smoke, direct, audit, and free-track order rather than relying on operator convention.

Tracked UTF-8 text uses LF-normalized SHA-256 plus Git blob IDs as execution gates. Raw byte hashes
and byte counts are diagnostics only. Parsed JSON/JSONL/CSV scientific content uses ordered object
hashes, so CRLF/LF or insignificant JSON formatting cannot change data identity.
