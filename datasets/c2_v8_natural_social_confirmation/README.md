# C2/V8 natural social confirmation dataset

This directory contains the frozen zero-GPU construction for C2/V8. It is derived from the
official SycoBench-600 `data/questions.json` at commit
`3b2d3680b0a6d456e618206e6281904c87cb49dd`.

The upstream dataset and this adapted dataset are distributed under CC BY 4.0. Retain attribution
to Debu Sinha and cite:

> Sinha, Debu. SycoBench-600: Measuring Sycophancy and Correction Selectivity in LLM Assistants.
> Findings of ACL 2026. DOI: 10.18653/v1/2026.findings-acl.1759.

## Files

- `source_syco600.official.json`: exact pinned upstream bytes.
- `source_syco600.jsonl`: canonical JSONL conversion of the same 600 objects.
- `c2_v8_base.jsonl`: 600 frozen bases.
- `c2_v8_ao_s_arms.jsonl`: 1,200 paired AO/S records.
- `c2_v8_free_generation_subset.jsonl`: preselected 240-base generation subset.
- `c2_v8_smoke12.jsonl`: engineering-smoke base set.
- `review/review_packet_a.csv` and `review_packet_b.csv`: independently ordered 120-pair
  human-review packets with blank judgment columns.
- `review/review_metadata_template.json` and `adjudication_template.csv`: templates for the
  reviewer identity attestation and any disagreement adjudication.
- `manifest.json`, `source_manifest.json`, and audit JSON files: provenance and hashes.

## Rebuild and validate

```bash
conda run -n cedar python scripts/build_c2_v8_natural_confirmation.py --offline
conda run -n cedar python \
  datasets/c2_v8_natural_social_confirmation/validate_c2_v8.py --stage source
conda run -n cedar python \
  datasets/c2_v8_natural_social_confirmation/validate_c2_v8.py --stage construction
```

## Human review gate

Give `review_packet_a.csv` and `review_packet_b.csv` to two different reviewers. They must work
independently without seeing model outputs. Each reviewer fills every `*_yes_no` field with exactly
`yes` or `no`, then returns files named `review_a_completed.csv` and `review_b_completed.csv`.

Create `review_metadata.json` manually with real identities:

```json
{
  "reviewer_a": "real reviewer identifier",
  "reviewer_b": "different real reviewer identifier",
  "independent": true,
  "model_outputs_seen": false
}
```

If reviewers disagree, add `adjudication.csv` containing the frozen `review_id` and final values for
the five judgment fields. Items jointly rejected by both reviewers cannot be adjudicated into a
pass; the builder/templates must be corrected and both reviewers must review the rebuilt packets.

Then run:

```bash
conda run -n cedar python \
  datasets/c2_v8_natural_social_confirmation/validate_c2_v8.py --stage review
```

Until `HUMAN_REVIEW_GATE.json` is generated with `status: passed`, no C2/V8 GPU run is authorized.
After it passes, follow `docs/C2_V8_NATURAL_SOCIAL_CONFIRMATION_RUN_LOG.md`; the runner enforces
the frozen smoke, direct, audit, and free-track order rather than relying on operator convention.
