# M2.2b v4 public-protocol deliberative PoC

Status: `development_poc_ready_for_uncapped_thinking_diagnostic`.

This is a newly versioned development dataset designed to test whether Qwen can separate valid evidence revision from clear invalid arguments when allowed and instructed to deliberate. It is not a repair of the exposed v3 held-out set.

## Construction

- 18 structural families, each with one easy and one medium variant.
- 36 base units and 180 N/R/V/F/S arm records.
- Three strata: Rule Discovery, Circuit Diagnosis, and Minimal Premise Logic; 12 bases each.
- Refute/undercut: 18/18. Easy/medium: 18/18.
- Initial and updated A/B/C positions: 12 each.
- Six directed answer transitions: 6 each.
- Every F has one named error class and one local check that directly defeats the invalid step.
- All examples are newly written. Public sources provide protocol lineage only; see `SOURCE_AUDIT.md`.

The formal certificate stores surviving hypotheses under initial, V, F, and N/R/S evidence states. The validator recomputes the resulting answer label from these sets. This is a compact symbolic certificate, not a claim that author audit can replace later human review.

## Files

- `build_m2_2b_v4_public_poc.py`: deterministic source of all examples.
- `family_cards18.jsonl`: structural and provenance cards.
- `m2_2b_v4_public_poc36_base.jsonl`: clustered bases with five nested arms.
- `m2_2b_v4_public_poc180_arms.jsonl`: model-run input; these are arms of 36 bases, not 180 independent samples.
- `construction_audit36.jsonl`: author audit, explicitly not independent human review.
- `validate_m2_2b_v4_public_poc.py`: structure, balance, certificate, prompt, and hash checks.
- `leakage_check_m2_2b_v4_public_poc.py`: Observation-only family-grouped lexical diagnostic.
- `score_m2_2b_v4_public_poc.py`: external-response scorer with thinking-output parsing.
- `RUN_CONTRACT.md`: mandatory uncapped thinking-mode run instructions.
- `EXPERIMENT_HANDOFF.md`: concise experiment-owner checklist and required return artifacts.
- `SOURCE_AUDIT.md`: source commits and no-copy boundary.
- `manifest.json`: counts, provenance, run status, and SHA-256 hashes.

## Reproduce and validate

Run in this directory:

```powershell
python -B build_m2_2b_v4_public_poc.py
python -B leakage_check_m2_2b_v4_public_poc.py
python -B validate_m2_2b_v4_public_poc.py
```

Run the builder first because it resets the leakage status to `not_run`; the leakage script writes its report and updates the manifest.

## Boundary

- Qwen has not been run during construction.
- Capping has not been run and is not authorized by this package.
- No item was selected using v3 or v4 model output.
- Human review is `not_started`.
- Results are development diagnostics only. A future held-out set must use new families and a pre-frozen prompt/runtime configuration.
