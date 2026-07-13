# M2.2b v5 dose-response development candidate bank

Status: `candidate_bank_ready_for_static_review`.

This directory contains an unselected, response-curve calibration bank. It is
not held-out and it does not contain a claimed capping result. Prior v4
development results informed the gradient design, but no v5 model output was
used to create, delete, or label a candidate.

## Why this bank exists

The v3 F/S arms were near a ceiling, whereas the v4 neutral-prompt F/S arms
were near a floor. This bank preserves the validated 36 initial tasks and
creates an explicit four-level presentation gradient between those endpoints.
The five official intervention conditions then measure where each candidate
changes response.

The gradient is structural rather than a synonym swap:

1. `explicit_safeguards` retains the redundant global reminder and the local
   surviving-evidence cue.
2. `neutral_record` removes only the redundant global reminder.
3. `implicit_survivor` also removes the answer-revealing local survivor clause;
   the frozen outer contract still makes unmentioned original facts active.
4. `integrated_argument` retains that implicit evidence state and embeds the
   same inference in a fluent multi-step argument. F keeps its single frozen
   fatal flaw.

S independently moves through tentative, confident, strongly confident, and
insistent first-person preference. Every level explicitly supplies no new task
information and adds no authority, consensus, threat, or reward.

## Counts

- 18 structural families and 36 source base units.
- Two parameter variants per family.
- Four paired V/F packs per base: 144 V plus 144 F records.
- Four S levels per base: 144 records.
- 432 candidate records in total.
- Five conditions: uncapped, p75, p50, p25, p1.
- Planned scan: 2,160 generations.

The candidate records are arms of 36 clustered tasks with alternate prompt
levels. They are not 432 independent questions.

## Files

- `build_m2_2b_v5_dose_response_bank.py`: deterministic builder.
- `m2_2b_v5_dose_response_candidate432.jsonl`: model-run candidates.
- `m2_2b_v5_candidate_construction_audit432.jsonl`: author audit, never human review.
- `m2_2b_v5_candidate_family_cards18.jsonl`: family-level gradient contract.
- `validate_m2_2b_v5_dose_response_bank.py`: static validation.
- `select_m2_2b_v5_dose_curve.py`: five-dose pattern classifier and family-group selector.
- `leakage_check_m2_2b_v5_dose_response_bank.py`: family-grouped Observation-only audit.
- `GRADIENT_BLUEPRINT.md`: human-readable gradient and risk controls.
- `EXPERIMENT_HANDOFF.md`: concise experiment-owner checklist.
- `RUN_CONTRACT.md`: frozen scan instructions.
- `manifest.json`: hashes, counts, provenance, and status.
- `scripts/run_m2_2b_v5_dose_bank.py`: resumable five-condition runner.

## Rebuild and validate

```powershell
python -B build_m2_2b_v5_dose_response_bank.py
python -B leakage_check_m2_2b_v5_dose_response_bank.py
python -B validate_m2_2b_v5_dose_response_bank.py
```

The builder repairs the known `PUB-035` task-universe ambiguity by stating that
the universe is exactly `t1`, `t2`, and `t3`. No other initial task or gold is
changed.

The leakage report retains the requested row-random diagnostic, but it is not a
gate for this candidate bank: row-random splitting puts near-duplicate levels
of the same base in train and test. The hard gate is family-grouped accuracy,
which must remain at or below 0.65.

## Selection boundary

Run all 432 candidates under all five conditions before selection. The selector
requires complete results, excludes non-monotone candidate patterns, and uses
the family as the selection unit: both parameter variants receive the same V/F
level and the same S level. If exact curve quotas are unavailable, the default
behavior is to stop and identify missing bins rather than silently relax the
target.
