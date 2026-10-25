# Changeset

## Frozen E2A implementation

- Evidence result anchor: `a0ad950ef160b3aec1900d9c005e9b83853b291e`.
- E2A CPU Gate: `a2b058cee3a0f5c1ed48c0534ceb16af2329c1c0`.
- E2A result: `0c8517bb5fb4640f7d5e4d51356d0cf167046643`.
- Added a derived 96-row Answer view and an independent 48-row V/F
  Validity-sufficiency view without altering frozen prompt content.
- Added deterministic A/B mapping, tokenizer/asset preflight, release validator,
  E2A-only CPU Gate, phase-gated capping, telemetry, runner, audit, analysis, and
  tests.
- Executed the frozen 48-cell Smoke and 576-cell Formal matrices. The 192 Answer
  uncapped/full-p25 anchor cells passed exact equivalence auditing.

## Post-result hygiene

- Renamed the descriptive pooled Validity response rate from `criterion` to
  `overall_sufficient_rate`; it is not signal-detection criterion `c`.
- Represented scope-inapplicable audit checks as `not_applicable` with
  `passed=null`, rather than a false failure.
- Added explicitly labeled post-hoc secondary summaries for reasoning length,
  option mass, full-vocabulary entropy, and four-state Answer transitions.
- Updated stale run-status files. No raw record, prompt, model output, primary
  estimand, routing threshold, or Route C decision was changed.
- Froze separate pre-execution protocols for crossed fixed-trace E2B and the
  new-salt unselected instance confirmation. Neither protocol itself authorizes
  GPU execution.

No E2B, confirmation-bank Qwen run, patching, gradient/JVP, J-Lens, MUSE, SAE,
p1, tuning, new capping method, or model-output-based item selection occurred in
this changeset.
