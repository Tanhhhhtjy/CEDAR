# Change Set

- Baseline anchor: `8228cb11c4512e9bd1467f05dedb2186b1e51d4d`
- Experiment branch: `agent/prover-generator-development-v1-1`
- Scope: measurement-only development-bank amendment

Implemented:

- immutable parent-file hash binding;
- neutral V/F shell and neutral shared system prompt;
- AO/S1/S2 candidate templates;
- deterministic 12-transition answer-position balance;
- stratified 12-family calibration / 12-family validation split;
- machine-readable scientific gates;
- 336-arm candidate builder and independent validator;
- fixed-tokenizer preflight and prompt-fingerprint generator;
- uncapped template-calibration runner, engineering auditor, analyzer, and
  global social-template finalizer;
- 4096-token thinking limit with p25 hard-blocked.

Rollback: delete the v1.1 data directory and new v1.1 scripts/tests/docs; the
frozen development-v1 release and Step 5 results remain unchanged.
