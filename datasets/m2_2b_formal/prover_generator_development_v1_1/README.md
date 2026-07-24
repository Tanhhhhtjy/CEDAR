# Prover-Generator Development v1.1

This directory is a measurement-only amendment of the frozen development-v1
bank. It preserves the parent symbolic logic and emits a 336-arm candidate
bank containing both global social-template candidates.

Current state: automatic preflight passes; the pinned Qwen tokenizer gate must
run before GPU calibration. p25 is not authorized.

Primary commands:

```bash
python datasets/m2_2b_formal/prover_generator_development_v1_1/build_prover_generator_development_v1_1.py
python datasets/m2_2b_formal/prover_generator_development_v1_1/finalize_human_naturalization_review.py --reviewer Codex --attest-all-48
python datasets/m2_2b_formal/prover_generator_development_v1_1/run_qwen_tokenizer_preflight.py --cache-dir "$HF_HOME"
python datasets/m2_2b_formal/prover_generator_development_v1_1/validate_prover_generator_development_v1_1.py --require-release-gates
```

See `docs/PROVER_GENERATOR_DEVELOPMENT_V1_1_PROTOCOL.md` for the frozen order
and scientific gates.
