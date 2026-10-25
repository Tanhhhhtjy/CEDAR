# CEDAR Prover-Generator Program Status

## Outcome

- Step 0 is complete: the C2/V8 margin-choice addendum is frozen into the core-bank requirements.
- Step 1 is complete at Gate A: the v1.2 paired-justification human review passed 12/12.
- Step 2 is complete: the selected engineering architecture is `proof_graph_plus_atomic_edit_certification`. The retired v1 `100 vs 99` comparison remains `inconclusive` and is not selection evidence.
- Steps 3 and 4 are complete at Gate A: the development bank contains 24 families, 48 bases, and 288 N/R/V/F/AO/S arms, with all automatic and human zero-GPU gates passed.
- The next required action is Step 5 uncapped development calibration, which requires GPU model execution.

## Development-bank evidence

- Six semantic transitions: 8 bases each.
- Target positions A/B/C/D: 12 bases each.
- Six F error classes: 8 bases each.
- F error positions early/middle/late: 16 bases each.
- Easy-to-medium proof-depth gradients: 24/24.
- Deterministic complete rebuild: passed.
- Independent Python/Prover9/Mace4 replay: 48/48 passed.
- Fixed Qwen3-32B tokenizer AO/S full-chat token equality: 48/48 passed on CPU; model weights were not loaded.
- Bilingual human review: 48/48 passed. Final naturalness scores are 4/5 throughout because the prose is clear but deliberately formal and long.

## Review-driven fixes before freeze

1. Binary generic rules were changed to explicitly preserve all antecedent entities with `that same item`, `that other item`, and bilingual equivalents.
2. Middle/late administrative argument steps were aligned with their stored formal literals (`review_context_open` then `review_case_active`).

Both fixes were followed by a complete deterministic rebuild, tokenizer preflight, independent validation, and a newly generated completed review packet.

## Authoritative entrypoints

- Protocol: `docs/PROVER_GENERATOR_DEVELOPMENT_BANK_V1_PROTOCOL.md`
- Final zero-GPU report: `datasets/m2_2b_formal/prover_generator_development_v1/DEVELOPMENT_BANK_ZERO_GPU_REPORT.md`
- Final gate: `datasets/m2_2b_formal/prover_generator_development_v1/ZERO_GPU_GATE.json`
- Release manifest: `datasets/m2_2b_formal/prover_generator_development_v1/manifest.json`
- Completed review: `datasets/m2_2b_formal/prover_generator_development_v1/development_naturalization_review48_bilingual_completed.xlsx`

## Next permitted action

Run Step 5 uncapped development calibration with thinking and non-thinking kept separate. Do not start capped/p25 runs, held-out confirmation generation, LoFa, or J-Lens yet.
