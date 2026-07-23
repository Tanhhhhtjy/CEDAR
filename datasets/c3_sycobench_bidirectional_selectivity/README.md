# C3 SycoBench bidirectional selectivity data

This directory is built deterministically from the SycoBench-600 object already pinned for
C2/V8. C3 uses all 600 source records and does not select a foil, template, smoke item, or
review item from Qwen output.

The frozen files contain 600 bases, 2,400 harmful/helpful by AO/S records, a 96-record smoke
grid, a 32-case rendered review packet, and a four-template review packet. The construction
manifest records Git blob, raw-byte diagnostic, LF-canonical, and semantic object hashes.

## Amended review gate

Construction validation deliberately reports `gpu_authorized=false`. Under the user's explicit
pre-GPU waiver, the executing agent must:

1. genuinely inspect all 32 rendered cases and record the judgments in
   `agent_review_completed.csv`;
2. genuinely inspect all four template pairs and record the judgments in
   `template_agent_review_completed.csv`;
3. run the agent attestation helper with the explicit user-waiver, pre-output, and pre-GPU
   confirmations;
4. commit the bound `agent_self_review_attestation.json` and passed `REVIEW_GATE.json` before
   execution preflight.

The attestation must disclose `agent_self_review_user_waiver`, `protocol_deviation=true`, no
human/data-lead review, no independent reviewer, no agreement estimate, and no Cohen kappa. It
must never be described as human approval. Until all three genuine review files exist,
`--stage review` must fail.

## Commands

    conda run -n cedar python scripts/build_c3_sycobench_bidirectional.py --offline
    conda run -n cedar python datasets/c3_sycobench_bidirectional_selectivity/validate_c3_sycobench_bidirectional.py --stage source
    conda run -n cedar python datasets/c3_sycobench_bidirectional_selectivity/validate_c3_sycobench_bidirectional.py --stage construction

After the agent has genuinely completed both CSVs:

    conda run -n cedar python scripts/freeze_c3_agent_self_review_attestation.py --review-agent-id codex_agent_self_review_20260723 --confirm-user-waiver --confirm-no-model-outputs --confirm-no-gpu-execution

Do not run any GPU command until the amended review stage and execution preflight both pass.
