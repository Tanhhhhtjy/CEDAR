# C3 SycoBench bidirectional selectivity data

This directory is built deterministically from the SycoBench-600 object already pinned for
C2/V8. C3 uses all 600 source records and does not select a foil, template, smoke item, or
review item from Qwen output.

The frozen files contain 600 bases, 2,400 harmful/helpful by AO/S records, a 96-record smoke
grid, a 32-case rendered review packet, and a four-template review packet. The construction
manifest records Git blob, raw-byte diagnostic, LF-canonical, and semantic object hashes.

## Human gate

Construction validation deliberately reports `gpu_authorized=false`. A real data lead must:

1. copy `review_packet32.csv` to `review_completed.csv` and fill every judgment with `yes` or
   stop and rebuild before any model run;
2. copy `template_review_packet.csv` to `template_review_completed.csv` and do the same;
3. run the attestation helper with a real reviewer ID and both explicit pre-output confirmations;
4. commit its bound `single_review_attestation.json` and passed `HUMAN_REVIEW_GATE.json` before
   execution preflight.

The attestation must disclose `single_data_lead_review`, no independent second review, no
agreement estimate, and no Cohen kappa. Blank packets and code cannot be treated as human
approval. Until all three genuine review files exist, `--stage review` must fail.

## Commands

    conda run -n cedar python scripts/build_c3_sycobench_bidirectional.py --offline
    conda run -n cedar python datasets/c3_sycobench_bidirectional_selectivity/validate_c3_sycobench_bidirectional.py --stage source
    conda run -n cedar python datasets/c3_sycobench_bidirectional_selectivity/validate_c3_sycobench_bidirectional.py --stage construction

After the human has completed both CSVs:

    conda run -n cedar python scripts/freeze_c3_single_review_attestation.py --reviewer-id ACTUAL_INTERNAL_ID --confirm-no-model-outputs --confirm-no-gpu-execution

Do not run the review stage or any GPU command until a person has completed the frozen packets.
