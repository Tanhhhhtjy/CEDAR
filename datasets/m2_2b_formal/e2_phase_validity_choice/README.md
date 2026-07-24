# CEDAR E2 Phase Validity–Choice v1

This directory freezes the CPU release for exploratory E2A phase localization.
It descends from Evidence result `a0ad950ef160b3aec1900d9c005e9b83853b291e`.

Contents:

- `answer_inputs96.jsonl`: unchanged N/R/V/F Answer prompts;
- `validity_inputs48.jsonl`: independent V/F sufficiency prompts;
- `validity_mapping.json`: deterministic A/B mapping and feasibility audit;
- `human_review.json`: one-real-reviewer 48/48 language/semantics audit;
- `tokenizer_preflight.json`: pinned Qwen tokenizer and p25 asset preflight;
- `ANALYSIS_PLAN.json`: frozen hierarchical models, ROPEs, and routing thresholds;
- `CPU_TEST_REPORT.json`: relevant-suite pass and disclosed legacy full-suite failures;
- `manifest.json`: canonical release hashes and smoke IDs;
- `E2_CPU_GATE.json`: the only GPU authorization object.

Build and CPU checks use no GPU:

```powershell
python datasets/m2_2b_formal/e2_phase_validity_choice/build_e2_phase_validity_choice.py
python datasets/m2_2b_formal/e2_phase_validity_choice/run_cpu_preflight.py --cache-dir <HF_CACHE>
python datasets/m2_2b_formal/e2_phase_validity_choice/validate_e2_phase_validity_choice.py --release
python -m pytest -q tests/test_e2_phase_validity_choice.py
```

Do not run E2A on GPU unless the committed `E2_CPU_GATE.json` says
`gpu_authorized=true`, the worktree is clean, and HEAD equals upstream.
