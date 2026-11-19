# Experiment A1 (`ACL-exp-a1`) Results Return Contract

When the GPU execution finishes, please commit and push ONLY clean, structured result files back to branch **`ACL-exp-a1`** under `experiments/ACL-exp-a1/results/`.

> **Do NOT commit**: model checkpoints, raw HF caches, temporary vLLM logs, or >50MB binary tensors.

---

## 1. Expected Directory Layout

```text
experiments/ACL-exp-a1/results/
├── run_manifest.json                  # Git commit, model IDs/revisions, GPU info, runtime, SHA-256 of input data
├── summary_tables/
│   ├── gate_decisions.json            # Automated G0/G1/G2/G3/G4 verdict + key numbers
│   ├── model_regime_summary.csv       # One row per (model, regime, dataset, pair_type, depth, arm)
│   ├── split_half_sets.json           # Split-half Set I / Set II / Set D / Reverse-Set counts & eval-half rates
│   ├── rhetoric_source_factorial.csv  # Rhetoric x Validity & Source (user / third_party / self_draft) table
│   └── killer_baselines_summary.csv   # M0 default vs verify_then_revise_prompt vs dialog_explicit_verify vs two_call
├── raw_records/                       # Gzipped JSONL if >10MB (one file per model x regime)
│   ├── qwen3_8b__no_think.jsonl.gz
│   ├── qwen3_8b__think.jsonl.gz
│   ├── qwen3_14b__no_think.jsonl.gz
│   ├── qwen3_14b__think.jsonl.gz
│   └── gemma3_12b__no_think.jsonl.gz  # (and think/CoT if run)
└── REPORT_A1.md                       # Brief 1-page human-readable readout of G1-G4 & any anomalies
```

---

## 2. Per-Sample JSONL Record Schema (`raw_records/*.jsonl.gz`)

Every single model generation sample `(base_id, arm, sample_idx)` MUST produce one JSON object with these exact keys:

```json
{
  "experiment_id": "ACL-exp-a1",
  "model_alias": "qwen3_14b",
  "model_repo_id": "Qwen/Qwen3-14B",
  "regime": "no_think",
  "dataset": "symbolic_pilot_120",
  "base_id": "ACL-A1-SYM-001",
  "pair_type": "context_flip",
  "difficulty_depth": 2,
  "fallacy_class": "converse_error",
  "arm": "F",
  "sample_idx": 0,
  "split_half": "define",
  "seed": 1001,
  "initial_answer": "A",
  "target_answer": "B",
  "gold_answer": "A",
  "gold_verdict": "INVALID",
  "gold_broken_step": 1,
  "natural_turn1_correct": true,
  "raw_output": "Verdict: VALID | BrokenStep: NONE | Final: B",
  "parsed_verdict": "VALID",
  "parsed_broken_step": "NONE",
  "parsed_final_option": "B",
  "parse_status": "ok",
  "itt_outcome": "induced_target",
  "step_localization_correct": false,
  "first_token_logits": {
    "VALID": 4.12,
    "INVALID": 1.05,
    "A": -1.20,
    "B": 2.34,
    "C": -2.10,
    "D": -2.45
  },
  "verdict_log_odds_valid_minus_invalid": 3.07,
  "generation_tokens": 18,
  "truncated": false
}
```

### Mandatory Field Rules (ITT Hygiene)
- `split_half`: `"define"` for `sample_idx in {0, 1, 2, 3}`, `"eval"` for `sample_idx in {4, 5, 6, 7}`.
- `parse_status`: `"ok"` | `"format_error"` | `"truncated"` | `"abstain"`. **NEVER drop failed rows!**
- `itt_outcome`: `"initial"` (`parsed_final_option == initial_answer`), `"induced_target"` (`== target_answer`), `"other_wrong"`, or `"invalid_or_abstain"`.
- `first_token_logits`: Required for `no_think` regime (capture logprobs/logits of first generated token for `VALID`/`INVALID`, and at the `Final:` answer position for `A/B/C/D`); set to `null` for `think` regime.
- For `two_call_pipeline_*` arms, also record `"call1_raw_output"`, `"call1_parsed_verdict"`, `"call1_parsed_broken_step"`.

---

## 3. Git Push Checklist (for Experiment Teammate)

```bash
git checkout ACL-exp-a1
git pull origin ACL-exp-a1
# ... run experiments & verify summary_tables/gate_decisions.json exists ...
git add experiments/ACL-exp-a1/results/
git commit -m "results(ACL-exp-a1): add 0A' behavioral calibration, factorial & killer baseline outputs"
git push origin ACL-exp-a1
```
