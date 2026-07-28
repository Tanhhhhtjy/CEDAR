---
type: results-report
date: 2026-07-28
experiment_line: cross-model-transport
round: 1
purpose: transfer-summary
status: complete-with-provenance-warnings
source_artifacts:
  - artifacts/cross_model_transport/analysis-output/analysis-report.md
  - artifacts/cross_model_transport/analysis-output/stats-appendix.md
  - artifacts/cross_model_transport/analysis-output/figure-catalog.md
linked_experiments:
  - docs/plans/2026-07-26-cedar-llama-cross-model-replication-decision.md
linked_results:
  - artifacts/cross_model_transport/cross_model_summary.csv
---

# Cross-model Transport / Round 1 / Transfer Summary / 2026-07-28

## 1. Executive Summary

This delivery audited 384 archived deterministic forced-choice forwards: 192 for Qwen-FC and 192 for Llama. No model was run during delivery. Both model-specific official p25 configurations produced a positive within-model shift in general answer geometry: Qwen-FC `+2.075 [1.372, 2.791]` logits and Llama `+1.391 [0.908, 1.865]` logits. Their V−F continuous-selectivity changes were `+0.042 [-0.438, 0.573]` and `+0.115 [-0.365, 0.615]` logits, respectively.

The frozen routing rule therefore gives **G-A transport**: the overall shifts share a direction, the Llama interval excludes zero, and neither V−F interval supports a reliable selectivity improvement. This supports a bounded cross-model statement under the archived matched forced-choice regime; it does not establish universality or equivalence.

## 2. Experiment Identity and Decision Context

The experiment asks whether Qwen's general answer-geometry shift and V/F non-selectivity appear under another officially supported model-specific Assistant-Axis capping configuration. The matched anchor prevents a Qwen thinking/free-generation result from serving as the Llama comparator.

The archived source records came from `LLama_data`. The delivery branch is `agent/cross-model-transport-delivery-v1`, based on `bff1e19a71f2051ca4b2270c5d57706114593550`. The source runner commit was not stored in the supplied JSONL and remains unknown.

## 3. Setup and Evaluation Protocol

- Bank: 24 bases from 12 structural families, with N/R/V/F arms (96 prompts).
- Conditions: uncapped and official p25; prompt-paired within model.
- Qwen-FC: `Qwen/Qwen3-32B`, revision `9216db5781bf21249d130ec9da846c4624c16137`, official `layers_46:54-p0.25`, actual layers 46–53.
- Llama: `meta-llama/Llama-3.3-70B-Instruct`, revision `6f6073b423013f6a7d4d9f39144961bfbfbc386b`, official `layers_56:72-p0.25`, actual layers 56–71.
- Readout: reasoning off, deterministic A/B/C/D forced-choice logits.
- Primary contrast: within-model `p25 − uncapped`.
- Inference: 20,000 percentile family-cluster bootstrap draws, seed `20260725`; family is the independent unit.
- Missing / invalid / truncated cross-model records: `0 / 0 / 0`.

The Qwen axis/config hashes are `a207fe7a…` / `6aec1220…`; the Llama hashes are `9497f109…` / `5a1e7e60…`. The prompt-bank hash is `c783f3c780c429ef22892ec806ca97db66ee337617a073044debf4f4ada57d7f` for both tracks.

## 4. Main Findings

- Overall `delta m_general`: Qwen-FC `+2.075 [1.372, 2.791]`; Llama `+1.391 [0.908, 1.865]` logits.
- N/R/V/F `delta m_general` was positive in every arm for both models; exact arm estimates and intervals are in `cross_model_summary.csv`.
- Pooled `delta m_specific`: Qwen-FC `−0.028 [−0.314, 0.265]`; Llama `−0.055 [−0.381, 0.274]` logits.
- V−F continuous selectivity change: Qwen-FC `+0.042 [−0.438, 0.573]`; Llama `+0.115 [−0.365, 0.615]` logits.
- Target follow: Qwen-FC V `37.5% → 50.0%` (`+12.5pp`) and F `16.7% → 33.3%` (`+16.7pp`); Llama V `33.3% → 50.0%` (`+16.7pp`) and F `20.8% → 33.3%` (`+12.5pp`).
- N/R argmax stability: Qwen-FC `24/24` and `23/24`; Llama `23/24` and `22/24`.
- Every p25 record triggered and produced a correction on the configured layers; uncapped records showed no hook intervention.

The secondary paired-family Llama-minus-Qwen interaction for overall `delta m_general` was `−0.684 [−1.456, +0.080]` logits. This descriptive interval crosses zero; the primary claims remain within-model.

## 5. Statistical Validation

All prompt pairs were retained. Family-level effects were calculated before resampling, so the 12 families—not 96 prompts or 384 forwards—define the independent bootstrap unit. The same seed, draw count, percentile interpolation, and resampling definition used by frozen E2A were reused.

E2A export independently reproduced the frozen overall `delta m_general = +3.130208333333` and 95% CI `[1.834190538194, 4.486121961806]` exactly. Its 192 selected uncapped/full-p25 records contain no missing, invalid, or truncated entries.

Arm and telemetry intervals are descriptive secondary breakdowns. The route uses only the prespecified overall and V−F criteria. A significant/non-significant contrast across models was not used as a model-difference test.

## 6. Figure-by-Figure Interpretation

### E2A family geometry

- Purpose: replace Figure 0 placeholder points with actual 12-family paired values.
- Observation: most families move toward a less negative `m_general`, but two family effects are slightly negative; V/F `delta m_target` is heterogeneous.
- Supported interpretation: the frozen overall shift is an aggregate family-level geometry change, not a hand-placed point cloud.
- Decision implication: Figure 0 can be updated without rerunning E2A.

### Figure 2D cross-model forest

- Purpose: show model-internal overall and V−F effects with family-bootstrap intervals.
- Observation: both overall intervals lie above zero; both V−F intervals cross zero.
- Supported interpretation: the archived results satisfy G-A, while not showing a reliable selectivity improvement.
- Decision implication: the bounded transport result may enter Figure 2D with its provenance caveats.

### Cross-model paired family audit

- Purpose: expose arm- and family-level endpoints with redundant marker/fill encoding.
- Observation: every arm has a positive mean shift in both models, with family heterogeneity and a smaller Llama magnitude.
- Supported interpretation: the effect is not confined to V or F.
- Decision implication: retain arm breakdowns in the audit or appendix; do not convert raw endpoint levels into the primary cross-model comparison.

## 7. Failure Cases, Negative Results, and Limitations

The supplied cross-model JSONL lacks four provenance fields: a separate threshold-file hash, chat-template identifier/hash, source runner commit, and a separately labelled smoke artifact. They are recorded as `null`; the config hash is not relabelled as a threshold hash. Core tokenization, cap-exposure, output-space, and pairing gates are directly reproducible from the records and pass.

“No reliable V/F improvement” is not equivalence and does not prove a zero selectivity effect. Two model configurations do not define a population-level model effect. The paired-family audit shows raw within-model endpoints in separate panels and must not be read as a direct comparison of cross-model raw-logit levels.

## 8. What Changed Our Belief

This round strengthens the bounded transport hypothesis: the broad geometry shift is not unique to the archived Qwen matched anchor. It does not strengthen the hypothesis that p25 selectively accepts valid evidence while rejecting fallacies. Magnitude differs across models, so model heterogeneity remains part of the result.

## 9. Next Actions

- Promote the G-A forest inputs and calibrated wording into the manuscript-facing Figure 2D workflow.
- Keep exact family/arm, migration, output-integrity, and telemetry tables in the audit or appendix.
- Do not rerun E2A, retune thresholds, add Gemma, or claim cross-model equivalence.
- If strict provenance completeness is required before submission, recover the original chat-template identifier, separate threshold hash, smoke artifact, and runner commit from the experiment host; do not infer them from these files.

## 10. Artifact and Reproducibility Index

- Raw normalized forwards: `raw_records.jsonl`; exact supplied sources: `source_qwen_fc_records.jsonl`, `source_llama_records.jsonl`.
- E2A exports: `e2a_family_geometry.csv`, `e2a_family_arm_geometry.csv`, `e2a_family_plot_audit.md`.
- Cross-model outputs: `cross_model_family_effects.csv`, `cross_model_summary.csv`, `cross_model_target_follow.csv`, `figure2d_forest_input.csv`.
- Audits: `gate_report.md`, `pair_audit.csv`, `cross_model_output_integrity.csv`, `cross_model_four_state_migrations.csv`, `cross_model_telemetry.csv`.
- Scripts: `scripts/build_delivery.py`, `scripts/plot_e2a_family_geometry.m`, `scripts/plot_cross_model_transport.m`.
- Figures: editable SVG, vector PDF, and 600-dpi PNG files under `figures/`.
- No Obsidian write-back was attempted; this repository is not treated as a bound Obsidian project for this delivery.
