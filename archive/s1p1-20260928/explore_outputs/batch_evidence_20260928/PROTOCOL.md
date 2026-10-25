# CEDAR ARR revision: supplemental experiment protocol

Date: 2026-09-27. Status: existing-data analyses completed; GPU acquisition protocol prepared before new GPU outcomes. This is a development analysis, not retrospective preregistration of the original paper.

## Question and ordering

How much of capping's answer-logit change is predictable from a general readout scale, and what reason-related change remains after that scale is removed?

1. S1: fixed-trace, held-out non-option-token scale prediction and final-normalization decomposition.
2. P1: bidirectional, leave-family-out validity probes with the same learned probe transferred across capping conditions.
3. C1: paired, bidirectional component interventions with training-family directions and matched controls, evaluated both before and after scale correction.
4. H1: genuinely new structural confirmation after prediction rules and sample-size criteria are frozen. No new-salt bank will be relabeled as new structural evidence.

## Data and numerical controls

- Preserve the AAAI supporting package and all historical run directories.
- Reuse the original E2B prompt and fixed trace token IDs, Qwen3-32B checkpoint revision, BF16, attention implementation, cap vectors and cap thresholds wherever available. Compare hashes before use.
- Cache-faithful replay means the original prompt is prefetched once, then each fixed token is decoded incrementally with KV cache. Capture raw logits before forcing the answer token.
- Main analysis uses treatment-naive traces; cap-generated traces are a separate sensitivity stratum. Keep N/R/V/F separate before any aggregate.
- Save full vocabulary logits, pre-final-normalization state, post-normalization state, and residual stream at prespecified layers 53, 60 and 63 (zero-based, verify module indexing). Capture only the answer boundary in S1. P1 additionally needs the pre-answer, pre-rationale prompt boundary to diagnose answer leakage; these are separate endpoints.
- Same-prompt, same-trace uncapped repeat is the numerical floor. Compare within the new hardware/runtime. Historical anchor mismatch is reported separately and cannot be silently accepted as exact reproduction.
- Record environment versions, model and asset hashes, device map, exact input tokens, cap activation counts, hook cleanup, elapsed time, peak memory, and each failed cell. No quantization or truncation substitution without explicit labeling.

## S1: separate fit tokens, prediction tokens and families

- Exclude all A/B/C/D candidate token IDs from fitting, including single-token whitespace variants when identified by the tokenizer.
- Define fitting/evaluation token sets from uncapped logits only, using a fixed seed. Stratify non-option tokens by uncapped rank: 1–256, 257–4096, and 4097–vocabulary end. Deterministically sample at most 1024 per stratum and split 50/50 into fit and test. Store exact token IDs. Report each stratum separately; use equal stratum weights when pooling.
- Compare identity, positive scale-only, and positive affine prediction. Fit on non-option fit tokens and evaluate untouched non-option test tokens and A/B/C/D. An unrestricted affine slope is a diagnostic; a nonpositive slope invalidates its interpretation as temperature scaling.
- Report MAE/RMSE, centered option-logit error, option-rank changes, and raw and scale-restored broad/specific margins. Avoid "explained share" when the denominator is near zero. Option R-squared from the fitting sample is descriptive only.
- Within-prompt fitting may use that prompt's paired capped non-option logits; it is not an unseen-family predictive model. Separately evaluate a simple family-out global scale fitted only on training-family non-option tokens, with no capped test-family logits used for fitting or model selection.
- For terminal RMSNorm, save its learned weights and epsilon. Verify reconstruction of logits from the captured postnorm state. Compute the counterfactual capped residual with the uncapped RMS denominator while preserving the capped residual direction, and the converse denominator effect. Distinguish exact numerical decomposition from a causal claim about reason use.
- Resample paired structural families (20,000 draws; seed 20260927), not tokens. Report the 12 family effects and heterogeneous signs. With few clusters, confidence intervals are descriptive and not proof of equivalence.

## P1 and C1: validity information and causal use

- V/F pairs stay in the same family fold. For each outer family, fit preprocessing and the validity direction only on the other families. Any hyperparameter selection is nested by family; use a fixed regularization if the pilot is too small for a stable inner split.
- Evaluate U→U, U→P, P→P and P→U on exactly the same held-out examples. Report AUC and balanced accuracy by family and position. Compare with shuffled V/F labels within paired bases.
- Evaluate answer identity and target identity as nuisance endpoints. Fit any nuisance projection only on training families, report both unadjusted and adjusted validity probes, and do not treat an answer-boundary probe alone as semantic validity.
- At one prespecified residual position/layer, transfer only the scalar component along the training-only validity direction from the same-item opposite-condition donor. Run U→P and P→U. Include no-op, equal-dimensional/equal-realized-norm random directions, and complete state replacement as a labeled positive control.
- Patch and readout scaling are distinct controls. The primary causal contrast is the held-out-family change in V–F selectivity remaining after S1 scale correction, relative to random controls. Generic margin restoration alone does not establish a validity-use pathway.
- Do not search layers on the test families. First complete S1; if residuals are at the numerical floor, stop escalation of the reason-use claim and document the scale finding.

## H1 and resource policy

- Choose new inference structures only after S1/P1 predictions are frozen. Certify V/F with the independent solver. Entity renaming and a new random salt do not count as structural novelty.
- Determine number of independent structures using pilot family variance and a stated smallest scientifically meaningful effect; publish precision/power curves rather than pick a favorable count after observing confirmation outcomes.
- Check GPUs and process owners; never terminate unrelated jobs. Reuse checkpoint caches; avoid redundant 65GB downloads on the nearly full shared volume. Start with a bounded smoke containing N/R/V/F and a repeat. Extrapolate runtime and storage before the full matrix.
- GPU startup, smoke completion, full acquisition, and scientific analysis are separate status fields. A running process is not a completed experiment.

## Completed CPU work

`scripts/reanalyze_existing.py` has executed against the frozen local supporting package. It provides normalized-stem-cluster Social contrasts, leave-one-option-out and family-out scale diagnostics on E2B, and RG1 absolute/relative components. Outputs and input SHA-256 hashes are under `repro_outputs/`. These computations do not replace S1/P1/C1/H1.

## Execution amendment frozen before new-model outcomes, 2026-09-27

- Remote root: `/data4/WuMingrui/TianJinyu/CEDAR`. The directory was empty on intake. Historical checkouts are read-only sources; all new files are isolated here.
- Original model and cap cache files were absent. Restore the exact pinned public revisions into this workspace. Require both official cap SHA-256 values and the SHA-256 of every model shard to match its Hugging Face LFS blob name before any model inference. No precision or model-size substitution.
- Recovered the original 192 frozen traces from a historical checkout, and verified all prompt, trace-prefix and concatenated-token hashes against all 384 submitted E2B result records.
- Use the submitted `run_e2b_fixed_trace_cached.run_one` without source edits. Observation hooks collect the final prompt token and the answer boundary. Layer 53 is observed at the input to layer 54, after the original layer-53 cap hook. Layers 60 and 63 are observed at block output.
- Engineering smoke: lexicographically first base, all N/R/V/F from uncapped source traces. Run U, P, and repeat U for each (12 forwards). The smoke is not a scientific confirmation set. The repeat numerical gate is maximum absolute difference <= 1e-4 across logits and all saved states; the terminal-head BF16 reconstruction gate is <= 0.25 logit. Record actual errors, including all historical option-logit differences. A historical mismatch is explicitly labeled and never silently called exact replication.
- Formal S1/P1 acquisition: all 192 frozen traces crossed with U/P (384 forwards). Uncapped-source traces are primary; capped-source traces are a separate sensitivity stratum. This supplements full-vocabulary and state evidence without new natural generation. Fixed-trace argmax changes must not be called generated behavioral changes.
- Preserve the historical four-device budget: physical GPUs 1, 2, 4, 6; 18 GiB maximum model allocation per device. Require at least 22 GiB free per device immediately before loading. Other users' jobs are never stopped. Actual device mapping and peak allocation are recorded.
- S1 summaries give equal weight to the 12 structural families. The four V-F contrasts (raw/restored x broad/specific) use exact family sign-flip tests with Holm correction, assuming independent symmetric family effects. Bootstrap confidence intervals and all arm/rank-stratum comparisons are descriptive. Within-prompt fits and family-out fits are explicitly distinct.
- Primary P1 endpoint: prompt boundary, layer 53, unit-RMS activations, training-family-only regression of initial and target answer identity, then train-only standardization and fixed logistic C=1. Evaluate U->U, U->P, P->P, P->U on complete held-out families. Report equal-family AUC and balanced accuracy. Raw activations, layers 60/63, answer-boundary states, and unadjusted probes are declared sensitivity analyses before observing results.
- P1 null: 100 V/F label-swap permutations within paired bases, shared across conditions, with full retraining within every outer fold. Holm correction over the four primary transfer directions. The minimum attainable Monte Carlo p-value is 1/101. Fold-AUC bootstrap intervals are descriptive because outer training sets overlap. No semantic-validity or causal-use conclusion follows from decoding alone.
- Separate analysis virtual environment with system packages visible; add plotting dependencies there, leaving the historical inference environment unchanged.
- The one-shot remote campaign waits for verified assets, runs engineering smoke, proceeds only after the technical gate, acquires the formal matrix, then runs S1 and P1 analysis. It stops on errors. It does not automatically claim causality or launch C1/H1 before the S1/P1 evidence is reviewed.
