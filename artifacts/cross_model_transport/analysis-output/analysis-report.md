# Analysis report

## Analysis question

Does official model-specific p25 capping shift answer-revision geometry within both matched forced-choice model tracks without a reliable V/F selectivity improvement?

## Main findings

- Qwen-FC overall `delta m_general`: +2.075 [+1.372, +2.791] logits.
- Llama overall `delta m_general`: +1.391 [+0.908, +1.865] logits.
- Qwen-FC V-F continuous selectivity change: +0.042 [-0.438, +0.573] logits.
- Llama V-F continuous selectivity change: +0.115 [-0.365, +0.615] logits.

Allowed wording: under the archived matched forced-choice regime, both audited model-specific p25 configurations shifted general answer geometry in the same direction, while neither V-F interval showed a reliable selectivity improvement.

Forbidden wording: universal transport; cross-model equivalence; proof of zero selectivity; direct comparison of unstandardized raw-logit levels; or a claim that every pre-run provenance field was archived.

## Claim candidates

- Claim: Both archived model-specific configurations show a positive within-model general-margin shift.
  - Source evidence: `cross_model_summary.csv`, `raw_records.jsonl`.
  - Allowed wording: bounded to these models, prompts, and official p25 configurations.
  - Forbidden stronger wording: universal model-family law or equal effect magnitude.
  - Uncertainty: 12-family percentile bootstrap.
  - Decision: keep.

- Claim: V/F selectivity did not reliably improve.
  - Source evidence: both V-F continuous-selectivity CIs include zero.
  - Allowed wording: no reliable improvement under this assay.
  - Forbidden stronger wording: equivalence or proof of a zero effect.
  - Decision: keep with calibrated wording.

## Caveats

The supplied JSONL omits a separate threshold hash, chat-template identifier, source runner commit, and separately archived smoke artifact. Core result-space gates pass, but those provenance fields cannot be reconstructed.
