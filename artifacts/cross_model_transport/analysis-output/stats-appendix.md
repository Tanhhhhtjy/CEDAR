# Statistics appendix

- Unit of analysis: 12 independently generated structural families; 24 bases nested within family.
- Pairing: exact stable prompt ID within model; cross-model prompt hashes checked.
- Bootstrap: 20,000 family-cluster draws with replacement; seed 20260725; percentile 2.5% and 97.5% endpoints with linear interpolation.
- Main effects: within-model p25 minus uncapped.
- ITT: no model-output exclusions; invalid and truncated records retained in integrity accounting.
- Multiplicity: arm and telemetry rows are descriptive secondary breakdowns; routing is based only on frozen overall and V-F criteria.
- Cross-model interaction: secondary paired-family difference, not inferred from one significant and one nonsignificant result.

Secondary Llama-minus-Qwen overall interaction: -0.684 [-1.456, +0.080] logits.
