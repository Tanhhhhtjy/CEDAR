# Frozen protocol: matched Qwen--Llama transport

- Bank: 24 bases, 12 structural families, N/R/V/F (96 prompts).
- Qwen: `Qwen/Qwen3-32B`, reasoning off, forced choice, uncapped vs official `layers_46:54-p0.25` (layers 46--53).
- Llama: `meta-llama/Llama-3.3-70B-Instruct`, reasoning off, forced choice, uncapped vs official `layers_56:72-p0.25` (layers 56--71).
- Primary contrast: within-model `p25 - uncapped`; raw logit levels are not compared across models.
- Pairing: stable prompt ID and prompt hash. No prompt, answer-order, family, or response-based exclusions.
- Inference: 20,000 percentile family-cluster bootstrap draws, seed `20260725`, family as the independent unit.
- ITT: invalid and truncated records remain in integrity counts; continuous metrics require archived option logits.
- This delivery does not run a model or reinterpret frozen C1/C2/C3 evidence.
