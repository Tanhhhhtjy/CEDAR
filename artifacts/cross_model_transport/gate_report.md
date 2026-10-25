# Gate report

The four G-D core gates (single-token option readout, cap exposure, output space, and strict pairing) pass for both archived model tracks.

## Qwen-FC

- Status: `PASS_CORE_GATES_WITH_PROVENANCE_WARNINGS`
- Revision: `9216db5781bf21249d130ec9da846c4624c16137`
- Axis hash: `a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211`
- Capping config hash: `6aec1220487473aaeab80b05d5d960ac54b5dd9080b51ac4bb0bbd1f4330db24`
- Prompt-bank hash: `c783f3c780c429ef22892ec806ca97db66ee337617a073044debf4f4ada57d7f`

- PASS `forward_count_192`
- PASS `paired_96`
- PASS `families_12`
- PASS `bases_24`
- PASS `arms_24_prompts_each`
- PASS `model_repo_locked`
- PASS `model_revision_locked`
- PASS `official_experiment_locked`
- PASS `reasoning_off_forced_choice`
- PASS `single_token_abcd`
- PASS `answer_mapping_consistent`
- PASS `within_model_prompt_pairs_equal`
- PASS `cross_model_prompt_hash_match`
- PASS `axis_hash_locked`
- PASS `config_hash_locked`
- PASS `hook_layers_exact`
- PASS `uncapped_no_intervention`
- PASS `p25_hooked`
- PASS `p25_triggered_and_corrected`
- PASS `complete_option_logits`
- PASS `option_argmax_recomputes`
- PASS `output_integrity`

Provenance warnings: the archived JSONL does not contain a separate threshold-file hash, chat-template identifier/hash, runner commit, or separately labelled smoke artifact. These fields remain `null`; the capping-config hash is not relabelled as a threshold hash.

## Llama

- Status: `PASS_CORE_GATES_WITH_PROVENANCE_WARNINGS`
- Revision: `6f6073b423013f6a7d4d9f39144961bfbfbc386b`
- Axis hash: `9497f1097803096678fd7b7a75ee9b1355f1478c3b08cbd6d0cff92b3f3ca33d`
- Capping config hash: `5a1e7e60a8d8c4a3871500e698f28a112f687a62c66d061e2c0a4b241240128d`
- Prompt-bank hash: `c783f3c780c429ef22892ec806ca97db66ee337617a073044debf4f4ada57d7f`

- PASS `forward_count_192`
- PASS `paired_96`
- PASS `families_12`
- PASS `bases_24`
- PASS `arms_24_prompts_each`
- PASS `model_repo_locked`
- PASS `model_revision_locked`
- PASS `official_experiment_locked`
- PASS `reasoning_off_forced_choice`
- PASS `single_token_abcd`
- PASS `answer_mapping_consistent`
- PASS `within_model_prompt_pairs_equal`
- PASS `cross_model_prompt_hash_match`
- PASS `axis_hash_locked`
- PASS `config_hash_locked`
- PASS `hook_layers_exact`
- PASS `uncapped_no_intervention`
- PASS `p25_hooked`
- PASS `p25_triggered_and_corrected`
- PASS `complete_option_logits`
- PASS `option_argmax_recomputes`
- PASS `output_integrity`

Provenance warnings: the archived JSONL does not contain a separate threshold-file hash, chat-template identifier/hash, runner commit, or separately labelled smoke artifact. These fields remain `null`; the capping-config hash is not relabelled as a threshold hash.

