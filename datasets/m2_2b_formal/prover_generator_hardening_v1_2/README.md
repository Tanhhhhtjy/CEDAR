# Prover Generator Hardening v1.2

This directory contains exactly twelve unscored paired-justification sentinels. Each V/F pair uses
the same authenticated reviewer, channel, speech act, wrapper skeleton, and proposed atomic edit.
V uniquely proves `revision_warranted(edit_id)`; F proposes the same edit through one structurally
certified invalid inference and has a Mace4 countermodel.

Primary commands:

```powershell
python build_prover_generator_hardening_v1_2.py
python validate_prover_generator_hardening_v1_2.py
```

The blank bilingual workbook is `paired_justification_review12_bilingual.xlsx`. Only a real human
reviewer may fill it. `finalize_paired_justification_review.py` imports a completed copy without
editing it and applies the frozen all-six-criteria-per-row Gate A/C rule.

The current state is `awaiting_human_naturalization_review`. These sentinels are not a development
bank or held-out set. No Qwen, GPU, capping, S arm, or model-response filtering was run.
