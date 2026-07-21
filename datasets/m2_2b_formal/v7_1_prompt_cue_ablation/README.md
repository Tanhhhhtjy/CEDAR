# C1/V7.1 prompt-cue ablation

This directory contains a deterministic sensitivity variant of C1/V7. It does not introduce new tasks and does not select items using model outputs.

The sole prompt transformation is deletion of this exact sentence from the final N, AO, and S user turn:

```text
No new task information is supplied.
```

V is unchanged and therefore is not duplicated. The 144 no-cue records comprise 48 N, 48 AO, and 48 S records paired to the frozen V7 source. Running uncapped and p25 requires 288 new generations.

Build and validate without a model:

```bash
python datasets/m2_2b_formal/v7_1_prompt_cue_ablation/build_c1_v7_1_prompt_cue_ablation.py
python datasets/m2_2b_formal/v7_1_prompt_cue_ablation/validate_c1_v7_1_prompt_cue_ablation.py
```

The inherited V7 limitation must remain disclosed: one data lead globally reviewed the bilingual cards, but no row-level labels, second reviewer, agreement, or Cohen's kappa are available.
