# CEDAR Datasets

Datasets for the CEDAR project (Cost of Epistemic Deference under Activation Restriction).
All data is **English**.

## Contents

### `formal_domain/` — formal-domain belief-revision test set
Derived from **Belief-R** (Wilie et al., EMNLP 2024; HF `CAiRE/belief_r`), reformatted for
measuring warranted belief revision.
- `formal_full.jsonl` (1744) — full set, main result / cross-paper comparability
- `formal_clean.jsonl` (786) — high-strength subset (agreement_lv≥5), robustness check
- `formal_balanced_200.jsonl` (200) — balanced (should-update / should-not each 100), for pilots
- Metric: BU-Acc (warranted-update accuracy) / BM-Acc (belief-maintain accuracy)
- `build_formal_dataset.py` reproduces from source (seed=42). `eval_formal_protocol.md` = eval protocol.

### `force_stimuli/` — minimal-pair stimuli for three forces
Self-built minimal pairs for extracting activation directions of three stance-change forces.
- `force_social.jsonl` / `force_evidence.jsonl` / `force_identity.jsonl` (300 pairs each)
- Each pair: shared context + `positive` (force applied) vs `negative` (neutral, shared baseline).
- `build_force_stimuli.py` reproduces (seed=42).

## Notes
- Build scripts contain absolute local paths (for regeneration); the `.jsonl` outputs are self-contained.
- Belief-R source license: verify on HF page before redistribution.
