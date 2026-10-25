# M2.2b v3 held-out uncapped gate — Phase 2 construction

Status: `awaiting_phase2_1_construction_review`

Phase 2.1 minimally reconstructs the four V observations identified in `PHASE2_CODEX_REVIEW.md`. No human review or model run may begin until this reconstruction passes review.

This directory contains the deterministic Phase 2 instantiation of the 30 family cards approved in `PHASE1_1_CODEX_REVIEW.md`. It has 60 base units and a five-arm N/R/V/F/S expansion of 300 records. No Qwen or capping run was performed, and no item was selected using model output.

## Frozen inputs and construction

- Repository freeze: `f7cfb489b41f105420973836ebcd39bc23558038`
- Family source: `family_cards30.jsonl`
- Frozen family-card SHA-256: `2ca36d449c61a1620e37d426786705a5ecf45b3f334634b6fa13bd3c8ff2bf4e`
- Construction seed: `20260713`
- Variant 1 is easy and variant 2 is medium for every family.
- `build_m2_2b_v3_heldout.py` contains the explicit construction parameters. Randomness is used only for balanced answer-label placement, blind A/B placement, and reviewer row ordering.

The builder refuses a changed family-card hash and checks that every frozen family is instantiated exactly twice. It copies the cards to `m2_2b_v3_family_cards30.jsonl` as a byte-identical delivery alias.

## Files

- `m2_2b_v3_heldout60_base.jsonl`: 60 clustered base units with all five arms nested.
- `m2_2b_v3_heldout300_arms.jsonl`: 300 flat arm records; these are arms of 60 bases, not 300 independent samples.
- `m2_2b_v3_construction_audit60.jsonl`: author construction gold, explicitly not independent human review.
- `reviewer_1_blind.csv`, `reviewer_2_blind.csv`: independently ordered, empty blind-review sheets.
- `m2_2b_v3_blind_review_key.jsonl`: hidden V/F mapping and construction key. Do not give this file to reviewers.
- `adjudication_template.csv`: blank third-reviewer template for disputed items only.
- `validate_m2_2b_v3_heldout.py`: structure, balance, prompt, frozen-field, hash, blind-sheet, and concrete-countermodel validation.
- `leakage_check_m2_2b_v3.py`: Observation-only unigram bag-of-words diagnostics, including grouped CV, random split, leave-one-family-out, predictive tokens, and source/authority cues.
- `leakage_report.json`: generated leakage results.
- `score_m2_2b_v3_heldout.py`: scores later external response JSONL; it does not invoke a model.
- `manifest.json`: hashes and pending human-review fields. The manifest omits its own hash to avoid recursive hashing.

## Reproduction and static gates

From this directory:

```powershell
python -B build_m2_2b_v3_heldout.py
python -B leakage_check_m2_2b_v3.py
python -B validate_m2_2b_v3_heldout.py
```

Run the builder before leakage because it intentionally resets `leakage_report.json` to `not_run`. The leakage script writes the report and updates its hash and metrics in `manifest.json`.

The validator recomputes both concrete models for all 30 undercut base units. The grouped Observation-only accuracy must be at most 0.65. A failure requires wording revision, not threshold relaxation or family leakage across the hard-gate split.

## Human-review boundary

Both reviewer sheets have blank label columns. Save the two raw completed sheets before computing agreement or adjudicating. Do not force one text to be valid and the other invalid. Only disputed items go to a third reviewer. Until those labels exist, agreement, Cohen's kappa, rhetorical-quality difference, unclear count, and minimal-pair failures remain `null` in the manifest.

This Phase 2 package must receive construction review before any uncapped Qwen run. Capping remains prohibited until the uncapped gate is approved.
