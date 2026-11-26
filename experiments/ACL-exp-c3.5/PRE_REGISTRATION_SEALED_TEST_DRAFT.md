# Pre-Registration Protocol for Final One-Shot Unsealing (`c1_test_id` & `c1_test_ood`) — **SUPERSEDED BY SIGNED PROTOCOL**

> **SUPERSEDED**: This pre-Stage-6 draft has been superseded by the formal, signed Stage 7 pre-registration protocol [`PRE_REGISTRATION_SEALED_TEST.md`](PRE_REGISTRATION_SEALED_TEST.md) and machine-verified lockfile [`data/stage7_sealed_config.json`](data/stage7_sealed_config.json) (which lock every full 64-char adapter/PEFT/merged-shard SHA-256, per-arm backend, frozen CAA vector spec, Holm-Bonferroni multiplicity, and open-data dry-run verification). Retained for historical audit transparency only.

---

## 1. Pre-Unseal Statistical & Schema Corrections (Locked from Astra's `794c9b3` Audit)

1. **Exact 50/50 Equal-Slice OOD Weighting (`c1_test_ood`)**:
   - Per `ACL-exp-c1/PROTOCOL.md:99` and `ACL-exp-c1/scripts/c1_build_datasets.py:1231`, `c1_test_ood` contains `60` `ood_converse` items (grouped under `8` family IDs) and `60` `ood_depth4` items (grouped under `60` family IDs).
   - Naive family-equal averaging across `c1_test_ood` would assign `60/68 = 88.2%` weight to `ood_depth4` and only `11.8%` to `ood_converse`.
   - **Locked Rule**: Total `c1_test_ood` score is defined as **exact 50/50 slice-equal average**:
     `Score_OOD = 0.50 * Mean(ood_converse, n=60) + 0.50 * Mean(ood_depth4, n=60)`
     computed via `c35_stage6_stats.score_ood_equal_slice_50_50(...)` with within-slice stratified paired bootstrap (`n_boot = 5000`). Both slices MUST ALSO be reported separately with exact item counts (`n=60` each), raw scores, and paired 95% CIs!
2. **Strict Fail-Closed Record Integrity Validation**:
   - Every `no_think` and `think` evaluation file MUST pass `c35_stage6_stats.validate_records_strict(...)` before aggregation: non-empty records, exact 1-to-1 `(base_id, arm)` coverage, explicit boolean `parse_ok`, non-null required fields (verified on `c3_dev` prior to unsealing).
3. **Bootstrap Reporting Convention**:
   - Primary inference uses paired 95% bootstrap confidence intervals (`n_boot = 5000`); two-sided p-values below bootstrap resolution are reported as `p < 0.0002` (never `p = 0.0000`).
4. **Dual Denominators & Metrics**:
   - Every model is evaluated ONCE on `c1_test_id` and `c1_test_ood` (`ood_converse` + `ood_depth4`) in BOTH `no_think` and `think` regimes, reporting:
     - Full set (`all`) AND shared `M0` Turn-1 dual-eligible subset (`shared_m0_eligible`);
     - `4-Cell`, `Joint (T1 ∧ 4-Cell)`, and `Turn-1 Dual Acc`.

---

## 2. Locked Final Evaluation Matrix (Weight SHA-256 Hashes To Be Frozen Post-Stage 6)

### Qwen3-8B Matrix
- `M0` (base model)
- `u1_base` (`seed=42`, `seed=1024`)
- `u2_corridor` (`seed=42`, `seed=1024`)
- `M3-v4-PureMid-MLP+` (`seed=42`, `seed=1024`)
- `M3-v4-PureMid-MLP+-Random-Mask` (`Mask1`, `seed=42`, `seed=1024`)
- `M3-v4-PureMid-MLP+-Random-Mask2` (`Mask2`, `seed=42`, `seed=1024`)
- External main-table baselines selected after Stage 6 (pre-specified prior to unsealing): `Pinpoint-DLA-Corridor`, `Pinpoint-DLA-SFT`, `Joint-DPO (dev-best beta*)`, `CAA-Steering (dev-best layer/alpha*)`.

### Qwen3-14B Matrix
- `M0` (base model)
- `u1_base` (`seed=42`, `seed=1024`)
- `u2_corridor` (`seed=42`, `seed=1024`)
- `M3-v4-PureMid-MLP+` (`seed=42`, `seed=1024`)
- `M3-v4-PureMid-MLP+-Random-Mask` (`Mask1`, `seed=42`, `seed=1024`)

---

## 3. Post-Unseal Prohibitions
Once `--allow-sealed-final` is executed:
- ZERO hyperparameter changes;
- ZERO checkpoint re-selection or model retraining;
- ZERO post-hoc alteration of weighting, parser, or statistical test definitions.
