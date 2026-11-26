# ACL-exp-c3.5 Stage 7 Execution Plan — Final One-Shot Sealed Evaluation (`c1_test_id` & `c1_test_ood`)

- **Branch**: `ACL-exp-c3.5`
- **Status**: **SIGNED PRE-REGISTRATION LOCKED & USER-AUTHORIZED FOR ONE-SHOT UNSEALING — READY TO AUTO-RUN IMMEDIATELY ON 8 × A800!**
- **Scope**: Pure inference only (**ZERO** retraining, **ZERO** checkpoint re-selection, **ZERO** CAA dev grid search, **ZERO** post-hoc threshold edits).
- **Signed Lock**: Governed by [`PRE_REGISTRATION_SEALED_TEST.md`](PRE_REGISTRATION_SEALED_TEST.md) and [`data/stage7_sealed_config.json`](data/stage7_sealed_config.json).
- **Hardware & Runtime**: 8 × NVIDIA A800 40GB (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`), **50 total jobs** (`14B = 18 jobs`, `8B = 32 jobs`), estimated total wall time **~1.5–2.0 hours**.

---

## 1. Pre-Unseal Verification Gate (MUST PASS 100% BEFORE `--execute`)

Run all CPU self-tests + server asset hash verification FIRST (ZERO sealed rows opened):

```bash
cd /mnt/kcl/tjy/projects/CEDAR
git fetch origin && git checkout ACL-exp-c3.5 && git pull --ff-only origin ACL-exp-c3.5
PY=${PY:-python}

# 1. CPU self-tests + open-data dry-run + fault injections
$PY experiments/ACL-exp-c3.5/scripts/c35_eval.py --selftest-tiny
$PY experiments/ACL-exp-c3.5/scripts/c35_eval_caa.py --selftest-tiny
$PY experiments/ACL-exp-c3.5/scripts/c35_stage6_stats.py --selftest-tiny
$PY experiments/ACL-exp-c3.5/scripts/c35_stage7_stats.py --dry-run-open
$PY experiments/ACL-exp-c3.5/scripts/c35_run_stage7_sealed.py --dry-run-plan

# 2. Verify ALL 23 adapter .pt, 21 PEFT .safetensors, Pinpoint BF16 merged shards, CAA frozen vector (L2==42.4911), and dataset SHA-256s!
# Note: --recreate-pinpoint-merged is a safe no-op if merged_8b/ already exists; if cleaned up after Stage 6,
# it verifies source PEFT SHA-256 (ee3454fdf4cb3aba...), re-merges FP32->BF16, and verifies all 4 shard SHA-256s bit-for-bit!
$PY experiments/ACL-exp-c3.5/scripts/c35_run_stage7_sealed.py --verify-only --recreate-pinpoint-merged
```

- **Hard Stop**: If `--verify-only` reports ANY missing path or hash/L2 mismatch, STOP immediately and report — do **NOT** run `--execute`.

---

## 2. One-Shot Sealed Evaluation Execution (`--execute --allow-sealed-final`)

Once Section 1 exits `0`, launch the single deterministic parallel runner:

```bash
$PY experiments/ACL-exp-c3.5/scripts/c35_run_stage7_sealed.py \
  --execute \
  --allow-sealed-final \
  --recreate-pinpoint-merged \
  --gpus 0,1,2,3,4,5,6,7 \
  --log-dir /tmp/c35_stage7_logs
```

### What It Runs Automatically
1. **Asset Lock & Receipt**: Re-verifies every file SHA-256 + CAA vector `L2 == 42.4911` and writes `experiments/ACL-exp-c3.5/results/stage7_sealed/stage7_prelaunch_asset_verification.json`.
2. **Wave 7A — Qwen3-14B (18 jobs across 4 × 2-GPU pairs: `0,1`, `2,3`, `4,5`, `6,7`)**:
   - 9 locked models (`M0`, `u1_base × 2`, `u2_corridor × 2`, `M3-v4-PureMid-MLP+ × 2`, `M3-v4-Random-Mask × 2`) × 2 regimes (`no_think` HF BF16 + `think` vLLM `tp=2` native LoRA).
3. **Wave 7B — Qwen3-8B (32 jobs across 8 single GPUs: `0..7`)**:
   - 14 locked models (`M0`, `u1_base × 2`, `u2_corridor × 2`, `M3-v4 × 2`, `RandMask1 × 2`, `RandMask2 × 2`, `Pinpoint-DLA-SFT`, `u2_corridor_only`, `u2_tone_inv_only`) × 2 regimes (`no_think` HF + `think` vLLM native LoRA).
   - `Pinpoint-DLA-Corridor` × 2 regimes (`no_think` HF + `think` vLLM **BF16 merged**, 4-shard SHA-256 verified).
   - `Joint-DPO (β=0.05@step300)` × 1 regime (**`no_think` exact HF only** — `think` excluded per pre-registered vLLM parity failure).
   - `CAA-Steering (anti_syco L22 α=+1.0)` × 1 regime (**`no_think` frozen-vector HF hook only** — exact L2 `42.4911` & vector SHA verified; zero dev grid search).
4. **Write-Once & Resume Safety**:
   - Every job enforces `--allow-sealed-final`, `--no-overwrite`, and full 64-char SHA-256 verification before loading weights.
   - Completed jobs are safely skipped on restart; partial/corrupted directories fail closed and NEVER silently overwrite.
5. **Final Strict Aggregation**:
   - Automatically invokes `c35_stage7_stats.py --config-json experiments/ACL-exp-c3.5/data/stage7_sealed_config.json --out-dir experiments/ACL-exp-c3.5/results/stage7_sealed --n-boot 5000`.
   - Writes `experiments/ACL-exp-c3.5/results/stage7_sealed/C35_STAGE7_SEALED_READOUT.md` and `c35_stage7_stats.json`.

---

## 3. Post-Run Cleanup & Mandatory Git Push (Auto-Publish Results)

After `c35_run_stage7_sealed.py --execute` exits `0` and both `C35_STAGE7_SEALED_READOUT.md` and `c35_stage7_stats.json` are generated:

```bash
# Optional disk cleanup if Pinpoint merged dir was recreated under $WORK:
# rm -rf /mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3.5/stage6-runtime/merged_8b/s6_8b_s42_pinpoint_dla_corridor

# Commit & push ONLY lightweight Stage 7 sealed results (NEVER .pt / .safetensors / .bin / logs!)
git add experiments/ACL-exp-c3.5/results/stage7_sealed/
git status --short
git commit -m "results(ACL-exp-c3.5): publish Stage 7 one-shot sealed test readout (c1_test_id & c1_test_ood)"
git push origin ACL-exp-c3.5
```

- **Absolute Post-Unseal Prohibition**: Do NOT re-run, re-tune, drop seeds, change weighting, or modify any Stage 7 sealed output after unsealing completes. Publish the exact readout verbatim and STOP.
