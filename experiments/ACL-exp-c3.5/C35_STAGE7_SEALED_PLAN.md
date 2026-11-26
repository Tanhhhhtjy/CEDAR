# ACL-exp-c3.5 Stage 7 Execution Plan — Final One-Shot Sealed Evaluation (`c1_test_id` & `c1_test_ood`)

> **Scope**: Pure inference only (**ZERO** retraining, **ZERO** checkpoint re-selection, **ZERO** CAA dev grid search, **ZERO** post-hoc threshold edits).
> **Signed Lock**: Governed by [`PRE_REGISTRATION_SEALED_TEST.md`](PRE_REGISTRATION_SEALED_TEST.md) and [`data/stage7_sealed_config.json`](data/stage7_sealed_config.json).
> **Hardware**: 8 × NVIDIA A800 40GB (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`), estimated total wall time **~1.5–2.0 hours** across 50 jobs (`14B = 18 jobs`, `8B = 32 jobs`).

---

## 1. Pre-Unseal Verification Gate (MUST PASS 100% BEFORE `--execute`)

Run all CPU self-tests + server asset hash verification FIRST (ZERO sealed rows opened):

```bash
cd /mnt/kcl/tjy/projects/CEDAR
git fetch origin && git checkout ACL-exp-c3.5 && git pull --ff-only origin ACL-exp-c3.5

# 1. CPU self-tests + open-data dry-run + fault injections
python experiments/ACL-exp-c3.5/scripts/c35_eval.py --selftest-tiny
python experiments/ACL-exp-c3.5/scripts/c35_eval_caa.py --selftest-tiny
python experiments/ACL-exp-c3.5/scripts/c35_stage6_stats.py --selftest-tiny
python experiments/ACL-exp-c3.5/scripts/c35_stage7_stats.py --dry-run-open
python experiments/ACL-exp-c3.5/scripts/c35_run_stage7_sealed.py --dry-run-plan

# 2. Verify ALL 23 adapter .pt, 21 PEFT .safetensors, Pinpoint BF16 merged shards, CAA frozen vector, and dataset SHA-256s!
# (If merged_8b/s6_8b_s42_pinpoint_dla_corridor was cleaned up after Stage 6, add --recreate-pinpoint-merged.)
python experiments/ACL-exp-c3.5/scripts/c35_run_stage7_sealed.py --verify-only
```

- If `merged_8b/s6_8b_s42_pinpoint_dla_corridor` was deleted to save disk space after Stage 6, pass `--recreate-pinpoint-merged` (it first verifies source PEFT `ee3454fdf4cb3aba...`, re-merges in FP32 → BF16, and verifies all 4 shard SHA-256 hashes match `vllm_merged_parity.json` bit-for-bit!).
- If `caa_directions_train.pt` is located under `$WORK`, pass `--caa-vector-file <path>` (the runner automatically checks default Stage 6 paths as well and verifies exact `L2 == 42.4911`!).
- **Hard Stop**: If `--verify-only` reports ANY missing path or hash mismatch, STOP immediately and report — do **NOT** run `--execute`.

---

## 2. One-Shot Sealed Evaluation Execution (`--execute --allow-sealed-final`)

Once `--verify-only` exits `0`, launch the single deterministic parallel runner:

```bash
python experiments/ACL-exp-c3.5/scripts/c35_run_stage7_sealed.py \
  --execute \
  --allow-sealed-final \
  --gpus 0,1,2,3,4,5,6,7 \
  --log-dir /tmp/c35_stage7_logs
```

### What It Runs Automatically
1. **Asset Lock & Receipt**: Re-verifies every file SHA-256 and writes `experiments/ACL-exp-c3.5/results/stage7_sealed/stage7_prelaunch_asset_verification.json` (including resolved `caa_directions_train.pt` exact SHA-256).
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
   - Writes `experiments/ACL-exp-c3.5/results/stage7_sealed/C35_STAGE7_SEALED_READOUT.md` and `c35_stage7_stats.json` (verifying every arm's `provenance` block, exact `100` ID + exact `60+60` OOD item coverage, separate per-`(scale, regime, dataset)` `M0` dual-eligible denominators, exact unrounded 50/50 OOD scores, `parse_fail`/`truncation` counts, and Holm-Bonferroni primary p-values).

---

## 3. Post-Run Hygiene & Git Push Rules

- **NEVER** commit `.pt`, `.safetensors`, `.bin`, merged model dirs, or `/tmp/c35_stage7_logs`.
- If `--recreate-pinpoint-merged` created a temporary `merged_8b/` directory under `$WORK`, delete it after Stage 7 aggregation passes.
- Commit and push ONLY lightweight results under `experiments/ACL-exp-c3.5/results/stage7_sealed/` (`*.json`, `*.jsonl`, `*.md`) on branch `ACL-exp-c3.5`!
- **Absolute Post-Unseal Prohibition**: Do NOT re-run, re-tune, drop seeds, change weighting, or modify any Stage 7 sealed output after unsealing completes. Report the exact readout verbatim.
