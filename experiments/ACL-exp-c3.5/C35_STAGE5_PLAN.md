# ACL-exp-c3.5 Stage 5 Plan: 8B Rank-Matched Closure + 14B 4-Arm Prospective Replication + Official External EX-FEVER (8 × A800)

- **Branch**: `ACL-exp-c3.5` (Stage 4 complete at `0628626`, Stage 5 READY TO AUTO-RUN IMMEDIATELY!)
- **Models**:
  - `BASE8=/mnt/kcl/devuser/ckpts/Qwen3-8B`
  - `BASE14=/mnt/kcl/devuser/ckpts/Qwen3-14B`
- **Absolute Red Lines**:
  - NEVER open, read, or evaluate sealed `c1_test_id_bases.jsonl` / `c1_test_ood_bases.jsonl`; strictly exclude `converse_error` and `depth-4`.
  - NEVER modify frozen `ACL-exp-a1/`, `ACL-exp-b1/`, `ACL-exp-c1/`, `ACL-exp-c2/`, or `ACL-exp-c3/`.
  - **NEVER commit `.pt` / `.safetensors` to Git**; push only lightweight `json / jsonl / md` results under `experiments/ACL-exp-c3.5/results/stage5_8b/` and `experiments/ACL-exp-c3.5/results/stage5_14b/` incrementally to `origin/ACL-exp-c3.5`.
  - **14B Seed Policy**: Run `Qwen3-14B` `seed=42` 4-arm prospective showdown + `EX-FEVER` + `Probe 2 v2` FIRST; DO NOT launch 14B `seed=1024` automatically until `seed=42` readout is reviewed!

---

## 1. Frozen Artifacts & SHA-256 Registry (Verified Locally on CPU)

| Artifact | Path | Key Invariants | SHA-256 (prefix) |
|---|---|---|---|
| Frozen Architectures (8B & 14B) | `experiments/ACL-exp-c3.5/data/c35_components_frozen.json` | Adds `M3-v4-PureMid-MLP+-Random-Mask` (`Mask1`, seed 2037) & `M3-v4-PureMid-MLP+-Random-Mask2` (`Mask2`, seed 2038). Exact match: 8B `6,994,944` (`-1.17%`, `L14–27`, `max_r=68`); 14B `10,348,800` (`-1.31%`, **`L16–30`**, `max_r=78`). | `a28a18b22468` |
| Prospective Blind v3 | `experiments/ACL-exp-c3.5/data/c3_blind_confirm_v3_bases.jsonl` | `1,080` bases / `900` units, 6 brand-new domains, 2 unseen styles, `240/240` CE decoys, `840/840` independent dual-premise DAG, `60` **fixed-style** `(before_rules vs after_rules)` pairs, `0` skeleton overlap vs `c3_train` & `v2`, `G0 AUC=0.5085`. | `32cdfab58ecc` |
| Prospective Blind Think v3 | `experiments/ACL-exp-c3.5/data/c3_blind_think_v3_bases.jsonl` | `240` stratified CoT items (`192` units). | `7dee7d61d636` |
| Official Ma et al. EX-FEVER Test | `experiments/ACL-exp-c3.5/data/exfever_official_test1000.jsonl` | Exact 1,000-row official `test` split (`507 True / 493 False`) from `dependentsign/sycophancy-rational-updating` (`emnlp-2026-camera-ready`), bit-for-bit upstream SHA256 verified! | `977ceccb3cfd` |

---

## 2. Exact Server Paths & Reusable 8B Checkpoints

- `WORK=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3.5/stage5-<SHORT_SHA>`
- `RES8=experiments/ACL-exp-c3.5/results/stage5_8b`
- `RES14=experiments/ACL-exp-c3.5/results/stage5_14b`
- `COMP=experiments/ACL-exp-c3.5/data/c35_components_frozen.json`
- `KL_CORPUS=experiments/ACL-exp-c1/data/c1_kl_wikitext2_200.jsonl`
- `EXF_DATA=experiments/ACL-exp-c3.5/data/exfever_official_test1000.jsonl`
- **Existing Frozen 8B Checkpoints on Server (DO NOT RETRAIN — reuse verbatim!)**:
  - `u1_base@42`: `pt=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3/showdown-8438e5e2/selected_42/u1_base/selected_adapter_state.pt`, `peft=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3/showdown-8438e5e2/peft_42/u1_base`, `prev_res=experiments/ACL-exp-c3.5/results/stage3/u1_base`
  - `u2_corridor@42`: `pt=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3/showdown-8438e5e2/selected_42/u2_corridor/selected_adapter_state.pt`, `peft=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3/showdown-8438e5e2/peft_42/u2_corridor`, `prev_res=experiments/ACL-exp-c3.5/results/stage3/u2_corridor`
  - `u1_base@1024`: `pt=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3.5/stage4-fb5c03cb/selected/s4_s1024_u1_base/selected_adapter_state.pt`, `peft=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3.5/stage4-fb5c03cb/peft/s4_s1024_u1_base`, `prev_res=experiments/ACL-exp-c3.5/results/stage4/s4_s1024_u1_base`
  - `u2_corridor@1024`: `pt=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3.5/stage4-fb5c03cb/selected/s4_s1024_u2_corridor/selected_adapter_state.pt`, `peft=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3.5/stage4-fb5c03cb/peft/s4_s1024_u2_corridor`, `prev_res=experiments/ACL-exp-c3.5/results/stage4/s4_s1024_u2_corridor`
  - `v4_mlpplus@42`: `pt=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3.5/stage4-fb5c03cb/selected/s4_s42_v4_mlpplus/selected_adapter_state.pt`, `peft=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3.5/stage4-fb5c03cb/peft/s4_s42_v4_mlpplus`, `prev_res=experiments/ACL-exp-c3.5/results/stage4/s4_s42_v4_mlpplus`
  - `v4_mlpplus@1024`: `pt=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3.5/stage4-fb5c03cb/selected/s4_s1024_v4_mlpplus/selected_adapter_state.pt`, `peft=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3.5/stage4-fb5c03cb/peft/s4_s1024_v4_mlpplus`, `prev_res=experiments/ACL-exp-c3.5/results/stage4/s4_s1024_v4_mlpplus`

---

## 3. Pre-Run CPU Self-Tests (~30s)

Run before allocating GPUs:
```bash
$PY experiments/ACL-exp-c3.5/scripts/c35_14b_preflight.py --selftest-tiny
$PY experiments/ACL-exp-c3.5/scripts/c35_eval_cross_letter_v2.py --selftest-tiny
$PY experiments/ACL-exp-c3.5/scripts/c35_eval_exfever.py --selftest-tiny
$PY experiments/ACL-exp-c3.5/scripts/c35_stage5_stats.py --selftest-tiny
```

---

## 4. Wave 5A: 8B Rank-Matched Closure + 8B `v3`/EX-FEVER/Probe2-v2 + 14B Preflight & `M0` Anchor (All 8 GPUs Parallel, ~2.5h)

### 4.1 GPUs 0–3: Train & Full-Eval 4 New 8B Rank-Matched Random-Mask Arms (Parallel, 1 GPU each)
| GPU | Tag | Seed | Arch | Loss / Z1 |
|---:|---|---:|---|---|
| **GPU 0** | `s5_8b_s42_v4_randmask1` | `42` | `M3-v4-PureMid-MLP+-Random-Mask` | `corridor`, `lr_z1=0.0, prox=0.0` |
| **GPU 1** | `s5_8b_s1024_v4_randmask1` | `1024` | `M3-v4-PureMid-MLP+-Random-Mask` | `corridor`, `lr_z1=0.0, prox=0.0` |
| **GPU 2** | `s5_8b_s42_v4_randmask2` | `42` | `M3-v4-PureMid-MLP+-Random-Mask2` | `corridor`, `lr_z1=0.0, prox=0.0` |
| **GPU 3** | `s5_8b_s1024_v4_randmask2` | `1024` | `M3-v4-PureMid-MLP+-Random-Mask2` | `corridor`, `lr_z1=0.0, prox=0.0` |

Exact per-GPU sequence for each `<TAG>`:
1. `c35_train.py --model $BASE8 --model-key qwen3_8b --components-file $COMP --arch <ARCH> --loss-mode corridor --iit-target none --lr 2.5e-5 --lr-z1-mult 0.0 --lambda-prox-z1 0.0 --max-steps 300 --grad-accum 20 --microbatch 2 --save-steps 120,200,250,300 --seed <SEED> --out-dir $WORK/ckpts_8b/<TAG>` (copy `train_meta.json` -> `$RES8/<TAG>/`).
2. `c35_select_dev_ckpt.py --model $BASE8 --model-key qwen3_8b --arch <ARCH> --ckpt-dir $WORK/ckpts_8b/<TAG> --batch-size 16 --out-dir $WORK/selected_8b/<TAG>` (copy `dev_selection.json` -> `$RES8/<TAG>/`).
3. `c35_export_peft_lora.py --model-key qwen3_8b --arch <ARCH> --adapter-state $WORK/selected_8b/<TAG>/selected_adapter_state.pt --base-model $BASE8 --out-dir $WORK/peft_8b/<TAG>` (copy `c3_peft_export_meta.json` -> `$RES8/<TAG>/`).
4. `c35_check_vllm_lora_parity.py --base-model $BASE8 --model-key qwen3_8b --arch <ARCH> --adapter-state $WORK/selected_8b/<TAG>/selected_adapter_state.pt --peft-lora-dir $WORK/peft_8b/<TAG> --dataset c3_dev --limit 40 --max-abs-du-mean 0.50 --no-fail-exit --out-json $RES8/<TAG>/vllm_lora_parity.json`.
5. `c35_eval.py --model $BASE8 --base-model $BASE8 --model-key qwen3_8b --arch <ARCH> --adapter-state $WORK/selected_8b/<TAG>/selected_adapter_state.pt --alias <TAG> --regime no_think --batch-size 8 --datasets c3_blind_confirm_v2 c3_blind_confirm_v3 c2_confirm_bidir c2_stealth_stress --kl-corpus $KL_CORPUS --out-dir $RES8/<TAG>/no_think`.
6. `c35_eval.py --model $BASE8 --backend vllm --peft-lora-dir $WORK/peft_8b/<TAG> --alias <TAG> --regime think --max-new-tokens 3072 --datasets c3_blind_think_v2 c3_blind_think_v3 --arms turn1_F turn1_V V_hes V_auth F_hes F_auth --out-dir $RES8/<TAG>/think`.
7. `experiments/ACL-exp-c1/scripts/c1_eval_capabilities.py --model $BASE8 --peft-lora-dir $WORK/peft_8b/<TAG> --ifeval-jsonl $IFEVAL_JSONL --gsm8k-jsonl $GSM8K_JSONL --ifeval-lib-dir $IFEVAL_LIB --alias <TAG> --out-dir $RES8/<TAG>/capabilities`.
8. `c35_eval_exfever.py --model $BASE8 --peft-lora-dir $WORK/peft_8b/<TAG> --data-jsonl $EXF_DATA --regime no_think --tp 1 --alias <TAG> --out-dir $RES8/<TAG>/exfever` AND `--regime think --tp 1 --alias <TAG> --out-dir $RES8/<TAG>/exfever`.
9. Incrementally commit & push `$RES8/<TAG>/` to `origin/ACL-exp-c3.5`.

### 4.2 GPUs 4–5: Existing 8B Arms Prospective `v3` + Official `EX-FEVER` + 8B Probe 2 v2
- Stage each of the 6 existing frozen 8B arms (`s5_8b_s42_u1_base`, `s5_8b_s1024_u1_base`, `s5_8b_s42_u2_corridor`, `s5_8b_s1024_u2_corridor`, `s5_8b_s42_v4_mlpplus`, `s5_8b_s1024_v4_mlpplus`) by copying its existing `no_think/`, `think/`, `capabilities/`, `vllm_lora_parity.json`, `dev_selection.json` from `prev_res` into `$RES8/<TAG>/`.
- Evaluate ONLY the NEW prospective datasets on each existing 8B arm (split cleanly across GPU 4 & GPU 5, ~35 min total!):
  - `c35_eval.py --model $BASE8 --model-key qwen3_8b --arch <ARCH> --adapter-state <PT> --alias <TAG> --regime no_think --batch-size 8 --datasets c3_blind_confirm_v3 --out-dir $RES8/<TAG>/no_think_v3` -> copy `records_no_think_c3_blind_confirm_v3.jsonl` into `$RES8/<TAG>/no_think/` (preserving original `eval_summary_no_think.json` & `reference_kl`!).
  - `c35_eval.py --model $BASE8 --backend vllm --peft-lora-dir <PEFT> --alias <TAG> --regime think --max-new-tokens 3072 --datasets c3_blind_think_v3 --arms turn1_F turn1_V V_hes V_auth F_hes F_auth --out-dir $RES8/<TAG>/think_v3` -> copy `records_think_c3_blind_think_v3.jsonl` into `$RES8/<TAG>/think/`.
  - `c35_eval_exfever.py --model $BASE8 --peft-lora-dir <PEFT> --data-jsonl $EXF_DATA --regime no_think --tp 1 --alias <TAG> --out-dir $RES8/<TAG>/exfever` AND `--regime think`.
- Also run `8B M0` on `EX-FEVER` (`no_think` & `think`) into `$RES8/M0/exfever` (no `--peft-lora-dir`)!
- Once GPUs 0 & 2 finish `s5_8b_s42_v4_randmask1` and `s5_8b_s42_v4_randmask2`: run **8B Probe 2 v2** (`c35_eval_cross_letter_v2.py --model $BASE8 --model-key qwen3_8b --dataset-file experiments/ACL-exp-c3.5/data/c3_blind_confirm_v3_bases.jsonl --out-dir $RES8/probe2_v2`) across `u2_corridor@s42`, `v4_mlpplus@s42`, `v4_randmask1@s42`, `v4_randmask2@s42`, then run `--aggregate-common`!
- Run `c35_stage5_stats.py` for 8B (`m0_anchor: {ifeval: 440, gsm8k: 496}`, both `c3_blind_confirm_v2` & `c3_blind_confirm_v3`, `exfever_regimes: ["no_think", "think"]`) -> publish `$RES8/c35_stage5_qwen3_8b_stats.json` & `$RES8/C35_STAGE5_QWEN3_8B_READOUT.md`!

### 4.3 GPUs 6–7 (2×A800): MANDATORY 14B Pre-Flight Gate + 14B `M0` Anchor (~45 min)
1. **MANDATORY HARD GATE (MUST PASS BEFORE WAVE 5B!)**:
   ```bash
   CUDA_VISIBLE_DEVICES=6,7 $PY experiments/ACL-exp-c3.5/scripts/c35_14b_preflight.py \
     --model $BASE14 --components-file $COMP --arch M3-v4-PureMid-MLP+ \
     --tp 2 --microbatch 1 --grad-accum 40 --parity-limit 20 \
     --out-dir $RES14/preflight
   ```
   Verify `$RES14/preflight/preflight_14b_passed.json` (`passed == true`), immediately push `$RES14/preflight/` to `origin/ACL-exp-c3.5`!
2. **14B `M0` Full Anchor (`CUDA_VISIBLE_DEVICES=6,7`, `--tp 2`)**:
   - Capabilities (`IFEval 541` + `GSM8K 1,319`, locks 14B's OWN relative health floor!):
     `c1_eval_capabilities.py --model $BASE14 --tp 2 --ifeval-jsonl $IFEVAL_JSONL --gsm8k-jsonl $GSM8K_JSONL --ifeval-lib-dir $IFEVAL_LIB --alias m0_14b --out-dir $RES14/M0/capabilities`
   - `EX-FEVER` (`1,000` official test, BOTH `no_think` & `think`):
     `c35_eval_exfever.py --model $BASE14 --tp 2 --data-jsonl $EXF_DATA --regime no_think --alias M0 --out-dir $RES14/M0/exfever`
     `c35_eval_exfever.py --model $BASE14 --tp 2 --data-jsonl $EXF_DATA --regime think --alias M0 --out-dir $RES14/M0/exfever`
   - `no_think` (`c3_blind_confirm_v2`, `c3_blind_confirm_v3`):
     `c35_eval.py --model $BASE14 --model-key qwen3_14b --alias m0_14b --regime no_think --batch-size 4 --datasets c3_blind_confirm_v2 c3_blind_confirm_v3 --out-dir $RES14/M0/no_think`
   - Push `$RES14/M0/` incrementally!

---

## 5. Wave 5B: `Qwen3-14B` Compact 4-Arm Prospective Showdown (`seed=42`, All 8 GPUs = 4 × 2-GPU Parallel, ~4–5h)

> **HARD PRECONDITION**: ONLY launch after `$RES14/preflight/preflight_14b_passed.json` exists with `"passed": true`!

| GPUs | Arm Tag | Seed | Architecture | Loss Mode / Z1 | Params (`Δ%`) |
|---|---|---:|---|---|---:|
| **GPU 0,1** | `s5_14b_s42_u1_base` | `42` | `Uniform-LoRA` | `base`, `lr_z1=1.0, prox=0.0` | `10,485,760` (`0.00%`) |
| **GPU 2,3** | `s5_14b_s42_u2_corridor` | `42` | `Uniform-LoRA` | `corridor`, `lr_z1=1.0, prox=0.0` | `10,485,760` (`0.00%`) |
| **GPU 4,5** | `s5_14b_s42_v4_mlpplus` | `42` | `M3-v4-PureMid-MLP+` (`L16–30`) | `corridor`, `lr_z1=0.0, prox=0.0` | `10,348,800` (`-1.31%`) |
| **GPU 6,7** | `s5_14b_s42_v4_randmask` | `42` | `M3-v4-PureMid-MLP+-Random-Mask` (`L16–30`) | `corridor`, `lr_z1=0.0, prox=0.0` | `10,348,800` (`-1.31%`) |

Exact per-2-GPU sequence for each `<TAG>` (zero-tuning cross-scale transfer matching 8B bit-for-bit: `grad_accum=20` examples/window × `300` optimizer steps = 1 epoch of `6,000` examples, with `--microbatch 1` for clean 14B VRAM headroom; resolves `wave5b_budget_evidence.json`!):
1. `c35_train.py --model $BASE14 --model-key qwen3_14b --components-file $COMP --arch <ARCH> --loss-mode <LOSS> --iit-target none --lr 2.5e-5 --lr-z1-mult <LRZ1> --lambda-prox-z1 0.0 --max-steps 300 --grad-accum 20 --microbatch 1 --save-steps 120,200,250,300 --seed 42 --out-dir $WORK/ckpts_14b/<TAG>` (copy `train_meta.json` -> `$RES14/<TAG>/`, push!).
2. `c35_select_dev_ckpt.py --model $BASE14 --model-key qwen3_14b --arch <ARCH> --ckpt-dir $WORK/ckpts_14b/<TAG> --batch-size 8 --out-dir $WORK/selected_14b/<TAG>` (copy `dev_selection.json` -> `$RES14/<TAG>/`).
3. `c35_export_peft_lora.py --model-key qwen3_14b --arch <ARCH> --adapter-state $WORK/selected_14b/<TAG>/selected_adapter_state.pt --base-model $BASE14 --out-dir $WORK/peft_14b/<TAG>` (copy `c3_peft_export_meta.json` -> `$RES14/<TAG>/`).
4. `c35_check_vllm_lora_parity.py --base-model $BASE14 --model-key qwen3_14b --arch <ARCH> --adapter-state $WORK/selected_14b/<TAG>/selected_adapter_state.pt --peft-lora-dir $WORK/peft_14b/<TAG> --dataset c3_dev --limit 40 --tp 2 --max-abs-du-mean 0.50 --no-fail-exit --out-json $RES14/<TAG>/vllm_lora_parity.json`.
5. `c35_eval.py --model $BASE14 --base-model $BASE14 --model-key qwen3_14b --arch <ARCH> --adapter-state $WORK/selected_14b/<TAG>/selected_adapter_state.pt --alias <TAG> --regime no_think --batch-size 4 --datasets c3_blind_confirm_v3 c3_blind_confirm_v2 c2_confirm_bidir c2_stealth_stress --kl-corpus $KL_CORPUS --out-dir $RES14/<TAG>/no_think`.
6. `c35_eval.py --model $BASE14 --backend vllm --tp 2 --peft-lora-dir $WORK/peft_14b/<TAG> --alias <TAG> --regime think --max-new-tokens 3072 --datasets c3_blind_think_v3 c3_blind_think_v2 --arms turn1_F turn1_V V_hes V_auth F_hes F_auth --out-dir $RES14/<TAG>/think`.
7. `experiments/ACL-exp-c1/scripts/c1_eval_capabilities.py --model $BASE14 --tp 2 --peft-lora-dir $WORK/peft_14b/<TAG> --ifeval-jsonl $IFEVAL_JSONL --gsm8k-jsonl $GSM8K_JSONL --ifeval-lib-dir $IFEVAL_LIB --alias <TAG> --out-dir $RES14/<TAG>/capabilities`.
8. **Official External `EX-FEVER` (`1,000` test, BOTH `no_think` & `think`, `--tp 2`)**:
   - `c35_eval_exfever.py --model $BASE14 --peft-lora-dir $WORK/peft_14b/<TAG> --data-jsonl $EXF_DATA --regime no_think --tp 2 --alias <TAG> --out-dir $RES14/<TAG>/exfever`
   - `c35_eval_exfever.py --model $BASE14 --peft-lora-dir $WORK/peft_14b/<TAG> --data-jsonl $EXF_DATA --regime think --tp 2 --alias <TAG> --out-dir $RES14/<TAG>/exfever`
9. Incrementally commit & push `$RES14/<TAG>/` to `origin/ACL-exp-c3.5`!

### 5.1 14B Upgraded Probe 2 v2 + Final Strict Stage 5 Aggregation
- Run **14B Probe 2 v2** (`c35_eval_cross_letter_v2.py --model $BASE14 --model-key qwen3_14b --dataset-file experiments/ACL-exp-c3.5/data/c3_blind_confirm_v3_bases.jsonl --out-dir $RES14/probe2_v2`) in parallel across 3 × 2-GPU pairs for `s5_14b_s42_u2_corridor`, `s5_14b_s42_v4_mlpplus`, `s5_14b_s42_v4_randmask`, then run `--aggregate-common`!
- Generate strict 14B config JSON (using exact pair names `Corridor_vs_Base`, `M3-v4-PureMid-MLP+_vs_u2`, `M3-v4-PureMid-MLP+-Random-Mask_vs_u2`, `v4_MLPplus_vs_v4_RandMask`) and run:
  ```bash
  $PY experiments/ACL-exp-c3.5/scripts/c35_stage5_stats.py \
    --config-json $RES14/stage5_14b_config.json \
    --m0-cap-json $RES14/M0/capabilities/summary.json \
    --n-boot 5000 --out-dir $RES14
  ```
- Push `$RES14/c35_stage5_qwen3_14b_stats.json`, `$RES14/C35_STAGE5_QWEN3_14B_READOUT.md`, and `$RES14/probe2_v2/` to `origin/ACL-exp-c3.5`!
