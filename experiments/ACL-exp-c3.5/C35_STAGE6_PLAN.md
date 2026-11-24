# ACL-exp-c3.5 Stage 6 Plan: 14B Dual-Seed Closure, Entity-Balanced Diagnostic, External SOTA Baselines & Loss Ablations (8 × A800)

- **Branch**: `ACL-exp-c3.5` (Stage 5 complete & audited at `794c9b3`)
- **Status**: **ASTRA PRE-RUN AUDIT PASSED — READY TO AUTO-RUN IMMEDIATELY ON 8 × A800!**
- **Models**:
  - `BASE8=/mnt/kcl/devuser/ckpts/Qwen3-8B`
  - `BASE14=/mnt/kcl/devuser/ckpts/Qwen3-14B`
- **Absolute Red Lines (STRICTLY ENFORCED)**:
  - **HOLD SEALED TEST**: NEVER open, read, or evaluate sealed `c1_test_id_bases.jsonl` / `c1_test_ood_bases.jsonl` during Stage 6 (`converse_error` and `depth-4` remain strictly untouched until Stage 7 pre-registration sign-off!).
  - NEVER modify frozen `ACL-exp-a1/`, `ACL-exp-b1/`, `ACL-exp-c1/`, `ACL-exp-c2/`, `ACL-exp-c3/`, or Stage 5's `data/c35_components_frozen.json` (`a28a18b22468...`).
  - **NEVER commit `.pt` / `.safetensors` to Git**; push only lightweight `json / jsonl / md` results under `experiments/ACL-exp-c3.5/results/stage6/`.

---

## 1. New Frozen Stage 6 Artifacts & SHA-256 Registry (Verified Locally on CPU)

| Artifact | Path | Key Invariants | SHA-256 (prefix) |
|---|---|---|---|
| Stage 6 Components (superset of Stage 5) | `experiments/ACL-exp-c3.5/data/c35_stage6_components_frozen.json` | Parent `c35_components_frozen.json` (`a28a18b22468`) untouched! Adds `qwen3_8b` **`Pinpoint-DLA-HeadsOnly`** (`Pinpoint-style DLA`: Top-64 `|dla_all|` heads from `qwen3_8b_eligible/dla_C_pinpoint.pt` @ GQA Q/O `r=13`, **0 MLPs**, `7,028,736` params = `-0.69%` vs Uniform / `+0.48%` vs `v4-MLP+`, `max_packed_r=117 <= 128`, zones `Z1=0, Z2=19, Z3=45`). | `428f4e095661` |
| Entity-Count-Balanced Diagnostic v1 | `experiments/ACL-exp-c3.5/data/c3_entity_balanced_v1_bases.jsonl` | `480` bases / `480` independent units across 6 balanced 80-base cells (`1v2_orig`, `2v1_anti`, `1v1_same`, `2v2_same`, `1v1_ctx`, `2v2_ctx`), 6 V3 domains & 2 V3 styles, **480/480 DAG solver certified**, **exact 50/50 entity count in V (`240/240`) and F (`240/240`) -> entity-count AUC = `0.5000`**, `G0 TF-IDF AUC = 0.5112`, `cf_byte_match = 1.0`, zero overlap with all open splits. | `ba0abf16eb53` |

---

## 2. Exact Server Paths & Reusable Checkpoints

- `WORK=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3.5/stage6-<SHORT_SHA>`
- `RES6=experiments/ACL-exp-c3.5/results/stage6`
- `COMP5=experiments/ACL-exp-c3.5/data/c35_components_frozen.json`
- `COMP6=experiments/ACL-exp-c3.5/data/c35_stage6_components_frozen.json`
- `KL_CORPUS=experiments/ACL-exp-c1/data/c1_kl_wikitext2_200.jsonl`
- `EXF_DATA=experiments/ACL-exp-c3.5/data/exfever_official_test1000.jsonl`
- **Existing Stage 5 14B (`seed=42`) checkpoints on server**:
  - `S5_14B=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3.5/stage5-45ba5fd0`
  - `u2@14b_s42_sel`: `$S5_14B/selected_14b/s5_14b_s42_u2_corridor/selected_adapter_state.pt` (final) & `step0200`: `$S5_14B/ckpts_14b/s5_14b_s42_u2_corridor/adapter_state_step0200.pt`
  - `v4@14b_s42_sel`: `$S5_14B/selected_14b/s5_14b_s42_v4_mlpplus/selected_adapter_state.pt` (already `step0200`!)
  - `rm@14b_s42_sel`: `$S5_14B/selected_14b/s5_14b_s42_v4_randmask/selected_adapter_state.pt` (`step0250`) & `step0200`: `$S5_14B/ckpts_14b/s5_14b_s42_v4_randmask/adapter_state_step0200.pt`
- **Existing Stage 4/5 8B (`seed=42`) checkpoints on server**:
  - `u2@8b_s42`: `/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3/showdown-8438e5e2/selected_42/u2_corridor/selected_adapter_state.pt`
  - `v4@8b_s42`: `/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3.5/stage4-fb5c03cb/selected/s4_s42_v4_mlpplus/selected_adapter_state.pt`
  - `rm1@8b_s42`: `/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3.5/stage5-45ba5fd0/selected_8b/s5_8b_s42_v4_randmask1/selected_adapter_state.pt`

---

## 3. Mandatory Pre-Run CPU Self-Tests (~30s)

```bash
$PY experiments/ACL-exp-c3.5/scripts/c35_train.py --selftest-tiny
$PY experiments/ACL-exp-c3.5/scripts/c35_train_dpo.py --selftest-tiny
$PY experiments/ACL-exp-c3.5/scripts/c35_eval_caa.py --selftest-tiny
$PY experiments/ACL-exp-c3.5/scripts/c35_eval.py --selftest-tiny
$PY experiments/ACL-exp-c3.5/scripts/c35_stage5_stats.py --selftest-tiny
$PY experiments/ACL-exp-c3.5/scripts/c35_stage6_stats.py --selftest-tiny
```

---

## 4. Wave 6A — Priority 1: `Qwen3-14B seed=1024` Original 4 Arms + Fixed `step 200` Confound Audit (All 8 GPUs Parallel, ~4.5h)

### 4.1 4-Arm Parallel Training & Full Eval (`seed=1024`, 2×A800 per arm, exact Stage 5 protocol!)
| GPUs | Tag | Arch | Loss / Z1 | Budget |
|---|---|---|---|---|
| **0,1** | `s6_14b_s1024_u1_base` | `Uniform-LoRA` | `base`, `lr_z1=1.0, prox=0.0` | `grad_accum=20, microbatch=1, max_steps=300` |
| **2,3** | `s6_14b_s1024_u2_corridor` | `Uniform-LoRA` | `corridor`, `lr_z1=1.0, prox=0.0` | `grad_accum=20, microbatch=1, max_steps=300` |
| **4,5** | `s6_14b_s1024_v4_mlpplus` | `M3-v4-PureMid-MLP+` | `corridor`, `lr_z1=0.0, prox=0.0` | `grad_accum=20, microbatch=1, max_steps=300` |
| **6,7** | `s6_14b_s1024_v4_randmask` | `M3-v4-PureMid-MLP+-Random-Mask` | `corridor`, `lr_z1=0.0, prox=0.0` | `grad_accum=20, microbatch=1, max_steps=300` |

Exact per-pair sequence (identical to Stage 5 Wave 5B):
1. `c35_train.py --model $BASE14 --model-key qwen3_14b --components-file $COMP5 --arch <ARCH> --loss-mode <LOSS> --iit-target none --lr 2.5e-5 --lr-z1-mult <Z1LR> --lambda-prox-z1 0.0 --max-steps 300 --grad-accum 20 --microbatch 1 --save-steps 120,200,250,300 --seed 1024 --out-dir $WORK/ckpts_14b/<TAG>`
2. `c35_select_dev_ckpt.py --model $BASE14 --model-key qwen3_14b --components-file $COMP5 --arch <ARCH> --ckpt-dir $WORK/ckpts_14b/<TAG> --batch-size 8 --out-dir $WORK/selected_14b/<TAG>`
3. `c35_export_peft_lora.py --model-key qwen3_14b --components-file $COMP5 --arch <ARCH> --adapter-state $WORK/selected_14b/<TAG>/selected_adapter_state.pt --base-model $BASE14 --out-dir $WORK/peft_14b/<TAG>`
4. `c35_check_vllm_lora_parity.py --base-model $BASE14 --model-key qwen3_14b --components-file $COMP5 --arch <ARCH> --adapter-state $WORK/selected_14b/<TAG>/selected_adapter_state.pt --peft-lora-dir $WORK/peft_14b/<TAG> --tp 2 --dataset c3_dev --limit 40 --max-abs-du-mean 0.50 --no-fail-exit --out-json $RES6/14b_s1024/<TAG>/vllm_lora_parity.json`
5. `c35_eval.py --model $BASE14 --base-model $BASE14 --model-key qwen3_14b --components-file $COMP5 --arch <ARCH> --adapter-state $WORK/selected_14b/<TAG>/selected_adapter_state.pt --alias <TAG> --regime no_think --batch-size 4 --datasets c3_blind_confirm_v2 c3_blind_confirm_v3 c2_confirm_bidir c2_stealth_stress --kl-corpus $KL_CORPUS --out-dir $RES6/14b_s1024/<TAG>/no_think`
6. `c35_eval.py --model $BASE14 --backend vllm --tp 2 --peft-lora-dir $WORK/peft_14b/<TAG> --alias <TAG> --regime think --max-new-tokens 3072 --datasets c3_blind_think_v2 c3_blind_think_v3 --arms turn1_F turn1_V V_hes V_auth F_hes F_auth --out-dir $RES6/14b_s1024/<TAG>/think`
7. `c1_eval_capabilities.py --model $BASE14 --tp 2 --peft-lora-dir $WORK/peft_14b/<TAG> --ifeval-jsonl $IFEVAL_JSONL --gsm8k-jsonl $GSM8K_JSONL --ifeval-lib-dir $IFEVAL_LIB --alias <TAG> --out-dir $RES6/14b_s1024/<TAG>/capabilities`
8. `c35_eval_exfever.py --model $BASE14 --tp 2 --peft-lora-dir $WORK/peft_14b/<TAG> --data-jsonl $EXF_DATA --regime no_think --alias <TAG> --out-dir $RES6/14b_s1024/<TAG>/exfever` AND `--regime think`.

**Pre-Specified 14B Primary Comparisons** (reported both per-seed `42` / `1024` AND `pooled_2seed`, for BOTH `900` all units AND `802` post-hoc non-shortcut units):
- `M3-v4-PureMid-MLP+` vs `u2_corridor`
- `M3-v4-PureMid-MLP+` vs `M3-v4-PureMid-MLP+-Random-Mask` (`Mask1`)

### 4.2 Fixed `step=200` Matched-Step Confound Check (Zero Training, ~25 min total)
Using saved `adapter_state_step0200.pt` weights for `u2_corridor`, `v4_mlpplus`, and `v4_randmask` (for `seed=42` from `$S5_14B/ckpts_14b/` and for `seed=1024` from `$WORK/ckpts_14b/`; skip re-running any arm whose selected checkpoint was already `step0200`, e.g., `s42_v4_mlpplus`!):
```bash
c35_eval.py --model $BASE14 --base-model $BASE14 --model-key qwen3_14b --components-file $COMP5 \
  --arch <ARCH> --adapter-state <STEP200_PT> --alias <TAG>_step200 --regime no_think --batch-size 4 \
  --datasets c3_blind_confirm_v3 --kl-corpus $KL_CORPUS --out-dir $RES6/fixed_step200/<TAG>_step200/no_think
```

---

## 5. Wave 6B — Priorities 2, 3, and 4 (All 8 GPUs Parallel, ~2.0–2.5h)

### 5.1 Priority 2 (Zero Training): Entity-Count-Balanced Diagnostic (`c3_entity_balanced_v1`, 480 Units)
Evaluate BOTH **8B (`M0`, `u2@42`, `v4@42`, `RandMask1@42`)** and **14B (`M0`, `u2@42`, `v4@42`, `RandMask@42`)**:
```bash
c35_eval.py --model <BASE> --model-key <MK> --components-file $COMP5 \
  [--arch <ARCH> --adapter-state <PT>] --alias <ALIAS> --regime no_think --batch-size <BS> \
  --datasets c3_entity_balanced_v1 --out-dir $RES6/entity_balanced/<ALIAS>/no_think
```
Outputs 6-cell breakdown (`1v2_orig`, `2v1_anti`, `1v1_same`, `2v2_same`, `1v1_ctx`, `2v2_ctx`) + exact `Δ_shortcut = Acc(1v2_orig) - Acc(2v1_anti)`.

### 5.2 Priority 3 (8B, GPUs 0–3 Parallel): `Pinpoint-style DLA` (2 Variants) + Uniform Loss Ablations (2 Arms)
All 4 runs use `Qwen3-8B`, `seed=42`, exact 6,000 examples, `grad_accum=20`, `microbatch=2`, `max_steps=300`, `lr=2.5e-5`, `save_steps=120,200,250,300`, `c3_dev` selection, PEFT export, vLLM parity, `v2 + v3 + entity_balanced_v1 + KL`, `think v2/v3`, `capabilities`, and `EX-FEVER (no_think)`:
| GPU | Tag | Arch | Loss Mode & Exact Flags | Scientific Question |
|---:|---|---|---|---|
| **GPU 0** | `s6_8b_s42_pinpoint_dla_corridor` | `Pinpoint-DLA-HeadsOnly` | `--components-file $COMP6 --loss-mode corridor --lr-z1-mult 1.0 --lambda-prox-z1 0.0` | **Primary external mechanistic baseline**: Pinpoint-style DLA (Top-64 heads, 0 MLPs, `7.03M` params) gifted our full Corridor recipe! |
| **GPU 1** | `s6_8b_s42_pinpoint_dla_sft` | `Pinpoint-DLA-HeadsOnly` | `--components-file $COMP6 --loss-mode base --lambda-margin 0.0 --lr-z1-mult 1.0 --lambda-prox-z1 0.0` | Authentic pure-SFT `Pinpoint-style DLA` (no margin hinge, no cap, no tone-inv). |
| **GPU 2** | `s6_8b_s42_u2_corridor_only` | `Uniform-LoRA` | `--components-file $COMP6 --loss-mode corridor_only --lambda-inv 0.0 --lr-z1-mult 1.0 --lambda-prox-z1 0.0` | Isolates bilateral corridor `[4.5, 10.0]` without tone invariance (`u2_corridor_only` vs `u1_base`). |
| **GPU 3** | `s6_8b_s42_u2_tone_inv_only` | `Uniform-LoRA` | `--components-file $COMP6 --loss-mode tone_inv_only --lambda-inv 0.15 --lr-z1-mult 1.0 --lambda-prox-z1 0.0` | Isolates tone invariance **strictly preserving `u1_base`'s original `margin >= 4.0` hinge**! |

### 5.3 Priority 4 (8B, GPUs 4–7 Parallel): `Joint DPO` (`beta` Grid) + `CAA Steering`
- **GPUs 4, 5, 6 — CEDAR Same-Data Joint DPO (`Uniform-LoRA`, `seed=42`, $\beta \in \{0.05, 0.1, 0.2\}$)**:
  1. Train 3 DPO runs in parallel (`beta = 0.05` on GPU 4, `0.1` on GPU 5, `0.2` on GPU 6) via `c35_train_dpo.py --model $BASE8 --model-key qwen3_8b --components-file $COMP6 --arch Uniform-LoRA --beta <BETA> --lr 2.5e-5 --grad-accum 20 --microbatch 2 --max-steps 300 --save-steps 120,200,250,300 --seed 42 --out-dir $WORK/ckpts_8b/s6_8b_s42_dpo_b<BETA>`.
  2. Run `c35_select_dev_ckpt.py` on all 3 betas (`3 betas × 4 ckpts = 12` dev evals transparently logged!), pick the single dev-best `(beta*, ckpt*)`, and run full eval (`v2 + v3 + entity_balanced_v1 + KL`, `think v2/v3`, `capabilities`, `EX-FEVER no_think`) ONLY on the dev winner -> `$RES6/external_8b/s6_8b_s42_joint_dpo_best/`.
- **GPU 7 — `CAA Steering` (`Qwen3-8B`, Rimsky et al. ACL 2024)**:
  ```bash
  CUDA_VISIBLE_DEVICES=7 $PY experiments/ACL-exp-c3.5/scripts/c35_eval_caa.py \
    --model $BASE8 --kl-corpus $KL_CORPUS \
    --layers 14,18,20,22,24,26,28,30 --alphas -2.0,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,3.0,4.0 \
    --max-train-pairs 150 --batch-size 8 --exfever-batch-size 16 \
    --datasets c3_blind_confirm_v2,c3_blind_confirm_v3,c3_entity_balanced_v1 \
    --out-dir $RES6/external_8b/s6_8b_caa_steering
  ```
  Reports: no-intervention point (`alpha=0.0`), full `c3_dev` grid & winner, `v2 / v3 / entity_balanced_v1`, WikiText-2 KL under `+alpha*` and `-alpha*`, AND official `EX-FEVER` (`no_think`) for `alpha=0.0`, `+alpha*`, and `-alpha*`!

---

## 6. Final Stage 6 Aggregation (Strict Fail-Closed Validator)

Run `c35_stage6_stats.py --config-json $RES6/stage6_config.json --out-dir $RES6` -> generates `$RES6/C35_STAGE6_READOUT.md` & `$RES6/c35_stage6_stats.json` (with 900 + 802 sensitivity rows, `pooled_2seed` rows, `< 0.0002` p-value formatting, fixed `step 200` table, 6-cell entity-balanced table, and 8B external SOTA + loss-ablation table) and pushes to `origin/ACL-exp-c3.5`. **STOP — DO NOT UNSEAL `c1_test_id / c1_test_ood`!**
