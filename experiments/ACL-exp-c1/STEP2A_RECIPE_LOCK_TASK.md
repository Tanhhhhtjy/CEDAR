# ACL-exp-c1 · Step 2A 任务单（配方锁门禁）：3 卡寻找 100% 全绿健康配方 + M0 新代理集基线

- **状态**：**READY FOR EXECUTION**（监控拉取后立刻执行本任务单；**严禁提前执行 `STEP2_CAUSAL_SHOWDOWN_PLAN.md`（Step 2B）**，Step 2A 推完 `STEP2A_READOUT.md` 后暂停等待三方对齐）
- **核心目标**：解决 Step 1 遗留的最后两个问题，在投入 8 卡跑双随机对照（Step 2B）前，用低成本（4 卡 ~60–75 分钟）锁定一套**同时通过全部健康门禁**的 `Qwen3-8B` 配方：
  1. 解决 `zero_rep_s295` 在 `no_think` 下普通 `V`（`68/80`）与 `V_hes/V_auth`（`71/68`）略低于 90% 的问题：对比 **原权重 `@200` 中间点** vs **新增 `--accept-weight 1.7`（精确平衡 1,000 接受 vs 1,709 拒绝样本）在 `@200` / `@295`**；
  2. 验证 Step 0 发现的 `Uniform-LoRA` 6-token 坍缩在新配方（`lr=3e-5` + 零回放）下是否已彻底根治；
  3. 同步冻结基座 `M0` 在两份独立代理集（`c1_proxy_ood` 240 题换域集 + `c1_proxy_template_ood` 120 题 0% 模板骨架重叠集）上的双侧 Turn-1 合格分母 `eligible_ids.json`。

---

## ⚠️ 严格安全护栏（必须遵守）
1. **严禁触碰封存测试集**：所有 `c1_eval.py` 命令**必须显式传 `--datasets c1_dev c1_proxy_ood c1_proxy_template_ood`**（或子集）；绝对禁止省略 `--datasets` 触发 `c1_test_id / c1_test_ood`！
2. **严禁使用 `--backend hf` 跑 `think`**：全部 `think / IFEval / GSM8K` 评测**必须**先通过 `scripts/c1_merge_and_check.py` 合并验收，再用 **`--backend vllm`** 执行（单组总耗时约 25–30 分钟，绝对禁止 10 小时/次的 HF 逐条生成！）。
3. **严禁修改旧结果目录**：Step 2A 全部新产物统一写入 **`experiments/ACL-exp-c1/results/step2a_recipe_lock/`**；严禁触碰 `phase0*` 目录。
4. **空闲卡检查 & 启动前 CPU 自检**：启动前确认目标 GPU 无外部常驻进程，并先跑 4 项 CPU 自检（~15 秒），任一失败立即中止：
   ```bash
   python experiments/ACL-exp-c1/scripts/c1_eval.py --selftest-tiny
   python experiments/ACL-exp-c1/scripts/c1_train.py --selftest-tiny
   python experiments/ACL-exp-c1/scripts/c1_merge_and_check.py --selftest-tiny
   python experiments/ACL-exp-c1/scripts/c1_step2_stats.py --selftest
   ```

---

## 一、Wave 2A.1（4 卡并行，约 35–40 分钟）：3 组训练 + GPU 3 跑 M0 新代理集基线

所有训练严格使用 `Qwen3-8B`（`/mnt/kcl/devuser/ckpts/Qwen3-8B`），统一超参：`--lr 3e-5 --epochs 3 --max-steps 295 --save-steps 200,295 --replay-file experiments/ACL-exp-c1/data/c1_replay_norobots_529.jsonl --zero-replay-loss --microbatch 2 --grad-accum 12 --seed 42 --no-merge-save`。

| GPU | 代号 | Arm (`c1_components_frozen.json`) | `--accept-weight` | 预注册主评测检查点 | 任务角色 |
|---|---|---|---:|---|---|
| **GPU 0** | `s2a_dual_w10` | `Targeted-Dual` (`6.87M`) | `1.0` | **`step 200`** | 验证 150–295 正中间步数能否同时保住 `V ≥ 90%` 与 `GSM8K ≥ 470` |
| **GPU 1** | `s2a_dual_w17` | `Targeted-Dual` (`6.87M`) | **`1.7`** | **`step 200` & `step 295`** | 精确平衡 `1,000` 接受 vs `1,709` 拒绝样本梯度，根治 `no_think` 过度保守 |
| **GPU 2** | `s2a_unif_w10` | `Uniform-LoRA` (`7.08M`, `r=4`) | `1.0` | **`step 295`** | 复验新配方下 `Uniform-LoRA` Step 0 的 6-token `think` 故障是否根治 |
| **GPU 3** | `M0` | 基座（零训练） | — | — | 跑 `M0` 在 `c1_dev` + `c1_proxy_ood` + `c1_proxy_template_ood` 的 `no_think` (HF) 与 `think` (vLLM)，导出冻结 `eligible_ids.json` |

**训练命令模板（GPUs 0–2）**：
```bash
CUDA_VISIBLE_DEVICES=<GPU> python experiments/ACL-exp-c1/scripts/c1_train.py \
  --model /mnt/kcl/devuser/ckpts/Qwen3-8B --model-key qwen3_8b \
  --components-file experiments/ACL-exp-c1/data/c1_components_frozen.json \
  --arm <ARM> --accept-weight <WEIGHT> \
  --train-file experiments/ACL-exp-c1/data/c1_train_examples_nothink.jsonl \
  --replay-file experiments/ACL-exp-c1/data/c1_replay_norobots_529.jsonl \
  --zero-replay-loss --lr 3e-5 --epochs 3 --max-steps 295 --save-steps 200,295 \
  --microbatch 2 --grad-accum 12 --seed 42 --no-merge-save \
  --out-dir /mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c1/step2a/ckpts/<TAG>
```

**GPU 3 M0 基线评测命令**：
```bash
CUDA_VISIBLE_DEVICES=3 python experiments/ACL-exp-c1/scripts/c1_eval.py \
  --model /mnt/kcl/devuser/ckpts/Qwen3-8B --alias qwen3_8b__M0 \
  --regime no_think --backend hf --datasets c1_dev c1_proxy_ood c1_proxy_template_ood \
  --eligible-ids-out experiments/ACL-exp-c1/results/step2a_recipe_lock/M0/no_think/eligible_ids.json \
  --out-dir experiments/ACL-exp-c1/results/step2a_recipe_lock/M0/no_think

CUDA_VISIBLE_DEVICES=3 python experiments/ACL-exp-c1/scripts/c1_eval.py \
  --model /mnt/kcl/devuser/ckpts/Qwen3-8B --alias qwen3_8b__M0 \
  --regime think --backend vllm --max-new-tokens 3072 --datasets c1_dev c1_proxy_ood c1_proxy_template_ood \
  --eligible-ids-out experiments/ACL-exp-c1/results/step2a_recipe_lock/M0/think/eligible_ids.json \
  --out-dir experiments/ACL-exp-c1/results/step2a_recipe_lock/M0/think
```

---

## 二、Wave 2A.15（快速合并门禁检查，4 卡并行 ~2 分钟）

Wave 2A.1 完成后，先并行对全部 **4 个待测检查点**执行标准化合并校验（不通过则立刻停止报错，绝不浪费后续评测时间）：
- `s2a_dual_w10_s200`（`s2a_dual_w10 / adapter_state_step0200.pt`）
- `s2a_dual_w17_s200`（`s2a_dual_w17 / adapter_state_step0200.pt`）
- `s2a_dual_w17_s295`（`s2a_dual_w17 / adapter_state_step0295.pt`）
- `s2a_unif_w10_s295`（`s2a_unif_w10 / adapter_state_step0295.pt`）

```bash
CUDA_VISIBLE_DEVICES=<GPU> python experiments/ACL-exp-c1/scripts/c1_merge_and_check.py \
  --base-model /mnt/kcl/devuser/ckpts/Qwen3-8B --model-key qwen3_8b --arm <ARM> \
  --components-file experiments/ACL-exp-c1/data/c1_components_frozen.json \
  --adapter-state /mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c1/step2a/ckpts/<RUN>/adapter_state_step<STEP>.pt \
  --out-merged-dir /mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c1/step2a/merged/<CKPT_TAG> \
  --out-check-json experiments/ACL-exp-c1/results/step2a_recipe_lock/<CKPT_TAG>/merge_check.json \
  --tag <CKPT_TAG>
```

---

## 三、Wave 2A.2（4 卡并行，约 40–45 分钟）：4 个检查点全量评测（全走合并 + vLLM）

每张 GPU 负责 1 个 `<CKPT_TAG>`，依次运行以下 3 项（全部基于合并权重，**严禁 `--backend hf` 跑 `think`**）：

1. **`no_think` + `WikiText-2 KL`（~8 分钟）**：
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU> python experiments/ACL-exp-c1/scripts/c1_eval.py \
     --model /mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c1/step2a/merged/<CKPT_TAG> \
     --base-model /mnt/kcl/devuser/ckpts/Qwen3-8B \
     --kl-corpus experiments/ACL-exp-c1/data/c1_kl_wikitext2_200.jsonl \
     --alias qwen3_8b__<CKPT_TAG> --regime no_think --backend hf \
     --datasets c1_dev c1_proxy_ood c1_proxy_template_ood \
     --eligible-ids-in experiments/ACL-exp-c1/results/step2a_recipe_lock/M0/no_think/eligible_ids.json \
     --out-dir experiments/ACL-exp-c1/results/step2a_recipe_lock/<CKPT_TAG>/no_think
   ```
2. **`vLLM think`（~20 分钟）**：
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU> python experiments/ACL-exp-c1/scripts/c1_eval.py \
     --model /mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c1/step2a/merged/<CKPT_TAG> \
     --alias qwen3_8b__<CKPT_TAG> --regime think --backend vllm --max-new-tokens 3072 \
     --datasets c1_dev c1_proxy_ood c1_proxy_template_ood \
     --eligible-ids-in experiments/ACL-exp-c1/results/step2a_recipe_lock/M0/think/eligible_ids.json \
     --out-dir experiments/ACL-exp-c1/results/step2a_recipe_lock/<CKPT_TAG>/think
   ```
3. **`IFEval`（541 题）+ `GSM8K`（1,319 题）（~18 分钟）**：严格复用 `results/phase0/capabilities/spec.json` 的 vLLM 评测口径，保存至 `results/step2a_recipe_lock/<CKPT_TAG>/capabilities/{summary.json,items.jsonl}`。
   - **主分**严格沿用冻结 v1（闭合思考块外独立 `#### <num>` 行，与 M0 `496/1,319` 对齐）；可同步另列 `gsm8k_relaxed_boxed`（识别 `\boxed{}`）作为辅助诊断列（不覆盖 v1 主分）。

---

## 四、验收汇总与推送（随后暂停等待三方对齐）

生成 **`experiments/ACL-exp-c1/results/step2a_recipe_lock/STEP2A_READOUT.md`**，对照以下全部健康门禁（含 Step 1 参照 `M0` 与 `zero_rep_s295`）给出完整通过/未通过表，并将 `experiments/ACL-exp-c1/results/step2a_recipe_lock/`（不含大型 `merged` 权重）提交推送至 `origin/ACL-exp-c1`：

- `merge_check.passed == True`（v2 四项联合门槛 + `eos_token_id=[151645, 151643]`）
- `IFEval prompt_strict ≥ 430/541`（同步汇报 `length` 撞限题数）
- `GSM8K v1 ≥ 470/1,319`（同步汇报 `n_unparsed` 与已解析正确率）
- `WikiText-2 KL ≤ 0.005`
- **`c1_dev` 与 `c1_proxy_ood` 与 `c1_proxy_template_ood`（`no_think` & `think` 双制度）**：
  - `V / V_hes / V_auth ≥ 90%`（普通真证据与双口吻真证据均达标）
  - `V/F 双对率`、`F` 拒绝率、`shell_only` 拒绝率、Turn-1 双侧 `≥ 95%`、`think` 解析失败率 `≤ 2.0%`
- **推送完成后保持 GPU 空闲，等待三方对齐结果后再下发 Step 2B（8 卡因果对决）任务单。**
