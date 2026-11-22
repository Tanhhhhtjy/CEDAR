# ACL-exp-c1 · Step 2A-diag 零训练同 KL 诊断任务单（~25 分钟，4 卡并行）

- **状态**：**READY FOR EXECUTION**（监控拉取后立刻执行本任务单；**零训练、绝不重训任何权重、绝不执行 Step 2B、绝不触碰封存测试集 `c1_test_id / c1_test_ood`**，推完 `STEP2A_DIAG_READOUT.md` 后保持空闲等待下一步指令）
- **基线提交**：`ACL-exp-c1@07f8a3e`（全部使用 Step 2A 已存盘的 `Qwen3-8B` `adapter_state` 权重）
- **三个明确科学目标**：
  1. **同权重公平比对（`w=1.0 @ 295`）**：补评磁盘上现成的 `s2a_dual_w10/adapter_state_step0295.pt`（与 `s2a_unif_w10 @ 295` 同为 `accept_weight=1.0`、295 步），在 `no_think`（全 3 集 + KL）与 `vLLM think`（全 3 集）上直接一对一配对比较；
  2. **零训练同 KL Pareto 曲线（$W_0 + \gamma \Delta W$）**：使用 `c1_eval.py --adapter-scale <GAMMA>` 对现成 `Dual (w=1.0)`, `Dual (w=1.7)`, `Uniform (w=1.0)` 扫 $\gamma \in \{0.75, 1.0, 1.25, 1.5\}$（仅测轻量 `no_think` + `WikiText-2 KL`，每点 ~5 分钟），在**重合 KL 区间（`~0.0009 – 0.0020`）**内直接判定更新幅度能否解释两者差距；
  3. **Live vs Merged 逐段 KL 复查**：使用 `scripts/c1_diag_merge_kl.py` 对 `s2a_unif_w10 @ 295` 与 `s2a_dual_w17 @ 295` 在 20 条 WikiText 段落 + 40 个 `c1_dev` 决策位置上复核 live adapter vs merged 误差，彻底查清位置 218 的 `8.953` 是否有实质行为影响。

---

## ⚠️ 严格安全护栏（必须遵守）
1. **零训练、绝不触碰 14B**：本轮不跑 `c1_train.py`。
2. **严禁触碰封存测试集**：所有 `c1_eval.py` 必须显式传 `--datasets c1_dev c1_proxy_ood c1_proxy_template_ood`！
3. **严禁使用 `--backend hf` 跑 `think`**：仅 GPU 3 上的 1 组 `think` 必须先合并校验再用 `--backend vllm`！
4. **独立产物目录**：全部新产物统一写入 **`experiments/ACL-exp-c1/results/step2a_diag_kl_pareto/`**，严禁修改 `phase0*` 与 `step2a_recipe_lock/` 内已有文件。
5. **启动前 CPU 自检（~10 秒）**：
   ```bash
   python experiments/ACL-exp-c1/scripts/c1_eval.py --selftest-tiny
   python experiments/ACL-exp-c1/scripts/c1_diag_merge_kl.py --selftest-tiny
   ```

---

## 一、待复用存盘权重 SHA-256 清单（与 `07f8a3e` 一致，启动前校验）

| 简称 | 存盘路径（`/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c1/step2a/ckpts/...`） | Arm | SHA-256 |
|---|---|---|---|
| `dual_w10_s295` | `s2a_dual_w10/adapter_state_step0295.pt` | `Targeted-Dual` | `e201a03005778fc77389f4eac706f4b89ec235e2456cf6619bc639af4cc00214` |
| `dual_w17_s295` | `s2a_dual_w17/adapter_state_step0295.pt` | `Targeted-Dual` | `882ab0fb2264fb91d2ff72021ce36a51d1fb8cab414bf27bc301bccd399854b6` |
| `unif_w10_s295` | `s2a_unif_w10/adapter_state_step0295.pt` | `Uniform-LoRA` | `c5795252db8a76ae6354e7a06b9297b83c3035f1e1cddaa32175cf92ef1b3d30` |
| `unif_w10_s200` | `s2a_unif_w10/adapter_state_step0200.pt` | `Uniform-LoRA` | `0f6cc3d79d32d902e6874326527325e20d1677175c115876b8a98e3a24be19cc` |

共用冻结 M0 合格分母：
- `no_think`: `experiments/ACL-exp-c1/results/step2a_recipe_lock/M0/no_think/eligible_ids.json`
- `think`: `experiments/ACL-exp-c1/results/step2a_recipe_lock/M0/think/eligible_ids.json`

---

## 二、4 卡并行执行清单（总耗时约 20–25 分钟）

### 统一 `no_think` + `WikiText-2 KL` 命令模板（使用 live `--adapter-state` + `--adapter-scale`，无需合并）
```bash
CUDA_VISIBLE_DEVICES=<GPU> python experiments/ACL-exp-c1/scripts/c1_eval.py \
  --model /mnt/kcl/devuser/ckpts/Qwen3-8B --base-model /mnt/kcl/devuser/ckpts/Qwen3-8B \
  --kl-corpus experiments/ACL-exp-c1/data/c1_kl_wikitext2_200.jsonl \
  --model-key qwen3_8b --arm-name <ARM> \
  --components-file experiments/ACL-exp-c1/data/c1_components_frozen.json \
  --adapter-state <ADAPTER_PATH> --adapter-scale <GAMMA> \
  --alias qwen3_8b__<TAG> --regime no_think --backend hf \
  --datasets c1_dev c1_proxy_ood c1_proxy_template_ood \
  --eligible-ids-in experiments/ACL-exp-c1/results/step2a_recipe_lock/M0/no_think/eligible_ids.json \
  --out-dir experiments/ACL-exp-c1/results/step2a_diag_kl_pareto/<TAG>/no_think
```

### 各卡分配（每卡串行 3 个轻量点，4 卡全并行 ~20 分钟完结）
- **GPU 0（`dual_w10_s295` 同权重基准 + 放大扫描，3 点）**：
  1. `dual_w10_s295_g100`（`--adapter-scale 1.0`）
  2. `dual_w10_s295_g125`（`--adapter-scale 1.25`）
  3. `dual_w10_s295_g150`（`--adapter-scale 1.5`）
- **GPU 1（`dual_w17_s295` 缩放扫描，3 点；`γ=1.0` 已有）**：
  1. `dual_w17_s295_g075`（`--adapter-scale 0.75`）
  2. `dual_w17_s295_g125`（`--adapter-scale 1.25`）
  3. `dual_w17_s295_g150`（`--adapter-scale 1.5`）
- **GPU 2（`unif_w10` 降 KL 对齐扫描 + `@200`，3 点；`@295 γ=1.0` 已有）**：
  1. `unif_w10_s295_g075`（`unif_w10_s295`, `--adapter-scale 0.75`）
  2. `unif_w10_s295_g125`（`unif_w10_s295`, `--adapter-scale 1.25`）
  3. `unif_w10_s200_g100`（`unif_w10_s200`, `--adapter-scale 1.0`）
- **GPU 3（Live vs Merged 逐段 KL 复查 + `dual_w10_s295` `vLLM think` 补齐，约 20 分钟）**：
  1. 跑 `c1_diag_merge_kl.py` 两次（~2 分钟）：
     - `unif_w10_s295` → `results/step2a_diag_kl_pareto/merge_kl_diag_unif_w10_s295.json`
     - `dual_w17_s295` → `results/step2a_diag_kl_pareto/merge_kl_diag_dual_w17_s295.json`
  2. 用 `c1_merge_and_check.py` 合并校验 `dual_w10_s295`（保存 `results/step2a_diag_kl_pareto/dual_w10_s295_g100/merge_check.json`），并用 **`--backend vllm`** 跑完 `dual_w10_s295` 的 `think`（`c1_dev c1_proxy_ood c1_proxy_template_ood`），写入 `results/step2a_diag_kl_pareto/dual_w10_s295_g100/think/`（评测完可删临时 `merged` 目录）。

---

## 三、汇总输出与推送

生成 **`experiments/ACL-exp-c1/results/step2a_diag_kl_pareto/STEP2A_DIAG_READOUT.md`**，包含三张清晰表格并推送至 `origin/ACL-exp-c1`：
1. **同 KL Pareto 总表**：按 `WikiText-2 KL` 排序并列所有点（含 `step2a_recipe_lock` 已有的 `dual_w17_s295_g100` 和 `unif_w10_s295_g100`），列出 `KL`、以及三套数据集在 `no_think`（全题 & M0 共享合格）下的 `V/F 双对`、`CF 双对`、`AF 双对`、`V`、`V_hes`、`V_auth`、`u(V)-u(F)`、`u(shell)-u(AO)`。
2. **同权重一对一 `think` 对比表**：`dual_w10_s295` vs `unif_w10_s295` vs `dual_w17_s295`（全 3 集）。
3. **Live vs Merged 逐段 KL 复查表**：汇总 `merge_kl_diag_unif_w10_s295.json` 与 `merge_kl_diag_dual_w17_s295.json` 的 20 段 WikiText 平均/最大 KL 与 40 个 C1 决策位 argmax 一致率及最大 `|Δu|`。

推送后保持 GPU 空闲等待 Step 2B 最终任务单。
