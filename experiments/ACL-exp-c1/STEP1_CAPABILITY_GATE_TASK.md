# ACL-exp-c1 · Step 1 能力门禁收尾任务单：现成 `@150`（含 `@295` 诊断）权重补测 `think` + `IFEval` + `GSM8K`

- **状态**：READY FOR EXECUTION（与 Astra 对齐确认，监控拉取后即可立刻开跑）
- **基线提交**：`ACL-exp-c1@20ddece`（仅使用 GPU 机上已存在的 Step 1 `Qwen3-8B` `Targeted-Dual` `seed=42` `adapter_state` 权重）
- **预计耗时**：约 **15–25 分钟**（零训练，4 组可 4 卡并行合并 + vLLM 评测）

---

## ⚠️ 严格安全护栏（必须遵守）
1. **零训练、绝不触碰 14B**：本轮仅对 4 个现成 8B `adapter_state` 做合并与推理评测。
2. **严禁触碰封存测试集**：跑 `c1_eval.py` 时**必须显式传 `--datasets c1_dev`**（绝对禁止省略 `--datasets` 触发默认的 `c1_test_id / c1_test_ood`！）。
3. **禁止覆盖任何旧目录**：所有新产物统一写入 **`experiments/ACL-exp-c1/results/phase0_step1_cap/`**。
4. **空闲卡检查**：启动前确认目标 GPU 无外部常驻进程（如 `decision-serve`）占用显存，避免 OOM。

---

## 一、待测 4 个现成权重清单（已锁 SHA-256，防加载错位）

均来自 `Qwen3-8B` `Targeted-Dual`（`seed=42`，`scaling_mode="rslora"`, `base_scale=2.0`, `anchor_rank=8`）：

| 代号 | 来源运行与步数 | 必须校验的 `adapter_state` SHA-256 | 角色 |
|---|---|---|---|
| **`ctrl_15e5_s150`** | `S1_ctrl_15e5` / `adapter_state_step0150.pt` | `8871e9c4d5300d5a097ce259116b6b326edb0404108ad2602364385aceb21b0c` | 主候选 #1（早停：`DC=78/80`, `V_hes=98.8%`, `KL=0.01298`） |
| **`zero_rep_s150`** | `S1_zero_rep` / `adapter_state_step0150.pt` | `53bfce5f43b7caa5d46e29f802dbe604c44ec8ac6ea9ce52e6337395d9483a08` | 主候选 #2（低 LR 零回放：`V_hes=95.0%`, `V_auth=98.8%`, `KL=0.00095`） |
| **`lr_3e5_s150`** | `S1_lr_3e5` / `adapter_state_step0150.pt` | `bd6aab2102730062e6d03e0f4332d4f8966fddd04e8c199cd01eac6b04cbc662` | 同一步共同对照（低 LR 人类回放：`V_hes=91.3%`, `V_auth=95.0%`, `KL=0.00123`） |
| **`zero_rep_s295`** | `S1_zero_rep` / `adapter_state_step0295.pt` | `9e1979532b6d0e3b9eb67ed14660f0aedcd5f781ab4bf82aae1f5e0db24ee79a` | 补充诊断点（验证 `KL=0.00099` 走满 1 轮时 IFEval 是否仍满血） |

### 合并与合并前一致性自检（每组约 1 分钟）
1. 校验 `adapter_state_stepNNNN.pt` 的 SHA-256 与上表完全一致。
2. 从干净 `M0`（`/mnt/kcl/devuser/ckpts/Qwen3-8B`，`bfloat16`）加载：
   `c1_modules.apply_c1_adapters(model, arm_spec, scaling_mode="rslora", base_scale=2.0, anchor_rank=8)` → `load_c1_adapter_state` → `merge_c1_adapters` → `save_pretrained(<tmp_merged_dir>, safe_serialization=True)` + `tok.save_pretrained(<tmp_merged_dir>)`。
3. 合并前在 1 条 prompt 上核对 unmerged vs merged 的 logits 最大绝对误差在 bf16 容差内（`< 5e-2`），并确认合并目录的 `eos_token_id` 仍为 `[151645, 151643]`；评测完成后临时 `merged_hf` 目录可不入库（保留合并校验 JSON 即可）。

---

## 二、统一评测口径（与 Phase 0 `M0` 100% 一致，4 组全测并全部汇报）

1. **`c1_dev` `think` 模式（1,200 条 = 80 题 × 15 臂）**：
   ```bash
   python experiments/ACL-exp-c1/scripts/c1_eval.py \
     --model <TMP_MERGED_DIR> \
     --alias qwen3_8b__<TAG> \
     --regime think --backend vllm --max-new-tokens 3072 \
     --datasets c1_dev \
     --eligible-ids-in experiments/ACL-exp-c1/results/phase0/M0/qwen3_8b/think/eligible_ids.json \
     --out-dir experiments/ACL-exp-c1/results/phase0_step1_cap/<TAG>/think
   ```
2. **`IFEval`（541 题）**：严格复用 `experiments/ACL-exp-c1/results/phase0/capabilities/spec.json`（`google/IFEval` `966cd89545d6b6acfd7638bc708b98261ca58e84`，`no_think` 即 `enable_thinking=False`，无 system message，vLLM `temperature=0`, `seed=42`, `max_tokens=3072`, `bfloat16`，官方 `test_instruction_following_strict`）。
   - 保存 `items.jsonl` 与 `summary.json` 至 `results/phase0_step1_cap/<TAG>/capabilities/`，统计 `prompt_strict`、`instruction_strict` 及 `finish_reason`（`stop` vs `length` 题数）。
3. **`GSM8K`（1,319 题）**：严格复用 `results/phase0/capabilities/spec.json`（`openai/gsm8k` main test `e53f048856ff4f594e959d75785d2c2d37b678ee`，`enable_thinking=True`，同后缀，vLLM `temperature=0`, `seed=42`, `max_tokens=3072`, `bfloat16`）。
   - 主分严格沿用冻结 v1（闭合思考块外独立 `#### <num>` 行）与 Phase 0 `M0`（`496/1,319 = 37.60%`）对齐；可同步另列 `n_unparsed` 与 `stop/length` 分布（不覆盖 v1 主分）。

---

## 三、验收汇总与推送

生成 **`experiments/ACL-exp-c1/results/phase0_step1_cap/STEP1_CAP_READOUT.md`**，对照以下 8B 预注册健康门槛给出一目了然的通过/未通过表，并将 `experiments/ACL-exp-c1/results/phase0_step1_cap/` 提交并推送至 `origin/ACL-exp-c1`：

- **IFEval 整题严格通过**：`≥ 430/541`（相对 M0 `440/541` 降幅 `≤ 2.0 pp`），同步汇报 `length` 撞限题数（M0 为 `13/541`）
- **GSM8K 冻结 v1**：`≥ 470/1,319`（相对 M0 `496/1,319` 降幅 `≤ 2.0 pp`）
- **`c1_dev think` 解析失败率**：`≤ 2.0%`（`≤ 24/1,200`）且 `think` 双对、双侧 Turn-1 保持
- **`c1_dev` 双口吻真证据保持**（`no_think` & `think`）：`V_hes ≥ 90%`、`V_auth ≥ 90%`
- 汇总表同时并列已测的 `no_think` 双对率与 `WikiText-2 KL`，便于直接拍板是否进入 **Step 2（8B 定位 vs 双随机掩码代理 OOD 对决）**，或补跑 `lr=1.5e-4 + 零回放 @ 150 步` 组合。
