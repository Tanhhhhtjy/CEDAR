# ACL-exp-c1 · Step 1 执行计划（本地草案，待 Astra 审计通过后再推送）

- **目标**：在 `Qwen3-8B` 上用一次 **4 卡并行单因子归因短跑（~40 分钟）**，精准拆解 Phase 0 中 `IFEval` 掉点（`-17.56 pp`）与 `Uniform-LoRA` 6-token 坍缩的真实归因，并锁定满足通用能力硬门槛的健康配方。
- **前提**：先等远端 Step 0（`eae6b33`）回传原始生成文本与 WikiText-2 KL 读数；本草案与 `c1_train.py` / `c1_eval.py` 补丁先经 Astra 审计，确认无误后再推给实验机执行。

---

## 一、本轮代码补丁摘要（待审计）

1. **`scripts/c1_eval.py`**：
   - 修复第 472 行 `--ablate-hubs` 访问 `m_info["head_sets"]["hub_heads"]` 的 `KeyError`，改为 `resolve_hub_heads(comp, model_key)` 读取真实字段 `m_info["hub_heads"]`；CPU 自检已覆盖 `qwen3_8b` 与 `qwen3_14b` 真实冻结文件及 `adapter_state` 重载与 `merged_hf` logits 一致性（`< 1e-4`）。
   - 新增 `--regime no_think_free`（不预填 `Final:`，默认 `max_new_tokens=64`，支持 HF 与 vLLM 后端），用于抽检自由生成 `no_think` 格式完整性（防止 GPU 3 遮蔽格式 CE 后丢失 `Final: X` 输出契约）。
2. **`scripts/c1_train.py`**：
   - 新增 `--max-steps`：在保留完整 `--epochs 3`（885 步 cosine 调度、53 步 warmup）时域的前提下，在指定步数（如 `295`）精确截停，确保 GPU 0 前 10/40/100 步与 Phase 0 旧轨迹 LR 与 loss 严格一致。
   - 新增 `--save-steps`（如 `25,50,100,150,295`）：在不干扰训练 RNG 的前提下保存轻量化中间 `adapter_state_stepNNNN.pt`。
   - 新增 `--zero-replay-loss`：保留全部 529 条回放的窗口位置、前向计算（对齐 dropout RNG）、窗口分母 `win_n` 与总步数，仅将回放样本 CE loss 乘以 `0.0`（纯回放窗产生显式 `0.0` 梯度并正常推进 AdamW 与 LR 调度）。
   - 新增 `--mask-rule-format-ce`：对**全部 3,000 条非回放规则样本**（同时覆盖 `2,709` 条 `margin_sign != 0` 与 `291` 条 `margin_sign == 0`！）独立对齐选项字母位置，将 `Final`/`:`/`<|im_end|>` 设为 `-100`，并保持原有效 token 分母 `ce_denom`（即精确 `CE(letter) / 4.0`，不放大 4 倍）；回放样本完全不受影响。
   - 日志新增逐窗口 `rule_sft_loss`、`replay_sft_loss`、`hinge_active_rate` 与 `grad_norm`。

---

## 二、Step 1 四卡单因子短跑设计（`Qwen3-8B`，`Targeted-Dual`，`seed=42`）

四张卡使用**完全一致的数据顺序、完整 885 步调度（`--epochs 3` + 53 步 warmup）+ `--max-steps 295` 截停**，并在 **`25,50,100,150,295`** 步保存中间 `adapter_state`：

| GPU | 运行代号 | `--lr` | 回放处理 | 规则格式 CE | 核心隔离变量 |
|---|---|---|---|---|---|
| **GPU 0** | `S1_ctrl_15e5` | `1.5e-4` | 保留 `no_robots` | 标准 CE | 复现 Phase 0 第 1 轮旧轨迹（核对 step 10/40/100 LR 与 loss 逐位对齐，追踪 IFEval 何时开始受损） |
| **GPU 1** | `S1_lr_3e5` | `3.0e-5` | 保留 `no_robots` | 标准 CE | 仅降学习率（隔离 **LR / 步数** 效应） |
| **GPU 2** | `S1_zero_rep` | `3.0e-5` | **`--zero-replay-loss`** | 标准 CE | 同窗口/同前向/同分母下零化回放梯度（隔离 **`no_robots` 净效应**） |
| **GPU 3** | `S1_mask_fmt` | `3.0e-5` | 保留 `no_robots` | **`--mask-rule-format-ce`** | 遮掉占 63.8% 名义权重的 `Final/:/EOS`（仅监督 `CE(letter)/4` + margin，隔离**短答格式监督效应**） |

### 参考启动命令模板（4 卡并行）
```bash
COMMON="--model <BASE_QWEN3_8B> --model-key qwen3_8b --arm Targeted-Dual \
  --replay-file experiments/ACL-exp-c1/data/c1_replay_norobots_529.jsonl \
  --epochs 3 --max-steps 295 --save-steps 25,50,100,150,295 \
  --microbatch 2 --grad-accum 12 --seed 42"

CUDA_VISIBLE_DEVICES=0 python experiments/ACL-exp-c1/scripts/c1_train.py $COMMON --lr 1.5e-4 --out-dir ckpts/step1/S1_ctrl_15e5 &
CUDA_VISIBLE_DEVICES=1 python experiments/ACL-exp-c1/scripts/c1_train.py $COMMON --lr 3.0e-5 --out-dir ckpts/step1/S1_lr_3e5 &
CUDA_VISIBLE_DEVICES=2 python experiments/ACL-exp-c1/scripts/c1_train.py $COMMON --lr 3.0e-5 --zero-replay-loss --out-dir ckpts/step1/S1_zero_rep &
CUDA_VISIBLE_DEVICES=3 python experiments/ACL-exp-c1/scripts/c1_train.py $COMMON --lr 3.0e-5 --mask-rule-format-ce --out-dir ckpts/step1/S1_mask_fmt &
wait
```

---

## 三、中间 Checkpoint 筛选与量化分支裁决

1. **多 Checkpoint 轻量筛选**（严禁触碰 `c1_test_id / c1_test_ood`！）：
   - 在 25/50/100/150/295 步的 `adapter_state` 上测：
     - `c1_dev`（`no_think` 预填读出 + 小型 **`no_think_free`** 自由生成格式检查 + `think`）
     - `WikiText-2` 参考 KL（`--datasets` 显式传空）
     - 独立指令保持小集（筛选最优步数）；入选点再跑完整 **541 题 IFEval** 与 **GSM8K** 过门禁。
2. **预注册量化触发条件**：
   - **回放分支（GPU 2 vs GPU 1）**：若 `S1_zero_rep` 相对 `S1_lr_3e5` 满足 **IFEval 领先 ≥ +3 pp** 或 **WikiText-2 KL 降低 ≥ 50%** → 立即在同一批 529 条 `no_robots` prompt 上生成 **`M0` 自蒸馏回放**，同轮数对比 `人类 no_robots` vs `M0 自蒸馏` vs `零回放`。
   - **格式 CE 分支（GPU 3 vs GPU 1）**：若 `S1_mask_fmt` 相对 `S1_lr_3e5` 显著改善（如 IFEval 挽回至 **≥ 428/541** 且 `no_think_free` + `think` 解析失败增幅 **≤ 2%**）→
     1. 采纳 `--mask-rule-format-ce`；
     2. **Uniform 6-token 闭环验证（2 卡短跑）**：对 `Uniform-LoRA @ lr=1.5e-4` 跑一对（标准 CE vs `--mask-rule-format-ce`，同数据顺序与步数至标准组复现 6-token 故障），实证遮格式 CE 能否直接根治 Uniform 的 6-token 坍缩。
   - **组合与保底**：若 GPU 2 与 GPU 3 各自有效，组合采用前必须实跑一次组合验证；若四组均未达 `IFEval ≥ 430/541`（`-2 pp`），则在独立聊天前缀上补轻量显式 KL 正则后再测（如怀疑早期 hinge 过强，单补 `λ=0` 对照）。

---

## 四、Step 2 预注册衔接备忘（仅在 Step 1 锁定健康配方后启动）

1. **代理 OOD 确认集**：所有 Step 2 模型从干净 `M0` 重启。整类剔除 `cross_entity_transfer`（剩 2,400 规则样例 + 冻结 424 条 15% 回放），另建**独立新模板族**代理 OOD 确认集——先用 40–50 道 M0 合格题筛 `{3e-5, 5e-5}` 与早停步数，再对入选模型扩测 **200–250 道便宜的 `no_think` 合格基础题**做配对显著性确认（避免 8–50 题高方差/低功效）。
2. **放行 14B 门槛**：满足全部健康门禁（8B `IFEval ≥ 430/541`、`GSM8K` 冻结主口径降 ≤ 2 pp、`think` 失败增 ≤ 2 pp、`V_hes / V_auth ≥ 90%`、Turn-1 保持），且对两个独立随机掩码（`Random-LM`, `Random-LM-seed2`，与训练种子交叉）点估计均 `≥ +10 pp`、配对置信区间下界 `> 0`、第二训练种子同向。
3. **预注册紧预算档（`~1.77M`，若 `6.87M` 仍打平）**：保持全部 47 头 + 5 层 MLP 位置不变，仅支持逐层 `head_rank` 降秩（30 头 `@r4` + 17 头 `@r3` + 5 MLP `@r2` = **`1,772,288`** vs `Uniform-LoRA r=1 = 1,769,472`，仅差 `0.16%`；两随机掩码逐层沿用同秩），并如实报告 `6.87M` 的平局。若两档预算均打平，则理性止损，不开 14B 全矩阵。
