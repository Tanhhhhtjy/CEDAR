# ACL-exp-c1 · Step 1 执行计划

- 在 `Qwen3-8B Targeted-Dual` 上做四卡并行单因子短跑，检验学习率、回放监督和规则格式监督能解释多少 IFEval 损失。四卡训练预计约 40 分钟，实际用时和评测用时分别记录。Uniform 的六 token 故障由同臂配对实验检验。
- Step 0 已归档。样例 1000 的重复生成和 DEV-001 上 Uniform 的六 token 提前终止已复现，1001 的长度表现有批次差异。四个 checkpoint 均有 WikiText-2 KL 基线。[Step 0 原文与读数](results/phase0_step0/STEP0_READOUT.md)

---

## 一、本轮代码补丁摘要

1. **`scripts/c1_eval.py`**：
   - 修复第 472 行 `--ablate-hubs` 访问 `m_info["head_sets"]["hub_heads"]` 的 `KeyError`，改为 `resolve_hub_heads(comp, model_key)` 读取真实字段 `m_info["hub_heads"]`；CPU 自检已覆盖 `qwen3_8b` 与 `qwen3_14b` 真实冻结文件及 `adapter_state` 重载与 `merged_hf` logits 一致性（`< 1e-4`）。
   - `--regime no_think_free` 默认读取 `c1_dev`，不预填 `Final:`，生成上限为 64。HF 与 vLLM 都要求正常停止并且全文恰好为一行 `Final: X`（可带末尾换行）。额外解释、多答案、思考标签和因长度上限截断的输出计入格式失败。在最后一个允许的位置生成 EOS 算正常停止。
2. **`scripts/c1_train.py`**：
   - `--max-steps 295` 保留 `--epochs 3` 的 885 步 cosine 调度、53 步 warmup。GPU 0 用旧日志核对 10/40/100 步的 LR 和原记录精度下的 loss，出现偏差时记录软件版本、GPU 和实际偏差。
   - `--save-steps`（如 `25,50,100,150,295`）保存轻量化中间 `adapter_state_stepNNNN.pt`。保留同目录的 `adapter_config.json`，供 HF 恢复训练时的 LoRA 缩放、臂和掩码；旧运行可读取已有的 `train_meta.json`。
   - 新增 `--zero-replay-loss`：保留全部 529 条回放的窗口位置、前向计算（对齐 dropout RNG）、窗口分母 `win_n` 与总步数，仅将回放样本 CE loss 乘以 `0.0`（纯回放窗产生显式 `0.0` 梯度并正常推进 AdamW 与 LR 调度）。
   - 新增 `--mask-rule-format-ce`：对**全部 3,000 条非回放规则样本**（同时覆盖 `2,709` 条 `margin_sign != 0` 与 `291` 条 `margin_sign == 0`！）独立对齐选项字母位置，将 `Final`/`:`/`<|im_end|>` 设为 `-100`，并保持原有效 token 分母 `ce_denom`（即精确 `CE(letter) / 4.0`，不放大 4 倍）；回放样本完全不受影响。
   - 每 10 步、存盘步和末步记录规则与回放的 CE、未置零的回放 CE、hinge 激活率和裁剪前总梯度范数。四臂同开 `--log-source-grad-norms`，记录按实际窗口分母归一化的 `rule_grad_norm`（含 margin）及 `replay_grad_norm`。有两类有效梯度的记录窗口需额外反传。

---

## 二、Step 1 四卡单因子短跑设计（`Qwen3-8B`，`Targeted-Dual`，`seed=42`）

四张卡使用**完全一致的数据顺序、完整 885 步调度（`--epochs 3` + 53 步 warmup）+ `--max-steps 295` 截停**，并在 **`25,50,100,150,295`** 步保存中间 `adapter_state`：

| GPU | 运行代号 | `--lr` | 回放处理 | 规则格式 CE | 核心隔离变量 |
|---|---|---|---|---|---|
| **GPU 0** | `S1_ctrl_15e5` | `1.5e-4` | 保留 `no_robots` | 标准 CE | 与 Phase 0 第 1 轮的 LR、loss 核对，追踪指令保持何时开始受损 |
| **GPU 1** | `S1_lr_3e5` | `3.0e-5` | 保留 `no_robots` | 标准 CE | 仅降学习率（隔离 **LR / 步数** 效应） |
| **GPU 2** | `S1_zero_rep` | `3.0e-5` | **`--zero-replay-loss`** | 标准 CE | 同窗口/同前向/同分母下零化回放梯度（隔离 **`no_robots` 净效应**） |
| **GPU 3** | `S1_mask_fmt` | `3.0e-5` | 保留 `no_robots` | **`--mask-rule-format-ce`** | 遮掉占 63.8% 名义权重的 `Final/:/EOS`（仅监督 `CE(letter)/4` + margin，隔离**短答格式监督效应**） |

### 参考启动命令模板（4 卡并行）
```bash
: "${BASE_QWEN3_8B:?请先设置实验机上的 Qwen3-8B 目录}"
common=(--model "$BASE_QWEN3_8B" --model-key qwen3_8b --arm Targeted-Dual
  --replay-file experiments/ACL-exp-c1/data/c1_replay_norobots_529.jsonl
  --epochs 3 --max-steps 295 --save-steps 25,50,100,150,295
  --microbatch 2 --grad-accum 12 --seed 42 --log-source-grad-norms)

pids=()
launch() {
  local gpu="$1" run="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" python experiments/ACL-exp-c1/scripts/c1_train.py \
    "${common[@]}" --out-dir "ckpts/step1/$run" "$@" &
  pids+=("$!")
}
launch 0 S1_ctrl_15e5 --lr 1.5e-4
launch 1 S1_lr_3e5 --lr 3e-5
launch 2 S1_zero_rep --lr 3e-5 --zero-replay-loss
launch 3 S1_mask_fmt --lr 3e-5 --mask-rule-format-ce
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
```

---

## 三、中间 Checkpoint 筛选与量化分支裁决

1. **多 Checkpoint 轻量筛选**（严禁触碰 `c1_test_id / c1_test_ood`！）：
   - 在 25/50/100/150/295 步的 `adapter_state` 上测：
     - 三个 C1 模式都显式传 `--datasets c1_dev`。使用 `--backend hf --adapter-state <该步文件>` 配合同一 M0、模型键和训练臂读取中间存盘。为每个模型、步数和模式指定独立输出目录。
     - WikiText-2 KL 单独运行，传 `--regime no_think --datasets --base-model <M0> --kl-corpus <冻结语料>`，同时指定待测权重和输出目录。
     - 独立指令保持小集（筛选最优步数）；入选点再跑完整 **541 题 IFEval** 与 **GSM8K** 过门禁。
2. **预注册量化触发条件**：
   - **回放分支（GPU 2 vs GPU 1）**：若 `S1_zero_rep` 相对 `S1_lr_3e5` 满足 **IFEval 领先 ≥ +3 pp** 或 **WikiText-2 KL 降低 ≥ 50%** → 立即在同一批 529 条 `no_robots` prompt 上生成 **`M0` 自蒸馏回放**，同轮数对比 `人类 no_robots` vs `M0 自蒸馏` vs `零回放`。
   - **格式 CE 分支（GPU 3 vs GPU 1）**：比较同一步数，`S1_mask_fmt` 的 IFEval 须优于 `S1_lr_3e5` 且达到 **≥ 430/541**。`no_think_free` 和 `think` 解析失败率各自相对同模式 M0 增幅须 **≤ 2 pp**，分母和题单固定。
     1. 满足上述条件后将 `--mask-rule-format-ce` 列入候选；最终配方还须满足全部健康门槛。
     2. **Uniform 6-token 闭环验证（2 卡短跑）**：对 `Uniform-LoRA @ lr=1.5e-4` 跑一对（标准 CE vs `--mask-rule-format-ce`，同数据顺序与步数至标准组复现 6-token 故障），实证遮格式 CE 能否直接根治 Uniform 的 6-token 坍缩。
   - **组合与保底**：若 GPU 2 与 GPU 3 各自有效，组合采用前必须实跑一次组合验证；若四组均未达 `IFEval ≥ 430/541`（`-2 pp`），则在独立聊天前缀上补轻量显式 KL 正则后再测（如怀疑早期 hinge 过强，单补 `λ=0` 对照）。

---

## 四、Step 2 预注册衔接备忘（仅在 Step 1 锁定健康配方后启动）

1. **代理 OOD 确认集**：所有 Step 2 模型从干净 `M0` 重启。整类剔除 `cross_entity_transfer`（剩 2,400 规则样例 + 冻结 424 条回放，约 15%）。各臂在同一批 40–50 道 M0 双侧合格基础题上筛 `{3e-5, 5e-5}` 与早停步数，再用另行冻结的 200–250 道合格基础题做配对确认。确认集与筛选集的基础题互斥；确认模板与训练集归一化 Turn-2 重叠为零。查看确认结果前固定入选配置、效应指标和各预算的判定规则。
2. **放行 14B 门槛**：满足全部健康门禁（8B `IFEval ≥ 430/541`、`GSM8K` 冻结主口径降 ≤ 2 pp、`think` 失败增 ≤ 2 pp、`V_hes / V_auth ≥ 90%`、Turn-1 保持），且对两个独立随机掩码（`Random-LM`, `Random-LM-seed2`，与训练种子交叉）点估计均 `≥ +10 pp`、配对置信区间下界 `> 0`、第二训练种子同向。
3. **预注册紧预算档（`~1.77M`，若 `6.87M` 仍打平）**：保持全部 47 头 + 5 层 MLP 位置不变。启用此档前需实现并验证逐层 `head_rank`（30 头 `@r4` + 17 头 `@r3` + 5 MLP `@r2` = **`1,772,288`** vs `Uniform-LoRA r=1 = 1,769,472`，仅差 `0.16%`；两随机掩码逐层沿用同秩），并如实报告 `6.87M` 的平局。若两档预算均打平，则理性止损，不开 14B 全矩阵。
