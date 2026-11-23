# ACL-exp-c3.5 Stage 1 · 零重训双探针（分投影 KL + 跨字母因果互换）与修复版新盲集 `v2` 方案

- **状态**：**`STAGE 1 READY TO EXECUTE`（零重训，3 卡并行约 12–15 分钟全部出完；完成后自动锁定 Stage 2 的 `M3-v2` 预算与 5 臂训练矩阵！）**
- **分支**：`ACL-exp-c3.5`（从已永久冻结的 `ACL-exp-c3@96f2859` 切出；`ACL-exp-a1/b1/c1/c2/c3` 一律只读冻结）
- **绝对红线**：**严禁读取或评测密封集 `c1_test_id / c1_test_ood`；复用 C3 训练集 `c3_train_examples_nothink.jsonl` 与 C3 磁盘现成权重 `u1_base / u2_corridor / u3_b1_iit / m2_corridor`，零浪费重训！**

---

## 第一部分 · 从 C3 审计到 C3.5 的完整科学思考过程

> 本节记录 C3（`c9def7c`）完成后，我们与 Astra 只读审计团队逐条核对、纠偏并确定 C3.5 两阶段冲刺方案的完整思考脉络。

### 1.1 C3 的真正价值与被 Astra 精准揪出的 4 处暗雷
C3 8 臂在 8B 上虽然因预注册旧主终点未过 `+3 pp` 而触发 `HOLD` 与旧配方 `14B` 全矩阵止损，但它暴露出了两条极具价值的正向信号：
- 在全量盲集上，**`m2_corridor − m1_base = +3.80 pp [+2.45, +5.21] (p<0.0001)`**；
- 在完全相同的全层 Uniform 容量和固定 5 层 B1 MLP 下，**`u3_b1_iit − u4_rnd_iit = +1.41 pp [+0.42, +2.45] (p=0.0076)`**。

同时，Astra 对 `c9def7c` 的独立只读复核精准指出了必须先修再跑的 4 个核心问题：
1. **盲集存在两个可客观修复的代码缺陷**：
   - 240 道跨实体（CE）题中，由于渲染器匹配 `Question:` 而 CE 问句以 `Based only ...` 开头，120 道 `before_rules` 题全漏插了干扰条款、120 道 `after_rules` 题全保留了干扰条款；
   - 840 道非 CE 复合题的 `s1 ∧ s2` 中 `s2` 原本就依赖 `s1`，拿掉 `s1` 必然失去 `s2`，未能真正考察整合**两条独立证据前提**。
   - **修复方案（`c3_blind_confirm_v2`）**：改为真正独立的双前提 DAG（`obsA → bA` 与 `obsB → bB` 互不蕴含，`bA ∧ bB → trig`），并在生成期对每道题硬断言“**单独拿掉 `obsA` 或单独拿掉 `obsB` 均将标准答案翻转回 `a0`**”；修复 CE 干扰条款插入并硬断言 **`240/240` 全部包含**；并内置 60 组同语义图成对 `before_rules / after_rules` 对照；主终点同时预注册 **全量 900 独立生成单元四格全对** 与 **首轮双对 ∧ 四格全对（联合成功率）**，防止 Turn-1 退化被 teacher-forced 提示掩盖。
2. **Wave 3.0 探针还需拆细到“分投影（Q/O vs MLP）+ 净 KL 变化”才能科学定死 `M3-v2` 预算**：
   - Wave 3.0 关掉了整个区，尚未区分 `Uniform` 底层是 **Q/O** 还是 **MLP** 承载 Row B，也未直接测出单关 `m2_corridor` 顶层 3 个 MLP `[29, 32, 35]` 能净省多少 Wiki KL。
   - **Stage 1 Probe 1（`c35_probe_subprojections.py`，零重训 ~6 分钟）**：直接加载 C3 现成 `u2_corridor`（SHA `cefc89b572ec...`）与 `m2_corridor`（SHA `197c5e137431...`），分别关闭 `Z1:Q/O`、`Z1:MLP`、`Z2:Q/O`、`Z2:MLP`、`Z3:MLP`、`Z3:All`，同步量出 `c3_dev` 四格、`Row B` 和精确 WikiText-2 KL，用实测数据把 `M3-v2`（匹配 Uniform 在 `±3%` 以内）的每一笔预算拍死！
3. **术后留出互换升级为“跨字母映射打乱”挑战（论文级机制证据！）**：
   - 此前源与受体同题同映射，输出源字母还无法彻底排除“复制答案字母”与“传递抽象逻辑真伪判断”；且 60 道探针缺少 `argument_flip`。
   - **Stage 1 Probe 2（`c35_eval_cross_letter_interchange.py`，零重训 ~8 分钟）**：在 `c3_blind_confirm_v2` 抽取覆盖 **`30 context_flip + 30 argument_flip + 20 CE (10A+10B)`** 的 80 道题，对受体做**完全不交的 A/B/C/D 映射置换**（使源的 `{a0, at}` 与受体的 `{a0', at'}` 完全无交集！），并同步做 **7 路组件拆分**（`b1_full`, `b1_mlp_only`, `b1_heads_only`, `b1_heads_rnd_mlp`, `rnd_heads_b1_mlp`, `rnd_full`, `identity`），一次性测清受体是输出**自身反事实真伪选项（抽象逻辑转移）**还是**源字面字母（字母复制）**！
4. **严格统计门禁修漏**：`c35_stats.py` 彻底废除 `None` 静默放过（任何缺失健康项或缺失对照严格判 FAIL），且统一记录 Phase-0 `M0` 准确 GSM8K 参考值为 **`496/1319` 正确（502 可解析）**。

---

## 第二部分 · Stage 1 冻结新盲集与审计结果（`c35_datasets_manifest.json`）

- **`c3_blind_confirm_v2_bases.jsonl`**：`1,080` 题 / **`900` 独立生成单元**（720 独立双前提非 CE + 60 组同语义图成对前后位置非 CE [=120] + 120 组 2×2 CE [=240]）
  - CE 干扰条款覆盖率：**`240/240 (100%)`**（`before_rules=120`, `after_rules=120`）
  - 独立双前提求解器翻转认证：**`840/840 (100%)`**（`depth(bA)=1, depth(bB)=1, depth(trig)=2`，去 `obsA` 或去 `obsB` 均翻转）
  - 文风与位置平衡：`style_c_dossier=540 / style_d_memo=540`，`before_rules=540 / after_rules=540`，因素独立性 χ² 全部 `p > 0.37`
  - 与 `c3_train` 骨架重合：**`0 / 0`**；`Only if` 计数：**`0`**；G0 TF-IDF AUC = **`0.5348`（`cf_byte_match = 1.0`）**
- **`c3_blind_think_v2_bases.jsonl`**：分层抽样 `240` 题（用于 Stage 2 vLLM `think` 非劣性检查）。

---

## 第三部分 · Stage 1 执行指令（3–4 卡并行，零重训，~12–15 分钟！）

> **服务器已核对路径（来自 C3 `dev_selection.json`）**：
> - `PY=/mnt/vllm310/bin/python`
> - `BASE=/mnt/kcl/devuser/ckpts/Qwen3-8B`
> - `C3_SEL=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3/showdown-8438e5e2/selected_42`
>   - `u1_base`：`selected_adapter_state.pt` (SHA `7f46eaae1ec1...`)
>   - `u2_corridor`：`selected_adapter_state.pt` (SHA `cefc89b572ec...`)
>   - `u3_b1_iit`：`selected_adapter_state.pt` (SHA `f5a3b5b9e229...`)
>   - `m2_corridor`：`selected_adapter_state.pt` (SHA `197c5e137431...`)
> - `RES=experiments/ACL-exp-c3.5/results/stage1`
> - `KL_CORPUS=experiments/ACL-exp-c1/data/c1_kl_wikitext2_200.jsonl`

### 并行任务（4 张空闲 GPU 同时跑，~12–15 分钟全部落地！）

1. **GPU 0 — Probe 1：分投影 Q/O vs MLP + 精确 Wiki KL 探针（`u2_corridor` & `m2_corridor`，~6 分钟）**：
   ```bash
   CUDA_VISIBLE_DEVICES=0 $PY experiments/ACL-exp-c3.5/scripts/c35_probe_subprojections.py \
     --model $BASE --model-key qwen3_8b \
     --u2-adapter-state $C3_SEL/u2_corridor/selected_adapter_state.pt \
     --m2-adapter-state $C3_SEL/m2_corridor/selected_adapter_state.pt \
     --kl-corpus $KL_CORPUS --batch-size 8 \
     --out-dir $RES/probe1_subproj
   ```

2. **GPU 1 — Probe 2A：跨字母打乱 7 路因果互换（`M0`, `u1_base`, `u2_corridor`，~10 分钟）**：
   ```bash
   CUDA_VISIBLE_DEVICES=1 $PY experiments/ACL-exp-c3.5/scripts/c35_eval_cross_letter_interchange.py \
     --model $BASE --alias M0 --out-dir $RES/probe2_cross_letter
   CUDA_VISIBLE_DEVICES=1 $PY experiments/ACL-exp-c3.5/scripts/c35_eval_cross_letter_interchange.py \
     --model $BASE --arch Uniform-LoRA --adapter-state $C3_SEL/u1_base/selected_adapter_state.pt \
     --alias u1_base --out-dir $RES/probe2_cross_letter
   CUDA_VISIBLE_DEVICES=1 $PY experiments/ACL-exp-c3.5/scripts/c35_eval_cross_letter_interchange.py \
     --model $BASE --arch Uniform-LoRA --adapter-state $C3_SEL/u2_corridor/selected_adapter_state.pt \
     --alias u2_corridor --out-dir $RES/probe2_cross_letter
   ```

3. **GPU 2 — Probe 2B：跨字母打乱 7 路因果互换（`u3_b1_iit`, `m2_corridor`，~7 分钟）**：
   ```bash
   CUDA_VISIBLE_DEVICES=2 $PY experiments/ACL-exp-c3.5/scripts/c35_eval_cross_letter_interchange.py \
     --model $BASE --arch Uniform-LoRA --adapter-state $C3_SEL/u3_b1_iit/selected_adapter_state.pt \
     --alias u3_b1_iit --out-dir $RES/probe2_cross_letter
   CUDA_VISIBLE_DEVICES=2 $PY experiments/ACL-exp-c3.5/scripts/c35_eval_cross_letter_interchange.py \
     --model $BASE --arch M3-DAC --adapter-state $C3_SEL/m2_corridor/selected_adapter_state.pt \
     --alias m2_corridor --out-dir $RES/probe2_cross_letter
   ```

4. **GPU 3 — 新盲集 `c3_blind_confirm_v2` 现有基准零重训重评（`M0, u1_base, u2_corridor, m2_corridor`，~12 分钟）**：
   ```bash
   CUDA_VISIBLE_DEVICES=3 $PY experiments/ACL-exp-c3.5/scripts/c35_eval.py \
     --model $BASE --alias M0 --regime no_think --batch-size 8 \
     --datasets c3_blind_confirm_v2 --out-dir $RES/baselines_v2/M0/no_think

   for PAIR in "u1_base:Uniform-LoRA" "u2_corridor:Uniform-LoRA" "m2_corridor:M3-DAC"; do
     TAG=${PAIR%%:*}; ARCH=${PAIR##*:}
     CUDA_VISIBLE_DEVICES=3 $PY experiments/ACL-exp-c3.5/scripts/c35_eval.py \
       --model $BASE --model-key qwen3_8b --arch $ARCH \
       --adapter-state $C3_SEL/$TAG/selected_adapter_state.pt \
       --alias $TAG --regime no_think --batch-size 8 \
       --datasets c3_blind_confirm_v2 --out-dir $RES/baselines_v2/$TAG/no_think
   done

   $PY experiments/ACL-exp-c3.5/scripts/c35_stats.py \
     --results-root $RES/baselines_v2 --stage1-baselines-only
   ```

*(完成后请只提交推送轻量 `json/jsonl/md` 结果至 `origin/ACL-exp-c3.5`！我方自动监控检测到结果后，将立刻根据 Probe 1 实测锁定 `M3-v2` 精确规格，并推送 Stage 2 的 5 新臂 `A, B, C, D, E` 训练代码与指令！)*

---

## 第四部分 · Stage 2 预登记路线图（等 Stage 1 探针落地后立刻冻结上线）

复用 C3 训练集与现成 `u1_base, u2_corridor, m2_corridor` 对照，仅训练 **5 个精准新臂**：
- **Arm A**：旧 `M3-v1` + Corridor，仅放宽底层 LR/prox（`A − m2` = 纯优化策略净增益）；
- **Arm B**：新 `M3-v2`（由 Probe 1 + `c3_dev` 锁定预算，匹配 Uniform 在 `±3%` 内）+ Corridor（`B − A` = 纯参数重分配净增益）；
- **Arm C**：`B` + **Soft-Gated B1-IIT**（共享前 120 步权重与样本顺序、真实 4 选源题 margin 门控、软插值 `α` 配方向性软目标、同 batch 分开计算主梯度与 IIT 梯度及其余弦夹角、执行 `0.30×` 独立限幅后再全局裁剪，`C − B` = 纯柔和 IIT 净增益）；
- **Arm D**：`B` + **Soft-Gated Matched-Random-Head IIT**（与 C 完全共享前 120 步状态、样本顺序、门控规则与固定 5 层 B1 MLP，`C − D` = 纯头定位证明）；
- **Arm E**：`Uniform-LoRA` + 同配方 **Soft-Gated B1-IIT**（`C − E` = 三区 vs 全层终极对决）。
