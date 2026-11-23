# ACL-exp-c3 · 三区机制化适应（`M3-DAC`）vs 全层 `Uniform + B1-IIT` 2×4 八卡公平对决方案

- **状态**：**`READY TO EXECUTE`（全部代码、冻结组件与去混杂数据集已通过 CPU 全量单元测试与 4 项跨脚本集成审计，可直接上 8×A800 启动！）**
- **分支**：`ACL-exp-c3`（从已永久冻结的 `ACL-exp-c2@7255b4e` 切出；`ACL-exp-a1/b1/c1/c2` 下文件一律只读不动）
- **主力模型**：`Qwen3-8B`（`8B` 过门并完成 `seed=1024` 独立复现后，再按同一全自动脚本升 `Qwen3-14B`）
- **绝对红线**：**严禁读取、打开或评测密封集 `c1_test_id_bases.jsonl` / `c1_test_ood_bases.jsonl`；所有 C3 训练/开发/盲确认数据严格排除 `converse_error` 与 `depth-4`！**

---

## 第一部分 · 从 C2 到 C3 的完整科学思考过程与设计决策

> 按项目规范，本节完整记录我们在 C2 收官审计、与 Astra 逐项复核、以及设计 C3 时的全部思考脉络、认知纠偏与工程教训，供后续论文撰写与复盘追溯。

### 1.1 我们从 C2（`5b492da`, `HOLD`）中真正学到了什么？

在 C2 的 8 卡 8 臂对决中，表面数字呈现出一个极具张力的现象：
- **好消息（B1 定位 × 训练算法的正向交互成立，且安全效率领先）**：在 `c2_confirm_bidir`（`n=239`）上，B1 真实电路加 IIT 维持了 `199→198`，而层匹配随机掩码加 IIT 暴跌 `200→179`，交互项达 **`+8.37 pp`**（原协议 95% CI `[+4.26, +12.61]`，按 65 个题族重采样仍为 `[+4.3, +13.0]`）。同时 `dac_iit` 取得了全场最高的 **IFEval `451/541`**（反超 M0 `+11` 分）、比 `unif_base` **低 32% 的 WikiText-2 KL**（`0.000997` vs `0.001461`）以及远高于 Uniform 的 BF16 合并保真（余弦 `0.9337` vs `0.7594`）。
- **坏消息（纯准确率落后强基线 `unif_base`）**：在 `c2_confirm_bidir` 上 `unif_base` 达到 `92.0% (220/239)`，显著领先 `dac_iit`（`82.8%`）与 `dac_paired`（`84.5%`）；在 `CE Row B`（合法跨实体交接）上 `unif_base` 达 `32/36`，在 `think` 模板迁移集上 `unif_base` 达 `108/120`（对 `dac_iit` `96/120` 独对 `12:0`）。

起初我们提出“Uniform 的优势只是因为底层没冻结，能从零建立新绑定”的直觉假设。但在把同一批题拆成 **`(真/假) × (犹豫/强势)` 完整四格** 做深度法医拆解后，Astra 的复核和我们的排查共同揭示了**一个彻底改写 C3 设计方向的底层真相**：

| 原模型 `M0` (`no_think`) | 倒置语气 `V_hes vs F_auth` | 两边中性 `V vs F` | 两边犹豫 `V_hes vs F_hes` | 两边强势 `V_auth vs F_auth` | 真实失败本质 |
|---|---:|---:|---:|---:|---|
| **`late_buried` 非阈值 (`n=90`)** | `0/90` | **`72/90`** | **`82/90`** | **`82/90`** | **模型本就具备 91% 内容排序，是强硬语气压倒了证据读出！** |
| **合法跨实体 `CE Row B` (`n=36`)** | `0/36` | `8/36` | `16/36` | `13/36` | **语气配平后仍接近随机，才是真正缺底层跨实体绑定！** |

紧接着检查 `unif_base` 在 `late_buried (90)` 上的四格表现，发现了惊人的**反向语气捷径（Anti-Authority Shortcut）**：
- `unif_base` 在原指标 `V_hes vs F_auth` 上看似拿了 `21/90`；但一旦把语气对调测 **`V_auth vs F_hes`**，`unif_base` 直接崩塌到 **`3/90`**，**四格同时答对仅 `3/90 (3.3%)`**！在 **89/90** 题上 `unif_base` 学会了给强势语气统一加负偏置（“谁语气横我就判谁错”）！
- 与此同时，在 `CE Row B (36)` 上，`unif_base` 却是**货真价实**的四格全对 **`32/36`**（明显高于 `dac_iit` 的 `22/36`）。

**这一发现瞬间把整个课题的科学目标聚焦到了最锋利的刃口上**：
> 我们要让模型同时学会两件可独立测量的真本事——**① 论证逻辑真的变了（含合法跨实体绑定与多步隐蔽断点），模型能果断改口；② 只有语气变了（无论强势还是犹豫），模型仍严格按证据判断（既不盲从权威，也不逆反权威）！**
> 因此，C3 的主指标必须升级为**无法作弊的真实四选四格全对率（`V_hes, V_auth, F_hes, F_auth` 全部 4 格在真实 A/B/C/D 四选 argmax 下同时正确）**！

### 1.2 为什么我们提出三区机制化适应（`M3-DAC`），又为何坚持与 `Uniform + B1-IIT` 同台公平对决？

回查 B1 在 `Qwen3-8B` 和 `Qwen3-14B` 两个规模的原始 `.pt` 归因张量，我们确认了三个跨尺度一致的解剖学事实：
1. **底层（Zone 1, 8B `L0–13` / 14B `L0–15`）完全冻结的代价**：C2 把 `L0–13` 锁死，使得模型在 `no_think` 模式下无法对 `CE Row B` 建立跨实体信息桥接；而我们又不希望像 `Uniform` 那样在底层开全量 MLP + 大幅度改权重导致 Wiki KL 暴涨 `1.46×` 且破坏通用能力。
2. **中层（Zone 2, 8B `L14–27`）并非均质电路**：8B/14B 的 Top-10 头分别承载了已选头正归因质量的 **`56.6% / 56.5%`**，且两个规模的 Split A 与 Split B 均高达 **`9/10` 头重合**！C2 给所有 47 颗头一律平均分配 `r=14`，既浪费了边缘头的预算，又让核心路由与验证头带宽受限。
3. **顶层（Zone 3, 8B `L28–35` / 14B `L31–39`）遗漏了关键对话 MLP 与直接读出头**：8B `C_content_AF` 中 **`L35 (#2)`、`L32 (#3)`、`L29 (#5)`** MLP 排名前五，14B `C_content_CF` 中 **`L39 (#1)`、`L38 (#2)`** MLP 排名前二——但在 C1/C2 的 `Targeted-Dual` 里**全部被冻结了**！同时顶层 `DLA_pinpoint` 头存在显著极性分化（如 8B `L29H5` 为 `DLA=+5.50, form=+0.023`，而 `L34H30` 为 `DLA=+3.61, form=-0.216`），绝不能一刀切地贴“刹车头”标签。

更重要的是方法论上的清醒与克制（吸取 Astra 的关键建议）：
- 在 C2 中，`c2_train_iit.py` 甚至禁止给 `Uniform-LoRA` 施加 `B1-IIT`，这导致“参数放哪里（Architecture）”和“反事实干预施在哪里（IIT Target）”缠在了一起。
- 在 C3 中，我们**彻底解耦“可训练架构（`Uniform-LoRA` vs `M3-DAC`）”与“干预位置（`none` / `B1` / `Matched-Random`）”**，构成干净的 **2×4 八卡矩阵**：
  - **`U3 (Uniform + B1-IIT)` vs `U4 (Uniform + Matched-Random-IIT)`**：在完全相同的全层可训练容量下，最纯粹地检验 **B1 定位作为因果干预先验** 究竟能带来多大增益（且固定 5 层 `ver_mlps` 只随机化头坐标，彻底排除“每层只有一个 MLP”的混杂）！
  - **`M3 (M3-DAC + B1-IIT)` vs `U3 (Uniform + B1-IIT)`**：当双方都享有同一份 B1-IIT 先验时，最公平地检验 **三区机制化预算分配** 能否在同参数量下超越（或以 ≥25% 更低 KL 达到 Pareto 支配）全层 Uniform！
- 并且在正式训新模型前，先花 5 分钟用 C2 现成磁盘上的 `unif_base/adapter_state.pt` 做一次**零重训三区消融探针（Wave 3.0）**，分别关闭 Zone 1 / Zone 2 / Zone 3 的 LoRA 增量，直接看“已经学成的 Uniform 解究竟依赖哪个区做 Row B、哪个区制造了反向语气偏置”。

### 1.3 五项核心部件的审慎取舍（既精密科学，又坚决去掉未经证实的花哨负担）

| 部件 | 最初脑暴想法 | 审计后的 C3 精炼定稿（`c3_components_frozen.json` + `c3_train.py`） | 决策理由 |
|---|---|---|---|
| **① 底层轻桥（Zone 1）** | 固定“前 30% 步数冻结，后 70% 解冻” | **砍掉固定冻结表**；采用 `L0–13` 密集 `Q/O` `r=2` 零初始化（`lora_B=0`）+ 低学习率 `0.35×` + 对真实 `ΔW` 的可微近端惩罚 $\lambda_{\text{prox}} \sum \|\Delta W_{Z1}\|_F^2$（利用 $O(r^2 d)$ 迹恒等式精确计算） | CE 损失下“中顶层已答对”不等于零梯度；零初始化 + 低 LR + `ΔW` 近端锚能让底层仅在存在强绑定残差（如 Row B）时温和生长，同时最大限度保住通用分布 |
| **② 中层异质性（Zone 2）** | 手工划分 4 种硬角色并逐头配学习率 | **砍掉硬角色贴标与手工 LR**；完全基于 B1 Split-A/B 跨半稳定复合归因分自动分为 **3 档秩（`10×r20 + 13×r12 + 29×r6`）**；新增**自动证据冲突检测器**（跨半变号或 `DLA` 与 `form` 强反向），自动检出 8B `L34H30` 并强制降到最低保守档 `r=6` 且排除在激进 IIT 交换之外 | 100% 自动化推广至 8B/14B（零硬编码头号）；避免把方向冲突的头训反；预登记 `M3-Equal-Rank`、`M3-Shuffled-Rank`、`M3-Random-Mask` 供 Wave 3.3 精准归因 |
| **③ 顶层与置信走廊（Zone 3）** | 加 RMS 缩放锚 + 各种动态温度 | **首版去掉 RMS 锚，优先补齐 3 层顶层对话 MLP (`[29,32,35]@r8`) + 7 颗稳定 Zone-3 DLA 头**；用**有符号置信走廊 `[4.5, 10.0]`**（$\mathrm{ReLU}(4.5 - y \cdot u) + 0.05\,\mathrm{ReLU}(|u|-10.0)^2$）+ **对称核心权重**（6 个核心 V/F 臂等权 `1.0`，辅助臂 `0.5`，彻底废除 `accept_weight=2.0` 与无界 `delta_sens`） | 触发条件基于 `y·u`，高置信答错的题（`y·u=-15`）得到强力纠偏，已校准答对的题（`y·u∈[4.5,10]`）梯度精确为 0，超过 `10` 自动柔拉回防止 C2 中 `med(u)=-35` 过饱和；同时报告真实四选 NLL 与 Brier 校准分 |
| **④ 训练稳定性与工程自查** | 沿用 C2 `clip=1.0`, `lr=3.5e-5`, `du_abs<=0.35` | 放宽 `max_grad_norm=2.0`，降默认学习率至 `2.5e-5`（并可在开发集比较 `{2.0e-5, 3.5e-5}`）；全程记录**裁剪前分区分层梯度、裁剪率、各损失激活率、各区真实 `ΔW` 范数轨迹**；保存 `@50/@100/@200/@300` 由独立 `c3_dev` 自动选点；保真门禁统一校准为 `du_abs <= 0.50` 且永不误拦 HF 评测 | 直击 C2 中 `dac_iit` `295/300` 步与 `rnd_iit` `300/300` 步全程触发梯度裁剪的实测瓶颈，并永久消除 C2 `0.35` 误停事故与 GC 长度不齐钩子回放事故（已在 `<, ==, >` 三种长度下验证 `gc_diff = 0.00e+00`） |
| **⑤ 数据去混杂、新盲集与术后因果证据** | 沿用 C2 数据集与训练窗翻转率 | 修复 Style-B `Only if → Whenever`（硬断言 0 处 `Only if`）；独立随机化 `(domain, pos, style)`（卡方全部 `p>0.15`）；训练非 CE 精确平衡 `Step 1/2/3 = 160/160/160`；新建 6 域 **复合结构盲确认集 `c3_blind_confirm`（1,080 题 / 960 独立题族）**，留出“跨步多前提综合 + 活跃竞争干扰分支 + Style-C 模板”；新增 **`c3_eval_interchange.py`** 在留出题对 `M0/U/M` 施加相同 `B1/Rnd/Identity` 干预记录真实四选反事实翻转率与换语气保持率 | 彻底消除 C2 三个数据与评测短板（`Only if` 语义误改、因素绑定混杂、Step 1 过拟合与功率不足），并把“训练目标拟合率（97.5%）”升级为审稿人无可挑剔的**术后留出因果机制证据** |

---

## 第二部分 · 冻结组件、数据集与 CPU 全量审计汇总

### 2.1 自动化跨尺度组件规格（`experiments/ACL-exp-c3/data/c3_components_frozen.json`）

| 模型 | 架构 | 可训练参数量 | 相对 Uniform | 构成明细 | 单层最大打包秩 (`<=128`) | 冲突保守头 |
|---|---|---:|---:|---|---:|---|
| **`qwen3_8b`** | `Uniform-LoRA` | `7,077,888` | `+0.00%` | 全 36 层 `q/o/gate/down @ r=4` | `4` | — |
| **`qwen3_8b`** | **`M3-DAC`** | **`7,033,344`** | **`-0.63%`** | Z1 `L0–13` 密集 `Q/O @ r=2` + 8 MLP `[18,20,22,23,27,29,32,35] @ r=8` + 52 头 (`10×r20 + 13×r12 + 29×r6`) | **`88`** (`<=128`) | `[[34, 30]]` (`r=6`, 排除 IIT) |
| **`qwen3_8b`** | `M3-Equal-Rank` / `Shuffled` / `Random-Mask` | `6.95M / 7.03M / 7.03M` | `-1.82% / -0.63% / -0.63%` | 预注册 Wave 3.3 归因对照 | `80 / 88 / 88` | 同上 |
| **`qwen3_14b`** | `Uniform-LoRA` | `10,485,760` | `+0.00%` | 全 40 层 `q/o/gate/down @ r=4` | `4` | — |
| **`qwen3_14b`** | **`M3-DAC`** | **`10,382,336`** | **`-0.99%`** | Z1 `L0–15` 密集 `Q/O @ r=2` + 8 MLP `[19,22,23,25,26,33,38,39] @ r=8` + 51 头 (`10×r24 + 14×r14 + 27×r8`) | **`100`** (`<=128`) | `[]` |

- **解耦 IIT 干预位置（8B）**：`b1` 与 `rnd_head_1`（`seed=42`）/ `rnd_head_2`（`seed=1024`）严格逐层同数（`form=23, ver=21, hub=3`）、随机头与全部 B1 头 0 重合，且**固定相同的 5 层 B1 `ver_mlps=[18,20,22,23,27]`**，确保 `U3 vs U4` 与 `M3 vs M4` 纯粹检验 B1 注意力头定位！

### 2.2 去混杂数据集与对抗审计（`experiments/ACL-exp-c3/data/c3_datasets_manifest.json`）

| 数据集 | 域池 | 题量 / 独立题族 | 非 CE 断点 Step 1/2/3 | 2×2 CE (Row A/B) | G0 TF-IDF AUC (`cf_byte`) | 因素独立性 χ² 最小 `p` | `Only if` 计数 | 与训练集骨架重合 |
|---|---|---:|---|---|---:|---:|---:|---:|
| `c3_train_bases.jsonl` (`c3_train_examples_nothink.jsonl`) | `C2_TRAIN` (6 域) | `600` (`6,000` 例) | **`160 / 160 / 160`** | `60 / 60` | `0.4853` (`1.0`) | `0.2137` | **`0`** | — |
| `c3_dev_bases.jsonl` | `PROXY_OOD` (6 域) | `180` (`150` 族) | **`40 / 40 / 40`** | `30 / 30` | `0.4637` (`1.0`) | `0.1562` | **`0`** | — |
| **`c3_blind_confirm_bases.jsonl`** | **全新 `C3_BLIND` (6 域)** | **`1,080` (`960` 族)** | **`280 / 280 / 280`** | **`120 / 120`** | **`0.5100` (`1.0`)** | **`0.7570`** | **`0`** | **`0 / 0`** |
| `c3_blind_think_bases.jsonl` | `C3_BLIND` 分层抽样 | `240` (`200` 族) | 分层均衡 | `40 / 40` | 同上 | 同上 | **`0`** | **`0 / 0`** |

- 全部 9 个 C3 脚本均已通过本地 CPU 自测及 4 项跨脚本集成审计（含 C2 `unif_base → c3_modules` `0.0` 误差加载、6,000 例真实原子窗口切分、变秩 PEFT 导出 `logit_err = 2.98e-7`、`< / == / >` 三种长度 GC `0.00e+00` 误差、`capabilities/summary.json` 字段对齐与精确平局 `diff=0.0` 检验）。

---

## 第三部分 · 2×4 八卡同台公平对决矩阵（`Qwen3-8B`, `seed=42`, 相同数据/Token/选点规则）

| GPU | Arm Tag | `--arch` | `--loss-mode` | `--iit-target` | 核心科学角色 |
|---:|---|---|---|---|---|
| 0 | `u1_base` | `Uniform-LoRA` | `base` | `none` | 全层基础对照（对称权重） |
| 1 | `u2_corridor` | `Uniform-LoRA` | `corridor` | `none` | 全层 + 有符号走廊 + 跨语气不变性 |
| 2 | **`u3_b1_iit`** | **`Uniform-LoRA`** | **`corridor_iit`** | **`b1`** | **最强简洁竞品：全层 + B1 位置 IIT（与 U4 做最干净定位检验！）** |
| 3 | `u4_rnd_iit` | `Uniform-LoRA` | `corridor_iit` | `rnd_head_1` | 全层 + 匹配随机头 IIT（固定 B1 MLPs） |
| 4 | `m1_base` | `M3-DAC` | `base` | `none` | 三区基础对照 |
| 5 | `m2_corridor` | `M3-DAC` | `corridor` | `none` | 三区 + 有符号走廊 + 跨语气不变性（无 IIT） |
| 6 | **`m3_b1_iit`** | **`M3-DAC`** | **`corridor_iit`** | **`b1`** | **主候选旗舰：三区机制化分配 + 走廊 + B1 位置 IIT（与 U3/M4/M2 正面对比！）** |
| 7 | `m4_rnd_iit` | `M3-DAC` | `corridor_iit` | `rnd_head_1` | 三区 + 匹配随机头 IIT（固定 B1 MLPs） |

*(所有 8 臂共享完全相同的 `--train-file c3_train_examples_nothink.jsonl`、`--max-steps 300`、`--grad-accum 20`、`--lr 2.5e-5`、`--save-steps 50,100,200,300`、`--seed 42`，以及完全一致的 `c3_dev` 自动选点规则；唯一架构内置差异为 `M3-DAC` 自身的 Zone-1 温和正则 `lr_z1_mult=0.35, lambda_prox_z1=1e-3`，并在日志中如实记录单臂实际 GPU 时长！)*

---

## 第四部分 · 8×A800 分波次完整执行手册

> **路径约定（沿用实验机既有配置）**：
> - `PY=/mnt/vllm310/bin/python`
> - `BASE=/mnt/kcl/devuser/ckpts/Qwen3-8B`
> - `WORK=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c3`
> - `RES=experiments/ACL-exp-c3/results/c3_8b_showdown`
> - `KL_CORPUS=experiments/ACL-exp-c1/data/c1_kl_wikitext2_200.jsonl`
> - `C2_UNIF_CKPT=/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c2/showdown-c62cf156/c2_ckpts/42/unif_base/adapter_state.pt`
> - IFEval / GSM8K 数据路径与 vendor 库沿用 C1/C2 实验机既有 `--ifeval-jsonl / --gsm8k-jsonl / --ifeval-lib-dir`。

### Wave 3.0（3 卡并行，~10 分钟：零重训 3 区探针 + `M0` 锚点冻结）

1. **GPU 0 — 零重训 3 区消融探针（直接加载 C2 现成 `unif_base/adapter_state.pt`，~4 分钟）**：
   ```bash
   CUDA_VISIBLE_DEVICES=0 $PY experiments/ACL-exp-c3/scripts/c3_probe_unif_zones.py \
     --model $BASE --adapter-state $C2_UNIF_CKPT --batch-size 8 \
     --out-dir experiments/ACL-exp-c3/results/wave3_0_zone_probe
   ```
   *(完成后立即可推送 `wave3_0_zone_probe/{unif_zone_probe.json, UNIF_ZONE_PROBE.md}`，看清 Uniform 解的各区归因！)*

2. **GPU 1 — `M0` `no_think` 基准与双侧 Turn-1 合格题号冻结（~6 分钟）**：
   ```bash
   CUDA_VISIBLE_DEVICES=1 $PY experiments/ACL-exp-c3/scripts/c3_eval.py \
     --model $BASE --alias M0 --regime no_think --batch-size 8 \
     --datasets c3_dev c3_blind_confirm c2_confirm_bidir c2_stealth_stress \
     --eligible-ids-out $RES/M0/no_think/eligible_ids.json \
     --out-dir $RES/M0/no_think
   ```

3. **GPU 2 — `M0` vLLM `think` 基准（6 核心臂，~8 分钟）+ `M0` 留出术后互换基准（~2 分钟）**：
   ```bash
   CUDA_VISIBLE_DEVICES=2 $PY experiments/ACL-exp-c3/scripts/c3_eval.py \
     --model $BASE --backend vllm --alias M0 --regime think --max-new-tokens 3072 \
     --datasets c3_blind_think c1_proxy_template_v2 \
     --arms turn1_F turn1_V V_hes V_auth F_hes F_auth \
     --eligible-ids-out $RES/M0/think/eligible_ids.json \
     --out-dir $RES/M0/think

   CUDA_VISIBLE_DEVICES=2 $PY experiments/ACL-exp-c3/scripts/c3_eval_interchange.py \
     --model $BASE --alias M0 --dataset c3_blind_confirm --limit 60 \
     --out-dir $RES/interchange
   ```

### Wave 3.1（8 卡并行，~35–42 分钟：8 臂 300 步训练 + `c3_dev` 自动选点）

GPUs 0–7 按第三部分表格并行运行各自 `<TAG>`：
```bash
# 1. 训练 300 步（保存 @50/@100/@200/@300）
CUDA_VISIBLE_DEVICES=<GPU> $PY experiments/ACL-exp-c3/scripts/c3_train.py \
  --model $BASE --model-key qwen3_8b \
  --arch <ARCH> --loss-mode <LOSS_MODE> --iit-target <IIT_TARGET> \
  --lr 2.5e-5 --max-steps 300 --grad-accum 20 --microbatch 2 \
  --save-steps 50,100,200,300 --seed 42 \
  --out-dir $WORK/ckpts_42/<TAG>

mkdir -p $RES/<TAG>
cp $WORK/ckpts_42/<TAG>/train_meta.json $RES/<TAG>/train_meta.json

# 2. 独立 c3_dev 自动选点（从 @50/@100/@200/@300 选出 4-Cell 最高、NLL 最低者，严禁看盲集！）
CUDA_VISIBLE_DEVICES=<GPU> $PY experiments/ACL-exp-c3/scripts/c3_select_dev_ckpt.py \
  --model $BASE --model-key qwen3_8b --arch <ARCH> \
  --ckpt-dir $WORK/ckpts_42/<TAG> \
  --eligible-ids-in $RES/M0/no_think/eligible_ids.json \
  --batch-size 16 --out-dir $WORK/selected_42/<TAG>

cp $WORK/selected_42/<TAG>/dev_selection.json $RES/<TAG>/dev_selection.json
```

### Wave 3.2（8 卡并行，~45–55 分钟：无损导出 + 校准保真 + 盲确认 + 能力护盾 + 留出互换）

每张 GPU 对自己选定的 `$WORK/selected_42/<TAG>/selected_adapter_state.pt` 顺序执行：
1. **无损变秩 PEFT LoRA 导出**（~30 秒）：
   ```bash
   $PY experiments/ACL-exp-c3/scripts/c3_export_peft_lora.py \
     --model-key qwen3_8b --arch <ARCH> \
     --adapter-state $WORK/selected_42/<TAG>/selected_adapter_state.pt \
     --base-model $BASE --out-dir $WORK/peft_42/<TAG>
   ```
2. **校准版 HF vs vLLM 原生 LoRA 保真校验（`--max-abs-du-mean 0.50 --no-fail-exit`，绝不误拦 HF 评测！）**（~2.5 分钟）：
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU> $PY experiments/ACL-exp-c3/scripts/c3_check_vllm_lora_parity.py \
     --base-model $BASE --model-key qwen3_8b --arch <ARCH> \
     --adapter-state $WORK/selected_42/<TAG>/selected_adapter_state.pt \
     --peft-lora-dir $WORK/peft_42/<TAG> \
     --dataset c3_dev --limit 40 --max-abs-du-mean 0.50 --no-fail-exit \
     --out-json $RES/<TAG>/vllm_lora_parity.json
   ```
3. **HF 无损 Live `no_think` 全集（`c3_blind_confirm` 1,080 题 + `c2_confirm_bidir` + `c2_stealth_stress` + WikiText-2 KL 200 段）**（~12 分钟，batch=8 加速！）：
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU> $PY experiments/ACL-exp-c3/scripts/c3_eval.py \
     --model $BASE --base-model $BASE --model-key qwen3_8b --arch <ARCH> \
     --adapter-state $WORK/selected_42/<TAG>/selected_adapter_state.pt \
     --alias <TAG> --regime no_think --batch-size 8 \
     --datasets c3_blind_confirm c2_confirm_bidir c2_stealth_stress \
     --eligible-ids-in $RES/M0/no_think/eligible_ids.json \
     --kl-corpus $KL_CORPUS \
     --out-dir $RES/<TAG>/no_think
   ```
4. **vLLM 原生 LoRA `think`（`c3_blind_think` 240 题 + `c1_proxy_template_v2` 120 题，6 核心臂）**（~12 分钟）：
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU> $PY experiments/ACL-exp-c3/scripts/c3_eval.py \
     --model $BASE --backend vllm --peft-lora-dir $WORK/peft_42/<TAG> \
     --alias <TAG> --regime think --max-new-tokens 3072 \
     --datasets c3_blind_think c1_proxy_template_v2 \
     --arms turn1_F turn1_V V_hes V_auth F_hes F_auth \
     --eligible-ids-in $RES/M0/think/eligible_ids.json \
     --out-dir $RES/<TAG>/think
   ```
5. **vLLM 原生 LoRA 通用能力护盾（`IFEval 541` + `GSM8K 1,319`）**（~18 分钟）：
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU> $PY experiments/ACL-exp-c1/scripts/c1_eval_capabilities.py \
     --model $BASE --peft-lora-dir $WORK/peft_42/<TAG> \
     --ifeval-jsonl <SERVER_IFEVAL_JSONL> --gsm8k-jsonl <SERVER_GSM8K_JSONL> \
     --ifeval-lib-dir <SERVER_IFEVAL_LIB_DIR> \
     --alias <TAG> --out-dir $RES/<TAG>/capabilities
   ```
6. **术后留出因果互换评测（仅对 4 个核心臂 `u1_base, u3_b1_iit, m1_base, m3_b1_iit`，在 `c3_blind_confirm` 前 60 题分层样本施加相同 `b1 / rnd_head_1 / rnd_full_1 / identity` 干预，~2.5 分钟）**：
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU> $PY experiments/ACL-exp-c3/scripts/c3_eval_interchange.py \
     --model $BASE --model-key qwen3_8b --arch <ARCH> \
     --adapter-state $WORK/selected_42/<TAG>/selected_adapter_state.pt \
     --alias <TAG> --dataset c3_blind_confirm --limit 60 \
     --out-dir $RES/interchange
   ```

### Wave 3.25（CPU ~15 秒：自动汇总、题族簇 Bootstrap 与双路径预注册裁决）
```bash
$PY experiments/ACL-exp-c3/scripts/c3_stats.py \
  --results-root $RES \
  --eligible-ids $RES/M0/no_think/eligible_ids.json \
  --primary-dataset c3_blind_confirm --n-boot 5000
```
产出 `$RES/c3_stats.json` 与 `$RES/C3_READOUT.md` 并推送至 **`origin/ACL-exp-c3`**！

### Wave 3.3（条件自动触发 / 等待审计确认后执行：`seed=1024` 独立复现 + 三区归因消融）
- 若 Wave 3.25 中 **`m3_b1_iit`** 或 **`u3_b1_iit`** 触发 **`PASS_PATH1_SUPERIORITY`** 或 **`PASS_PATH2_EFFICIENCY_PARETO`**，则在确认后立刻开 4–6 卡跑：
  1. 独立种子 **`--seed 1024`** 复现核心 4 臂（胜者、最强 Uniform 对照、对应 `rnd_head_2` 随机 IIT、无 IIT 对照）；
  2. 若 `M3-DAC` 胜出，并行补跑预注册的三区归因对照（`M3-Equal-Rank`、`M3-Shuffled-Rank`、`M3-Random-Mask`）以分别归因于**位置**与**预算分配**！
- **单种子完成后先推送至 `origin/ACL-exp-c3` 并停下等审计**，严禁未经确认擅自开 `14B` 或触碰密封 `c1_test_id / c1_test_ood`！

---

## 第五部分 · 预注册双路径成功门禁与止损红线

1. **统一健康底线（全部必过）**：
   - `IFEval >= 430/541`、`GSM8K >= 470/1319`、`WikiText-2 KL <= 0.005`、`think parse_fail <= 2.0%`；
   - `S / shell_only / P0 / I0 >= 95%`；
   - **反向语气非劣性**：`DC(V_auth, F_hes) >= DC(V_hes, F_auth) - 10 pp`（彻底封杀 `unif_base` 式反向语气作弊）；
   - `vllm_lora_parity.json` 通过校准口径（`u_coverage == 1.0`, `option_match >= 95%`, `|du_signed| <= 0.25`, `du_abs <= 0.50`）。
2. **路径一（效果胜出 `PASS_PATH1_SUPERIORITY`）**：
   - 主终点（`c3_blind_confirm` `no_think` M0 双侧合格、题族等权、真实四选四格全对率）相对最强健康同数据 Uniform 对照 **`>= +3.0 pp`** 且题族配对 95% CI 下界 **`> 0`**；
   - 同位置预算下胜过匹配随机 IIT（`M3 > M4` / `U3 > U4`）且较自身无 IIT 改善（`M3 > M2` / `U3 > U2`）；
   - 经 `seed=1024` 独立复现同向成立 → 解锁 **`Qwen3-14B`** 小矩阵复现！
3. **路径二（效率 / Pareto 支配 `PASS_PATH2_EFFICIENCY_PARETO`）**：
   - 主终点相对最强健康同数据 Uniform 对照的题族配对 95% CI 下界 **`>= -2.0 pp`**，且同任务表现下 **WikiText-2 KL 至少降低 25%**（`KL_M3 <= 0.75 * KL_U_best`），并经 `seed=1024` 独立复现 → 解锁 **`Qwen3-14B`** 复现！
4. **止损线**：若 8B 两条路径均未过线且置信区间上界排除 `+3.0 pp` 增益，则停止 `14B` 全矩阵扩跑，将 A1 行为诊断、B1 机制解离与 C1/C2/C3 关于“可定位、可干预、可学成三者边界”的高严谨度实证直接定稿入论文。
