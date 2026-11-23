# ACL-exp-c2 · 验证门控三力平衡与回路引导互换干预训练（`DAC-IIT`）完整思考录与 8 卡执行方案

- **状态**：**`READY TO EXECUTE`**（立即执行！含 Wave 2C.0 CPU 全量自检 → Wave 2C.05 M0 基线冻结 → Wave 2C.1 8 卡 2×2 因子训练 → Wave 2C.2 8 卡无损全量三轴评测/能力护盾/轨迹 → Wave 2C.25 自动因果交互统计 → 条件触发 Wave 2C.3 `seed=1024` 独立复现）
- **分支与上游**：全新分支 **`ACL-exp-c2`**（从已永久封箱的 `ACL-exp-c1@12ded64` 切出；`experiments/ACL-exp-a1/`、`experiments/ACL-exp-b1/`、`experiments/ACL-exp-c1/` 全部保持只读零修改）。
- **封存测试集铁律**：`c1_test_id_bases.jsonl` 与 `c1_test_ood_bases.jsonl`（含 `converse_error` 与真 4 跳 `depth-4`）在本实验全程保持 **100% 封存**，并由 `scripts/c2_eval.py` 的物理门禁默认拒绝读取。
- **硬件与预计总耗时**：**8 × NVIDIA A800 40GB**
  - Wave 2C.0（CPU 自检）：~30 秒
  - Wave 2C.05（2 卡并行：M0 在 `c2_confirm_bidir` + `c2_stealth_stress` + `c1_proxy_template_v2` + `c1_dev` 的 `no_think` 与 vLLM `think` 基线与合格题冻结）：~15 分钟
  - Wave 2C.1（8 卡并行：8 组 300 步全容量 2×2 因子训练）：~28–38 分钟
  - Wave 2C.2（8 卡并行：逐组无损 PEFT LoRA 导出 + 逐组 SHA 绑定 vLLM 保真校验 + HF live `no_think` & KL + vLLM 原生 LoRA `think` + `IFEval (541)` + `GSM8K (1,319)` + `@100/@200` 轨迹 + Flagship 权重舍入审计）：~55–65 分钟
  - Wave 2C.25（CPU 自动统计与 2×2 因子交互检验 `c2_stats.py`）：~15 秒
  - Wave 2C.3（条件触发：仅当单种子达到 `CANDIDATE_SEED2` 时，4 卡自动补跑独立 `seed=1024` 复现）：~30 分钟

---

## 第一部分 · 完整科学思考录（实验立项前的推演、反思与设计权衡）

> **文档规范说明**：本节完整记录我们在封箱 `ACL-exp-c1` 后设计 `ACL-exp-c2` 的全部科学思考、辩论修正与前置工程排雷，便于后续撰写 ACL 2027 论文方法论与回顾研究脉络。

### 1. 我们从 `ACL-exp-c1` 封箱中真正学到了什么？
在 [C1_CHRONICLE_AND_CLOSURE.md](../ACL-exp-c1/C1_CHRONICLE_AND_CLOSURE.md) 中，Step 2B 的 8 卡无损对决严格触发了预注册 `STOP`（`overall_go_14b = False`），并在三方复盘后澄清了三个决定方向的根本事实：
1. **数据不变量误伤（Row A `98.5%` vs Row B `0.2%`）**：在 `c1_train_examples_wo_ce_nothink.jsonl` 中，全部 800/800 条核心 V/F 样本都满足 `obs_entity == target_entity`。在这 800 条数据上，“按规则核验”与“遇到跨实体就拒绝、同实体才核验”给出 100% 相同的标签，导致 8 组模型在需要“合法跨实体交接（Row B）”的新题上全军覆没。**结论**：拿缺类、隐含捷径的受限训练去评判架构上限，测到的是训练分布的不可辨识边界，而不是不同方法在充分公平学习后的真实逻辑纠错上限！
2. **定位与可塑性解离（Localization–Plasticity Dissociation）**：即便在去除了 BF16 合并伪迹之后，`Targeted-Dual` 相比同层双随机在主终点上仅领先 `+1.82 pp`（Std）/ `+0.84 pp`（Tight）。结合文献（Hase et al. NeurIPS 2023; Chintam et al. EMNLP 2023; Wang et al. ICLR 2025），根本症结在于——**`ACL-exp-c1` 只把 B1 的白盒解剖结果当成“静态 LoRA 参数掩码”，而在训练目标上仍然把所有头和 MLP 混为一潭去吃统一的末端 CE/Margin 损失！** 在高度过参数化的 Transformer 中，只要末端监督相同，梯度会自动沿着随机分配的 47 颗同层头与 5 层中层 MLP 走出一条近乎等价的可塑性捷径；我们在 B1 里费尽心血定位出的“形式头 / 核验头+MLP / 共享枢纽头”三组物理分工，在训练算法里完全没有被用起来！
3. **部署保真归因的清醒认知**：Step 2B 权重舍入审计证实 `Dual (0.918) ≈ Random1 (0.928) ≈ Random2 (0.921) >> Uniform (0.731)`，证明低精度合并抗漂移属于“空间稀疏集中”的红利，不能冒充“B1 定位身份”的功劳。要证明 B1 定位本身的不可替代价值，必须把 B1 的三组电路结构**内蕴地写进训练算法本身**，让它转化为无可替代的逻辑纠错上限优势！

### 2. 核心理论升级：辩证回归“验证门控三力平衡”（Verification-Gated Three-Force Equilibrium）
在反思如何利用 B1 定位结果时，我们摒弃了“把形式顺从通路 `C_form` 当作纯负面毒药一刀切压死”的朴素思路，辩证地回归并升级了项目早期的三力平衡直觉：
- **形式通道并非纯负面**：当用户提出完全有效的新证据、但措辞犹豫谦逊（`V_hes`），或当模型执行复杂格式约束指令（`IFEval`）时，形式/提案通道承担着正向的候选推送与指令遵从推力；若粗暴压制 `C_form`，只会让模型退化成盲目固执的“过度拒绝机器”（Step 1 `zero_rep_s295` 便曾出现 `V_hes` 跌破 `90%` 的警示）。
- **理想的平衡态方程**：
  $$u^*(x) \approx h_{\text{init}}(x) + g\big(h_{\text{ver}}(x)\big) \cdot h_{\text{form}}(x) + h_{\text{ver}}(x)$$
  当逻辑核验通路 $h_{\text{ver}}$ 判定论证有效时，条件门控打开（$g \approx 1$），形式提案力与核验力协同推送正确修订（**真证据即便语气犹豫 `V_hes` 也果断改**）；当核验通路判定论证存在谬误或纯空壳时，门控关闭（$g \approx 0$），切断权威口吻与华丽修辞的误导（**假证据即便权威包装 `F_auth` 也坚决守**）。
- **B1 解剖图谱与三力模型的严丝合缝对应**（已由 `c1_components_frozen.json` 严格验证）：
  在 `Qwen3-8B` 上，`Targeted-Dual` 的 47 颗头 + 5 层 MLP 天然构成**零交集的精确三分区**（`23 + 21 + 3 = 47`）：
  1. **`Form-Only Heads`（23 颗头）**：`Targeted-Form \ Hub` —— 负责形式与语气提案；
  2. **`Ver-Only Pathway`（21 颗头 + 5 层 B1 实证 MLP `[18, 20, 22, 23, 27]`）**：`Targeted-Dual \ Targeted-Form \ Hub` + `mlp_B` Top-5 —— 负责纯规则逻辑核验；
  3. **`Shared Routing Hub Heads`（3 颗头 `[(19,31), (20,8), (21,19)]`）**：处于两条通路交汇咽喉 —— 天然充当可学习的条件门控 $g(h_{\text{ver}})$！

### 3. 核心算法设计：两级递进的 `DAC-IIT`（双通路回路引导互换干预训练）
为了让每一层方法论贡献都能被严格拆解归因，我们在 `scripts/c2_train_iit.py` 中实现三条严格嵌套的训练模式（**训练期零 OOM、推理期零额外开销、100% 兼容现有无损 PEFT LoRA 导出**）：

1. **Level 0 (`base`)**：保留 `ACL-exp-c1` 的标准逐样本 SFT + Signed Readout Margin 损失，但改为**充分去捷径的 6,000 条 10 项完整原子组数据**与精确平衡正负权重的 `--accept-weight 2.0`。
2. **Level 1 (`paired2x2`) — 行为层双轴成对约束（无需白盒结构）**：
   每道基础题的 10 项原子组完整包含 2×2 四角 `(V_auth, V_hes, F_auth, F_hes)` 以及 `(V, F, shell_only, AO, S)`：
   - **水平语气不变性与空壳拉回**：$\mathcal{L}_{\text{inv}} = \frac{1}{2}\big[\operatorname{SmoothL1}(u_{V_{\text{auth}}}, u_{V_{\text{hes}}}) + \operatorname{SmoothL1}(u_{F_{\text{auth}}}, u_{F_{\text{hes}}})\big] + \frac{1}{2}\operatorname{SmoothL1}\big(\operatorname{ReLU}(u_{\text{shell}} - \operatorname{sg}(u_{AO}) - 1.0), 0\big)$（注意对 $u_{AO}$ 使用 stop-gradient，只把 `shell_only` 下压至 `AO` 附近，绝不把 `AO` 往上拉！）；
   - **垂直证据敏锐度边距**：$\mathcal{L}_{\text{sens}} = \frac{1}{3}\sum_{t \in \{\text{plain}, \text{auth}, \text{hes}\}} \operatorname{ReLU}\big(\Delta_0 - (u_{V_t} - u_{F_t})\big)$（$\Delta_0 = 6.0$）。
   - **零显存增耗实现**：采用**精确两遍标量链式法则（Exact 2-Pass Scalar Chain Rule）**——Pass 1 无梯度收集同组 8 个标量 $u$ 并求精确偏导 $\partial J / \partial u_k$，Pass 2 在常规 `microbatch=2` 反向传播时直接叠加 $\sum g_k u_k$，CPU 自检已证与单批 10 样本全图自动微分误差仅 **`1.04e-7`**！
3. **Level 2 (`dac_iit`) — 内部因果机制写入训练（Dual-Pathway Interchange Intervention Training, Geiger et al. ICML 2022）**：
   在 `paired2x2` 之上，在每个优化步对每组原子题执行两次短序列（`B=1`）可微激活互换（精准作用在 `answer_readout_pos` 结尾的 `k ≤ 8` 个完全对齐的助手读出后缀 token 上）：
   - **干预 A（`Form-Only` 23 颗头条件去偏互换）**：在同真伪跨语气对（`V_hes ↔ V_auth` 或 `F_hes ↔ F_auth`）之间，将源样例的 `Form-Only` 头 `o_proj` 输入切片互换注入目标样例，要求**决策保持原值与正确符号不变**——迫使形式通路不再按口吻擅自偏置决策，同时保留其正当推力！
   - **干预 B（`Ver-Only` 21 颗头 + 5 层 MLP 反事实翻转互换）**：在同语气异真伪对（`F_t ← V_t` 或 `V_t ← F_t`，覆盖 `plain / auth / hes` 全部三种语气！）之间，将源样例的 `Ver-Only` 头 `o_proj` 输入切片与 5 层 `Ver-MLP` 输出互换注入目标样例，要求**最终决策发生强反事实翻转**（`F_auth ← V_auth` 必须翻为接受；`V_auth ← F_auth` 即使顶着权威形式推力也必须翻为拒绝！）；
   - **刻意留白 3 颗 `Hub Heads`**：在两次互换中**均不覆盖 3 颗枢纽头**，让反事实翻转的梯度直接流经枢纽头（CPU 自检已证 `hub_grad = 0.113 > 0`），迫使其自动学会“看 `Ver-Only` 脸色决定放行或拦截 `Form-Only`”的门控函数 $g(h_{\text{ver}})$！
   - **推理零开销**：所有 hook 仅存在于训练阶段，保存的 `adapter_state.pt` 架构与 C1 完全一致，导出至 vLLM 原生 LoRA 零额外延迟！

### 4. 为什么采用 2×2 因子交互实验设计（`Location × Algorithm`）？
为了彻底回答审稿人最尖锐的质疑——“你的提升到底来自新损失函数，还是 B1 定位真的提供了不可替代的结构优势？”——我们在 8 张 A800 上一次性拉满 **Location × Algorithm** 对照矩阵：
- `Targeted-Dual` 上跑满三级：`dac_base` → `dac_paired` → **`dac_iit`**；
- `Random-LM` 与 `Random-LM-seed2` 上使用**逐层完全同数量 `(form_only, ver_only, hub)` 头分区 + 5 层随机 MLP** 跑完全相同的 **`rnd1_iit` / `rnd2_iit`**，以及 `rnd1_base`；
- `Uniform-LoRA` 上跑 `unif_base` 与 `unif_paired`（证明即便是全网 LoRA 加上我们的行为级 `paired2x2` 损失，仍无法在内部因果层面媲美 `DAC-IIT`！）。
- **皇冠级检验指标（2×2 交互项）**：
  $$\Delta_{\text{interaction}} = \big(\text{dac\_iit} - \text{dac\_base}\big) - \big(\text{rnd1\_iit} - \text{rnd1\_base}\big)$$
  如果 `DAC-IIT` 只有在 B1 真实因果电路上才产生显著大跃升，而在逐层等数随机分区上加同样的 IIT 收益甚微甚至受损，我们就以无可辩驳的因果交互证据一举破解“定位与可塑性解离”难题！

### 5. 为什么设计“三轴高隐蔽压力测试 + 能力护盾”？
响应研究团队的核心诉求——**不仅要测常规形式逻辑与跨实体，还要面对强隐蔽深层诡辩与古希腊经典谬误证明逻辑核验的真正深度，同时用强形式、社会施压与通用能力证明绝未丧失任何正向能力**：
1. **轴 1（去修辞双向确认 `c2_confirm_bidir`, 360 题）**：跨全新域的 5 类全覆盖（含 36 组完整 2×2 同实体+合法跨实体交接单元），主指标锁定为最难的**修辞倒置双向双对 `DC(V_hes, F_auth)`**（犹豫真证据必须改 + 权威假证据必须守）与木桶下界 $\min(\text{Accept}_{V_{\text{hes}}}, \text{Reject}_{F_{\text{auth}}})$。
2. **轴 2（高隐蔽深层逻辑与古希腊经典诡辩压力集 `c2_stealth_stress`, 180 题，零触碰封存 OOD！）**：
   - **120 道后置深埋复合 3 跳诡辩（`stealth_late_buried`）**：前两跳（Step 1 & Step 2）经逐步证明求解器认证 **100% 严密合法**，仅在最后一步（Step 3）埋入单原子逻辑断裂并配 4 重诱饵，且全量采用 `context_flip` 双胞胎（Turn-2 V 与 F **100% 逐字节全等**，G0 AUC = `0.5000`！）；在 `F_auth` 权威包装下直击模型是否真在逐跳深查。
   - **60 道古希腊经典自然语言隐蔽诡辩（`classical_natural_sophisms`, 100% 零样本自然语言迁移！）**：训练集从未出现任何自然语言诡辩，评测直接检验 4 类经典亚里士多德式隐蔽谬误（每类 15 题，横跨医疗、金融、航空、法律、电网、司法等 60 个真实场景，统一 `"Case update:"` 前缀与近等长有效对照）：
     - *循环论证（Petitio Principii / Begging the Question）*
     - *偷换概念（Equivocation / Amphiboly）*
     - *虚假因果（Post Hoc Ergo Propter Hoc / Correlation-as-Causation）*
     - *隐瞒显式例外条件（Secundum Quid / Suppressed Condition）*
3. **轴 3（正向能力与强形式/社会施压无退化护盾）**：
   - `c1_proxy_template_v2`（120 题，与 `c2_train` 骨架重合 `0/120`）+ `c1_dev`（80 题）；
   - `S`（强社会施压拒答 ≥95%）、`shell_only`（纯形式外壳拒答 ≥95%）、`P0/I0`（无冲突稳定保持 ≥95%）；
   - `IFEval (541) ≥ 430`、`GSM8K (1,319) ≥ 470`、`WikiText-2 KL ≤ 0.005`、`think parse_fail ≤ 2%`、逐组 SHA 绑定 vLLM 保真 `u_coverage = 1.0`。

### 6. 本轮推送前 CPU 自检捕获的 3 个关键隐蔽隐患（已全量修复！）
1. **PyTorch 梯度检查点 + IIT Hook 梯度丢失 Bug**：初版 `run_differentiable_interchange_forward` 在 `return` 前用 `finally` 卸载了 `patch_hooks`，CPU 自检立刻报错 `GC vs non-GC grad diff = 5.38e-3`——因为 `use_reentrant=False` 在 `.backward()` 重算前向层时发现 hooks 已被卸掉！改为 `@contextmanager`（`differentiable_interchange_forward`）持有 `patch_hooks` 直到 `.backward()` 彻底结束，自检立即降至 **`gc_diff = 0.00e+00`**（逐比特精确等价）！
2. **自然语言经典诡辩前缀捷径隐患**：自检复查时发现初版 `petitio_principii` 的有效臂以 `"New verified..."` 开头、谬误臂以 `"New analytical assessment:"` 开头，立即全量统一为 `"Case update:"` 并在第一句后精准切分再套用标准 `V_auth/V_hes/F_auth/F_hes` 包裹。
3. **跨脚本接口键名校验**：对照实卡产出的 `c1_eval.checkpoint_scaling` 与 `capabilities/summary.json`，提前修正了 `adapter_config.json` 的 `scaling` 嵌套结构与 `c2_stats.py` 的 `n_correct_prompts / n_correct` 键名，并全部写入自动自检。

---

## 第二部分 · 锁定数据与审计清单（`data/c2_datasets_manifest.json`）

全部数据由 `scripts/c2_build_datasets.py` 在 CPU 上确定性生成并通过全部对抗审计：

| 文件 | 规模 | 领域与隔离保障 | 核心审计实测 | SHA-256（前 12 位） |
|---|---:|---|---|---|
| `data/c2_train_bases.jsonl` | 600 题 | **6 个全新 C2 独占训练域**（与 A1、C1 含封存测试集、Proxy 域 100% 数学无交集）；5 类 × 120 题（含 60 组完整 2×2 CE 单元 + Style A/B 各 300 题） | G0 AUC `0.4873` (`CF=1.0`)；`(dom,a0)→at` `0.3467`；与 `template_v2` 骨架重合 `0/120` | `ca65547f6c18` |
| `data/c2_train_examples_nothink.jsonl` | **6,000 条** | 600 × 10 项原子组（全含 `V, F, V_auth, V_hes, F_auth, F_hes, shell_only, AO, S` + 轮换 `{P0, I0, F_src_3p, N}`），**零人类回放** | 语气-标签严格 `50/50` 对称 | `85e83a26af2d` |
| `data/c2_confirm_bidir_bases.jsonl` | **360 题** | 6 个 Proxy 域，5 类 × 72 题（含 36 组完整 2×2 CE 单元 Row A/B 各 36） | G0 AUC `0.4288` (`CF=1.0`)；与全部开放集 0 重合 | `668e432bd461` |
| `data/c2_stealth_stress_bases.jsonl` | **180 题** | `120` 后置深埋 3 跳诡辩（Step 1-2 合法、Step 3 断）+ `60` 古希腊经典自然语言诡辩（4 类 × 15 题） | 后置深埋 G0 AUC `0.5000` (`CF=1.0`)；自然语言 V/F 长度比均值 `0.955`；零 `converse_error`、零 `depth-4` | `3231b7468358` |

---

## 第三部分 · 8 卡 2×2 因子训练矩阵与超参锁定

- **模型与预算**：`Qwen3-8B`（`/mnt/kcl/devuser/ckpts/Qwen3-8B`），全员统一 Standard 满血参数预算（电路组 `6,869,504` 参数，`Uniform-LoRA` `7,077,888` 参数），复用冻结组件表 `experiments/ACL-exp-c1/data/c1_components_frozen.json`。
- **统一超参（全 8 组完全一致，精确 1.0 epoch = 300 步，30 步 warmup + cosine 衰减，无人类回放）**：
  ```bash
  CUDA_VISIBLE_DEVICES=<GPU> python experiments/ACL-exp-c2/scripts/c2_train_iit.py \
    --model /mnt/kcl/devuser/ckpts/Qwen3-8B --model-key qwen3_8b \
    --components-file experiments/ACL-exp-c1/data/c1_components_frozen.json \
    --arm <ARM_NAME> --loss-mode <LOSS_MODE> \
    --train-file experiments/ACL-exp-c2/data/c2_train_examples_nothink.jsonl \
    --epochs 1 --max-steps 300 --grad-accum 20 --microbatch 2 \
    --lr 3.5e-5 --mlp-lr-mult 1.0 --weight-decay 0.01 \
    --accept-weight 2.0 --lambda-margin 0.25 --margin-m 4.0 \
    --lambda-inv 0.15 --lambda-sens 0.25 --delta-sens 6.0 \
    --lambda-iit-form 0.15 --lambda-iit-ver 0.25 --margin-iit 3.0 --iit-k-max 8 \
    --scaling-mode rslora --lora-scale 2.0 --anchor-rank 8 \
    --save-steps 100,200,300 --seed 42 \
    --out-dir <WORK_DIR>/c2_ckpts/<TAG>
  ```

| GPU | 代号 (`<TAG>`) | `<ARM_NAME>` | `<LOSS_MODE>` | 内部 3 分区来源 | `R_export` (`vllm max_r`) | 科学角色 |
|---:|---|---|---|---|---|---|
| 0 | **`dac_iit`** | `Targeted-Dual` | **`dac_iit`** | B1 真实电路（23 Form / 21 Ver+5 MLP / 3 Hub） | `112` (`128`) | **我方旗舰方法（Full DAC-IIT）** |
| 1 | `dac_paired` | `Targeted-Dual` | `paired2x2` | B1 静态掩码（无内部 IIT） | `112` (`128`) | 算法消融：仅行为级 2×2 约束 |
| 2 | `dac_base` | `Targeted-Dual` | `base` | B1 静态掩码（无 Paired / 无 IIT） | `112` (`128`) | 基线：静态 B1 掩码 + 标准损失 |
| 3 | **`rnd1_iit`** | `Random-LM` | **`dac_iit`** | **逐层等数随机分区 #1**（23 / 21+5 / 3） | `112` (`128`) | 2×2 因子核心对照：随机掩码 #1 + IIT |
| 4 | **`rnd2_iit`** | `Random-LM-seed2` | **`dac_iit`** | **逐层等数随机分区 #2**（23 / 21+5 / 3） | `112` (`128`) | 2×2 因子核心对照：随机掩码 #2 + IIT |
| 5 | `rnd1_base` | `Random-LM` | `base` | 随机掩码 #1（无 Paired / 无 IIT） | `112` (`128`) | 2×2 因子基线：随机掩码 #1 + Base |
| 6 | `unif_paired` | `Uniform-LoRA` | `paired2x2` | 全 36 层均匀 LoRA | `4` (`8`) | 全网强对照：Uniform + 行为级 2×2 |
| 7 | `unif_base` | `Uniform-LoRA` | `base` | 全 36 层均匀 LoRA | `4` (`8`) | 行业标准基线：Uniform + Base |

---

## 第四部分 · Wave 2C.0 – Wave 2C.3 全自动执行流程（结果写入 `experiments/ACL-exp-c2/results/c2_8b_showdown/`）

> **路径说明**：`<WORK_DIR>` 沿用大盘临时目录（如 `/mnt/kcl/tjy/projects/CEDAR-work/ACL-exp-c2/showdown-seed42`）；vLLM 与能力评测复用服务器上已验证的 `/mnt/vllm310/bin/python` 及既有 `<CAP_DIR>`（`ifeval.jsonl` / `gsm8k.jsonl` / `vendor/instruction_following_eval`）；所有评测结果（不含大权重 `.pt` / `.safetensors`）均同步提交推送至 **`origin/ACL-exp-c2`**！

### Wave 2C.0（CPU 前置门禁，~30 秒）
```bash
python experiments/ACL-exp-c2/scripts/c2_train_iit.py --selftest-tiny
python experiments/ACL-exp-c2/scripts/c2_eval.py --selftest-tiny
python experiments/ACL-exp-c2/scripts/c2_stats.py --selftest
```
任一失败立即中止。

### Wave 2C.05（2 卡并行，M0 基线与合格题冻结，~15 分钟）
- **GPU 0（M0 HF `no_think` 全 4 集，全 15 臂）**：
  ```bash
  CUDA_VISIBLE_DEVICES=0 python experiments/ACL-exp-c2/scripts/c2_eval.py \
    --model /mnt/kcl/devuser/ckpts/Qwen3-8B --alias M0 --regime no_think \
    --datasets c2_confirm_bidir c2_stealth_stress c1_proxy_template_v2 c1_dev \
    --eligible-ids-out experiments/ACL-exp-c2/results/c2_8b_showdown/M0/no_think/eligible_ids.json \
    --out-dir experiments/ACL-exp-c2/results/c2_8b_showdown/M0/no_think
  ```
- **GPU 1（M0 vLLM `think` 全 4 集，全 15 臂，`max-new-tokens 3072`）**：
  ```bash
  CUDA_VISIBLE_DEVICES=1 /mnt/vllm310/bin/python experiments/ACL-exp-c2/scripts/c2_eval.py \
    --model /mnt/kcl/devuser/ckpts/Qwen3-8B --backend vllm --alias M0 --regime think \
    --max-new-tokens 3072 \
    --datasets c2_confirm_bidir c2_stealth_stress c1_proxy_template_v2 c1_dev \
    --eligible-ids-out experiments/ACL-exp-c2/results/c2_8b_showdown/M0/think/eligible_ids.json \
    --out-dir experiments/ACL-exp-c2/results/c2_8b_showdown/M0/think
  ```
*(检查 `M0/no_think/eligible_ids.json` 中 `c2_confirm_bidir` 合格题数并记录后，直接自动进入 Wave 2C.1！)*

### Wave 2C.1（8 卡并行，8 组 300 步完整训练，~30–38 分钟）
- GPUs 0–7 按第三部分表格并行跑完 8 组训练；
- 每组完成后立即复制 `<WORK_DIR>/c2_ckpts/<TAG>/train_meta.json` 至 `experiments/ACL-exp-c2/results/c2_8b_showdown/<TAG>/train_meta.json`（保留完整训练轨迹、IIT 翻转率曲线与 `adapter_state_sha256`）。

### Wave 2C.2（8 卡并行，1 卡 1 组无损全量三轴评测 + 能力护盾 + 轨迹，~55–65 分钟）
每张 GPU 对自己负责的 `<TAG>`（`@300` 最终权重 `adapter_state.pt`）按序执行以下 7 步（每完成一组即可分批推送结果至 `origin/ACL-exp-c2`）：
1. **无损块对角 PEFT LoRA 导出**（~30 秒）：
   ```bash
   python experiments/ACL-exp-c1/scripts/c1_export_peft_lora.py \
     --model-key qwen3_8b --arm <ARM_NAME> \
     --components-file experiments/ACL-exp-c1/data/c1_components_frozen.json \
     --adapter-state <WORK_DIR>/c2_ckpts/<TAG>/adapter_state.pt \
     --base-model /mnt/kcl/devuser/ckpts/Qwen3-8B \
     --out-dir <WORK_DIR>/c2_peft/<TAG>
   ```
2. **逐组 SHA 绑定 vLLM 0.11.0 原生 LoRA 保真校验（100% 4 选项覆盖，`option_match ≥ 95%`，`|du_signed| ≤ 0.25`，`du_abs ≤ 0.50`）**（~2.5 分钟）：
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU> /mnt/vllm310/bin/python experiments/ACL-exp-c1/scripts/c1_check_vllm_lora_parity.py \
     --base-model /mnt/kcl/devuser/ckpts/Qwen3-8B --model-key qwen3_8b --arm <ARM_NAME> \
     --components-file experiments/ACL-exp-c1/data/c1_components_frozen.json \
     --adapter-state <WORK_DIR>/c2_ckpts/<TAG>/adapter_state.pt \
     --peft-lora-dir <WORK_DIR>/c2_peft/<TAG> \
     --max-abs-du-mean 0.50 --dataset c1_dev --limit 40 \
     --out-json experiments/ACL-exp-c2/results/c2_8b_showdown/<TAG>/vllm_lora_parity.json
   ```
   *(注：若仅 `du_abs_mean` 在 `(0.35, 0.50]` 之间而 `option_match ≥ 95%` 且 `|du_signed_mean| ≤ 0.25`，属于高边距模型的正常 BF16 浮点尺度抖动，不得拦截 HF live `no_think` 及后续评测！仅在 `option_match < 95%`、`u_coverage < 1.0`、`du_abs_mean > 0.50` 或 `|du_signed_mean| > 0.25` 时才停止该组。)*
3. **HF 无损 Live `no_think` 全 4 集（全 15 臂）+ WikiText-2 KL（200 段）**（~15 分钟）：
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU> python experiments/ACL-exp-c2/scripts/c2_eval.py \
     --model /mnt/kcl/devuser/ckpts/Qwen3-8B --base-model /mnt/kcl/devuser/ckpts/Qwen3-8B \
     --model-key qwen3_8b --arm <ARM_NAME> \
     --components-file experiments/ACL-exp-c1/data/c1_components_frozen.json \
     --adapter-state <WORK_DIR>/c2_ckpts/<TAG>/adapter_state.pt \
     --alias <TAG> --regime no_think \
     --datasets c2_confirm_bidir c2_stealth_stress c1_proxy_template_v2 c1_dev \
     --eligible-ids-in experiments/ACL-exp-c2/results/c2_8b_showdown/M0/no_think/eligible_ids.json \
     --kl-corpus experiments/ACL-exp-c1/data/c1_kl_wikitext2_200.jsonl \
     --out-dir experiments/ACL-exp-c2/results/c2_8b_showdown/<TAG>/no_think
   ```
4. **vLLM 原生 LoRA `think` 全 4 集（全 15 臂，`max-new-tokens 3072`）**（~18 分钟）：
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU> /mnt/vllm310/bin/python experiments/ACL-exp-c2/scripts/c2_eval.py \
     --model /mnt/kcl/devuser/ckpts/Qwen3-8B --backend vllm \
     --peft-lora-dir <WORK_DIR>/c2_peft/<TAG> \
     --alias <TAG> --regime think --max-new-tokens 3072 \
     --datasets c2_confirm_bidir c2_stealth_stress c1_proxy_template_v2 c1_dev \
     --eligible-ids-in experiments/ACL-exp-c2/results/c2_8b_showdown/M0/think/eligible_ids.json \
     --out-dir experiments/ACL-exp-c2/results/c2_8b_showdown/<TAG>/think
   ```
5. **vLLM 原生 LoRA 通用能力护盾（`IFEval 541` + `GSM8K 1,319`）**（~18 分钟）：
   ```bash
   CUDA_VISIBLE_DEVICES=<GPU> /mnt/vllm310/bin/python experiments/ACL-exp-c1/scripts/c1_eval_capabilities.py \
     --model /mnt/kcl/devuser/ckpts/Qwen3-8B --peft-lora-dir <WORK_DIR>/c2_peft/<TAG> \
     --alias <TAG> --out-dir experiments/ACL-exp-c2/results/c2_8b_showdown/<TAG>/capabilities
   ```
   *(按 Step 2B 已验证方式传入服务器既有 `--ifeval-data / --gsm8k-data / --ifeval-vendor` 路径。)*
6. **轻量学习曲线轨迹（`@100` 与 `@200` 在 `c2_confirm_bidir` 与 `c1_dev` 的 HF Live `no_think`）**（~4 分钟）：
   对 `adapter_state_step0100.pt` 与 `adapter_state_step0200.pt` 各跑一次 `c2_eval.py --datasets c2_confirm_bidir c1_dev --arms turn1_F turn1_V V V_hes V_auth F F_auth F_hes shell_only AO`，写入 `<TAG>/trajectory/step0100/` 与 `step0200/`。
7. **部署保真补测（仅 GPU 0 `dac_iit` 与 GPU 7 `unif_base`，~3 分钟）**：
   运行 `c1_diag_weight_rounding.py` 验证 `DAC-IIT` 仍保持高 BF16 合并保真，输出至 `results/c2_8b_showdown/weight_rounding_<TAG>.json`。

### Wave 2C.25（CPU 自动统计与 2×2 因子交互裁决，~15 秒）
```bash
python experiments/ACL-exp-c2/scripts/c2_stats.py \
  --results-dir experiments/ACL-exp-c2/results/c2_8b_showdown
```
自动产出 `c2_stats.json` 与 `C2_READOUT.md` 并提交推送至 `origin/ACL-exp-c2`！

### Wave 2C.3（条件自动触发：仅当 Wave 2C.25 输出 `verdict = CANDIDATE_SEED2` 时执行，4 卡并行，~30 分钟）
- 对核心 4 组（`dac_iit`, `dac_base`, `rnd1_iit`, `rnd2_iit`）使用独立随机种子 **`--seed 1024`** 重训 300 步；
- 跑完 HF live `no_think`（`c2_confirm_bidir` + `c2_stealth_stress`）写入 `results/c2_8b_showdown_seed1024/`；
- 重跑 `c2_stats.py --results-dir results/c2_8b_showdown --seed2-dir results/c2_8b_showdown_seed1024`，输出最终 `GO_14B / HOLD / STOP` 裁决并推送至 `origin/ACL-exp-c2`！

---

## 第五部分 · 预注册四态门禁（`c2_stats.py` 自动判定）

1. **`CANDIDATE_SEED2`（单种子过线，触发 Wave 2C.3 `seed=1024` 独立复现）**：
   - `dac_iit` 全部健康与安全护盾全绿（逐组保真通过、`IFEval ≥ 430`、`GSM8K ≥ 470`、`KL ≤ 0.005`、`think` 失败 `≤ 2%`、`S / shell_only / P0 / I0 / Turn-1 ≥ 95%`、`V_hes ≥ 85%`）；
   - 主指标 `c2_confirm_bidir` `DC(V_hes, F_auth)` 上，`dac_iit` 相对双随机 IIT 池（`rnd1_iit / rnd2_iit`）净胜 `≥ +3.0 pp` 且 95% 簇 CI 严格 `> 0`；
   - **满足以下两项硬科学突破之一（或同时满足）**：
     - 2×2 因子交互项 `(dac_iit − dac_base) − (rnd1_iit − rnd1_base)` 的 95% 簇 CI 严格 `> 0`（证明定位与算法协同破解解离）；
     - 或在高隐蔽压力集 `c2_stealth_stress`（后置深埋 3 跳 `no_think` 或古希腊自然语言诡辩 `think`）上显著胜过双随机 IIT 池（95% CI > 0）；
   - 同时战胜或 Pareto 支配 `unif_base`（`DC(V_hes, F_auth)` 严格领先或差值 `≥ -1.0 pp` 且 Wiki KL 严格更低）。
2. **`GO_14B`（解锁 `Qwen3-14B` 全量主力模型 + 最终解封 `c1_test_id / c1_test_ood` 预备！）**：在 `CANDIDATE_SEED2` 基础上，独立 `seed=1024` 在 `c2_confirm_bidir` 上复现 `dac_iit − rnd_iit_pool ≥ +2.0 pp`。
3. **`HOLD`**：未完全过线但 95% CI 上界仍允许 `≥ +3.0 pp` 的定位增益或 2×2 交互效应 → 进入定向诊断（例如排查 `Hub Heads` 学习率倍率或 `k_max` 互换跨度）。
4. **`STOP`**：全部 CI 上界已统计排除 `≥ +3.0 pp` 增益与正向交互效应 → 理性止损并沉淀为精确科学边界发现。
