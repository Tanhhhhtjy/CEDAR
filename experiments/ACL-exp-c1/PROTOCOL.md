# ACL-exp-c1 · 机制引导的双通路非对称定向微调：完整实验设计与执行计划（Draft v1.0，待三方对齐后冻结）

- **分支**：`ACL-exp-c1`（从已封箱的 `ACL-exp-b1@13b4899` 拉出）
- **上游依赖**：
  - A1 封箱结论：`origin/ACL-exp-a1@2c53088`（`experiments/ACL-exp-a1/A1_CLOSURE_SUMMARY.md`）
  - B1 封箱结论：`origin/ACL-exp-b1@13b4899`（`experiments/ACL-exp-b1/B1_CLOSURE_SUMMARY.md`）
- **核心定位**：论文第三幕（建设性破局主战局）。把 B1 解剖出的内部机制直接转化为可训练参数 `< 0.1%` 的微创手术，证明在**严格同等参数预算**下同时做到：**抗社会施压 (`S`) + 抗无关外壳 (`shell_only`) + 抗隐蔽谬误 (`F`，含未见结构 OOD) + 保留真证据与犹豫口吻真证据 (`V / V_hes`) + 极低 KL 散度与通用能力零退化**。

---

## 1. 机制动机与方法定义（`DAC-LoRA`：双通路非对称微创微调）

根据 A1 + B1 的物理模型，决策读出位的 logit 差可分解为：
$$u_\theta(x) = \frac{\Delta w_U^\top \operatorname{diag}(g)}{\operatorname{RMS}(h_L(x))} \Big( h_{\text{init}}(x) + \underbrace{h_{\text{soc+form}}(x)}_{\text{共模顺从偏置（头主导、MLP 无效）}} + \underbrace{h_{\text{ver}}(x)}_{\text{差模逻辑核验（中层头 + 5 层中层 MLP）}} \Big)$$

B1 已经证实了三条硬规律：
1. `C_form`（格式顺从）由中后层注意力头主导，其 MLP 层与随机无异；
2. `C_content_N_CF`（零词汇伪迹逻辑核验）与 `C_content_CF`（14B 对话态）由中层头与 **5 层中层 MLP**（14B：`L19,20,22,23,25`；8B：`L18,19,20,22,23`）联合承载（单靠 5 层 MLP 恢复率达 `83%–92%`），且两者双向 100% 互通；
3. 前人直接路径 `DLA(C_pinpoint)` 困在末层输出头，漏掉了中层共享枢纽头（14B：`L23H4, L24H17, L24H19, L25H15`；8B：`L19H31, L20H8, L21H19`）与全部 MLP。

因此，**`Targeted-Dual`（我方主方法）** 采用双通路非对称微创参数化（实现见 `scripts/c1_modules.py`）：
- **精确 GQA 头切片 LoRA（`HeadSliceLoRALinear`）**：仅对选中头 $h$ 的 `q_proj` 输出行切片 `[h·d_h : (h+1)·d_h, :]` 和 `o_proj` 输入列切片 `[:, h·d_h : (h+1)·d_h]` 施加低秩增量 $\frac{\alpha}{r} B_h A_h$，**严格不碰共享的 $W_K, W_V$ 及任何未选中头**（本地自检已验证合并后未选中头切片逐比特 diff 严格等于 `0.0`）。
- **选择性中层低秩 MLP LoRA（`StandardLoRALinear`）**：仅对 `C_content_N_CF` 的 Top-5 中层 MLP 层的 `gate_proj` 与 `down_proj` 施加超低秩 LoRA（`r=8`），以极少自由度打通非线性蕴含核验，规避 `Pinpoint` Table 3 全参调 MLP 破坏事实记忆的魔咒。
- **冻结其余 99.9% 参数**：Embedding、LM Head、全部 $K/V$、底层、非目标头与其余 35 层 MLP 全部冻死。
- **无损权重合并（`merge_c1_adapters`）**：训练完成后原地合并为标准 `Qwen3ForCausalLM` HF 权重（自检误差 `< 4e-7`），直接兼容 **vLLM 高速评测** 与 **`b1_run.py` 术后二次白盒 CT 复验**。

### 损失函数（SFT + 配对选择性边距损失 `WCA-Pair`）
$$\mathcal{L} = \mathcal{L}_{\text{SFT}} + \lambda_{\text{margin}} \cdot \ell_{\text{margin}}, \qquad \ell_{\text{margin}} = \begin{cases} \operatorname{ReLU}(m - u_\theta(x)) & x \in \{V, V_{\text{hes}}\} \\ \operatorname{ReLU}(m + u_\theta(x)) & x \in \{F, F_{\text{auth}}, F_{\text{src\_3p}}, \text{shell\_only}, S, AO\} \\ 0 & x \in \{P_0, I_0, N, \text{replay}\} \end{cases}$$
默认 $\lambda_{\text{margin}} = 0.25$、$m = 4.0$ logits；按 `pair_group_id` 分组采样，使每个 `grad_accum=12` 窗口内成对包含同一题的 $V / V_{\text{hes}} / F / \text{shell}$。关闭 $\lambda_{\text{margin}} = 0$ 即为 `Targeted-Dual-NoMargin`（`WCA-unpaired`）消融。

---

## 2. 实验对比组与严格参数量对齐（`data/c1_components_frozen.json`）

所有组件清单由 `scripts/c1_export_components.py` 从冻结的 B1 合格子集（`97e7cbd`）确定性导出。**6 个主对比组总可训练参数量在 14B 上对齐误差 ≤ 0.8%，在 8B 上 ≤ 3.0%**（主表按总参数量对齐；可选附录补按 rank 对齐）：

| 组别 | 所属阶段 | Qwen3-14B 组件与参数量（偏差） | Qwen3-8B 组件与参数量（偏差） | 科学对照意义 |
|---|---|---|---|---|
| **Base ($M_0$)** | 基准 | 0 | 0 | 原模型基线 |
| **Pinpoint-DLA** | 主表 | `DLA(C_pinpoint)` 20 头 @ r=50，无 MLP：`10.50M` (`+0.0%`) | 20 头 @ r=41，无 MLP：`6.93M` (`+0.8%`) | 前人 `Pinpoint Tuning` 忠实代理 |
| **Pinpoint-AtP** | 主表 | `AtP(C_pinpoint)` 20 头 @ r=50，无 MLP：`10.50M` (`+0.0%`) | 20 头 @ r=41，无 MLP：`6.93M` (`+0.8%`) | 加强版 Pinpoint（含枢纽头），堵死稻草人质疑 |
| **Targeted-Form** | 主表 | `C_form(20)∪C_soc(12)` 24 头 @ r=42，无 MLP：`10.58M` (`+0.8%`) | 25 头 @ r=33，无 MLP：`6.97M` (`+1.5%`) | 验证“仅拆外壳头、不加核验 MLP”的边界 |
| **Targeted-Dual** | **主方法** | 46 头 @ r=18 + 5 中层 MLP `[19,20,22,23,25]` @ r=8：`10.49M` (`0.0%`) | 47 头 @ r=14 + 5 MLP `[18,19,20,22,23]` @ r=8：`6.87M` (`0.0%`) | **主方法**：头压外壳 + 中层头/MLP 重耦合核验 |
| **Random-LM** | 主表 | 逐层同头数随机头(46) + 同中层带随机 5 MLP `[14,15,21,26,30]`：`10.49M` (`+0.0%`) | 逐层随机 47 头 + 中层带随机 5 MLP `[13,14,21,25,28]`：`6.87M` (`+0.0%`) | 严格位置+预算对照（排除层带偏置） |
| **Uniform-LoRA** | 主表 | 全 40 层 `q/o/gate/down` @ r=4：`10.49M` (`-0.1%`) | 全 36 层 `q/o/gate/down` @ r=4：`7.08M` (`+3.0%`) | 行业标准全网同预算基线 |
| **Hub-Only** | 消融 | 仅 4 颗共享枢纽头 `[(23,4),(24,17),(24,19),(25,15)]` @ r=18：`0.76M` | 仅 3 颗枢纽头 `[(19,31),(20,8),(21,19)]` @ r=14：`0.35M` | 检验枢纽头单独作为重耦合开关的效力 |
| **Dual-wo-Hub** | 消融 | `Targeted-Dual` 冻结 4 颗枢纽头（42 头 + 5 MLP）：`9.74M` | 冻结 3 颗枢纽头（44 头 + 5 MLP）：`6.51M` | 检验动枢纽头是否导致过度怀疑或必需重耦合 |
| **Dual-NoMargin** | 消融 | 同 `Targeted-Dual`，设 `--lambda-margin 0.0` | 同 `Targeted-Dual`，设 `--lambda-margin 0.0` | 验证配对边距损失 `WCA-Pair` 的净增益 |

---

## 3. 零泄漏训练与 Held-out / OOD 数据规划（`scripts/c1_build_datasets.py`）

- **100% 与 A1 隔离**：使用 6 个全新领域（`marine_berth`, `rail_signal`, `energy_substation`, `clinic_triage`, `water_plant`, `vault_custody`）、全新实体名/谓词/选项名及全新随机种子 `20270104`；构建脚本内置断言确认与 A1 v2.2 零实体重叠、零语境重叠、零题号重叠。
- **结构级 OOD 留出（Leave-One-Fallacy-Out + Depth-4）**：
  1. **训练集（`c1_train_bases.jsonl`，500 基础题 → `c1_train_examples_nothink.jsonl` 3,000 对话样例）**：
     - 仅含 **5 类已见谬误**（`scope_swap`, `cross_entity_transfer`, `missing_conjunct`, `negation_drop`, `threshold_violation`），深度 2/3（350 标准 + 150 难题 3 诱饵），`context_flip` 与 `argument_flip` 各 50%。
     - **严禁出现 `converse_error` 和 深度 4！**
     - 每题 6 条配对样例：接受类 2 条（`V` + `V_hes`，33.3%）、拒绝类 3 条（`F` + `shell_only` + 轮换 `S/AO/F_auth/F_src_third_party`，50.0%）、无冲突/中性确认 1 条（轮换 `P0/I0/N`，16.7%）。训练时按 `--replay-ratio 0.15` 混入外部通用指令/推理回放（如 OpenHermes / GSM8K-train 子集）。
  2. **Held-out 同分布测试集（`c1_test_id_bases.jsonl`，100 基础题）**：独立种子，同 5 类，50 标准 + 50 难题。
  3. **Held-out 结构 OOD 测试集（`c1_test_ood_bases.jsonl`，120 基础题）**：
     - `ood_converse`（60 题）：**训练从未见过的 `converse_error`**（30 深度 2 标准 + 30 深度 3 难题）；
     - `ood_depth4`（60 题）：**训练从未见过的 4 跳推理链（Depth-4）**（3 类结构谬误 + 3 诱饵规则，求解器严格认证）。
  4. **G0 仅看论证文本伪迹审计（已全部通过）**：
     - `c1_train`：`context_flip = 0.5000`, `argument_flip = 0.5005`, `overall = 0.4992`
     - `c1_test_id`：`context_flip = 0.5000`, `argument_flip = 0.5032`, `overall = 0.5019`
     - `c1_test_ood`：`context_flip = 0.5000`, `argument_flip = 0.3978`, `overall = 0.4431`

---

## 4. 三阶段执行流程与预案（4 × A800 40GB）

### Phase 0：8B 快速冒烟与配方 A/B 验证（约 1.5–2 小时，必先过门再上 14B！）
在 `Qwen3-8B` 上并行跑 4 组（4 卡每卡 1 组，每组 ~20 分钟）：`Targeted-Dual`、`Dual-wo-Hub`、`Random-LM`、`Uniform-LoRA`，并在 `Targeted-Dual` 上对比：
- **Recipe A**：纯 `no_think` 3,000 规则样例 + 15% 通用回放（含思考链样本），零样本测 `think` 迁移；
- **Recipe B**：`no_think` 3,000 样例 + ~600 条原模型自生成且经验证的 `think` 短核验链（可选，若 Recipe A 伤到 `think` 或 `GSM8K` 时启用）。

**Phase 0 体检门禁与应急预案**：
1. 检查 `train_meta.json` 的 `final_delta_norms`：确保 `mlp_delta_frob` 与 `attn_delta_frob` 同量级（若 MLP 几乎未动，调整 `--mlp-lr-mult` 如 `2.0`）。
2. 检查 **`Targeted-Dual` vs `Random-LM`**：在 `c1_test_ood` 上 `Double-Correct(V, F)` 必须领先 `Random-LM` ≥ `+0.10`（否则先查超参/定位再决定是否切 Plan B）。
3. 检查 **枢纽头防误伤**：若 `Targeted-Dual` 导致 `V_hes < 0.90` 或 `IFEval` 明显掉点，而 `Dual-wo-Hub` 保持 `V_hes ≥ 0.90` 且抗 `F` 相当，则将“冻结共享路由枢纽、仅微调入口头与核验 MLP”升级为发现与主配方！
4. 检查 **`think` 与通用能力**：若 Recipe A 导致 `think` 截断或 `GSM8K / IFEval` 掉点 > `2.0 pp`，切换 Recipe B 混合训练。

### Phase 1：全量主表（`Qwen3-14B` 主模型 + `Qwen3-8B` 复现，预计 1.5–2 天）
- **模型**：主力 `Qwen3-14B`（2 卡/组，两两并行，单组训练 ~45–60 分钟），`Qwen3-8B` 完整复现（单卡/组，~20 分钟）。
- **组别**：全部 6 组主表 + 3 组消融（核心 4 组 `Targeted-Dual / Uniform-LoRA / Random-LM / Pinpoint-DLA` 跑 2 个随机种子 `42, 1024` 报告均值±标准差）。
- **四维评测矩阵**：
  1. `c1_eval.py` 跑 `c1_test_id`、`c1_test_ood`（分报 `ood_converse` 与 `ood_depth4`）、`natural_probe`（`no_think` + `think` 双模式）及参考文本 **Next-Token KL 散度**；
  2. 公开自然语言反谄媚 OOD 评测（TruthfulQA / GSM8K-sycophancy 或 `Ma et al.` 公开自然更新集，在 GPU 机用 `lm-eval` / 现有评测脚本跑）；
  3. 通用能力防遗忘基准：**MMLU（5-shot）+ GSM8K + IFEval**（用 `lm-eval-harness` 在合并后的 `merged_hf` 上跑 vLLM 后端，全套只需数十分钟）。

### Phase 2：机制闭环复验（预计 ~1 小时 GPU，ACL 高分点睛之笔）
1. **术后二次白盒 CT 复验**：直接拿已封箱的 `experiments/ACL-exp-b1/scripts/b1_run.py --stages sanity,atp,verify` 跑在合并后的 **`Targeted-Dual`** 和 **`Pinpoint-DLA`** checkpoint 上，直接对比术前 $M_0$ vs 术后：
   - 共模外壳跳变 `u(shell_only) - u(AO)` 是否从 `+11.5` 塌缩至 `~0`；
   - 差模逻辑核验 `C_content_CF`（对话态 `context_flip`）与 `C_content_N_CF` 的效应量是否从 `+1.25` 跃升至两位数且中层 MLP 贡献保持高恢复。
2. **枢纽头注意力可视化**：在 $M_0$ 与 `Targeted-Dual` 上提取 4 颗枢纽头（`L23H4, L24H17, L24H19, L25H15`）在 `context_flip` 配对上从读出位指向 `Rule 1 / Step 1` 的注意力权重图。
3. *(可选加分项，仅在 Qwen3-14B / 8B 主表全部完成后如有余力再跑)*：`Gemma-3-12B` `no_think` 小规模跨家族复现（`Targeted-Dual` vs `Uniform-LoRA`）。

---

## 5. 本地 CPU 自检命令（已全部通过）
```bash
python experiments/ACL-exp-c1/scripts/c1_export_components.py
python experiments/ACL-exp-c1/scripts/c1_build_datasets.py
python experiments/ACL-exp-c1/scripts/c1_train.py --selftest-tiny
python experiments/ACL-exp-c1/scripts/c1_eval.py --selftest-tiny
```
