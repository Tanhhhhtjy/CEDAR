# ACL-exp-c1 · 机制引导的双通路非对称定向微调：完整实验设计与执行计划（v1.1，已纳入独立审计全量修复）

- **分支**：`ACL-exp-c1`（从已封箱的 `ACL-exp-b1@13b4899` 拉出）
- **上游依赖**：
  - A1 封箱结论：`origin/ACL-exp-a1@2c53088`（`experiments/ACL-exp-a1/A1_CLOSURE_SUMMARY.md`）
  - B1 封箱结论：`origin/ACL-exp-b1@13b4899`（`experiments/ACL-exp-b1/B1_CLOSURE_SUMMARY.md`）
- **核心定位**：论文第三幕（建设性破局主战局）。把 B1 解剖出的内部机制直接转化为可训练参数 `< 0.1%` 的微创手术，证明在**严格同等参数预算**下同时做到：**抗社会施压 (`S`) + 抗无关外壳 (`shell_only`) + 抗隐蔽谬误 (`F`，含未见结构 OOD) + 保留真证据与对称口吻真证据 (`V / V_hes / V_auth`) + 极低 WikiText-2 KL 散度与通用能力零退化**。

---

## 1. 机制动机与方法定义（`DAC-LoRA`：双通路非对称微创微调）

根据 A1 + B1 的物理模型，决策读出位的 logit 差可分解为：
$$u_\theta(x) = \frac{\Delta w_U^\top \operatorname{diag}(g)}{\operatorname{RMS}(h_L(x))} \Big( h_{\text{init}}(x) + \underbrace{h_{\text{soc+form}}(x)}_{\text{共模顺从偏置（头主导、MLP 无效）}} + \underbrace{h_{\text{ver}}(x)}_{\text{差模逻辑核验（中层头 + B1 实证 5 层中层 MLP）}} \Big)$$

B1 已经证实了三条硬规律：
1. `C_form`（格式顺从）由中后层注意力头主导，其 MLP 层与随机无异；
2. `C_content_N_CF`（零词汇伪迹逻辑核验）与 `C_content_CF`（14B 对话态）由中层头与 **B1 `mlp_B` 实证 Top-5 中层 MLP**（14B：`[19, 22, 23, 25, 26]`，$M@5=0.918$；8B：`[18, 20, 22, 23, 27]`，$M@5=0.831$）联合承载，且两者双向 100% 互通；
3. 前人直接路径 `DLA(C_pinpoint)` 困在末层输出头，漏掉了中层共享枢纽头（14B：`L23H4, L24H17, L24H19, L25H15`；8B：`L19H31, L20H8, L21H19`）与全部 MLP。

因此，**`Targeted-Dual`（我方主方法）** 采用双通路非对称微创参数化（实现见 `scripts/c1_modules.py`）：
- **精确 GQA 头切片 LoRA（`HeadSliceLoRALinear`）**：仅对选中头 $h$ 的 `q_proj` 输出行切片 `[h·d_h : (h+1)·d_h, :]` 和 `o_proj` 输入列切片 `[:, h·d_h : (h+1)·d_h]` 施加低秩增量，**严格不碰共享的 $W_K, W_V$ 及任何未选中头**（本地自检已验证合并后未选中头切片逐比特 diff 严格等于 `0.0`）。
- **选择性中层低秩 MLP LoRA（`StandardLoRALinear`）**：仅对 B1 实证的 `mlp_B` Top-5 中层 MLP 层的 `gate_proj` 与 `down_proj` 施加超低秩 LoRA（`r=8`），规避 `Pinpoint` Table 3 全参调 MLP 破坏事实记忆的问题。
- **秩稳定缩放（`rsLoRA`）**：全 arm 统一采用 $s(r) = 2\sqrt{8/r}$（在 $r=8$ 时等于 $2.0$），消除大 rank 单通路基线（`Pinpoint-DLA r=50`、`Targeted-Form r=42`）与小 rank 全网基线（`Uniform-LoRA r=4`）之间的有效步长失衡。
- **冻结其余 99.9% 参数 + 双向保存**：训练后同时保存轻量化未合并权重 **`adapter_state.pt`**（用于零重训通路消融 `--ablate-mlp` / `--ablate-hubs`）与原地无损合并的标准 HF 权重 **`merged_hf/`**（自检最大误差 `< 3e-7`，直接兼容 vLLM 与 `b1_run.py` 术后二次白盒 CT）。

### 损失函数（SFT + 有符号读出边距损失 `Signed Readout Margin Loss`）
$$\mathcal{L} = \mathcal{L}_{\text{SFT}} + \lambda_{\text{margin}} \cdot \ell_{\text{margin}}, \qquad \ell_{\text{margin}} = \begin{cases} \operatorname{ReLU}(m - u_\theta(x)) & x \in \{V, V_{\text{hes}}, V_{\text{auth}}\} \\ \operatorname{ReLU}(m + u_\theta(x)) & x \in \{F, F_{\text{hes}}, F_{\text{auth}}, F_{\text{src\_3p}}, \text{shell\_only}, S, AO\} \\ 0 & x \in \{P_0, I_0, N, \text{replay}\} \end{cases}$$
- 默认 $\lambda_{\text{margin}} = 0.25$、$m = 4.0$ logits；读出位精确锁定在 `assistant_target` 内部**最后一个 `Final:`** 之后并断言下一 token 严格等于金标选项 token（同时兼容 `no_think` 与 Recipe B `think` 多步思维链）。
- **原子成组打包 + 实际窗口归一化**：每道基础题的 6 条样例打包为不可拆分的原子组（replay 单条样本另打包为 ≤6 条的独立小组交错插入），永不跨 `grad_accum=12` 边界撕裂，并按实际窗口条数 `win_n` 归一化；`--microbatch 2/4` 采用逐样本精确 token 均值 CE（自检与 `microbatch=1` 误差 `0.00e+00`）。
- `--arm Targeted-Dual-NoMargin` 内置默认 `lambda_margin=0.0`，直接做有符号边距损失消融。

---

## 2. 实验对比组与严格参数量对齐（`data/c1_components_frozen.json`）

所有组件清单由 `scripts/c1_export_components.py` 从冻结的 B1 合格子集（`97e7cbd`）确定性导出。**主表对比组总可训练参数量在 14B 上对齐误差 ≤ 0.8%，在 8B 上 ≤ 3.0%**：

| 组别 | 所属阶段 | Qwen3-14B 组件与参数量（偏差） | Qwen3-8B 组件与参数量（偏差） | 科学对照意义 |
|---|---|---|---|---|
| **Base ($M_0$)** | 基准 | 0 | 0 | 原模型基线 |
| **Pinpoint-DLA** | 主表 | `DLA(C_pinpoint)` 20 头 @ r=50，无 MLP：`10.50M` (`+0.0%`) | 20 头 @ r=41，无 MLP：`6.93M` (`+0.8%`) | 前人 `Pinpoint Tuning` 忠实代理 |
| **Pinpoint-AtP** | 主表 | `AtP(C_pinpoint)` 20 头 @ r=50，无 MLP：`10.50M` (`+0.0%`) | 20 头 @ r=41，无 MLP：`6.93M` (`+0.8%`) | 加强版 Pinpoint（含枢纽头），堵死稻草人质疑 |
| **Targeted-Form** | 主表 | `C_form(20)∪C_soc(12)` 24 头 @ r=42，无 MLP：`10.58M` (`+0.8%`) | 25 头 @ r=33，无 MLP：`6.97M` (`+1.5%`) | 验证“仅拆外壳头、不加核验 MLP”的边界 |
| **Targeted-Dual** | **主方法** | 46 头 @ r=18 + B1 `mlp_B` 5 MLP `[19,22,23,25,26]` @ r=8：`10.49M` (`0.0%`) | 47 头 @ r=14 + B1 `mlp_B` 5 MLP `[18,20,22,23,27]` @ r=8：`6.87M` (`0.0%`) | **主方法**：头压外壳 + 中层头/B1 实证 MLP 重耦合核验 |
| **Random-LM** | 主表 | 逐层同头数随机头(46) + 中层带随机 5 MLP `[14,15,20,24,31]`：`10.49M` (`0.0%`) | 逐层随机 47 头 + 中层带随机 5 MLP `[13,14,24,28,32]`：`6.87M` (`0.0%`) | 严格位置+预算对照（随机掩码 #1，seed=2027） |
| **Random-LM-seed2** | 主表 | 独立随机掩码 #2（seed=2028，MLP `[14,16,20,28,31]`）：`10.49M` (`0.0%`) | 独立随机掩码 #2（seed=2028，MLP `[13,15,19,29,31]`）：`6.87M` (`0.0%`) | 评估随机选组件本身的结构方差 |
| **Uniform-LoRA** | 主表 | 全 40 层 `q/o/gate/down` @ r=4：`10.49M` (`-0.1%`) | 全 36 层 `q/o/gate/down` @ r=4：`7.08M` (`+3.0%`) | 行业标准全网同预算基线 |
| **Hub-Only** | 消融 | 仅 4 颗共享枢纽头 `[(23,4),(24,17),(24,19),(25,15)]` @ r=18：`0.76M` | 仅 3 颗枢纽头 `[(19,31),(20,8),(21,19)]` @ r=14：`0.35M` | 检验枢纽头单独作为重耦合开关的效力 |
| **Dual-wo-Hub** | 消融 | `Targeted-Dual` 冻结 4 颗枢纽头（42 头 + 5 MLP）：`9.74M` | 冻结 3 颗枢纽头（44 头 + 5 MLP）：`6.51M` | 检验动枢纽头是否导致过度怀疑或必需重耦合 |
| **Targeted-Dual-NoMargin** | 消融 | 同 `Targeted-Dual`，默认 `lambda_margin=0.0` | 同 `Targeted-Dual`，默认 `lambda_margin=0.0` | 验证有符号读出边距损失的净增益 |
| **零重训通路消融** | 事后消融 | 加载 `Targeted-Dual/adapter_state.pt`，分别 `--ablate-mlp` / `--ablate-hubs` | 同左 | 同一 checkpoint 上零成本量化 MLP / 枢纽头净贡献 |

---

## 3. 零泄漏、零捷径数据规划（`scripts/c1_build_datasets.py` + `c1_prepare_external.py`）

- **100% 与 A1 及跨切分隔离**：使用 6 个全新领域（`marine_berth`, `rail_signal`, `energy_substation`, `clinic_triage`, `water_plant`, `vault_custody`）、全新实体/谓词/选项及种子 `20270104`。构建脚本对 `(A1, train, dev, test_id, test_ood)` 全量执行**双向 V/F 语境去重与完整对话哈希去重**，断言 0 重叠。
- **四个严格隔离切分**：
  1. **训练集（`c1_train_bases.jsonl`，500 题 → `c1_train_examples_nothink.jsonl` 3,000 样例）**：
     - 仅含 **5 类已见谬误**（`scope_swap`, `cross_entity_transfer`, `missing_conjunct`, `negation_drop`, `threshold_violation`），深度 2/3（350 标准 + 150 难题 3 诱饵），**严禁出现 `converse_error` 与深度 4**。
     - **口吻—标签严格 50/50 对称**：每题原子组 6 条，同题内成对包含相同口吻的正反例：`V (500)` vs `F (500)`、`V_hes (250)` vs `F_hes (250)`、`V_auth (250)` vs `F_auth (250)`，外加 `shell_only (500)` 与轮换 `S (125) / P0 (125) / I0 (125) / AO (42) / F_src_third_party (42) / N (41)`。
  2. **独立开发集（`c1_dev_bases.jsonl`，80 题）**：同 5 类（40 深度 2 标准 + 40 深度 3 难题），独立种子。**Phase 0 配方/超参选择仅看 `c1_dev`，绝不触碰最终测试集！**
  3. **封存 ID 测试集（`c1_test_id_bases.jsonl`，100 题）**：同 5 类，50 标准 + 50 难题。
  4. **封存结构 OOD 测试集（`c1_test_ood_bases.jsonl`，120 题）**：
     - `ood_converse`（60 题）：训练从未见过的 `converse_error`（30 深度 2 + 30 深度 3 难题）；
     - `ood_depth4`（60 题）：**真正的 4 跳推理链（`Step 1..4 → Rule 5 override`）**，断点严格均衡分布在 **Step 1 (20) / Step 2 (20) / Step 3 (20)**，经逐步最短证明与首个无效步骤求解器逐题认证（零重言式、零早断）。
- **外部固定语料（由 `scripts/c1_prepare_external.py` 在 GPU 机一键冻结）**：
  - **15% 通用指令回放**：`HuggingFaceH4/no_robots`（`train` 分层抽样 529 条，`529 / 3529 = 14.99%`）；
  - **参考 Next-Token KL 语料**：`Salesforce/wikitext`（`wikitext-2-raw-v1/test`，200 段干净散文，截断 256 token；`c1_eval.py` 若缺失则直接报错中止，无静默回退）；
  - **公开自然语言 OOD 主评测**：Ma et al. `EX-FEVER` 测试集（1,000 题，官方评测器，报 $R_{\text{UY}}$ 与 $R_{\text{RU}}$）；A1 的 12 道 `natural_probe`（`dev_only`）仅作历史诊断项（opt-in，不进默认主评测）。
- **全量捷径与伪迹审计实测结果（`data/c1_dataset_manifest.json`，全部内置断言通过）**：
  - `(domain, a0) → at` 查表准确率：`train 0.336 / dev 0.325 / test_id 0.330 / test_ood 0.325`（理论随机 `0.333`）；
  - `cross_entity_transfer/argument_flip` 实体一致率：`train / dev / test_id / test_ood` 全切分严格 **`0.5000`**；
  - `threshold_violation/argument_flip` 单数值 ROC AUC：`train 0.5562 / dev 0.5938 / test_id 0.5750`（均 ≤ 0.60）；
  - 训练集口吻对称度：`plain / hes / auth` 接受率严格 **`0.5000`**；
  - G0 仅看 Turn-2 文本（TF-IDF 词+字 GroupKFold 双向 AUC）：`train 0.4997`、`dev 0.4981`、`test_id 0.4996`、`test_ood 0.5199`（所有切分 `context_flip` 逐字节一致率 `1.000`）。

---

## 4. 三阶段执行流程与预案（4 × A800 40GB）

### 准备步骤（5 分钟）
```bash
python experiments/ACL-exp-c1/scripts/c1_prepare_external.py
```
生成并冻结 `c1_replay_norobots_529.jsonl` 与 `c1_kl_wikitext2_200.jsonl`。并在 $M_0$（8B 与 14B）上先跑一次 `c1_eval.py --eligible-ids-out` 冻结双侧 Turn-1 合格题分母（`turn1_F == a0` 且 `turn1_V == a0`）。

### Phase 0：8B 快速冒烟与配方 A/B 选型（约 1.5–2 小时，仅看 `c1_dev`，封存测试集！）
在 `Qwen3-8B` 上并行跑 4 组（4 卡每卡 1 组，`--microbatch 2`）：`Targeted-Dual`、`Dual-wo-Hub`、`Random-LM`、`Uniform-LoRA`，对比 Recipe A（`no_think` 3,000 + `no_robots` 529）与 Recipe B（若 Recipe A 伤到 `think` 或 `GSM8K` 时补自验证短 `think` 链）。

**Phase 0 体检门禁（严格在 `c1_dev` + `IFEval/GSM8K` 上裁决，`c1_test_id / c1_test_ood` 全程封存）**：
1. 检查 `train_meta.json` 的 `final_delta_norms`：确保 `mlp_delta_frob` 与 `attn_delta_frob` 同量级（否则调 `--mlp-lr-mult`）。
2. 检查 **`Targeted-Dual` vs `Random-LM`** 在 `c1_dev`（共享 $M_0$ 合格分母）上的 `Double-Correct(V, F)` 领先 ≥ `+0.10`。
3. 检查 **枢纽头防误伤与口吻对称**：`V_hes` 与 `V_auth` 接受率均 ≥ `0.90` 且双侧 Turn-1 准确率无明显下降；若 `Targeted-Dual` 掉点而 `Dual-wo-Hub` 保持健康，则切换冻结枢纽头为主配方。
4. 检查 **`think` 与通用能力**：`c1_eval.py --regime think --backend vllm --max-new-tokens 3072` 的 `parse_fail_rate` / `truncation_rate` 与 `GSM8K / IFEval` 掉点 ≤ `2.0 pp`。

### Phase 1：全量主表（`Qwen3-14B` 主模型 + `Qwen3-8B` 复现，预计 1.5–2 天）
- **模型**：主力 `Qwen3-14B`（2 卡/组，两两并行，`--microbatch 2`），`Qwen3-8B` 完整复现（单卡/组）。
- **组别**：主表 7 组（含 `Random-LM` + `Random-LM-seed2` 双独立随机掩码）+ 3 组消融（`Hub-Only`, `Dual-wo-Hub`, `Targeted-Dual-NoMargin`）+ 2 组零重训事后通路消融（`--ablate-mlp`, `--ablate-hubs`）。核心 4 组跑 2 个训练种子 `42, 1024`。
- **统一评测矩阵**（全部同时报告共享 $M_0$ 合格分母与全集指标 + 术后 Turn-1 准确率 + `parse_fail / truncation` 率）：
  1. 封存符号基准：`c1_test_id` + `c1_test_ood`（分报 `ood_converse` 与真 4 跳 `ood_depth4`），`no_think`（HF 单次前向）+ `think`（vLLM 批量生成 `max_new_tokens=3072`）；
  2. 公开自然语言 OOD：Ma et al. `EX-FEVER`（1,000 题，$R_{\text{UY}}, R_{\text{RU}}$）；
  3. 通用能力与漂移：**MMLU（5-shot）+ GSM8K + IFEval** + 冻结 **WikiText-2 Next-Token KL 散度**。

### Phase 2：机制闭环复验（预计 ~1 小时 GPU）
1. **术后二次白盒 CT 复验**：在合并后的 `Targeted-Dual` 与 `Pinpoint-DLA` checkpoint 上运行已封箱的 `b1_run.py`（注意 14B 必须加 `--include-deferred` 才会跑 `C_content_CF`）：
   ```bash
   python experiments/ACL-exp-b1/scripts/b1_run.py \
       --model checkpoints/qwen3_14b/Targeted-Dual/merged_hf --alias qwen3_14b__Targeted-Dual \
       --stages sanity,atp,verify --include-deferred \
       --contrasts C_pinpoint C_soc C_form C_content_N_CF C_content_CF
   ```
   对比术前 $M_0$ vs 术后：共模外壳跳变 `u(shell_only) - u(AO)` 塌缩趋近 `0`，差模逻辑核验 `C_content_CF` 与 `C_content_N_CF` 跃升且 B1 `mlp_B` 贡献保持高恢复。
2. **零重训事后通路消融**：对 `Targeted-Dual` 加载 `adapter_state.pt` 分别跑 `--ablate-mlp` 与 `--ablate-hubs`，量化微调后 MLP 与枢纽头各自对 `ood_converse / ood_depth4 / shell_only` 的因果净贡献。

---

## 5. 本地 CPU 自检命令（已全部通过）
```bash
python experiments/ACL-exp-c1/scripts/c1_export_components.py
python experiments/ACL-exp-c1/scripts/c1_build_datasets.py
python experiments/ACL-exp-c1/scripts/c1_train.py --selftest-tiny
python experiments/ACL-exp-c1/scripts/c1_eval.py --selftest-tiny
python experiments/ACL-exp-c1/scripts/c1_prepare_external.py --selftest-offline
```
