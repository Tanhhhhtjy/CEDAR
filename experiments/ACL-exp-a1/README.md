# `ACL-exp-a1` — Week 1 Experiment A1 Handoff Guide (Data Certification, Artifact Audit & `0A'` Behavioral Calibration)

本分支对应 **ACL 2027 转型后的第一个正式前置实验（Experiment A1 / `0A'`）**，采用 **“一分支一实验”** 协作模式：本分支已冻结实验设计、认证数据、伪迹审计报告与回传格式规范；实验同学完成 GPU 运行后，请直接将结果提交并推送回 **`ACL-exp-a1`** 分支。

---

## 一、快速了解：这个实验要解决什么问题？

在投入白盒机制解剖（`ACL-exp-b1`）和反事实微调（`ACL-exp-c1`）之前，A1 要用 **5–7 小时 4×A800 机时** 把四个生死门禁一次测清：
1. **数据零伪迹确认（`G0`，已在 CPU 完成并通过！）**：引入 **语境翻转双胞胎（Context-Flip Twins）**（$V/F$ 第二轮用户论证文本逐字 100% 相同，只改第一轮一条规则），经仅论证分类器审计 AUC = **`0.5000`**（`argument_flip` 去泄漏后 AUC = **`0.4700`**）。
2. **模型能力与难度定标（`G1`）**：在 `Qwen3-8B`、`Qwen3-14B`、`Gemma-3-12B-it`（⚠️ 禁用 Gemma-2，因其 logit soft-capping 破坏线性分析）上，验证 2 步与 3 步题目的**单轮配对核验准确率 $\ge 75\%$** 且**两轮谬误 $F$ 跟随率 $\ge 25\%$、$V$ 接受率 $\ge 70\%$**，用 **Split-Half（$K=8$：前 4 次划分集合，后 4 次评估）** 无偏切出 **集合 I / 集合 II / 集合 D**。
3. **杀手基线压测（`G2`）**：实测 **两次调用管线（先单轮中立核验 $\to$ 再带回两轮作答）** 与 **格式匹配 `Verify-then-Revise` 提示词** 能否在集合 I 上直接闭合差距，决定后续论文叙事走“直接胜出”还是“单次调用内化摊销”。
4. **谄媚 vs. 核验失败识别（`G3`）**：通过 **修辞 × 有效性析因**（`{权威, 犹豫} × {V, F}`）、**来源操纵**（同一段 $F$ 分别标为 `用户 / 第三方日志 / 模型自己草稿`）与 **`shell_only` 纯外壳对照臂**，精确分离“社交谄媚”与“纯逻辑核验缺失”。

---

## 二、目录结构一览

```text
experiments/ACL-exp-a1/
├── README.md                              # 本快速指南
├── PROTOCOL.md                            # 完整冻结实验协议（必读！含所有 Prompt 模板、采样参数、门禁公式）
├── configs/
│   └── exp_a1_config.yaml                 # 冻结机器可读配置（模型、采样 K=8、19+ 实验臂、G0-G4 门禁）
├── scripts/
│   ├── build_symbolic_pilot_v2.py         # 120 题符号试点集生成器（60 context_flip + 60 argument_flip，双态逻辑认证）
│   └── audit_argument_only_artifacts.py   # 仅论证伪迹审计脚本（5 折 GroupKFold Word+Char N-gram LR）
├── data/
│   ├── symbolic_pilot_120_dev_only.jsonl  # 【已生成并校验】120 题 dev-only 认证符号集（严禁进入正式测试集！）
│   ├── symbolic_pilot_120_manifest.json   # 数据统计、7 项形式化认证断言与 SHA-256 哈希
│   ├── artifact_audit_report.json         # 【已生成】伪迹审计报告（context_flip=0.5000, argument_flip=0.4700）
│   └── natural_pilot_seed_12.jsonl        # 12 题高质量自然语言种子集（6 context_flip + 6 argument_flip，可扩至 60 题）
├── schemas/
│   └── RESULTS_RETURN_CONTRACT.md         # 实验结果回传目录结构、JSONL 字段规范与 Git 推送检查单
└── results/                               # 实验同学跑完后将结果放回此处并 push 到 ACL-exp-a1
```

---

## 三、实验同学执行步骤（Step-by-Step）

### Step 1：拉取分支并核验数据哈希（1 分钟）
```bash
git fetch origin
git checkout ACL-exp-a1
python experiments/ACL-exp-a1/scripts/audit_argument_only_artifacts.py
```
确认 `symbolic_pilot_120_dev_only.jsonl` 的 SHA-256 与 `configs/exp_a1_config.yaml` 一致，且审计输出 `context_flip AUC = 0.5000`。

### Step 2：（可选但推荐）扩充自然语言试点集（12 题 $\to$ 60 题）
- 参考 `data/natural_pilot_seed_12.jsonl`（特别是 `NAT-002..012` 的 override 规则写法，保证 $F$ 条件下 $a_0$ 依然严格成立），补齐至 60 题 `data/natural_pilot_60_dev_only.jsonl`（30 `context_flip` + 30 `argument_flip`），并跑一遍伪迹审计确认 `context_flip AUC <= 0.55`。
- *如果希望先快速看符号集结果，也可先直接跑 `symbolic_pilot_120_dev_only.jsonl` + `natural_pilot_seed_12.jsonl`，再补跑其余 48 题。*

### Step 3：按 `PROTOCOL.md` 运行 `0A'` GPU 推理（4 × A800 40G，vLLM BF16）
请仔细核对 `PROTOCOL.md` §3–§4 的四条硬要求：
1. **模型顺序**：先跑 **`Qwen/Qwen3-8B`**（`no_think` 约 20 分钟 + `think` 约 1.5 小时），再跑 **`Qwen/Qwen3-14B`**（`no_think` 约 30 分钟 + `think` 约 2.5 小时），最后跑 **`google/gemma-3-12b-it`**（无 soft-capping）。
2. **采样与 Split-Half**：每题每臂 **$K = 8$ 次采样**（`temp=0.6, top_p=0.95`，种子见 `exp_a1_config.yaml`），前 4 次（`0..3`）仅用于定义集合 I/II/D，后 4 次（`4..7`）用于评估汇报；`no_think` 额外记录 1 次 greedy 与**首字 `VALID/INVALID` 及 `Final:` 选项 logits**。
3. **Qwen3 多轮模板与 ITT 纪律**：Turn-1 历史剥离 `<think>`；**严禁静默丢弃格式失败或截断样本**，一律按 ITT 四分类记账。
4. **Module 5 必跑项**：务必包含 `standalone_neutral_{V,F}`、`dialog_explicit_verify_{V,F}`、`two_call_pipeline_{V,F}` 和 `verify_then_revise_prompt_{V,F}`。

### Step 4：汇总指标并推送结果回 `ACL-exp-a1`
按照 `schemas/RESULTS_RETURN_CONTRACT.md` 生成 `results/run_manifest.json`、`results/summary_tables/`、`results/raw_records/*.jsonl.gz` 与 `results/REPORT_A1.md`，然后：
```bash
git add experiments/ACL-exp-a1/results/
git commit -m "results(ACL-exp-a1): add 0A' behavioral calibration, factorial & killer baseline outputs"
git push origin ACL-exp-a1
```

> **重要提醒**：`ACL-exp-a1` 的结果回传并通过 `G1–G4` 门禁审计后，我们才会新建 **`ACL-exp-b1`**（轻量白盒解耦指数实验）与 **`ACL-exp-c1`**（400 对 × 3 种子微调探路实验）分支。如有任何 Prompt 细节或环境问题，请随时同步！
