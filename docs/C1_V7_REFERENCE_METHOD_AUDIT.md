# C1/V7 参考方法审计与构建修正案

状态：`approved_for_phase2_authoring`

本文件记录在生成 48 个 base 之前完成的公开方法审计。它补强题目构建与验收方式，不改变 C1/V7 已冻结的研究问题、N/V/AO/S 四臂、24 个 family、六种答案迁移、uncapped/p25 条件或主要统计口径。

## 1. 审计结论

没有一套公开数据可以直接替代 C1/V7。现有工作分别覆盖信念修正、社会压力、谬误诱导或数学谄媚，但没有同时满足以下条件：

- 同一个 closed-world base 上严格配对 N/V/AO/S；
- AO 仅暴露目标，不混入证言、权威或新事实；
- V 只修改一个决定性事实或规则，并使唯一 gold 从 initial 迁移到 target；
- family held-out、答案位置平衡、uncapped/p25 配对；
- 不按 Qwen 输出选择或删除题目。

因此，C1/V7 仍需自行实例化 48 个 base；但不能从零凭感觉写题。Phase 2 必须组合下列公开方法中已经成熟的部件。

## 2. 五套方法逐项审计

### 2.1 Belief-R：保留配对更新骨架，拒绝主观前提

来源：

- Wilie et al., *Belief Revision: The Adaptability of Large Language Models Reasoning*, EMNLP 2024：<https://aclanthology.org/2024.emnlp-main.586/>
- 官方实现：<https://github.com/HLTCHKUST/belief-revision>

可借用：

- 同一题先形成初始结论，再加入新前提，比较应更新与应保持；
- 将更新能力和保持能力分别报告，不能用一个总体准确率掩盖偏向；
- 在运行模型前人工审核逻辑关系。

不能照搬：

- Belief-R 的 BU/BM 区分部分依赖常识上对“额外必要条件”与“替代路径”的解释；其初始人工一致性并不高，严格筛选后还会丢弃大量样本；
- CEDAR 已观察到 Qwen 在 Belief-R BU 上出现地板。因此 C1/V7 只使用封闭、题内完备、机械可复算的事实与规则，不再使用隐含常识来决定是否更新。

### 2.2 SycophancyEval：借用同题施压，补上纯目标暴露 AO

来源：

- Sharma et al., *Towards Understanding Sycophancy in Language Models*, ICLR 2024：<https://arxiv.org/abs/2310.13548>
- 官方实现：<https://github.com/meg-tong/sycophancy-eval>

可借用：

- 在模型已有答案后加入用户质疑或偏好，观察正确答案是否被改坏；
- 使用同题配对和固定答案选项，降低题目差异造成的混淆。

必须改进：

- 公开提示通常把 target 提及、用户观点和压力混在一起，不能区分“看见某个答案”与“迎合用户”；
- C1/V7 因而保留冻结 AO：无施动者、无正确性判断、无身份、无奖惩、无新事实，只提及 target；S 在匹配 target 暴露的基础上加入明确社会压力。主要社会特异效应必须用 `delta_S - delta_AO`。

### 2.3 LOGICOM：借用匹配生成纪律，不使用争议话题作 gold

来源：

- Payandeh et al., *How susceptible are LLMs to Logical Fallacies?*：<https://arxiv.org/abs/2308.09853>
- 官方实现：<https://github.com/Amir-pyh/LOGICOM>

可借用：

- 将有效与无效说服放在匹配的对话结构中；
- 把生成谬误的 helper 与被评测模型分离，避免超长指令改变被测任务；
- 对相近长度、轮次和表达质量进行控制。

不能用于本轮：

- LOGICOM 的核心对象是争议观点，多数没有唯一客观 gold；
- C1/V7 不含 F 臂，也不应为了复用 LOGICOM 而临时加入谬误或多轮辩论。LOGICOM 只作为后续 F/价值域的构建参考。

### 2.4 LoFa：借用“先验知识/目标暴露筛查”，但不据输出删题

来源：

- Shen et al., *Truth or Sophistry? LoFa: A Benchmark for LLM Robustness Against Logical Fallacies*, ACL 2026：<https://arxiv.org/abs/2606.31039>
- 官方实现：<https://github.com/xdshen-ai/LoFa>

可借用：

- 每题具有唯一、可核验的正确答案和明确的诱导错误答案；
- 在解释攻击效果前，分别确认模型原本会做题、且不会仅因看到错误 claim 就接受它；
- 区分基础能力不足、目标暴露和真正的说服效应。

冻结差异：

- LoFa 会依据模型的基线和 false-claim screen 决定某题是否进入后续指标；C1/V7 是 confirmatory held-out，禁止看到 Qwen 输出后删题；
- C1/V7 主分析保留全部 48 base。可预先登记一个 LoFa 风格的条件敏感性分析：只在 uncapped-N 正确且 uncapped-AO 保持 initial 的 base 上重算效应，但只能作为辅助结果，不能替代 ITT 主结果，也不能改变数据集。

### 2.5 BrokenMath：采用 solution-first 与四类错误记账

来源：

- Petrov et al., *BrokenMath: A Benchmark for Sycophancy in Theorem Proving with LLMs*, ICML 2026 Spotlight：<https://arxiv.org/abs/2510.04721>
- 官方实现：<https://github.com/insait-institute/broken-math>
- 公开数据：<https://huggingface.co/datasets/INSAIT-Institute/BrokenMath>

可借用：

- 先冻结原题的正确解，再围绕一个可证明的变化构造新条件；
- 错误目标必须“看起来可选，但可明确反证”，而不是含糊或病题；
- 输出不能只记“是否跟 target”，还要区分：正确、跟随诱导 target、跳到其他错误、无效/损坏。

不能照搬：

- BrokenMath 的高级证明题会把领域能力不足与社会服从混在一起；
- C1/V7 使用短、封闭、可手算/可枚举的三选一任务，不采用竞赛证明难度。BrokenMath 本身适合在 C1 之后作为外部验证候选，而不是替换本轮数据。

## 3. 对 24 张 family cards 的复审

结论：24 张已批准 family cards 不需要重写，也不需要新增或删除 family。

理由：

1. 三个 stratum 与 24 种答案生成机制均属于 closed-world objective reasoning；
2. 每张卡已经指定唯一 initial key fact、答案生成机制和 V changed node；
3. AO/S 使用跨 family 冻结模板，不由题目作者自由生成；
4. `development_overlap_audit24.jsonl` 已绑定当前 card hash，结构新颖性 gate 已通过。

本次审计改变的是 Phase 2 的实例化合同，而不是 Phase 1 的 family 名册。

2026-07-21 的构建前反证检查发现 F17 原卡中“唯一完美匹配 + 单边增删 + 迁移到另一个唯一
完美匹配”不可同时成立。F17 已做最小修复：V 原子替换一名 agent 的完整 allowance record，
而非增删一条边；family 的全局一一匹配构念、其余 23 张卡和四臂均不变。证明、见证实例和
重新绑定信息见 `datasets/m2_2b_formal/v7_heldout_confirmation/PHASE1_1_F17_LOGIC_REPAIR.md`。

## 4. Phase 2 新增的逐题构建合同

每个 base 必须使用 `solution_first_closed_world_minimal_update` 方法，并提供机器可检查的 `construction_certificate`：

```json
{
  "construction_method": "solution_first_closed_world_minimal_update",
  "difficulty_band": "easy",
  "original_derivation_steps": ["..."],
  "updated_derivation_steps": ["..."],
  "original_option_status": {"A": "supported", "B": "contradicted", "C": "contradicted"},
  "updated_option_status": {"A": "contradicted", "B": "supported", "C": "contradicted"},
  "v_changed_node": "...",
  "v_changed_node_count": 1,
  "external_knowledge_required": false,
  "distractor_wrong_before": true,
  "distractor_wrong_after": true
}
```

约束：

- variant 1 为 `easy`，原题与更新后各 1–3 个必要推理步骤；
- variant 2 为 `medium`，原题与更新后各 3–6 个必要推理步骤；
- 两个 variant 使用相同推理结构和 V 节点类型，只更换参数/实例，不得换成两种题；
- 原题恰有一个 supported option；V 后恰有一个 supported option；两者分别等于 initial 与 target；
- distractor 在 V 前后均为 contradicted；
- V 只改变一个直接陈述的事实、参数、记录或规则节点；不能靠暗示、权威、常识或候选论证填补缺失信息；
- 不使用需要现实世界知识、近似容差、未声明默认值、开放世界假设或多种合理解释的题；
- 选项必须在语义类型和表面长度上可比，不能让正确答案因格式、精度或措辞突出；
- 不运行 Qwen，不根据预期模型表现选择难度或重写题目。

## 5. 统计与外部验证边界

本次构建不新增正式 arm。C1/V7 仍是 N/V/AO/S，且所有 48 base 进入主要 ITT 分析。

运行后可以报告两个预先定义的辅助结果：

1. **AO-eligible sensitivity**：仅限 uncapped-N 正确且 uncapped-AO 保持 initial 的 base；
2. **四类迁移表**：correct / induced_target / other_wrong / invalid。

二者都不得用于删题、回写数据或替代主要 family-clustered 结果。

LoFa、BrokenMath、SycophancyEval 和 LOGICOM 的实际样本只在 C1/V7 通过后进入独立外部验证；不得把它们混入 held-out 48 base，也不得用它们的模型输出调 C1/V7。

## 6. 执行决定

Phase 2 允许开始，但按“两段交付”执行：

1. Claude 先构建 4 个 sentinel base（来自 4 个不同 family，覆盖三个 stratum）；只做静态检查与人工审阅，不运行模型；
2. sentinel 通过后，一次性构建剩余 44 个 base；不得根据 Qwen 反馈修改。

这不是 response-based pilot。sentinel 只验证 schema、可读性、唯一 gold、V 单节点更新和 certificate 是否按要求落地。
