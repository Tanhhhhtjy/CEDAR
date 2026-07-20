# C1/V7：从 sentinel 批准到 held-out 确认结果的执行手册

状态：`active_after_sentinel_approval`

本手册把后续工作拆成可连续执行的阶段。它是操作说明，不修改
`docs/C1_V7_HELDOUT_PROTOCOL.md`、修正案 01、统计 Gate 或模型配置。发生冲突时，以冻结协议和
修正案为准。

## 总体纪律

- 完整 48-base 冻结之前，禁止运行 Qwen、GPU、capping 或用任何模型输出筛题。
- 题目作者可以进行确定性计算、枚举、静态 lint 和文字审查；这些不等于两名独立人工审核。
- 四条已批准 sentinel 必须原样进入完整 48-base，canonical hash 不得改变。
- 不删除失败、分歧、截断或不理想结果；修订必须有新提交和失效链说明。
- 不接触、不删除原有未跟踪 `datasets/m2_2b_formal/v2/`。
- `private/` 中的 salt、blind key、裁决和 final labels 永不提交 Git。

## Phase A：构建剩余 44 条并形成完整 48-base

### A0. 先完成 24/24 family 可实现性反证审计

F17 已证明，文字上合理的 family card 仍可能包含无法同时满足的条件。因此在写剩余 44 条前，
必须先对全部 24 张卡各给出一个小型可实现性见证，生成：

```text
datasets/m2_2b_formal/v7_heldout_confirmation/family_feasibility_audit24.jsonl
datasets/m2_2b_formal/v7_heldout_confirmation/PHASE2_FAMILY_FEASIBILITY_AUDIT.md
```

每个 family 的机器可读记录至少包含：当前 family-card canonical hash、原始实例、V 的唯一
changed node、更新实例、原始唯一答案、更新后不同的唯一答案、验证方法、
`original_unique=true`、`updated_unique=true`、`target_differs=true`、
`changed_node_count=1` 和 `decision=feasible`。

`original_witness`、`updated_witness`、`changed_node`、`original_answer`、`updated_answer`、
`rationale` 均使用非空文本；`original_answer` 与 `updated_answer` 必须不同。验证方法只能是
`exhaustive_enumeration`、`independent_calculation` 或 `formal_derivation`。construction/freeze
validator 会要求 24/24 覆盖、当前 card hash 全匹配和全部 feasible。

有限排列/集合/图任务用穷举验证；算术和规则任务用独立计算；不得只声称“应该能构造”。四个
sentinel 可以直接引用已批准实例，但仍要绑定当前 card hash。

若任一 family 无法给出见证，立即停止，不生成剩余 base，也不得自行悄悄修改冻结卡。单独写
矛盾证明和最小修订建议，重新走 card hash、overlap 绑定和批准流程。只有 24/24 feasible 才
进入 A1。这是结构可实现性 gate，不调用模型，也不依据 Qwen 行为筛题。

### A1. 固定 ID 与 family 映射

完整文件按 `C1V7-001` 至 `C1V7-048` 排序。family `Fxx` 的 variant 1/2 固定使用：

```text
base_number = 2 * (family_number - 1) + variant
```

例如 F01 使用 001/002，F11 使用 021/022，F17 使用 033/034，F24 使用 047/048。

四条 sentinel 直接复制，不得重写：001、021、033、048。构建其余 44 条时，继续遵守
`PHASE2_CLAUDE_CONSTRUCTION_BRIEF.md`、family card 和 Phase 1 风险护栏。

### A2. 难度与结构

- 每个 family 恰好两个 variant；variant 1 为 easy（1–3 个必要步骤），variant 2 为 medium
  （3–6 个必要步骤）。
- 两个 variant 使用相同答案生成机制与 V 节点类型，只改变参数/实例和答案位置。
- 原题唯一支持 initial；V 后唯一支持 target；distractor 前后均错误。
- V 只改变一个原子事实、参数、记录或规则节点；所有其他信息显式保持不变。
- 所有规则题内完备，不需要现实知识、默认约定、近似容差或开放世界推断。
- N/AO/S 不由作者撰写，base 不得包含自定义 treatment 字段。

### A3. 迁移配额

完整 48 条每种迁移各 8。四条 sentinel 已占用：

```text
A_to_B: 1
A_to_C: 0
B_to_A: 2
B_to_C: 0
C_to_A: 0
C_to_B: 1
```

剩余 44 条必须恰好补足：

```text
A_to_B: 7
A_to_C: 8
B_to_A: 6
B_to_C: 8
C_to_A: 8
C_to_B: 7
```

同一 family 两个 variant 不得使用同一迁移。先完成逻辑，再置换 A/B/C；不得为满足位置配额
改变题目事实或 gold。

### A4. 输出

生成：

```text
datasets/m2_2b_formal/v7_heldout_confirmation/m2_2b_v7_heldout48_base.jsonl
datasets/m2_2b_formal/v7_heldout_confirmation/PHASE2_CONSTRUCTION_REVIEW.md
```

`PHASE2_CONSTRUCTION_REVIEW.md` 必须逐题列出：原题复算、V 后复算、三个 option 前后 verdict、
唯一 changed node、步骤数、family 对齐、歧义检查和作者结论。对于有限组合任务必须枚举；对于
算术/规则任务必须用独立计算复核，不能只复述 certificate。

### A5. 四条 sentinel 内容门

在完整 48-base 中重新计算四条 canonical hash，必须等于：

```text
C1V7-001 e861c1a3900501bbd138f0f3f8156b5e5f4b31ab1e64b7fedbd1480a61049d8e
C1V7-021 5cc7474c1e8d0c1118149a73741c4feb0da242aec62de559e76411acbf49c6ec
C1V7-033 d44f4c1804823efb60a93a5200a7fa9bf423fcc59ca5c9f5add37546a196c115
C1V7-048 a1487c0ada6dc32bb4870d30df8d6ec1f4dfa8f2244225800d34ecf772a73042
```

不匹配立即停止，恢复 sentinel 原文；不得重新批准自己修改后的版本。

## Phase B：确定性展开与 construction gate

### B1. 顺序执行

```powershell
python datasets/m2_2b_formal/v7_heldout_confirmation/validate_m2_2b_v7_heldout.py --stage intake
python datasets/m2_2b_formal/v7_heldout_confirmation/build_m2_2b_v7_heldout.py
python datasets/m2_2b_formal/v7_heldout_confirmation/validate_m2_2b_v7_heldout.py --stage construction
```

然后运行 C1/V7 专项测试与 Ruff。若完整项目环境已安装，再运行全仓测试；依赖缺失必须如实
报告，不能写成测试通过或代码失败。

### B2. 硬性验收

必须同时满足：

- 24 families、48 bases、192 arms；N/V/AO/S 各 48；
- 三个 stratum 各 8 family / 16 base；
- 六种迁移各 8；initial/target/distractor 位置各 16；
- 每个 family variant 1/2 各一条且迁移不同；
- 24 easy、24 medium；48/48 construction certificates；
- 原题与 V 后 option-status 都恰好一个 supported；
- V changed-node count 全部为 1；external knowledge 全部 false；
- frozen N/AO/S、system prompt 与 JSON contract 逐字一致；
- AO/S 长度比、overlap audit、family-card hash、lexical leakage 全部通过；
- `construction_audit48.jsonl` 48/48，且明确不是独立人工审核；
- Qwen/GPU/model outputs 均不存在。

### B3. 重复构建

保存第一次构建的公开输出 SHA-256，再运行一次不带 `--prepare-review` 的 builder。两次的
192 arms、construction audit、leakage report 和 manifest 必须逐文件哈希一致。若不一致，
停止并定位非确定性来源。

### B4. checkpoint

若 B1–B3 全部通过且 `PHASE2_CONSTRUCTION_REVIEW.md` 没有 pending/unclear/fail：

1. 提交并推送一个独立 `full48 construction` commit；
2. 记录 commit、48-base SHA、192-arm SHA、测试结果；
3. 才允许进入 Phase C。

任何一题存在歧义时，在这个阶段修题并重跑完整 B；仍然禁止看模型输出。

## Phase C：实现盲审协调工具并生成审核包

### C1. 零 GPU 协调工具

在生成审核包前，实现并测试一个只处理审核文件的协调脚本，例如：

```text
scripts/coordinate_c1_v7_reviews.py
```

至少支持：

- `--score-only`：校验两份 CSV 完整性、内容/hash 绑定，调用冻结的 agreement 逻辑并写公开
  `review_agreement_report.json`；不得写 final labels；
- `--prepare-adjudication`：只列出真实分歧的 blind ID、字段和双方原始标签，不自动裁决；
- `--finalize`：读取人工裁决，保留所有双方一致的原标签，只对分歧项采用裁决，生成 private
  `final_review_labels.jsonl` 与 `adjudication_completed.json`；
- 拒绝覆盖已有 raw review、private key 或裁决文件；
- 在 review packet、completed review、adjudicated/frozen 三个状态分别维护独立 manifest，
  不能让旧 CSV 哈希冒充当前文件哈希；
- 测试 agreed label 不可被改写、分歧数一致、坏 final contract 必须失败、private 文件不被 Git
  跟踪。

该工具不得调用模型或根据预期 Gate 修改标签。

### C2. 只生成一次审核包

确认目录中不存在旧 review packet 后，只运行一次：

```powershell
python datasets/m2_2b_formal/v7_heldout_confirmation/build_m2_2b_v7_heldout.py --prepare-review
```

检查：

- `blind_review_a.csv`、`blind_review_b.csv` 各 48 行、全部标签列为空、顺序不同；
- 可见 prompt 与当前 48 base 精确匹配；
- `private/review_salt.txt`、`private/blind_review_key.jsonl` 存在且未跟踪；
- key 绑定 48 个 base canonical hash 与完整 dataset hash；
- `git ls-files .../private` 返回空；
- 没有模型输出。

生成后把此时的 `manifest.json` 原样复制为不可覆盖的 `review_packet_manifest.json`，用于记录
空白审核包哈希。后续 reviewer 填写 CSV 后文件哈希必然变化，不能继续把空白 manifest 冒充
当前状态。

提交并推送公开的空白审核包、协调代码、测试和 manifest。不要提交 `private/`。随后进入唯一的
强制人工等待点。

## Phase D：两名真实人员独立盲审

### D1. 分发

由数据负责人分别发送：

- reviewer A 只收到 `blind_review_a.csv`；
- reviewer B 只收到 `blind_review_b.csv`。

审核者不得访问：另一个人的表、仓库答案字段、private key、作者 construction certificate、
Qwen/model outputs。两人不得讨论标签。

### D2. 每人填写

不改表头、blind ID、行顺序和 prompt，只填写：

- `original_supported_answer`：A/B/C/unclear；
- `v_supported_answer`：A/B/C/unclear；
- 十个 yes/no/unclear 检查字段；
- `confidence_1_5`；
- `notes`。

审核标准以 blind sheet 中实际文字为准，不因“作者应该想表达什么”而脑补。若有第二答案、规则
缺失或语言歧义，必须如实标 `unclear/no`。

### D3. 返回与冻结 raw review

两份表完成后分别返回协调者。只有两份都收到后，Claude 才能同时导入；导入前记录两文件
SHA-256 和收到时间。不得先看 A 的标签再提示 B，不能修改 reviewer 原表。

导入两份完成表后，先提交一个只含 raw completed reviews、接收记录与当前 SHA 的 checkpoint，
再开始任何裁决。协调脚本必须写新的 `completed_review_manifest.json`；保留
`review_packet_manifest.json` 作为空白包历史，不覆盖或伪造旧哈希。

## Phase E：一致率、裁决与数据 freeze

### E1. 只评分，不裁决

运行协调脚本 `--score-only`。硬门：

- raw gold agreement ≥ 95%；
- reviewer A/B 各自 original/V gold accuracy ≥ 95%；
- V validity agreement ≥ 90%，kappa ≥ 0.8；
- S/AO contamination agreement ≥ 90%，kappa ≥ 0.8。

失败时不得通过改标签补门槛。检查分歧是否暴露题目歧义：若题目有问题，修订 base，并把旧数据、
审核包和报告归档为 superseded；重新构建、生成新 key，并让两人重新审核全部 48 条。不得只重审
修改题或隐藏失败周期。

### E2. 人工裁决

门槛通过后，运行 `--prepare-adjudication`。第三位裁决者或预先指定的数据负责人只裁决分歧项，
逐格写理由和时间；双方一致项禁止改动。

裁决完成后运行 `--finalize`。最终 48 条逐题必须满足冻结可接受性合同：原题答案为 initial、
V 答案为 target、八个正向检查为 yes、`development_isomorphic` 和
`other_reasonable_answer` 为 no。

如果真实裁决结论不满足合同，说明题目不合格，必须走完整修订与重审；不得把裁决写成“通过”。

### E3. freeze gate

```powershell
python datasets/m2_2b_formal/v7_heldout_confirmation/validate_m2_2b_v7_heldout.py --stage freeze
```

必须确认：review hash/base hash/dataset hash 一致，分歧数与裁决数一致，unresolved=0，private
文件未被 Git 跟踪。生成公开 `C1_V7_DATASET_FREEZE_REPORT.md`，记录所有公开交付 SHA、review
统计和 freeze 输出，但不泄露 private key/base 映射。

提交并推送公开冻结交付；private 文件只保留在授权实验机的安全目录。此时数据集才允许交给
实验同学。

## Phase F：冻结 runner/auditor/analyzer（仍然零 GPU）

按协议实现或复核：

```text
scripts/run_c1_v7_heldout.py
scripts/audit_c1_v7_heldout.py
scripts/analyze_c1_v7_heldout.py
tests/test_c1_v7_heldout_runner.py
tests/test_c1_v7_heldout_analysis.py
```

要求：精确模型 revision、thinking-greedy、batch=1、max_new_tokens=2048、uncapped/p25、layers
46–53、完整 logits/entropy/thinking/telemetry、严格 checkpoint resume、禁止输出导向重试。分析器
冻结 family-cluster bootstrap 10,000 次和 Gate A/B/C/D/Inconclusive，不得根据结果改阈值。

用 mock model 和小张量跑完整 dry-run；冻结 input manifest、代码 SHA 和配置 SHA，提交推送后
才可申请 GPU smoke。

## Phase G：GPU smoke 与主确认实验

### G1. 工程 smoke

按协议固定 2 个 base 的 8 条四臂：uncapped no-hook 与 monitor-only 必须输出逐字一致、A/B/C
logits 最大差 ≤ `1e-4`、无 hook 泄漏或 NaN/Inf。另固定 4 条 S 跑 p25 sentinel，只检查层、
方向、trigger/correction 和遥测完整性；不得按是否翻转决定继续。

工程失败立即停止修代码；数据与模型输出不得改写。

### G2. 主格

固定顺序：

1. 192/192 uncapped；
2. 192/192 p25；
3. 完成前不查看 S 翻转数或任何聚合行为；
4. 崩溃只允许从已验证 checkpoint 精确恢复，不允许重跑或替换内容失败、截断和格式错误。

### G3. 跨条件审计

确认 ID、顺序、prompt/token hash、模板、thinking、batch、max tokens 完全一致，唯一实验差异为
capping。截断原样保留。审计失败不得运行正式统计。

## Phase H：评分、Gate 与最终停止

依次运行：严格解析、宽松敏感性、四类迁移、截断最坏情况、family-cluster bootstrap、S/AO
社会特异性、V/N 控制、logit/output-space/telemetry 诊断。所有 48 base 进入 ITT 主分析。

可以另报预先定义的 LoFa 风格辅助敏感性：仅 uncapped-N 正确且 uncapped-AO 保持 initial 的
base；它不得替代 ITT、删题或回写数据。

严格按冻结 Gate 路由：

- Gate A：held-out S target-follow increase 复现；
- Gate A + social-specific 条件：才称社会压力特异候选；
- Gate B：更像 target exposure / boundary following；
- Gate D：未干净复现；
- Inconclusive：不自动扩样。

只有 Gate A 通过才运行 D1 thinking-sampling 与 D2 nonthinking-greedy；完成后停止。Gate A
未通过时禁止追加题目、seed、p1、steering 或机制工具。

## 每个 checkpoint 的固定汇报格式

每次提交必须报告：

1. 分支、commit、远端 SHA；
2. 新增/修改文件；
3. 数量与 hash；
4. validator/test/Ruff 结果；
5. 是否运行 Qwen/GPU/capping；
6. 是否存在 truncated/invalid/retry；
7. private 文件是否未跟踪；
8. 当前 Gate、允许的下一阶段和明确禁止事项。
