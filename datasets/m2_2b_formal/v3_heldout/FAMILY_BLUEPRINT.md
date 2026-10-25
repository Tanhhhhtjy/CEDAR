# M2.2b v3 held-out uncapped gate：Phase 1 family blueprint

状态：`approved_for_phase2_construction`

Codex 首轮审核结论见 `PHASE1_CODEX_REVIEW.md`，Phase 1.1 复审批准见 `PHASE1_1_CODEX_REVIEW.md`。现已授权按冻结 cards 进入 Phase 2 construction；仍未授权运行 Qwen 或 capping。

本文件只定义 30 个结构 family。当前目录没有生成 60 个 base units、五臂 records、模型输出或 capping 结果。未获数据负责人/Codex 批准前不得进入 Phase 2。

## 1. 冻结依据与审计范围

- 规范来源：`docs/M2_2B_V3_HELDOUT_INTAKE.md`。
- 冻结 commit：`f7cfb489b41f105420973836ebcd39bc23558038`。
- 已纳入 novelty 审计的 development：
  - `m2.2b-pilot-v1` 旧 60 题；
  - v2 的 30 个结构 family / 60 个参数化题；
  - v3 calibration12 的 12 个 family；
  - S-strength calibration（与 calibration12 同源）；
  - 两版 objective correction screen 共 40 题；
  - Belief-R 的 ponens/tollens、Event→Event、Event→MentalState 结构。
- v3 calibration 的模型输出仅用于理解项目已经冻结的范式决定；没有按模型逐题表现选择、删除或修改本蓝图。
- 项目说明书后续已由负责人以 Markdown 导出并完成通读；其确认 M2.2b 是五臂形式域任务。novelty 的可复现判据仍以冻结 commit 与仓库内 development 资产为准。

## 2. 结构新颖性的判定单位

每个 family 以四元结构签名审计，而不是以题面、名称或 `family_id` 审计：

1. 原答案所依赖的关键节点；
2. V 改变或撤销的节点/边；
3. 规范答案从 initial 到 updated 的变换机制；
4. F 的唯一无效桥接发生在哪里。

任一候选若同时复现 development 的这四项，即使换了背景、数字或标签也应删除。完整逐行比较见 `DEVELOPMENT_OVERLAP_MATRIX.md`。

## 3. 数量与难度合同

| stratum | subtype | family 数 | Phase 2 easy 变体 | Phase 2 medium 变体 | 计划 base 数 |
|---|---:|---:|---:|---:|---:|
| arithmetic_accounting | refute | 5 | 5 | 5 | 10 |
| arithmetic_accounting | undercut | 5 | 5 | 5 | 10 |
| unit_label | refute | 5 | 5 | 5 | 10 |
| unit_label | undercut | 5 | 5 | 5 | 10 |
| multistep_logic | refute | 5 | 5 | 5 | 10 |
| multistep_logic | undercut | 5 | 5 | 5 | 10 |
| **总计** |  | **30** | **30** | **30** | **60（尚未生成）** |

若获批准，每个 family 才能生成两个参数化变体：variant 1=`easy`，variant 2=`medium`。变体只能换实体与参数，并落实 card 中已写死的步数/干扰规则，不能改变结构图、V 节点、F 致命错误或 split。

### 3.1 Phase 1.1 F 与 countermodel 合同

- `F_error_class` 共 15 个取值；`cross_scope_transfer=11/30`，低于 12/30 上限。
- 三个 stratum 分别覆盖 5、7、7 类 F 错误机制，均不少于 3 类。
- 指定的 12 张 F 已改为 target-specific 的非识别性校验、生态推断、相关性推出映射、可能性当现实、缺少独立确认、局部规律全局化或 coarse-bin 当精确相等。
- 15 张 undercut card 均保存 `shared_facts`、`varying_node`、`model_1`、`model_2`；两个模型共享 V Observation 与未提及事实，只改变一个节点并导向不同结论。

## 4. Arithmetic/accounting · refute

| family_id | 核心图 | V 的决定性动作 | F 唯一失效点 |
|---|---|---|---|
| `hoa_ar_refute_marginal_tier_opening_state` | opening state + current quantity → 阈值分段 → 边际费率 | 直接更正 billing-cycle 开局累计量，使当前量跨档 | 偷换 maintenance cycle 与 billing cycle |
| `hoa_ar_refute_weighted_pool_composition` | 组值 + 贡献权重 → 加权分子/分母 → pooled mean | 更正一个组的 roster weight | 用 subteam 数量替代 contributor 数量 |
| `hoa_ar_refute_allocation_eligibility_class` | eligibility × driver hours → 分母 → 分摊份额 | 更正一台机器是否进入分摊分母 | 偷换自由文本描述与正式 eligibility class |
| `hoa_ar_refute_inventory_layer_reservation` | 有序成本层 + availability → earliest-available 消耗 → COGS | 证明最早层被预留，必须跳过 | 把同 shipment 中另一层的 reservation 泛化到 L1 |
| `hoa_ar_refute_noncommuting_adjustment_order` | 两个非交换算子 + order bit → 最终金额 | 更正算子执行顺序位 | 跨 transaction type 泛化执行顺序 |

## 5. Arithmetic/accounting · undercut

| family_id | 核心图 | V 的决定性动作 | F 唯一失效点 |
|---|---|---|---|
| `hoa_ar_undercut_set_overlap_cardinality` | marginal counts + intersection → union total | 撤销唯一 disjointness 支撑，overlap 未识别 | 把 pilot population 的 overlap 泛化到 target population |
| `hoa_ar_undercut_currency_basis_attachment` | 行数值 + currency attachment + rate → total | 证明页眉与行绑定丢失，存在多个货币赋值 | 把 prior batch 的 currency 泛化到 current batch |
| `hoa_ar_undercut_boundary_inclusivity` | X 与阈值 T + `<`/`≤` bit → eligibility | 局部校验失败，只剩两种 comparator | 用其他 clause 的 comparator 异质性否定 target clause |
| `hoa_ar_undercut_rounding_stage` | round-each-then-sum vs sum-then-round | 撤销 rounding-stage bit 的唯一支撑 | 用两种 stage 都通过的非识别性单行校验否定显式 stage |
| `hoa_ar_undercut_contingency_dependence` | marginals + joint dependence → union probability → expected liability | dependence flag 丢失，marginals 不识别 joint | 从 aggregate weekly co-movement 推出 item-level dependence |

## 6. Unit/label · refute

| family_id | 核心图 | V 的决定性动作 | F 唯一失效点 |
|---|---|---|---|
| `hoa_ul_refute_affine_zero_offset` | `Y=A*(X-B)` | 直接测得非零 offset B，scale A 不变 | 从同时兼容两个 offset 的区间强行点估计 B |
| `hoa_ul_refute_axis_permutation` | 两通道 + permutation → X/Y 语义 | 连通性追踪确认 channel swap | 在两参考轴共同变化时用相关性识别 channel map |
| `hoa_ul_refute_power_dimension` | prefix factor 的 D 次幂 | 读清 superscript，更正维度指数 | 把 adjacent field 的 exponent 迁移到 target field |
| `hoa_ul_refute_counter_rollover` | display + wraps×modulus → accumulated count | wrap log 证明发生一次回卷 | 把 rollover capability 当作实际发生 rollover |
| `hoa_ul_refute_reciprocal_unit_orientation` | numerator/denominator orientation → multiply 或 reciprocal | 读清 slash 方向，反转单位朝向 | 用 X=1 时两种 orientation 相同的校验点识别方向 |

## 7. Unit/label · undercut

| family_id | 核心图 | V 的决定性动作 | F 唯一失效点 |
|---|---|---|---|
| `hoa_ul_undercut_reference_datum_attachment` | relative X + datum baseline → absolute Y | datum tag 与 display 的绑定边丢失 | 用 `{D1,D2}` set-level 校验否定显式 D1 binding |
| `hoa_ul_undercut_compound_unit_grouping` | 同一 token 的两棵括号 parse tree | 仅括号区丢失，两种 parse 均合法 | 用 guide example 覆盖 target 的显式括号 |
| `hoa_ul_undercut_scale_interpolation_law` | between-tick position + linear/step law → Y | interpolation-law 字段失效 | 用两种 law 在 labeled tick 相同的校验点否定显式 law |
| `hoa_ul_undercut_temperature_coefficient_binding` | raw + k·ΔT，sensor→k binding | row key 失效，两个系数均可适用 | 用 ΔT=0 时所有 k 相同的校验点否定显式 binding |
| `hoa_ul_undercut_channel_synchronization` | A/B time series + pairing map → ratio/difference | 粗时间桶保留但 cross-channel pairing 丢失 | 把 coarse check 未独立确认误当作反证 |

## 8. Multistep logic · refute

| family_id | 核心图 | V 的决定性动作 | F 唯一失效点 |
|---|---|---|---|
| `hoa_lg_refute_parity_gate` | flags → XOR/parity → outcome | 更正一个 flag，翻转 parity | 从两个例子归纳 identifier parity→state 规则 |
| `hoa_lg_refute_directed_reachability_bridge` | directed edges → path existence | 纠正唯一 bridge 的方向 | 把部分节点的 degree 规律强制成全局约束 |
| `hoa_lg_refute_lexicographic_tiebreak` | primary comparator → tie 时 secondary | 更正 primary score 成 tie，激活 secondary | 把同一 coarse score band 当作 exact tie |
| `hoa_lg_refute_exactly_k_constraint` | Boolean variables + exactly-k → unique solution | 更正一个 Boolean input 后重解约束 | 对非穷尽日志作闭世界推断 |
| `hoa_lg_refute_modular_state_transition` | event increments → modulo finite-state endpoint | 更正一个 event type/transition | 把另一 event class 的 suffix convention 迁移到 target |

## 9. Multistep logic · undercut

| family_id | 核心图 | V 的决定性动作 | F 唯一失效点 |
|---|---|---|---|
| `hoa_lg_undercut_independent_support_collapse` | provenance parents → independence → two-source threshold | lineage 证明两报告同源，独立支持从 2 降到 1 | 偷换 shared subject data 与 shared report provenance |
| `hoa_lg_undercut_rule_directionality` | endpoints + arrow bit → implication direction | arrowhead 丢失，只剩 P→Q / Q→P 两解 | 把 companion section 的 rule direction 迁移到 target |
| `hoa_lg_undercut_exclusive_partition_loss` | A⊥B + A-membership → not-B → outcome | exclusivity edge 来自错误 namespace，active taxonomy 可重叠 | 错把 overlap 当作传递关系 |
| `hoa_lg_undercut_priority_relation_loss` | conflicting rules + priority edge → winner | 只撤销 priority edge，两个 rule 都保留 | 把另一 rule pair 的 priority exception 泛化到 target pair |
| `hoa_lg_undercut_quantifier_scope_binding` | `∃y∀x` vs `∀x∃y` → common witness | scope brackets 丢失，两种量词序均可 | 用 guide example 覆盖 target 的显式 scope brackets |

### 9.1 四个 medium 结构锁

- `hoa_ar_undercut_set_overlap_cardinality`：固定其余全部 pairwise overlap 与 triple intersection，只允许 `|A∩B|` 未知。
- `hoa_lg_refute_lexicographic_tiebreak`：第三候选在未改变的 primary score 上被严格支配；只能出现 X/Y 的一个 tie。
- `hoa_lg_undercut_rule_directionality`：第二条 `Q→R` 固定可靠，只允许第一条 P/Q arrow bit 变化，不加入其他 reverse rule。
- `hoa_lg_refute_exactly_k_constraint`：initial/updated assignment 均唯一，V 只改变目标 Boolean。

## 10. Phase 2 前的硬停点

以下事项必须由数据负责人/Codex 审核后明确批准：

1. 30 个结构图是否均与 development 的四元签名不同；
2. `ISOMORPHISM_RISKS.md` 中的中风险 family 是批准、替换还是要求改图；
3. 每个 undercut 是否确实存在至少两个与 Observation 相容、但答案不同的闭世界实例；
4. 每个 F 是否只有 card 所列的一个致命桥接错误；
5. easy/medium 规则是否可在不改变结构的前提下实例化。

在收到批准前，本 Phase 1 到此停止。
