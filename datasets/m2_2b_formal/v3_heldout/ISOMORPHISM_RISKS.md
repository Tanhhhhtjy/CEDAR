# Phase 1 潜在同构风险

状态：`approved_for_phase2_construction`。Codex 复审允许保留全部 30 个 family 和 14 个中风险项；Phase 2 仍须落实本文件护栏，批准范围与下一硬停点见 `PHASE1_1_CODEX_REVIEW.md`。

## 1. 已删除的高风险方向

最初工作区蓝图中的下列方向没有进入本版 cards，因为与 development 过近：

- settlement date / accounting period：接近 v2 `ar_refute_accounting_period`；
- regional codebook：接近 v2 `ul_refute_routing_version`；
- priority override：接近 v2 `lg_refute_exception`；
- witness identity / entity link：接近 v2 `lg_undercut_identity_ambiguity`；
- closed-world completeness：接近 v2 `ar_undercut_incomplete_export`；
- source/rule validity 的泛化版本：接近 v2 `source_failure` / `rule_provenance`；
- temporal granularity + page-order F：与 v2 `lg_refute_temporal_order` 的四轴签名过近。

这些方向不得在 Phase 2 中换名回流。

## 2. 中风险 family（14）与实例化护栏

| family | 风险来源 | Phase 2 必须守住的结构差异 | 触发替换的条件 |
|---|---|---|---|
| `hoa_ar_refute_marginal_tier_opening_state` | objective hard tier fee、v2 quantity | V 必须改 opening cumulative state；当前 quantity 与所有 tier 必须原先已正确使用 | 若题面变成“漏算一个已知 tier”或“把 quantity 改成另一个数”，删除 |
| `hoa_ar_refute_weighted_pool_composition` | objective mean、v2 quantity | weight 必须同时进入 numerator/denominator；V 不得改 group value | 若只是 sum→mean 或 quantity×price，删除 |
| `hoa_ar_refute_noncommuting_adjustment_order` | objective discount/tax/shipping、v2 temporal order | 两算子都必须从首轮起存在；V 只改 execution-order bit | 若 updated answer 来自补上漏项，删除 |
| `hoa_ar_undercut_currency_basis_attachment` | metric prefix、code truncation | 代码与数值必须完整；只丢 header-to-row attachment | 若 V 改成“标签字符读错/缺字”，删除 |
| `hoa_ar_undercut_boundary_inclusivity` | unreadable mode、rule provenance | rule/currentness/threshold 都保留；只丢 `<`/`≤` semantic bit | 若整条 rule 失效，删除 |
| `hoa_ul_refute_affine_zero_offset` | direct calibration | scale A 必须固定；V 只改 additive B | 若 V 改 per-mark factor，删除 |
| `hoa_ul_refute_axis_permutation` | routing mapping、range mode | 同时存在两个 channel；V 必须是 permutation，不得只是新 lookup row | 若只有一个 code→label，删除 |
| `hoa_ul_refute_power_dimension` | metric prefix | prefix K 固定；V 只改 exponent D | 若只是 m→c 等 prefix 更正，删除 |
| `hoa_ul_refute_reciprocal_unit_orientation` | metric prefix/direct calibration | numeric rate 固定；updated computation 必须从 multiply 变 reciprocal | 若只换 factor，删除 |
| `hoa_ul_undercut_reference_datum_attachment` | timezone/code truncation | datum 值均完整；只丢 relative-value→datum binding | 若变成纠正具体 timezone 或缺字符，删除 |
| `hoa_ul_undercut_temperature_coefficient_binding` | version conflict/expired calibration | 所有 coefficient row 都有效；只缺 sensor-binding key | 若两个 active versions 直接冲突或证书过期，删除 |
| `hoa_ul_undercut_channel_synchronization` | timezone/temporal order | channel identity/zone 完整；只缺 cross-channel same-instant pairing | 若问题退化为谁先谁后，删除 |
| `hoa_lg_undercut_rule_directionality` | Belief-R、rule provenance | endpoints 和 rule currentness 保留；V 只撤销 arrow bit，updated 必须是 unknown | 若 V 改 antecedent 事实或整条 rule 失效，删除 |
| `hoa_lg_undercut_priority_relation_loss` | exception、contradiction、rule provenance | 两条相反 rule 必须都已触发且可靠；只撤销独立 priority edge | 若 V 新增 exception 或第二条冲突记录，删除 |

其余 16 个当前标为低风险，但仍须接受审核：

- `hoa_ar_refute_allocation_eligibility_class`
- `hoa_ar_refute_inventory_layer_reservation`
- `hoa_ar_undercut_set_overlap_cardinality`
- `hoa_ar_undercut_rounding_stage`
- `hoa_ar_undercut_contingency_dependence`
- `hoa_ul_refute_counter_rollover`
- `hoa_ul_undercut_compound_unit_grouping`
- `hoa_ul_undercut_scale_interpolation_law`
- `hoa_lg_refute_parity_gate`
- `hoa_lg_refute_directed_reachability_bridge`
- `hoa_lg_refute_lexicographic_tiebreak`
- `hoa_lg_refute_exactly_k_constraint`
- `hoa_lg_refute_modular_state_transition`
- `hoa_lg_undercut_independent_support_collapse`
- `hoa_lg_undercut_exclusive_partition_loss`
- `hoa_lg_undercut_quantifier_scope_binding`

## 3. 新 family 之间的近邻风险

### 3.1 `set_overlap_cardinality` vs `exclusive_partition_loss`

- 前者是 arithmetic inclusion-exclusion：缺失的是 intersection cardinality，target 是数值 union total。
- 后者是 logic taxonomy：缺失的是 disjointness edge，target 是由 `not-B` 驱动的结论。
- Phase 2 禁止把两者都写成“两个标签可能重叠所以无法确定”；必须显式保留一个数值交集节点与一个逻辑负成员推导。

### 3.2 `noncommuting_adjustment_order` vs `priority_relation_loss`

- 前者的两个算子顺序一旦确定就给唯一数值，V 正向改成另一 order。
- 后者的两条规则已经给出相反结论，V 撤销 resolver 后得到 undercut/unknown。
- 禁止在 priority family 中计算费用顺序，也禁止在 adjustment family 中引入冲突规则。

### 3.3 `axis_permutation` vs `channel_synchronization`

- permutation 更改“哪个 channel 是哪个 axis”，并给出新的确定答案；
- synchronization 保留 channel identity，只丢“哪些时间点配成一对”，并得到 unknown。
- 两者不得共用“线缆接错/通道对调”的 Observation。

### 3.4 多个 metadata-undercut family 的泛化风险

`currency_basis_attachment`、`rounding_stage`、`reference_datum_attachment`、`compound_unit_grouping`、`interpolation_law`、`temperature_coefficient_binding`、`channel_synchronization` 都会使用“一个控制节点失去支撑”的 undercut 模式。它们只有在被撤销的节点类型和答案生成图不同的情况下才算不同 family。

Phase 1.1 已在全部 15 张 undercut card 中保存两个显式 countermodels；Phase 2 实例化时仍必须逐题落实：

- 两个 countermodel 共享 Observation 与所有未提及原事实；
- 只在 card 指定的节点取不同值；
- 两个 countermodel 导向不同答案；
- 不得额外诉诸“来源整体不可靠”。

若做不到，应在生成任何 base 前替换该 family。

## 4. F 的第二失效点风险

虽然 30 张卡均标记 `second_failure_point_absent=true`，Phase 2 仍有以下常见失败方式：

1. F Observation 同时是无关事实、来源不明、又与原事实冲突；
2. Candidate reasoning 除逻辑跳步外再算错数字；
3. 为凑 target 偷换题目规则或引入未说明现实知识；
4. Observation 其实已经足以 undercut，但 Candidate 又增加另一个谬误；
5. F 使用“审计、权威、官方、专家”等社会/权威线索。

实例化时应让 Candidate 在接受那一个无效桥接后，其余推理与计算全部正确。否则 family card 虽合格，具体 base 仍不合格。

首稿曾大量使用颜色、版式、编号等表面代理；Phase 1 首版又有约 24/30 集中为跨对象/跨范围迁移。Phase 1.1 已将指定 12 张重写为 target-specific 的其他错误机制，并给 30 张卡显式增加 `F_error_class`：

- `cross_scope_transfer=11/30`，不超过 12/30；
- 共 15 个类别；
- arithmetic/accounting、unit/label、multistep logic 分别覆盖 5、7、7 类；
- 新增机制包括非识别性校验强行点估计、生态推断、相关性推出 mapping、可能性当现实、缺少独立确认当反证、局部规律全局化、coarse bin 当 exact tie；
- Phase 2 不得重新集中为 `another/different/neighboring/previous/spare/staging/adjacent` 驱动的 Observation 模板。

每个 card 的 `vf_source_pairing` 是实例化硬约束：V/F 必须使用相同来源类别、相近数字精度和相近专业程度。

## 5. 难度漂移风险

- `contingency_dependence`、`compound_unit_grouping`、`quantifier_scope_binding` 最容易变成 hard。easy 版本必须直接给出两个可行解释/模型，避免要求外部概率论或形式逻辑知识。
- medium 只能增加到 2–4 步或加入中性干扰信息；不得增加第二个未知节点。
- `parity_gate`、`counter_rollover` 最容易过易。medium 版本必须增加组合步骤，而不是用更大的数字假装更难。

Phase 1.1 另冻结四个 medium 护栏：

- `hoa_ar_undercut_set_overlap_cardinality`：其余 pairwise overlaps 与 triple intersection 全部固定，只留一个 intersection 未知；
- `hoa_lg_refute_lexicographic_tiebreak`：第三候选在未改变的 primary score 上被严格支配，只允许一个 X/Y tie；
- `hoa_lg_undercut_rule_directionality`：第二条 edge 固定可靠，答案差异只能来自第一条 arrow bit；
- `hoa_lg_refute_exactly_k_constraint`：initial/updated assignment 均唯一，V 只改目标 Boolean。

## 6. 审核建议

Phase 1.1 复审应优先核对三项机器可判条件：`cross_scope_transfer<=12`、15/15 undercut countermodels 字段齐全、四个 medium 护栏逐字进入 cards；随后抽查 countermodels 是否真的只改变一个节点。任何 reviewer 若认为某 family 与 development 四轴同构，仍应直接按结构理由否决，不得靠改名或换背景补救。
