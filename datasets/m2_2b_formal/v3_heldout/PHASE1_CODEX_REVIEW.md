# M2.2b v3 held-out · Phase 1 Codex 审核

审核日期：2026-07-13  
审核结论：`changes_requested`。30 个 base family 的结构骨架可以保留；完成以下静态小修前不得进入 Phase 2。

## 已通过

- 30 个唯一 family，六个 `stratum × subtype` 单元各 5 个。
- 当前没有 base、arms、builder 或模型结果，阶段纪律正确。
- 14 个中风险 family 均可保留；未发现必须删除的 development 四轴同构项。
- refute/undercut 的概念边界基本正确。

## 阻断项 1：F 机制过度集中

若把“将另一对象、批次、范围、字段、设备、图、规则对或 guide example 的事实迁移到 target”统一记为 `cross_scope_transfer`，当前约 24/30 张 card 属于这一类。这会把构念缩窄为“能否拒绝跨对象类比”，并产生 `another/different/neighboring/previous/spare/staging/adjacent` 等 Observation-only 表面线索。

修订要求：

- 每张 card 新增 `F_error_class`；
- 至少 5 个实质 F 类别，每个 stratum 至少 3 类；
- `cross_scope_transfer` 最多保留 12/30；
- 优先改写 unit/label 的 F；
- 新 F Observation 本身不得 refute/undercut 原答案，只有接受一个唯一无效桥接后才能到 target；
- 不改变 base structural graph、V 节点、updated target 或 subtype。

为使复审具有确定性，至少改写以下 12 张，不得由生成者在看到模型结果后另行挑选：

- `hoa_ar_undercut_rounding_stage`
- `hoa_ar_undercut_contingency_dependence`
- `hoa_ul_refute_affine_zero_offset`
- `hoa_ul_refute_axis_permutation`
- `hoa_ul_refute_counter_rollover`
- `hoa_ul_refute_reciprocal_unit_orientation`
- `hoa_ul_undercut_reference_datum_attachment`
- `hoa_ul_undercut_scale_interpolation_law`
- `hoa_ul_undercut_temperature_coefficient_binding`
- `hoa_ul_undercut_channel_synchronization`
- `hoa_lg_refute_directed_reachability_bridge`
- `hoa_lg_refute_lexicographic_tiebreak`

这些改写应优先采用 target-specific 但不具识别力的观察，例如相关性推出映射、单个非识别性校验点推出参数、局部或汇总规律推出个体结论；不能只是把 `another device` 换成 `another batch`。

可选类别包括：非穷尽日志的闭世界推断、必要/充分条件倒置、不可识别参数的强行点估计、无效传递、聚合量与个体量偷换、共享数据与共享来源偷换、相关性推出映射、局部约束推出全局约束、量词/括号作用域误读。

## 阻断项 2：undercut 缺少显式 countermodels

15 张 undercut card 均需新增 `undercut_countermodels`，至少包含 `shared_facts`、`varying_node`、`model_1`、`model_2`。两个模型必须同时与 V Observation 和所有未提及事实相容，只在指定节点不同，并导向不同答案；不得额外诉诸来源整体不可靠。

## medium 变体收紧

- `hoa_ar_undercut_set_overlap_cardinality`：固定其余 pairwise overlap 与 triple intersection，只留一个 intersection 未知。
- `hoa_lg_refute_lexicographic_tiebreak`：第三候选必须在未改变的 primary score 上被严格支配，不能产生第二个 tie。
- `hoa_lg_undercut_rule_directionality`：第二条边必须固定可靠，答案差异只能来自第一条 arrow bit。
- `hoa_lg_refute_exactly_k_constraint`：medium 必须仍有唯一 initial/updated assignment，且 V 只改目标 Boolean。

## 复审通过条件

1. 30 张 card 均有 `F_error_class`，分布满足上述限制；
2. 15 张 undercut 均有显式 countermodels；
3. 四张 medium 规则已收紧；
4. blueprint、overlap matrix、risk report 与 cards 同步；
5. 静态验证仍为 30/30、六单元各 5、无 high-risk、无 base/arms/model output。

满足后可批准 Phase 2；仍不得跳过双人盲审、机器闸门和 uncapped gate，也不得运行 capping。
