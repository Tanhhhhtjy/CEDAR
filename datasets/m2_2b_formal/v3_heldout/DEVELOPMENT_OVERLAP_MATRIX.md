# Development overlap matrix

状态：`approved_for_phase2_construction`

冻结比较点：`f7cfb489b41f105420973836ebcd39bc23558038`。

本矩阵比较的是结构签名，不是词面。`风险=M` 表示存在一个抽象近邻，Phase 2 必须落实所列差异；`L` 表示在已盘点 development 中没有近似四元签名。没有把任何 `H`（四项高度一致）候选保留进 family cards。Phase 1.1 只重写 F 的无效桥接并增加审计字段，没有改变 initial/V/target/subtype 四轴骨架。

## A. Development 结构基线

| 基线 | 已覆盖结构 |
|---|---|
| v1 arithmetic | 三条目求和；V 证明一条重复并删除；undercut 撤销条目/来源可靠性 |
| v1 unit/label | 单一线性 factor×reading；code→destination lookup；V 更正 factor/mapping 或撤销来源 |
| v1 logic | 单个 if-then/二元状态；V 更正 antecedent 或撤销其记录可靠性 |
| v2 arithmetic refute | duplicate identity、transposed amount、sign reversal、accounting period、quantity×price |
| v2 arithmetic undercut | receipt integrity、draft provenance、unresolved amount conflict、measurement provenance、incomplete export |
| v2 unit refute | direct calibration、metric prefix、range mode、routing version、timezone metadata |
| v2 unit undercut | expired calibration、unreadable mode、mapping signature、truncated code、active-version conflict |
| v2 logic refute | corrected state、exception、universal counterexample、disjunction、temporal order |
| v2 logic undercut | source failure、identity ambiguity、rule provenance、missing conjunct、unresolved contradiction |
| v3 calibration12 / S calibration | 复用 v2 中 12 个 family；S calibration 不增加新题目结构 |
| objective correction v1 | 静态简单算术、日期时间、线性单位换算、漏步骤、passage fact 的错答纠正 |
| objective correction v2 hard | 静态多步题中的最终漏项、tier omission、单位后处理、排程/阈值/日期约束 |
| Belief-R | ponens/tollens；If-Event-Then-Event / If-Event-Then-MentalState；BU/BM 的 t→t1 更新 |

## B. 30-family 四轴比较

| 新 family | 最近 development 近邻 | 初始关键前提差异 | V 改变位置差异 | 答案变化方式差异 | F 错误位置差异 | 风险 |
|---|---|---|---|---|---|---:|
| `hoa_ar_refute_marginal_tier_opening_state` | objective hard tier fee；v2 quantity_price | 开局累计状态进入 piecewise tariff | 改 opening state，不改当前数量/漏掉已知 tier | 低档-only→边际跨档 | maintenance-cycle→billing-cycle scope equivocation | M |
| `hoa_ar_refute_weighted_pool_composition` | objective mean；v2 quantity_price | aggregation weight 同时进分子/分母 | 改 roster weight，不改 group value | equal-weight mean→weighted mean | subteam count→contributor count | M |
| `hoa_ar_refute_allocation_eligibility_class` | v2 quantity_price / incomplete_export | denominator 的正式 eligibility membership | 改 membership bit，行与小时都保留 | 重新归一化分摊 share | free-text class equivocation | L |
| `hoa_ar_refute_inventory_layer_reservation` | v2 accounting_period；objective quarantine | ordered layer availability | 改最早层 availability，不删数量/改 period | earliest-available selection 改层 | other-layer reservation→target layer | L |
| `hoa_ar_refute_noncommuting_adjustment_order` | objective hard_logic_07；v2 temporal_order | 两算子的 execution order bit | 改 operator order，不补漏项/改事件时间 | `K∘D`→`D∘K` | other transaction type→target order | M |
| `hoa_ar_undercut_set_overlap_cardinality` | v2 duplicate_identity / incomplete_export | aggregate disjointness fixes intersection=0 | 撤销 intersection knowledge；marginals 完整 | exact union→unidentified union | pilot-population overlap→target population | L |
| `hoa_ar_undercut_currency_basis_attachment` | v2 metric_prefix / code_truncation | header-to-row semantic attachment | 丢 binding edge，numerals/codes/rates 均完整 | exact conversion→multiple assignments | prior batch→current batch | M |
| `hoa_ar_undercut_boundary_inclusivity` | v2 unreadable mode / rule provenance | comparator bit at equality boundary | 局部撤销 `<` vs `≤`，rule 仍 active | qualified exact→unknown | cross-clause heterogeneity→target invalidity | M |
| `hoa_ar_undercut_rounding_stage` | objective arithmetic；v2 amount conflict | same operator 的 placement bit | 撤销 stage，不撤销输入或 rounding rule | line-round total→two stage totals | non-identifying one-line check→stage uncertainty | L |
| `hoa_ar_undercut_contingency_dependence` | v2 incomplete_export / conflict | marginals + independence identify joint | 仅撤销 dependence node，marginals 完整且一致 | point expectation→partial identification | aggregate co-movement→individual joint dependence | L |
| `hoa_ul_refute_affine_zero_offset` | v2 direct_calibration | additive offset B, not scale factor A | 改 B，A/X 不变 | affine intercept correction | non-identifying interval→exact offset | M |
| `hoa_ul_refute_axis_permutation` | v2 routing_version / range_mode | simultaneous channel-role permutation | 改 permutation edge，不改 value/multiplier/table version | values exchange semantic axes | coupled-axis correlation→channel mapping | M |
| `hoa_ul_refute_power_dimension` | v2 metric_prefix | dimensional exponent D | 改 exponent，prefix K 不变 | K→K^D | adjacent field→target exponent | M |
| `hoa_ul_refute_counter_rollover` | v2 range_mode；date rollover screens | hidden modular state R | 加 logged wrap，不改 displayed count/mode | X→X+M before scaling | rollover capability→actual wrap | L |
| `hoa_ul_refute_reciprocal_unit_orientation` | v2 metric_prefix / calibration | numerator-denominator orientation | 改 slash orientation，numeric rate 不变 | multiply→reciprocal/divide | X=1 non-identifying check→orientation | M |
| `hoa_ul_undercut_reference_datum_attachment` | v2 timezone / code_truncation | relative reading bound to datum baseline | 丢 display→datum edge，不丢 code characters | exact absolute→two baselines | allowed-set check→explicit binding uncertainty | M |
| `hoa_ul_undercut_compound_unit_grouping` | v2 metric_prefix / unreadable_mode | compound-unit parse tree | 丢 parentheses node，tokens/modes 完整 | one parse→two algebraic parses | guide example→target grouping | L |
| `hoa_ul_undercut_scale_interpolation_law` | v2 direct_calibration / unreadable_mode | between-tick decoding law | 丢 interpolation law，endpoints/position 完整 | exact linear value→linear/step alternatives | labeled-tick agreement→law uncertainty | L |
| `hoa_ul_undercut_temperature_coefficient_binding` | v2 version_conflict / expired_calibration | sensor→coefficient attachment | 丢 row key；raw value和各 coefficient 都可靠 | exact compensation→two valid k | zero-delta agreement→binding uncertainty | M |
| `hoa_ul_undercut_channel_synchronization` | v2 timezone / temporal_order | cross-channel same-instant pairing | 丢 pairing map，不改 zone/identity/order fact | exact pair-derived value→multiple pairings | lack of independent confirmation→pairing failure | M |
| `hoa_lg_refute_parity_gate` | v2 corrected_state / quantifier | n-ary XOR parity | 改一个 flag input | parity flip→opposite lookup | two examples→global identifier rule | L |
| `hoa_lg_refute_directed_reachability_bridge` | v2 temporal_order / disjunction | unique bridge in directed path | 反转 edge orientation | reachable→unreachable by path recomposition | local degree regularity→global constraint | L |
| `hoa_lg_refute_lexicographic_tiebreak` | v2 transposed_amount / exception | primary comparison determines whether secondary fires | 改 score into a tie | primary winner→secondary winner | shared coarse band→exact tie | L |
| `hoa_lg_refute_exactly_k_constraint` | v2 quantifier / disjunction | unique satisfying assignment under cardinality constraint | 改 Boolean input，再解全局约束 | one assignment→another unique assignment | nonmention→false | L |
| `hoa_lg_refute_modular_state_transition` | v2 corrected_state / temporal_order | event type selects transition increment in automaton | 改 transition label，event order不变 | replayed modular endpoint changes | other class convention→target increment | L |
| `hoa_lg_undercut_independent_support_collapse` | v2 source_failure / contradiction | two-source threshold depends on provenance independence | 合并 provenance parents；报告本身仍可靠一致 | established→only one independent support→unknown | shared subject data→shared report parent | L |
| `hoa_lg_undercut_rule_directionality` | Belief-R；v2 rule_provenance | rule endpoint direction bit | 丢 arrow direction，rule endpoints/currentness保留 | derived Q→Q unknown | companion rule→target edge | M |
| `hoa_lg_undercut_exclusive_partition_loss` | v2 disjunction / missing_conjunct | taxonomy disjointness licenses negative membership | 丢 exclusivity edge，positive facts完整 | derived not-B outcome→overlap-compatible unknown | invalid transitivity of overlap | L |
| `hoa_lg_undercut_priority_relation_loss` | v2 exception / contradiction / rule_provenance | explicit priority edge resolves two firing rules | 只丢 priority relation；rules/facts均保留 | selected conclusion→unresolved rule conflict | other-pair exception→target priority | M |
| `hoa_lg_undercut_quantifier_scope_binding` | v2 quantifier / identity ambiguity | `∃y∀x` common-witness binding | 丢 scope brackets，不改 entity identity/relation facts | common witness entailed→not entailed under alternate scope | guide example→target scope | L |

## C. 判定结果

- `H`：0。没有已知四轴同构项进入 cards。
- `M`：14。需要审核者重点查看，详见 `ISOMORPHISM_RISKS.md`。
- `L`：16。
- v3 calibration12 和 S-strength 因与 v2 同源，没有被误算为额外 novelty；凡与其结构相同者也已通过对应 v2 行检查。

本矩阵不是双人盲审，也不声称替代 Phase 3 的 validity review；它是作者在 Phase 1 的 construction/novelty audit。

## D. Phase 1.1 F 分类审计

- `cross_scope_transfer`：11/30。
- `non_identifying_check_to_uncertainty`：4/30。
- `equivocation`：3/30。
- 其余各 1/30：`aggregate_individual_mismatch`、`ecological_inference`、`non_identifying_measurement_to_point_estimate`、`correlation_to_mapping`、`possibility_to_actuality`、`non_identifying_check_to_parameter`、`absence_of_confirmation`、`inductive_overgeneralization`、`local_to_global_constraint`、`coarse_bin_to_exact_equality`、`closed_world_from_nonexhaustive_log`、`invalid_transitivity`。
- arithmetic/accounting、unit/label、multistep logic 分别覆盖 5、7、7 个类别。
