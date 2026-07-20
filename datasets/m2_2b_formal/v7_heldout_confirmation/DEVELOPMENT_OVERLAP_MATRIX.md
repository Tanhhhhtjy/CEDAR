# C1/V7 development structural-overlap matrix

状态：Phase 1 结构审核通过。所有结论均绑定 `family_cards24.jsonl` 的逐卡 SHA-256；修改任一卡片会使机器审计失败。

审核比较四个维度：初始关键事实、答案生成机制、V 改变节点/规则、与最近 development family 的结构关系。AO/S 使用统一冻结处理，不作为 novelty 依据。自动 lexical overlap 不能替代本表。

| new family | nearest development families | initial distinct | answer mechanism distinct | V node distinct | decision | concise rationale |
|---|---|---|---|---|---|---|
| C1V7-F01 | weighted_pool; lexicographic_tiebreak | yes | yes | yes | approved | order statistic; V changes rank occupancy |
| C1V7-F02 | tier_opening; layer_reservation | yes | yes | yes | approved | minimum of derated capacities |
| C1V7-F03 | boundary_inclusivity; rounding_stage | yes | yes | yes | approved | exact ceiling from indivisible containers |
| C1V7-F04 | contingency_dependence; weighted_pool | yes | yes | yes | approved | deterministic serial-yield product |
| C1V7-F05 | eligibility_class; currency_basis | yes | yes | yes | approved | class-defined algebraic sign |
| C1V7-F06 | reciprocal_orientation; adjustment_order | yes | yes | yes | approved | invert one gross/base equation |
| C1V7-F07 | weighted_pool; reciprocal_orientation | yes | yes | yes | approved | conserved distance/time, not mean of speeds |
| C1V7-F08 | contingency_dependence; modular_transition | yes | yes | yes | approved | finite absorbing recurrence |
| C1V7-F09 | version_selector; priority_relation | yes | yes | yes | approved | longest-prefix route selection |
| C1V7-F10 | version_selector; lexicographic_tiebreak | yes | yes | yes | approved | semantic-version object comparison |
| C1V7-F11 | boundary_inclusivity; interpolation | yes | yes | yes | approved | full uncertainty-set containment |
| C1V7-F12 | compound_grouping; quantifier_scope | yes | yes | yes | approved | escaped-delimiter sequential parse |
| C1V7-F13 | power_dimension; coefficient_binding | yes | yes | yes | approved | categorical logarithmic encoding |
| C1V7-F14 | axis_permutation; sensor_scope | yes | yes | yes | approved | exact protocol-port tuple lookup |
| C1V7-F15 | parity_gate; eligibility_class | yes | yes | yes | approved | componentwise bitmask then lookup |
| C1V7-F16 | counter_rollover; synchronization | yes | yes | yes | approved | finite nth-weekday computation |
| C1V7-F17 | exclusive_partition; reachability | yes | yes | yes | approved | global bijective matching |
| C1V7-F18 | boundary_inclusivity; reachability | yes | yes | yes | approved | interval-relation composition |
| C1V7-F19 | eligibility_class; exactly_k | yes | yes | yes | approved | weighted quorum plus independent veto |
| C1V7-F20 | reachability; adjustment_order | yes | yes | yes | approved | maximum weighted path in a DAG |
| C1V7-F21 | lexicographic_tiebreak; shared_positive | yes | yes | yes | approved | universal pairwise dominance |
| C1V7-F22 | disjunctive_alarm; witness_pattern | yes | yes | yes | approved | minimum hitting set across conflicts |
| C1V7-F23 | exclusive_partition; lexicographic_tiebreak | yes | yes | yes | approved | permutation CSP with adjacency constraints |
| C1V7-F24 | rule_directionality; reachability | yes | yes | yes | approved | set-valued closure plus key minimality |

完整 nearest-family ID、逐维布尔结论、理由与绑定哈希见 `development_overlap_audit24.jsonl`。
