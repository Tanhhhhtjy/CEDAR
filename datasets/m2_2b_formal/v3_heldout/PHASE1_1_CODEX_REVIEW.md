# M2.2b v3 held-out · Phase 1.1 Codex 复审

复审日期：2026-07-13  
复审结论：`approved_for_phase2_construction`

本批准只授权按冻结 family cards 生成 60 个 base units、300 条五臂 records 及配套盲审/验证材料；不授权运行 Qwen、capping，亦不授权根据模型输出筛题。

## 1. 复审结果

- 30/30 cards 可解析，family ID 唯一；
- 六个 `stratum × subtype` 单元均为 5 family；
- 30/30 含 `F_error_class`，共 15 类；
- `cross_scope_transfer=11/30`，满足不超过 12/30 的要求；
- arithmetic/accounting、unit/label、multistep logic 分别覆盖 5、7、7 类 F；
- 首轮指定的 12 张 F 均完成实质机制改写，不是同类措辞替换；
- 15/15 undercut cards 含 `shared_facts`、`varying_node`、`model_1`、`model_2`；
- 两个 countermodel 均被设计为共享 V Observation 与未提及事实，只改变指定节点并导向不同答案；
- 四个 medium 规则已按首轮意见收紧；
- 14 个 development 中风险 family 仍可保留，未新增 high-risk；
- 当前 base=0、arms=0，未运行模型或 capping。

## 2. 12 张 F 改写判定

改写后新增了不可识别校验强行选点、生态推断、相关性推出 mapping、可能性当现实、缺少确认当反证、局部规律全局化、coarse bin 当精确相等等机制。它们的 Observation 本身不足以推翻原答案；只有接受 card 指定的唯一无效桥接后才到达 target。因此通过 Phase 1.1。

## 3. Phase 2 实例化硬护栏

1. 不得改变 card 的 structural graph、V 节点、F_error_class、updated target、subtype 或 easy/medium 规则。
2. 每个 undercut base 必须把 card 的 countermodels 落成可计算的具体数值/关系，并由 validator 重新证明两个模型都可行。
3. `affine_zero_offset` 的 F 区间必须同时包含 B=0 与候选非零 B 的预测；`axis_permutation` 的 F 测试必须让两轴共同变化，不能意外识别唯一 mapping。
4. `contingency_dependence` 必须明确区分 period-level aggregate co-movement 与 item-level joint occurrence；不得让 Observation 直接给出联合事件记录。
5. `quantifier_scope_binding` 必须给出显式有限关系表或见证配置，证明弱量词序不蕴含共同见证。
6. `channel_synchronization` 必须至少存在两种满足全部粗时间约束的 pairing，并产生不同答案。
7. V/F 继续使用冻结的相同包装，来源类别、数字精度、专业程度与长度匹配；不得重新出现系统性的 `another/different/neighboring/...` arm 泄漏。
8. 每个 F 在接受唯一错误桥接后，其余计算必须全部正确；不得再引入算术错误、来源失效或未说明现实知识。

## 4. 下一硬停点

Phase 2 应交付 deterministic builder、60-base JSONL、300-arm JSONL、两份盲审模板、hidden key、validator、Observation-only 防泄漏脚本和待填 manifest。生成后必须先做数据负责人/Codex construction review；在双人盲审与机器闸门通过前不得运行 uncapped Qwen，在 uncapped gate 通过前不得运行 capping。
