# C1/V7 Phase 1 family-card review

## 审核结论

24 张 family cards 可以进入 Phase 2 实例化；本结论只批准结构蓝图，不批准尚未生成的 48 个 base、192 个 arm、人工审核标签或任何模型结果。

- 24 个唯一 family，三个 stratum 各 8 个。
- 每个 family 固定两个参数变体，尚未实例化。
- 所有 family 使用冻结的 AO/S targeting mechanism；AO/S 不被用来主张结构新颖性。
- `development_overlap_audit24.jsonl` 覆盖 24/24，并以 canonical SHA-256 绑定当前 card。
- intake validator 已通过；未运行 Qwen、GPU、capping 或 response-based 筛选。

这里的 `approved` 表示数据负责人阶段的结构审核，不等于后续两名独立审核者对 base 内容的盲审通过。

公开方法复审完成后，24 张 card 仍全部保留；Phase 2 改用 solution-first 构建证书与 easy/medium
步骤边界。依据见 `docs/C1_V7_REFERENCE_METHOD_AUDIT.md`，执行合同见
`PHASE2_CLAUDE_CONSTRUCTION_BRIEF.md`。这项修正不改变 family 名册和已绑定的 overlap hash。

Phase 1.1 反证审计随后发现 F17 的原“单边增删”迁移不可能实现，已最小修订为原子替换一名
agent 的完整 allowed-item record，并重新绑定 F17 hash。证明与批准范围见
`PHASE1_1_F17_LOGIC_REPAIR.md`；其余 23 张 card 未变。

## 重点风险 family

下列 family 获准进入 Phase 2，但实例化时必须优先复核。若违反护栏，应删除该实例或退回 Phase 1 改卡，不能只换叙事表面。

| family | 风险 | Phase 2 护栏 |
|---|---|---|
| C1V7-F03 | 可能退化为旧 rounding/boundary 题 | 使用精确整数与不可分容器；不得引入小数舍入或端点是否计入的歧义 |
| C1V7-F06 | 可能与 reciprocal/orientation 题表面接近 | 明示 gross = base × (1 + rate)；V 只能更正等式中的 rate/status，不得改单位朝向 |
| C1V7-F07 | 可能退化为 weighted-average 题 | 必须从总距离/总时间推导；不得把权重或样本数作为关键节点 |
| C1V7-F09 | 可能退化为 selector/priority 题 | 优先级必须由最长前缀机械产生；不得额外给出版本选择器或优先边 |
| C1V7-F10 | 可能退化为旧 version-selector 题 | 比较对象本身的 semantic version；不得问“哪一版规则生效” |
| C1V7-F15 | 可能退化为 parity/eligibility 题 | 保留逐位 mask 与精确 pattern lookup；不得把结果化简为奇偶性或集合成员计数 |
| C1V7-F17 | 可能退化为 partition/reachability 题，且单边增删无法实现唯一匹配迁移 | 必须求全局一一匹配；V 只能把一名 agent 的完整 allowed-item record 作为单个原子记录替换，其他行保持不变；不得改写成单边增删 |
| C1V7-F18 | 可能退化为 boundary/reachability 题 | 必须先推导两个区间，再判关系；不得以端点是否计入作为唯一难点 |
| C1V7-F20 | 可能退化为 directed reachability 题 | 图的连通性保持不变；答案由最大加权路径决定，V 只更正 duration |
| C1V7-F24 | 可能退化为 rule directionality/reachability 题 | 必须计算属性集合的固定点 closure 并检查 key minimality；不得只验证单条蕴含 |

其余 14 张为较低重叠风险，但仍必须接受相同的 base-level 双人审核和 overlap audit。

## Phase 2 不可改变项

1. 不改变 family 的答案生成机制、V changed node 或 V update mechanism。
2. 同一 family 的两个变体只改变参数和表面情境，不能变成两个不同结构。
3. 六种答案迁移必须按冻结协议平衡；不能在看到 Qwen 输出后调答案位置。
4. V 必须真正改变 gold，AO 只能无施动者地提及 target，S 只能施加社会压力而不添加事实。
5. 先生成 48 base，再由冻结 builder 确定性展开 192 arms 并完成 construction validation；在 construction review 与后续双人审核通过前不得运行模型。

## Phase 2 交接条件

Phase 2 必须按下列顺序推进：

1. 先构建固定 4 个 sentinel，检查 schema、唯一 gold、V 单节点更新与构建证书；不运行模型。
2. sentinel 获数据负责人批准后，48 个 base 全部由当前 24 张卡一次性实例化；development overlap 自动审计和作者 construction audit 通过。
3. 冻结 builder 确定性展开 192 arms，验证每臂数量、迁移平衡、AO/S 纯度、base/dataset hash 与重复构建一致性。
4. construction review 通过后才生成两份空白 blind review pack；private key 不得提交。
5. 两名独立审核者完成准确性、唯一 gold、V 有效性、AO/S 纯度审核；一致率、Cohen's κ 和各自 gold accuracy 达到冻结门槛。
6. 最终标签与逐题 base hash、完整 dataset hash 一致后才可 freeze；在此之前不得运行 Qwen 或 capping。
