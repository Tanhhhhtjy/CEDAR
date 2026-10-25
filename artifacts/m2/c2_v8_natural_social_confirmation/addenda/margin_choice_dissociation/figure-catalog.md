# C2/V8 补充图表目录

## figure-01-margin-choice-dissociation.svg

- Purpose：并列显示同一四格实验的连续 margin 与离散 target rate。
- Data source：600个严格配对的Direct base。
- Caption requirement：左图单位为target-minus-initial logit；右图为target选择率；连线只连接
  同一arm的uncapped/p25均值，不表示时间序列；本图为post-hoc补充。
- Observation：p25令AO/S margin均向target移动，但AO平移更大；离散target选择却在S中增加
  更多。
- Interpretation：相对margin压缩与临界选择恶化可以同时发生。
- Implication：不能把负的social margin interaction直接称为行为安全收益。
- Caveat：target只覆盖A/B，四格置信区间为未校正描述性区间。

## figure-02-boundary-transitions.svg

- Purpose：检查S arm的离散翻转是否集中于uncapped margin接近零的题。
- Data source：全部600个S-arm Direct base。
- Caption requirement：横轴为uncapped margin，纵轴为p25 margin；虚线为零边界；颜色表示
  完整迁移类别；不删除other-wrong或极端点。
- Observation：initial-to-target翻转集中在决策边界附近，而大多数题保持原类别。
- Interpretation：离散结果可由连续边界平移与阈值共同产生。
- Implication：后续实验必须同时报告连续margin和四态选择。
- Caveat：相关图不能确定内部机制，也不能区分字母偏置与语义偏移。
