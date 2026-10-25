# 图表说明

## figures/s1_selectivity_verified.pdf / .png

目的：比较同一主源上 raw 与 affine-restored 的 V−F margin 差异。点为 12 个 family 均值，方块为等权总体均值，误差线为 20,000 次 paired-family bootstrap 的描述性 95% 区间。数据来自核验过的 S1_SUMMARY。

读图：校正后差异缩小但仍存在；broad 全为负，specific 有 2 个正向 family。决策：可以继续设计匹配 C1，不能直接推出原因使用通路。检查：只含 uncapped-source；没有 token 独立性推断；误差线不是跨模型种子置信区间。

## figures/p1_transfer_verified.pdf / .png

目的：同时展示四方向的 family AUC 与固定阈值 balanced accuracy。左图散点为 12 个 fold-family AUC，方块及误差线为等权 AUC 和描述性 bootstrap CI；右图是 balanced accuracy。虚线为 0.5。数据来自核验过的 P1_SUMMARY。

读图：AUC 在 0.81–0.85，但 balanced accuracy 仅 0.58–0.60；不能只报高 AUC。决策：保留可解码性描述，因果用法留待 C1，跨新结构泛化留待 H1。检查：4 个方向齐全、prompt L53 主端点、每 family 4 例、fold 训练集重叠，CI 只能作描述。

原始 S1 heldout_prediction.pdf 仍在 analysis-output/formal/s1/figures：点为单条轨迹的 identity 与 affine 选项 RMSE，支持描述性的误差改善；不能当作 unseen-family 预测图，也不是因果解释率。
