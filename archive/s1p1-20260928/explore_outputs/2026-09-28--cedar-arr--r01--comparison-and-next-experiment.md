---
type: results-report
date: 2026-09-28
experiment_line: cedar-arr
round: 1
purpose: comparison-and-next-experiment
status: comparison-verified-next-experiment-proposed-not-started
source_artifacts:
  - comparison_audit_20260928.json
  - batch_review_20260928/analysis-report.md
  - batch_review_20260928/stats-appendix.md
  - batch_review_20260928/figure-catalog.md
  - batch_review_20260928/provenance.json
  - batch_integrity_20260928.json
linked_experiments:
  - ../PROTOCOL.md
linked_results:
  - 2026-09-28--cedar-arr--r01--round-review.md
---

# CEDAR ARR / Round 1 / Comparison and Next Experiment / 2026-09-28

## 1. 摘要与决定

建议：先补齐可由已保存状态完成的 S1/P1 配套诊断，再把第一笔新增 GPU 计算用于有明确对照的 C1 因果作用检验；随后冻结预测，在 H1 独立新结构上确认。当前不启动 C1/H1，不改原冻结协议。若资源只允许一个更大的补实验，选择取决于主张：机制主张优先 C1；跨结构稳健性和评价论文主张优先 H1。不可把开发集上的更多重复运行当作新结构证据。

本次 384 次采集是原 E2B 配对运行的新增观测，不是 384 个独立的新样本，也没有新增自然生成行为测试。真正新增的是：完整词表采集后的非选项留出尺度预测、末端归一化分解、生成推理前的 prompt 边界状态，以及同一个探针的双向跨条件测试。

## 2. 原论文与本次实验分别回答什么

原 E2B 的问题是：把可见推理文本固定以后，capping 引起的答案分数变化还会不会存在？它控制的是“生成了不同的推理文字”这个替代解释。冻结的是每条轨迹在 U/P 配对运行中的 token 序列，不是强迫 V/F 或所有样本共享一段文字，更不是固定模型内部隐藏状态。

本次 S1 的问题是：已知固定轨迹下仍有分数变化，其中有多少可以被一般尺度模型预测？用没有包含答案选项的 token 拟合以后，对答案的预测是否仍成立，校正后 V−F 差异是否仍在？

原 RG1 已做内部状态传播诊断和条件内探针。原探针在 U 状态训练并测试，在 P 状态另训练并测试，说明两边各自有可解码线索。本次 P1 把源条件的预处理和同一个分类器固定，再应用到另一条件，检验 U→P 和 P→U；跨条件成功仍不等于语义表征完全相同或模型因果使用该信息。

V/F 在数据构造中分别对应独立求解器认证的有效理由与含认证致命错误的谬误理由。上一条对话中把 V/F 仅说成“未解释的两类标签”不准确。真正需要谨慎的是：从 V/F 标签可解码，不能直接推出探针识别了抽象的逻辑有效性；它仍可能依赖其他相关特征。

## 3. 设置与逐项差别

| 项目 | 原论文 E2B / RG1 / 收缩分析 | 本次 ARR S1/P1 |
|---|---|---|
| 模型与干预 | Qwen3-32B BF16、原 p25 配置 | 同配置；不是模型升级或新方法 |
| 输入与规模 | 96 prompts × 2 trace sources × 2 forward conditions = 384 forwards | 同一组 192 轨迹、384 配对条件；另 12 次工程 smoke |
| 可见文本控制 | 同一轨迹的 U/P 重放逐 token 固定，使用增量 KV cache | 直接调用原 run_one，未改重放逻辑 |
| E2B 主读出 | 四选项 logits、margin、argmax、数值保真与 cap telemetry | 保留原读出，再捕获完整词表 logits 和额外状态 |
| 原收缩分析 | 每条样本用四个 A/B/C/D 点拟合 slope/intercept，并在这四点计算拟合质量；主要 RG1 记录也不是本轮新增数据 | 在非选项词表分层抽样后分 fit/test，A/B/C/D 及识别出的空白变体不参与拟合；另做 leave-family-out 参数预测 |
| 归一化 | 原主要是选项级描述性仿射分析 | 新存 norm 前/后状态、权重与 epsilon，计算 RMS 分母互换读出；这是数值分解，不是已完成 C1 |
| 内部表示 | RG1 已有 45–63 层答案边界传播诊断、随机参考和辅助探针 | 重点存 53/60/63 层 prompt 与 answer 两个边界；没有重新覆盖原 RG1 的全部随机控制 |
| 探针比较 | 条件内各自拟合；原文重点是 L60 答案边界，已有 family 留出 | 主端点为 L53 prompt 边界，训练 family 内 identity nuisance 处理、unit-RMS、固定 logistic；U→U/U→P/P→P/P→U 齐全 |
| 探针数值比较 | 原 pooled out-of-fold AUC，base 中心化、训练折标准化与 PCA | 主指标为 family 等权 AUC，预处理和取样位置不同，不能说 AUC 从 0.77 提升到 0.83 |
| 结论范围 | 可见推理文本变化不是固定轨迹分数变化的必要条件；已有尺度、传播和可解码性线索 | 尺度解释受到更严格的预测检验；新增加跨条件可读性证据；仍没有因果机制或独立结构确认 |

本次历史 runner SHA-256 与采集时 provenance 一致。新旧 384 条记录的配对键全部匹配；prompt/轨迹/拼接输入哈希无一不一致；四选项 logits 最大绝对偏差为 0。详见 comparison_audit_20260928.json。此核验仅覆盖保存的历史读出与输入，不表示未保存的历史全状态也被验证相等。

若历史完整隐藏张量可用，部分读出可复用历史缓存计算；发布包中的四选项记录不足以替代完整词表和 prompt 边界观测。本次重跑的作用是补采这些观测。后续优先盘点并使用现有缓存。

## 4. 主要结果与哪些不是新发现

原论文固定轨迹 U-source 的 broad shift 是 +2.078 logits，P-source 是 +0.700；每个来源 argmax 都为 0/96 翻转。原文也已经报告负向连续 V−F 对比：U-source 的 target-minus-initial 为 -2.421875（约 -2.422），P-source 为 -0.796875。

本次从旧记录直接重算得到 U-source：broad V−F = -1.6232639，specific V−F = -1.1979167。这正是新结果中的 raw 两项。三者的区别是指标定义：target margin = broad margin + (2/3) × specific margin。因此 -2.422、-1.623、-1.198 不是互相冲突或三次独立发现。

S1 新增的有效证据是：从非选项 token 估计变换后，选项 RMSE 从 identity 3.4204 降到 same-prompt affine 0.9328；family-out affine 为 1.2315。校正后的主源 broad V−F = -1.0043，specific V−F = -0.5209。它支持“本次拟合的尺度解释不完整”，不支持“所有一般尺度或非语义解释已被排除”。原收缩分析本来也承认 V/F 下有非仿射残差；本次是加强控制，不是第一次发现残差。

P1 新增的跨条件主结果：U→P family AUC 0.8333，P→U 0.8125；balanced accuracy 分别 0.5833、0.6042。AUC 0.83 不能读成 83% 分类准确率。原论文 L60 的 AUC 0.769/0.767 是条件内不同流水线，不能直接与本次 L53 prompt、family 平均 AUC 比较高低。

## 5. 统计与核验边界

正式独立结构单位仍为 12 个 family、24 个 base；不能把 384 个 forwards 或 192 条轨迹当作 384/192 个独立结构。S1 四个预定主对比使用 family 符号翻转与 Holm 校正；P1 使用 100 次配对标签置换和四方向 Holm 校正。原探针的 family 标签翻转、不同汇总指标与本次不可直接对比 p 值。

本次主 P1 pooled OOF AUC 诊断约为 0.608（U→P）与 0.628（P→U），低于 family 内 AUC；不同 fold 分数标尺影响跨折合并，因此它仅作诊断。这也说明“同 family 内相对排序不错”不等于“跨 family 的统一判定阈值可靠”。保留两种指标，禁止挑更高者替换主端点。

本说明用现有核验分析包，不重训探针，不再次运行模型。S1/P1 为开发集扩展分析，不能称为独立确认。

## 6. 现有图的含义

- S1 图 batch_review_20260928/figures/s1_selectivity_verified.png：raw 是旧固定轨迹效应的再现；restored 才展示新尺度控制后的差异。应突出配对校正，而不是把 raw 当新发现。
- P1 图 batch_review_20260928/figures/p1_transfer_verified.png：关键是两个跨条件方向及其有限的阈值分类表现；不能与旧论文不同流水线的 0.77 作性能提升图。
- 原 Figure 3 的传播/随机参考与本轮不是同一组完整控制；本轮没有重新跑这些随机参考，不能凭新 S1/P1 宣称 Assistant Axis 特异性。

## 7. 负结果、缺口与不能宣称的内容

两个来源 argmax 均未翻转。capped-source restored specific 均值约 +0.0902，描述性区间跨 0；不能把主源结果扩展成来源不变的结论。没有新自然生成结果，没有 H1 新结构，没有 C1 定向干预。原论文的 exposure/length 叙事仍主要是相关性证据。

附加诊断缺口：P1 单独的 initial/target identity nuisance endpoint；P1 capped-source 敏感性；S1 family-out 非选项测试误差。prompt 边界不会依赖后续生成轨迹来源，不能把源不同而 prompt 相同的 prompt 状态作为独立证据；来源敏感性重点应在 answer 边界展开，同时检查 prompt 状态重复一致性。

已导出的方向是 raw-space nuisance-adjusted 方向，而主 P1 使用 unit-RMS。C1 启动前必须写清楚方向的训练、标准化逆映射和施加空间，不能把 raw 导出向量冒充 unit-RMS 主探针方向。

## 8. 对论文主张的影响

可写：在相同冻结轨迹上扩展观测，以不含选项的拟合数据检验一般尺度预测，且观察到校正后 V−F 差异；同一预处理与探针在 U/P 间保持有限的跨条件可解码性。

不可写：新增 384 个独立样本；首次证明固定轨迹效应；首次发现非仿射残差；第一次做隐藏状态探针；已证明语义代码完全保持；已证明理由信息被因果使用；已排除长度或一般扰动解释。

本地论文正文仍有强于证据的句子，例如 main.tex 中对残差来源为 exposure 而非 assessment 的定论。新的 residual 结果不自动证明相反机制，但需要把该措辞降到待检验解释，并与补充材料已承认的限制一致。本轮不直接修改论文。

## 9. 下一步：可审阅的实验顺序

### 9.1 先用现有状态补诊断

补全 S1 留出非选项预测与归一化分母/方向的量化分解；补 P1 initial/target identity 独立端点、答案边界轨迹源敏感性及统一流水线下的 L53/L60 bridge table。保持全部原始主结果，新增产物明确标为事后诊断。无需为了这些端点再跑一整轮 384 次前向。

### 9.2 第一项新增 GPU 实验：有匹配对照的 C1

核心问题：沿候选 V/F 方向改变内部状态，会不会在留出 family 上特异性地改变理由选择性？以及同样的候选信号扰动，在 U/P 下是否具有不同的答案读出效应？

- 每个测试 family 的方向仅从其余 family 学习；V/F 配对不拆分。主位点沿现有方案定为 prompt L53，L60 如要检验原文下游猜想，应预先列为有单独 multiplicity 处理的次端点，而不是看到结果后挑层。
- 同条目、同位置执行 U→P 与 P→U 的单方向分量移植。冻结方向定义与小组剂量，不在测试 family 上选择层、阈值、剂量或最佳方向。
- 对照至少包括 no-op、等维且实际扰动范数匹配的多个预定随机方向，以及完整局部状态替换的管线/效应参考。完整局部状态替换不是全 KV cache 置换，不保证恢复全部历史条件。
- 核心终点是相对随机对照的 V−F selectivity 变化，同时报告 raw 和固定 S1 规则校正后的结果、N/R 的一般扰动、选项排序和 argmax。S1 校准参数在测试结果出现前锁定；不能为每个干预单元重新挑选最有利校正。
- 单独报告被移植的实际分量差和范数。若 U/P 在该分量原本几乎相同，移植接近 no-op，阴性结果不能说明模型不用该信息。
- 因此建议在 C1 中预先加入有限的等剂量正/负方向扰动，比较 U/P 对同一信号的 margin 响应曲线；这是针对“信息存在但影响答案的程度改变”的直接检验。该增补是待冻结的新 C1 设计，不回写原 S1/P1 协议。
- 若真实方向与随机方向一样，都仅引起尺度变化，则特异性机制不成立于该测试。若定向效应在校正后仍优于随机参考、且方向/剂量响应符合预先预测，则支持该位点与分量的因果参与，但仍不自动推出抽象语义机制。
- 即使 C1 阴性，也只能否定所测位置/方向/剂量的预测，不能推出模型所有位置都不使用理由信息。

### 9.3 H1 独立结构确认

冻结 S1/P1/C1 的主终点和最小有意义效应后，按 pilot family 方差制定独立结构数量与 precision/power 曲线。在独立 solver 验证的新结构上复验，不能用改实体名或换 random salt 充当新结构。如果 C1 不支持机制，可仍以评价/现象边界为主线做 H1，无需强求机制成功。

### 9.4 另一个有条件的优先项

若论文保留“效应由 exposure 决定”这一强主张，必须另做长度/干预暴露匹配的控制来区分内容与累计剂量；它目前不是已证实机制。可先降调该主张，避免在核心 C1/H1 之前铺开过多实验。跨模型全面扩跑与新缓解结构暂不优先。

## 10. 保存位置与证据索引

本批本地规范目录：D:/CEDAR/arr_revision_20260927。全部本地实验产物、图表、分析、运行/排障记录、脚本和已下载证据包从 C 盘原目录迁入，原始结果文件逐项 SHA-256 校验，不改结果值或冻结协议。服务器上的完整原始 NPZ 和模型权重继续保留原远程目录；本地证据镜像不是全量服务器备份。

- 新旧逐条核验：explore_outputs/comparison_audit_20260928.json。
- 第一批复盘与严格分析：explore_outputs/2026-09-28--cedar-arr--r01--round-review.md 与 batch_review_20260928/。
- 原论文本地副本：D:/CEDAR/CEDAR_overleaf_AAAI27/main.tex、supplementary.tex；本轮未声称核验当前在线 Overleaf 版本。
- 原始投稿支撑材料：D:/CEDAR/CEDAR_SUPPORTING_DATA/。
- 迁移凭证：explore_outputs/migration_20260928.json。
- 本说明未向 Obsidian 写回，未新建或启动 GPU 实验。
