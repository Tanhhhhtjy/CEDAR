# CEDAR 实验结果入口

更新日期：2026-09-28。用户要求：今后所有本地实验结果、图表、日志、证据镜像和运行脚本均放 D:/CEDAR，不再放 C:/Users/legion/Desktop/paper_proposal。

## 当前执行：C1 signed-dose（2026-09-28 晚）

用户已批准新方案。服务器独立目录 c1_signed_20260928 已启动，顺序为 52 次技术验证、通过工程与资源门禁后自动继续 3840 次正式前向、完整性与四项主端点分析。状态以该目录时间戳为准；下文 v2 blocked 状态为历史，不代表当前新任务。

- [本批运行说明](arr_c1_signed_20260928/RUNBOOK.md)
- [新冻结协议](arr_c1_signed_20260928/PROTOCOL.md)
- [启动与核验记录](arr_c1_signed_20260928/explore_outputs/)
- [H1 独立结构设计](arr_c1_signed_20260928/h1_design/DESIGN.md)

服务器已通过全 3892 个配置的 CPU 扰动检查及方向/原始数组校验；CPU 配置通过不等于模型前向完成。新建 cedar-c1-30 监控，每 30 分钟核实实际进度并保存证据。H1 当前只做设计和 CPU 精度规划，Jev 留作未来工作。

## 当前第一批 ARR

- 目录：[arr_revision_20260927](arr_revision_20260927/)
- 运行与恢复说明：[RUNBOOK.md](arr_revision_20260927/RUNBOOK.md)
- 与论文旧实验的逐项差别及下一步建议：[对照与实验建议](arr_revision_20260927/explore_outputs/2026-09-28--cedar-arr--r01--comparison-and-next-experiment.md)
- 已完成实验复盘：[第一批复盘](arr_revision_20260927/explore_outputs/2026-09-28--cedar-arr--r01--round-review.md)
- 严格分析、统计附录与图表：[分析包](arr_revision_20260927/explore_outputs/batch_review_20260928/)
- 完整性：[核验 JSON](arr_revision_20260927/explore_outputs/batch_integrity_20260928.json)
- 新旧记录逐条对照：[对照 JSON](arr_revision_20260927/explore_outputs/comparison_audit_20260928.json)
- 迁移与哈希：[迁移清单](arr_revision_20260927/explore_outputs/migration_20260928.json)

第一批 S1/P1 完成，不代表所有补实验完成。C1 有 32 次旧工程试跑；新版在 CPU 数值检查阶段被阻塞，正式 C1 和 H1 尚未完成。当前证据镜像含关键输出，完整原始数组与权重仍在 /data4/WuMingrui/TianJinyu/CEDAR。

## 历史 C1 v2 阻塞与已有状态补诊断（2026-09-28 白天）

- 新目录：[arr_c1_20260928](arr_c1_20260928/)
- 当前状态、失败原因与新增诊断：[核验报告](arr_c1_20260928/explore_outputs/STATUS_AND_FINDINGS.md)
- 安全续跑说明：[C1 RUNBOOK](arr_c1_20260928/RUNBOOK.md)
- 完整性：[证据下载清单](arr_c1_20260928/explore_outputs/evidence_download_manifest.json)
- 待确认的新范围：[52 次技术验证方案](arr_c1_20260928/explore_outputs/NEXT_STEP_PROPOSAL.md)
- 最新研究取舍与 Jev 未来工作：[研究优先级备忘录](arr_c1_20260928/explore_outputs/RESEARCH_PRIORITIES_20260928.md)

本次 384 个原数组校验通过，P1 主方向预测重建误差为 0。C1 v2 计划的 5612 个格子中，14 个随机对照无法满足 BF16 实际范数 0.5% 匹配门槛；已在模型加载前停止。旧 32 次试跑仅供工程审计。当前没有本批待监管的 GPU 进程，不启动空跑监控。

修正正负方向的独立候选设计已通过 3840 个格子的 CPU 检查，未运行模型。建议下一步先做 52 次技术验证，预计约 1.5–2 小时；该范围变更待确认，不能把候选检查通过当作原 C1 完成。

原论文材料在 CEDAR_overleaf_AAAI27/，原投稿支撑包在 CEDAR_SUPPORTING_DATA/；原 ZIP 镜像放 D:/CEDAR/CEDAR_SUPPORTING_DATA.zip。不要把同一冻结轨迹重复运行计为新独立样本。

旧 cedar 自动化在本次接口查询/更新中无法定位，未创建新的监控。以后恢复时应使用 D 盘路径、30 分钟间隔，仅在有待监管作业时运行。

迁移状态：D 盘全部本地副本与投稿证据 ZIP 已通过 SHA-256 核验。自动审批拒绝删除 C 盘原件，已向用户请求明确确认；确认前 C 盘原件保留，后续只向 D 盘写入。
