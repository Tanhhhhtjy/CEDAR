# CEDAR ARR 补实验运行说明

远程服务器：用户指定的 IPv6 SSH 主机，端口 50000。工作目录：`/data4/WuMingrui/TianJinyu/CEDAR`。密码未写入任何脚本、配置、报告或 Git 文件。

本批自动运行的范围是 S1（留出词表的尺度预测与末端归一化分解）和 P1（family 留出的双向跨条件探针）。C1 因果 patch 与 H1 独立结构确认，须根据这批实际结果继续设计和执行，当前不能标成已完成。

## 状态与日志

登录后执行：

```bash
cd /data4/WuMingrui/TianJinyu/CEDAR
cat explore_outputs/campaign_status.json
cat explore_outputs/assets_status.json
tail -n 20 logs/campaign.log
tail -n 20 logs/formal.log
```

`waiting_for_verified_weights` 表示恢复权重中，还没有真实模型前向；`smoke` 是 12 次配对/重复验证；`formal` 是 384 次固定轨迹前向；`s1_p1_complete_c1_h1_pending` 才表示本批采集和分析完成。出错会写 `failed` 和 traceback，不会绕过验证继续。

下载期间使用临时 SSH 反向转发，经本机现有代理访问公开 Hugging Face 文件。因此下载完成前需要保持本机网络和转发进程运行。权重校验完成后的 GPU 作业在服务器执行，脱离当前交互 SSH 会话。任何下载中断都需先查看错误，再恢复；已有分片可以续传。

2026-09-27 下载曾因 HTTP 流中断退出，约 15.9GB 缓存被保留。已增加每项下载最多 16 次的瞬时网络错误重试，间隔 15–60 秒，并按已有完整/未完成分片估算剩余空间。校验失败和不可恢复错误仍停止。重试记录位于 `explore_outputs/download_retry_events.jsonl`；首次失败日志保留为 `logs/fetch_assets_initial_failure.log`。下载完成前请保持本机代理及 Codex 会话运行，避免电脑休眠。

## 已固定的规模与边界

- Qwen3-32B 原版 BF16，revision `9216db5781bf21249d130ec9da846c4624c16137`；官方 p25 cap 原 SHA-256，SDPA，与历史依赖版本一致。
- 192 条原始冻结轨迹，哈希与投稿 E2B 记录一致。先 12 次 smoke，再 384 次 formal；不生成新的推理文本。
- 物理 GPU 1、2、4、6，每卡模型预算 18 GiB；启动前每卡需至少 22 GiB 可用。不终止其他用户作业。
- S1 的 A/B/C/D 及其单 token 空白变体不参与拟合；非选项拟合/测试 token 分离，另做 family-out 参数预测。
- P1 主分析使用第 53 层、生成推理前的 prompt 边界、unit-RMS 表示、训练 family 拟合的初始/目标答案身份控制。U→U、U→P、P→P、P→U 全部报告。
- CPU 工程测试和合成数据测试只验证程序，不能算论文新实验结果。

## 恢复与产物

仅在原任务已退出时恢复 campaign（脚本有进程锁，防止重复启动）：

```bash
nohup /home/WuMingrui/miniconda3/envs/cedar/bin/python -u scripts/run_campaign.py > logs/campaign.log 2>&1 < /dev/null &
```

已完成记录具有逐文件 SHA-256；同一代码、协议与运行配置可恢复。程序拒绝混合不同来源的记录。若修复会改变科学含义，应使用新的运行目录并记录协议变更。

主要结果：

- `runs/smoke/gate.json`：数值重复、末层重建与历史 anchor 偏差。
- `runs/formal/`：逐条 JSON、全词表 logits、三层状态、归一化状态和分母交换读出。
- `analysis-output/formal/s1/S1_SUMMARY.json`、`S1_DETAILS.json`：尺度与 family 分析。
- `analysis-output/formal/p1/P1_SUMMARY.json`：双向探针、敏感性结果和配对标签置换。

旧投稿包和旧服务器 checkout 保持为历史参照。当前记录不宣称论文已具备 ARR 投稿条件。

## 本地保存路径变更（2026-09-28）

用户指定本地实验产物统一保存到 D:/CEDAR。本批规范目录为 D:/CEDAR/arr_revision_20260927，运行说明为该目录 RUNBOOK.md；不要重新在 C:/Users/legion/Desktop/paper_proposal/arr_revision_20260927 创建产物。原始结果、历史来源路径与协议指纹保持原样；历史元数据中的 C 盘路径不是当前输出位置。

第一批已完成。后续实验使用 D:/CEDAR 下的新命名目录，先冻结新方案，再执行。服务器路径不变。D:/CEDAR/ARR_RESULTS_INDEX.md 提供入口。

2026-09-28 续跑记录：C1 旧版完成 32 次工程试跑，但对照不满足新版要求，不能用作正式因果结论；新版 C1 在 CPU 范数预检查阶段被阻塞，GPU 前向为 0。已有状态的 S1/P1 附加诊断已完成。当前无本批任务进程，H1 未启动。详见 D:/CEDAR/arr_c1_20260928/RUNBOOK.md 和该目录 explore_outputs/STATUS_AND_FINDINGS.md。不要按本文件的原第一批 campaign 命令重启 C1。

迁移期间尝试更新旧 cedar 监控项时，自动化接口返回该项不存在，未新建或重启监控。若以后恢复监控，须使用本节 D 盘路径和用户要求的 30 分钟间隔。
