from pathlib import Path
import json,datetime,shutil,hashlib
root=Path('D:/CEDAR/arr_c1_signed_20260928')
exp=root/'explore_outputs';closure=exp/'closure_20261003';e=closure/'evidence'
audit=json.loads((e/'explore_outputs/completion_audit_20261003T113216Z/completion_audit.json').read_text())
final=json.loads((e/'explore_outputs/completion_verified_20261003T114127Z/FINAL_VERIFICATION.json').read_text())
summary=json.loads((e/'analysis/formal/SUMMARY.json').read_text())
transfer=json.loads((closure/'transfer_verification.json').read_text())
assert final['status']=='passed' and transfer['status']=='passed'
now=datetime.datetime.now(datetime.timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%SZ')
tz=datetime.timezone(datetime.timedelta(hours=8))
def bj(epoch):return datetime.datetime.fromtimestamp(epoch,tz).strftime('%Y-%m-%d %H:%M:%S')
formal=json.loads((e/'runs/formal/status.json').read_text())
acquired=bj(formal['updated']);analyzed=bj(final['campaign']['updated'])
receipt=dict(time_utc=now.isoformat(),automation_id='cedar-c1-30',action='delete',status='deleted',tool='mcp__codex_app__automation_update',tool_response='Deleted automation in the app.',reason='C1 completed; integrity and primary analysis verified; no managed scientific processes or held locks; evidence transfer verified.',configuration_backup='automation_before_closure.toml')
(closure/'monitor_closure.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
for name in ['status.json','STATUS.md']:
 src=exp/name
 if src.exists():shutil.copy2(src,closure/('before_closure_'+name))
shutil.copy2(e/'explore_outputs/completion_verified_20261003T114127Z/FINAL_VERIFICATION.json',closure/'FINAL_VERIFICATION.json')
status=dict(time_utc=now.isoformat(),campaign=final['campaign']['stage'],current_stage=final['campaign']['stage'],health='complete_verified',smoke_completed=52,smoke_total=52,formal_completed=3840,formal_total=3840,formal_completed_beijing=acquired,analysis_completed_beijing=analyzed,delta={'formal':17,'smoke':0,'log_bytes':1788},active_cell=None,active_cell_age_seconds=None,locks=final['resources']['locks'],managed_processes=final['resources']['processes'],disk_free_gib=round(final['resources']['disk_free_bytes']/2**30,2),asset_verification={'status':'passed','files':26,'fresh_full_weight_hash_performed':True,'current_sizes_mtimes_match':True,'evidence':'closure_20261003/evidence/explore_outputs/completion_audit_20261003T113216Z/asset_rehash.json'},protocol_sha256=final['protocol_sha256'],prepared_manifest_sha256=final['prepared_manifest_sha256'],issues=[],experiment_restarted=False,protocol_changed=False,action='Completed development C1. Monitoring stopped. H1 remains design and precision planning only.',evidence='closure_20261003/FINAL_VERIFICATION.json',evidence_sha256=hashlib.sha256((closure/'FINAL_VERIFICATION.json').read_bytes()).hexdigest(),analysis_primary_recomputed=True,primary_holm_significant=0,monitor=receipt,local_transfer=transfer)
for p in [exp/'status.json',closure/('completion_status_'+stamp+'.json')]:p.write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
with (exp/'monitor_history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(status,ensure_ascii=False)+'\n')
labels=[('same_broad_common','broad：共同响应变化'),('same_broad_interaction','broad：V/F 交互'),('same_specific_common','specific：共同响应变化'),('same_specific_interaction','specific：V/F 交互')]
rows=[]
for key,label in labels:
 v=summary['primary'][key];ci=v['descriptive_ci95']
 rows.append(f"| {label} | {v['equal_family_mean']:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] | {v['exact_signflip_p']:.6f} | {v['holm_four_primary']:.6f} |")
table='\n'.join(rows)
report=f"""# CEDAR signed-dose C1 完成核验

更新：{now.astimezone(tz).strftime('%Y-%m-%d %H:%M:%S')}（北京时间）。本报告对应 c1_signed_20260928 开发集实验。

终端命令执行与 SSH 登录已恢复并实际验证。52/52 工程 smoke、3840/3840 正式前向和四主端点分析均已完成；采集完成时间为 **{acquired}**，分析完成时间为 **{analyzed}**。最终完整性核验通过，监控已结束。

## 执行与完整性

- 完成顺序：CPU 输入/全矩阵范数门禁 → 权重校验 → 52 smoke → 3840 正式前向 → 完整性与分析。smoke 不计入正式独立证据。
- 正式矩阵：96 prompts × 2 target conditions × [noop + replacement + 2 signs × (true + 8 random)]。覆盖 12 families、24 复用 bases。
- 3892 条记录的 JSON/NPZ 配对、唯一性、矩阵覆盖、聚合记录一致性、数组 SHA256 与有限值均已核验；原有 384 份捕获数组的哈希也通过。
- 所有新 no-op 与原始数组最大绝对误差 0；smoke 重复误差 0；终端 logit 重建误差 0。正式随机方向 BF16 实际范数匹配最大相对误差 0.000630547，低于冻结阈值 0.005。
- 冻结协议、代码、依赖、方向、尺度和 token 来源已核验。运行保持 BF16、SDPA、无 CPU/disk offload；hook 次数、单次修改与清理均符合协议。
- 最终重新完整读取并计算 26 个权重/cap 资产的 SHA256（含 17 个模型分片），全部匹配；随后文件大小和修改时间复核一致。
- 相比最后成功监控快照 2026-10-03 09:53:33 UTC，正式记录由 3823 增至 3840（+17），正式日志增加 1788 字节至 390410 字节。
- 核验时无本批 campaign/runner/analysis 进程，两个锁均空闲；原 campaign PID 1884511、runner PID 1948775 均已退出。GPU 1/2/4/6 各仅 17 MiB 占用、利用率 0%，无计算进程。远程磁盘剩余约 {final['resources']['disk_free_bytes']/2**30:.2f} GiB。
- 累计 GPU 阶段墙钟时间 {final['budget']['used_seconds']/3600:.2f} 小时，预算计费已结束；低于冻结 10 天上限。

## 四个冻结主端点

U 为 uncapped，P 为 full_p25；主分析使用已锁定的同 prompt S1 仿射尺度校正。先计算每个 base 的正负剂量有限差分，再减去 8 个随机方向的平均响应；共同响应为 V/F 平均的 P−U，交互为 [P,V−P,F]−[U,V−U,F]。broad 是其他三个选项均值减初始选项，specific 是目标选项减另外两个非初始/非目标选项均值。

| 端点 | 家族等权效应 | 描述性 95% CI | 原始 p | Holm p（四检验） |
|---|---:|---|---:|---:|
{table}

全部主统计已从记录 logits 独立重算，与保存结果在 1e-12 容差内一致。区间采用 20,000 次 family bootstrap，seed=20260928；检验枚举 4096 个 family 符号翻转，四个主检验统一 Holm 校正。统计汇总单位为 12 个 family 均值，3840 次前向和随机方向都不能作为独立样本数。

四主端点均未达到 Holm 校正后的 0.05 阈值。正式记录相对对应 no-op 的选项 argmax 改变为 0（3840 条包含 192 条 no-op，实际干预条目 3648）。完整条件/方向/剂量、N/R、replacement、raw 与 family-out 敏感性结果保存在分析 JSON 中。

这些是开发集探索性结果：已有 bases 被复用，cross-fit 训练集重叠，family 对称独立假设尚未建立。区间均跨 0，不能由“不显著”宣称等效或无效应。解释仅限固定轨迹、此方向/位置/剂量下的终端读出，不能推断自然生成行为、普遍是否使用理由或自然中介效应。U/P 的实际原始位移不完全相等，锁定 S1 校正也不保证消除全部干预后尺度变化。

## 连接恢复及审计修正

本轮已实际验证本地终端读写和 SSH 执行。旧 SSH 会话已不存在；首次直接连接超时，原 127.0.0.1:10090 代理端口拒绝连接。启动已有 FlClash 后直接 SSH 重新连通。凭据仅用于 SSH stdin，未写入命令参数或产物。没有足够证据从这些现象还原此前聊天工具不可用的根因；服务器实验在监控连接不可用期间继续执行，本次无需重启实验。

第一次收尾审计的独立脚本错误地要求 model_generate_used=False，因此把全部 3892 条记录标成 generation 错误。冻结历史 runner 实际通过 model.generate 的 KV-cache 接口与 FrozenTokenForcer 重放完整既定 token 序列，并校验 prefix 和终端 probe；model_generate_used=True 与 natural_generation=False、generated_text=False 同时成立，符合协议。纠正后的审计逐条检查该语义、固定 token 哈希、capture-before-force 顺序，3892 条全部通过。初次失败诊断完整保留；修正只涉及收尾审计，没有修改冻结实验代码、阈值、数据或统计结果。

旧 c1_20260928_v2 失败记录未覆盖、未重启；S1/P1 未重跑。本次只进行读取、校验、分析复算和证据归档。

## 证据与监控收尾

本地证据根目录：D:/CEDAR/arr_c1_signed_20260928/explore_outputs/closure_20261003。

- FINAL_VERIFICATION.json：通过的最终核验，包含对初次诊断的透明修正及资源状态。
- evidence/explore_outputs/completion_audit_20261003T113216Z/：全部记录哈希、26 个资产重哈希、原始审计与主端点独立复算。
- evidence/analysis/formal/：SUMMARY.json、base_endpoints.json、details.json。
- evidence/prepared/、evidence/scripts/、evidence/logs/、evidence/runs/：冻结来源、运行日志与所有记录 JSON。
- c1_completion_evidence_20261003.tar.gz：已下载并校验归档 SHA256，解包后逐文件验证 {transfer['files_verified']} 个文件全部匹配。SHA256：{transfer['archive_sha256']}。
- 原始 3892 份 NPZ 数组保留在远程 runs/smoke 和 runs/formal；本地保存全部哈希与 JSON，没有声称已将所有 NPZ 下载到 D 盘。
- monitor_closure.json：监控 cedar-c1-30 已通过应用工具删除，后续不再触发。automation_before_closure.toml 保留原配置备查。

远程证据位于 /data4/WuMingrui/TianJinyu/CEDAR/c1_signed_20260928/explore_outputs；本报告和收尾凭据同步保存到其 closure_20261003 子目录。监控结束仅表示本批 C1 完成，不表示全部论文实验完成。

## H1 下一步（仅设计与精度规划）

1. 构建真正独立的新结构候选库；对旧 PGF-01..08 的规则依赖图与理由推理图进行去标签、带符号和参数位置的同构审查。换名、换 salt、加长旧链或新 family ID 不能自动算独立结构。
2. 使用独立 Python 求解器与 Prover9 双重认证，以 Mace4 检查 F 反模型；记录全部认证失败。保持 N/R/V/F 匹配、答案转换平衡、理由长度与选项暴露可比，不按模型结果筛题。
3. 按独立结构组规划精度，先确定科学上有意义的效应范围/区间宽度，再冻结组数、缺失处理、主指标、多重比较及推理预算。旧 S1 方差不能直接用于 C1 交互或 P1 AUC 的功效声明。
4. 若将 C1 方向与端点带入 H1，使用开发集已冻结的版本；不得在 H1 标签上重新拟合后称为独立确认。H1 冻结协议与新题库门禁尚未完成，当前没有启动 H1 GPU 实验；Jev 仍是未来工作。
"""
(closure/'C1_COMPLETION_REPORT.md').write_text(report,encoding='utf-8')
short=f"""# C1 已完成并核验

更新时间：{now.isoformat()}。

- smoke：52/52；正式：3840/3840。
- 采集完成（北京时间）：{acquired}；分析完成：{analyzed}。
- 记录、输入、冻结来源、权重哈希与四主端点复算均通过。
- 无本批科学进程；锁已释放，GPU 已释放。
- 四主端点 Holm 检验均未显著；开发集探索性结果，不等同于等效。
- 监控 cedar-c1-30 已结束并删除，原配置已备份。
- H1 仍仅设计与精度规划，Jev 为未来工作。

详见 [完成报告](closure_20261003/C1_COMPLETION_REPORT.md) 和 [最终核验](closure_20261003/FINAL_VERIFICATION.json)。
"""
(exp/'STATUS.md').write_text(short,encoding='utf-8')
with (root/'RUNBOOK.md').open('a',encoding='utf-8') as f:
 f.write(f"\n\n## Verified completion — {stamp}\n\nC1 signed-dose finished: smoke52/52, formal3840/3840, analysis and integrity verified. No managed scientific processes or held locks remain. The heartbeat cedar-c1-30 was deleted after verified closure; its configuration is backed up. Do not restart the completed campaign. H1 remains design/precision planning only. See explore_outputs/closure_20261003/C1_COMPLETION_REPORT.md and FINAL_VERIFICATION.json.\n")
print(json.dumps({'report':str(closure/'C1_COMPLETION_REPORT.md'),'status':status['health'],'monitor':'deleted','files_verified':transfer['files_verified']},ensure_ascii=False))

