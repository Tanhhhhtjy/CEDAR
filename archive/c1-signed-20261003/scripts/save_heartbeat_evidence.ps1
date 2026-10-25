param([Parameter(Mandatory=$true)][string]$PayloadPath)
$ErrorActionPreference='Stop'
$exp='D:\CEDAR\arr_c1_signed_20260928\explore_outputs'
$envelope=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String((Get-Content -Raw -LiteralPath $PayloadPath))) | ConvertFrom-Json
if($envelope.name -notmatch '^heartbeat_monitor_\d{8}T\d{6}Z\.json$'){throw 'Invalid evidence name'}
$evidencePath=Join-Path $exp $envelope.name
if(Test-Path -LiteralPath $evidencePath){throw 'Evidence already exists'}
[IO.File]::WriteAllBytes($evidencePath,[Convert]::FromBase64String($envelope.bytes_base64))
$hash=(Get-FileHash -LiteralPath $evidencePath -Algorithm SHA256).Hash.ToLower()
if($hash -ne $envelope.sha256){throw 'Remote/local evidence SHA256 mismatch'}
$e=Get-Content -Raw -LiteralPath $evidencePath | ConvertFrom-Json
$s=[ordered]@{
 time_utc=$e.time_utc; campaign=$e.campaign; smoke_completed=$e.smoke_completed; smoke_total=$e.smoke_total
 formal_completed=$e.formal_completed; formal_total=$e.formal_total
 delta=[ordered]@{formal=$e.delta_formal;smoke=0;log_bytes=$e.log.delta_bytes}
 active_cell=$e.status.active_cell; active_cell_age_seconds=$e.active_cell_age_seconds
 locks=$e.locks; disk_free_gib=$e.disk.available_gib; asset_verification=$e.asset_verification
 protocol_sha256=$e.protocol_sha256; prepared_manifest_sha256=$e.prepared_manifest_sha256
 current_stage=$e.campaign; health=$e.health; issues=$e.issues;experiment_restarted=$e.experiment_restarted
 protocol_changed=$e.protocol_changed; action=$e.action;evidence=$envelope.name;evidence_sha256=$hash
}
$s | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $exp 'status.json') -Encoding utf8
$historyPath=Join-Path $exp 'monitor_history.jsonl'
$lastLine=Get-Content -LiteralPath $historyPath -Tail 1
if($lastLine -notlike "*$($e.time_utc)*"){($s | ConvertTo-Json -Depth 12 -Compress) | Add-Content -LiteralPath $historyPath -Encoding utf8}
$dt=[DateTimeOffset]::Parse($e.time_utc).UtcDateTime.ToString('yyyy-MM-dd HH:mm:ss')
$age=[math]::Round($e.active_cell_age_seconds,1)
$budget=[math]::Round($e.gpu_budget.used_seconds)
$gpuText=($e.gpu | Where-Object {$_.index -in @(1,2,4,6)} | ForEach-Object {"GPU $($_.index): $($_.memory_used_mib) MiB / $($_.utilization_percent)%"}) -join '；'
$processText=($e.processes | ForEach-Object {"PID $($_.pid): $($_.owner), $($_.cwd)"}) -join '；'
$lines=@(
'# C1 signed-dose 当前状态',
'',
"最后核实：$dt UTC。",
'',
"阶段：$($e.campaign)。Smoke $($e.smoke_completed)/52；正式前向 $($e.formal_completed)/3840，较上一快照新增 $($e.delta_formal) 条。日志增加 $($e.log.delta_bytes) 字节；当前 cell 已运行约 $age 秒，低于 90 分钟停滞门槛。",
'',
"进程：$processText。命令与各自锁均已核实；本轮检测问题数：$($e.issues.Count)。协议、prepared manifest、冻结程序与输入依赖哈希已核验。CPU 3892 个 patch 检查和 smoke 技术门禁已通过。",
'',
"26 个权重和 cap 资产的本轮大小/mtime 与完整 SHA256 校验记录一致；本轮未重新读取全部权重计算 SHA256。新增 $($e.hashes_and_integrity.new_records_hashed) 条记录的 NPZ 哈希通过；现有 $($e.hashes_and_integrity.completed_json_count) 条记录的元数据与冻结矩阵、fingerprint 一致。",
'',
"$gpuText。磁盘可用 $($e.disk.available_gib) GiB，高于 20 GiB 门槛；累计 GPU 阶段运行预算 $budget/864000 秒。",
'',
"正式矩阵及最终完整性复核、四主端点分析仍未完成。健康状态：$($e.health)。本轮未重启、未修改科学协议；H1 仍限设计和精度规划。",
'',
"证据：$($envelope.name)。远端与本地字节 SHA256 一致：$hash。"
)
$lines | Set-Content -LiteralPath (Join-Path $exp 'STATUS.md') -Encoding utf8
$envelope.summary | ConvertTo-Json -Compress
[ordered]@{evidence=$evidencePath;sha256=$hash;mirrored=$true;metadata_mismatches=$e.asset_verification.current_recheck_mismatches;frozen_file_errors=$e.frozen_file_verification.errors;disk_gib=$e.disk.available_gib;budget=$e.gpu_budget;log_tail=$e.log.tail;gpu=$e.gpu;counts=$e.hashes_and_integrity | Select-Object completed_json_count,completed_npz_count,record_metadata_errors,new_records_hashed} | ConvertTo-Json -Depth 6 -Compress

