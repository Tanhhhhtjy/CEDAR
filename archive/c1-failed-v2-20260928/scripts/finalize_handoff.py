import datetime,gzip,hashlib,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
candidate=root/'signed_control_candidate.retry1.json.gz'
expected='ea4cd42a9b3a10028f8fd734d98a7dfd1c662c30f46db15702092a95aa11859a'
assert hashlib.sha256(candidate.read_bytes()).hexdigest()==expected
data=gzip.decompress(candidate.read_bytes());result=json.loads(data)
assert result['status']=='prospective_cpu_only_passed' and len(result['checks'])==3840 and not result['failures']
(root/'explore_outputs/signed_control_candidate.json').write_bytes(data)
partial=root/'signed_control_candidate.json.gz'
if partial.exists():partial.rename(root/'signed_control_candidate.json.gz.partial')
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
state=dict(updated_utc=stamp,status='blocked_v2_prospective_signed_pilot_awaiting_scope_decision',
    server_root='/data4/WuMingrui/TianJinyu/CEDAR/c1_20260928_v2',
    local_root='D:/CEDAR/arr_c1_20260928',original_s1_p1='complete',v1_engineering_forwards=32,
    v1_formal_eligibility=False,v2_gpu_forwards=0,v2_preflight_cells=5612,v2_failed_main_cells=14,
    candidate_cpu_cells=3840,candidate_cpu_failures=0,candidate_max_norm_relative_error=result['max_norm_relative_error'],
    candidate_audit_utc=result['timestamp_utc'],candidate_gzip_sha256=expected,
    proposed_pilot_forwards=52,proposed_pilot_historical_hours=result['proposed_smoke']['historical_hours'],
    proposed_pilot_gpu_started=False,h1_started=False,active_codex_monitor=False,
    last_task_process_audit_utc='20260928T054910Z',last_task_processes=[],
    note='CPU candidate completed; no GPU stage launched. Original failed gates remain preserved.')
(root/'explore_outputs/status.json').write_text(json.dumps(state,indent=2)+'\n')
bundle=root/'handoff_notes_20260928.tgz'
paths=[root/'RUNBOOK.md',*sorted((root/'explore_outputs').glob('*.md')),root/'explore_outputs/status.json',root/'explore_outputs/evidence_download_manifest.json']
with tarfile.open(bundle,'w:gz') as archive:
    for path in paths:archive.add(path,arcname=path.relative_to(root))
print(json.dumps(dict(files=len(paths),bundle=str(bundle),sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),state=state)))
