"""Schema correction to completion audit; preserves original diagnostic and data."""
from pathlib import Path
import json, hashlib, datetime, os, pwd, fcntl, subprocess, shutil
root=Path('/data4/WuMingrui/TianJinyu/CEDAR')
out=root/'c1_signed_20260928'
audit=out/'explore_outputs/completion_audit_20261003T113216Z'
dest=out/'explore_outputs'/('completion_verified_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
dest.mkdir(exist_ok=False)
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
 with (dest/name).open('x') as f: json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n')
prior=read(audit/'completion_audit.json')
old_hash=sha(audit/'completion_audit.json')
errors=[]
def check(ok,msg):
 if not ok: errors.append(msg)
hashes=read(audit/'record_hashes.json')
expected_errors=['generation '+h['cell_key'] for h in hashes]
check(prior['errors']==expected_errors and len(expected_errors)==3892,'unexpected original audit errors')
m=read(out/'prepared/manifest.json')
code_evidence={}
for base,items in [(out,m['files']),(root,m['dependencies'])]:
 for rel,expected in items.items():
  p=base/rel
  h=sha(p);check(h==expected,'frozen hash '+str(p))
  if p.name in ['run_signed.py','run_arr_acquisition.py','run_e2b_fixed_trace_cached.py']:
   code_evidence[str(p)]=dict(sha256=h,expected_sha256=expected)
check(sha(out/'PROTOCOL.md')==prior['protocol_sha256'],'protocol hash')
check(sha(out/'prepared/manifest.json')==prior['prepared_manifest_sha256'],'manifest hash')
counts={s:0 for s in ['smoke','formal']}
for h in hashes:
 p=out/'runs'/h['scope']/(h['cell_key']+'.json')
 check(sha(p)==h['json_sha256'],'record changed '+str(p))
 r=read(p);cid=h['scope']+'/'+h['cell_key'];po=r['processor_order'];n=r['trace_prefix_tokens']
 check(r['model_generate_used'] is True and r['natural_generation'] is False
       and r['generated_text'] is False
       and r['replay_engine']=='model.generate_kv_cache_forced_tokens','forced replay schema '+cid)
 check(po['capture_calls']==po['force_calls']==n+1
       and po['capture_call_index']==n
       and po['capture_before_force_all'] is True
       and po['violations']==[], 'fixed replay order '+cid)
 check(r['forced_trace_token_ids_sha256']==r['trace_prefix_token_ids_sha256']
       and r['forced_trace_tokens']==n,'replay token consistency '+cid)
 counts[h['scope']]+=1
for name,h in prior['analysis_hashes'].items():check(sha(out/'analysis/formal'/name)==h,'analysis changed '+name)
assets=read(audit/'asset_rehash.json')['files']
for a in assets:
 st=Path(a['path']).stat()
 check(st.st_size==a['size'] and st.st_mtime_ns==a['mtime_ns'],'asset changed '+a['path'])
processes=[];historical_pids={}
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  args=(p/'cmdline').read_bytes().decode().split('\0')
  cwd=os.readlink(p/'cwd')
  task=any(str(out/'scripts') in a or 'c1_signed_20260928/scripts/' in a for a in args)
  task=task or (cwd.startswith(str(out)) and any(x.endswith(('campaign.py','run_signed.py','analyze_signed.py')) for x in args))
  if task or int(p.name) in [1884511,1948775]:
   item=dict(pid=int(p.name),owner=pwd.getpwuid(p.stat().st_uid).pw_name,cwd=cwd,argv=args)
   if task:processes.append(item)
   if int(p.name) in [1884511,1948775]:historical_pids[p.name]=item
 except (OSError,UnicodeError):pass
locks={}
for name in ['campaign.lock','runner.lock']:
 with (out/name).open('r') as f:
  try:
   fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks[name]='free';fcntl.flock(f,fcntl.LOCK_UN)
  except BlockingIOError:locks[name]='held'
campaign=read(out/'explore_outputs/campaign_status.json');budget=read(out/'explore_outputs/gpu_budget.json')
check(not processes and all(v=='free' for v in locks.values()),'managed process/lock remains')
check(campaign['stage']=='c1_signed_complete_h1_pending' and budget['active'] is False,'not complete')
resources=dict(processes=processes,historical_pids=historical_pids,locks=locks,
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True),
 gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True),
 disk_free_bytes=shutil.disk_usage(root).free)
result=dict(status='passed' if not errors else 'failed',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 errors=errors,counts=counts,original_audit=str(audit),original_audit_sha256=old_hash,
 correction=dict(original_predicate="not model_generate_used and not natural_generation",
 reason="Frozen historical run_one intentionally calls model.generate with capture then FrozenTokenForcer. Generated tokens are checked against the entire fixed prefix and terminal probe; no new text is generated. The original completion-only audit incorrectly rejected this API flag.",
 corrected_predicate="model_generate_used is True; natural_generation is False; generated_text is False; replay_engine is model.generate_kv_cache_forced_tokens; fixed-token hashes and processor order verified",
 producer="CEDAR_SUPPORTING_DATA/code/scripts/run_e2b_fixed_trace_cached.py:239-310,352-355",
 original_false_positive_count=len(expected_errors),original_diagnostic_preserved=True,
 scientific_code_protocol_or_records_changed=False),
 code_evidence=code_evidence,all_other_original_checks_passed=True,
 record_json_hashes_unchanged=len(hashes),npz_integrity_evidence=str(audit/'record_hashes.json'),
 assets_fully_rehashed_in_original_audit=len(assets),asset_stats_still_unchanged=True,
 analysis_hashes=prior['analysis_hashes'],primary_recomputation_matches=prior['primary_recomputation_matches'],
 protocol_sha256=prior['protocol_sha256'],prepared_manifest_sha256=prior['prepared_manifest_sha256'],
 resources=resources,campaign=campaign,budget=budget,delta=prior['delta'],
 h1_status='Design and precision planning only; no H1 or Jev execution.')
save('FINAL_VERIFICATION.json',result)
save('evidence_hashes.json',{p.name:sha(p) for p in audit.iterdir() if p.is_file()})
print('FINAL_VERIFICATION',dest, result['status'],errors,flush=True)

