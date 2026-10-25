"""Read-only completion audit; writes new evidence, never alters frozen inputs."""
import os, sys, json, time, hashlib, datetime, subprocess, pwd, fcntl, shutil, itertools
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
root=Path('/data4/WuMingrui/TianJinyu/CEDAR')
out=root/'c1_signed_20260928'
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
dest=out/'explore_outputs'/('completion_audit_'+stamp)
dest.mkdir(exist_ok=False)
def read(p): return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
 return h.hexdigest()
def save(name,obj):
 with (dest/name).open('x') as f: json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def key(c): return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:24]
started=time.time(); errors=[]; checked_files={}
def check(ok,label):
 if not ok: errors.append(label)
def filecheck(p,expected=None):
 digest=sha(p);checked_files[str(p)]=digest
 if expected:check(digest==expected,'hash mismatch: '+str(p))
 return digest
try:
 prep=read(out/'prepared/manifest.json'); ph=filecheck(out/'prepared/manifest.json')
 protocol=filecheck(out/'PROTOCOL.md')
 check(ph=='6ff92cc27dd14c3ad31ae2721abc5910899f067a01d4f42219dcf317bbda4599','prepared digest')
 check(protocol=='22514863f250e1abce981676ce77efdfa9d7fc7f0bcedbc11b859e3bf80b1afb','protocol digest')
 for base,entries in [(out,prep['files']),(root,prep['dependencies']),(root,prep['original_array_sha256'])]:
  for rel,h in entries.items():filecheck(base/rel,h)
 for name in ['directions','scales']:filecheck(out/f'prepared/{name}.json',prep[name+'_sha256'])
 for item in read(out/'prepared/directions.json').values():filecheck(out/item['file'],item['sha256'])
 filecheck(root/'runs/formal/results.json',prep['records_sha256'])
 pf=read(out/'prepared/patch_preflight.json')
 check(pf['status']=='passed' and not pf['failures'] and len(pf['patches'])==3892,'CPU preflight')
 gate=read(out/'runs/smoke/gate.json');check(gate['technical_pass'] and gate['completed']==52,'smoke gate')
 original=read(root/'runs/formal/results.json')
 trace_idx={t:i for i,t in enumerate(dict.fromkeys(r['trace_id'] for r in original))}
 original_by={(r['trace_id'],r['condition']):r for r in original}
 all_records={}; scope_reports={}; hashes=[]
 for scope,total in [('smoke',52),('formal',3840)]:
  run=out/'runs'/scope; rows=read(run/'results.json'); plan=read(run/'matrix.json'); fp=read(run/'manifest.json')
  state=read(run/'status.json'); runtime=read(run/'runtime.json')
  check(state['stage']=='complete' and state['completed']==total,'completion '+scope)
  check(runtime['precision']=='bfloat16' and runtime['attention']=='sdpa','runtime '+scope)
  check(not any(v in ('cpu','disk') for v in runtime['device_map'].values()),'offload '+scope)
  check(fp['prepared_sha256']==ph and fp['scope']==scope and fp['cuda_visible_devices']=='1,2,4,6','fingerprint '+scope)
  check(fp['plan_sha256']==hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),'plan hash '+scope)
  json_files=[p for p in run.glob('*.json') if len(p.stem)==24]
  npz_files=list(run.glob('*.npz'))
  check(len(rows)==len(plan)==len(json_files)==len(npz_files)==total,'counts '+scope)
  check(len({key(r['cell']) for r in rows})==total,'unique cells '+scope)
  check({key(r['cell']) for r in rows}=={key(c) for c in plan}=={p.stem for p in json_files},'coverage '+scope)
  noopmax=repeatmax=reconstructionmax=normmax=0.; noops=repeats=0
  for n,r in enumerate(rows):
   cid=key(r['cell']); p=run/(cid+'.json')
   check(read(p)==r,'aggregate differs: '+scope+'/'+cid)
   jp=filecheck(p); ap=run/r['array_file']; ah=filecheck(ap,r['arrays_sha256'])
   hashes.append(dict(scope=scope,cell_key=cid,json_sha256=jp,npz_sha256=ah))
   check(r['technical_pass'] and r['fingerprint']==fp,'record pass/provenance '+cid)
   ck=r['checks']
   check(ck['applied']==1 and ck['hook_calls']==ck['expected_calls'] and ck['hook_inventory_restored'],'hook '+cid)
   reconstructionmax=max(reconstructionmax,ck['reconstruction_max_abs'])
   normmax=max(normmax,r['patch'].get('random_norm_relative_error') or 0.)
   old=original_by[r['trace_id'],{'U':'uncapped','P':'full_p25'}[r['cell']['target']]]
   for field in ['prompt_token_ids_sha256','trace_prefix_token_ids_sha256','forced_trace_token_ids_sha256','teacher_forced_input_token_ids_sha256']:
    if field in old:check(r[field]==old[field],'token '+field+' '+cid)
   check(not r['model_generate_used'] and not r['natural_generation'],'generation '+cid)
   with np.load(ap,allow_pickle=False) as data:
    for k in data.files:check(bool(np.isfinite(data[k]).all()),'nonfinite '+cid+'/'+k)
    if r['cell']['operation']=='noop':
     noops+=1
     oldpath=root/'runs/formal'/('%04d_%s.npz'%(trace_idx[r['trace_id']],{'U':'uncapped','P':'full_p25'}[r['cell']['target']]))
     with np.load(oldpath,allow_pickle=False) as ref:
      err=max(float(np.max(np.abs(data[k].astype(float)-ref[k].astype(float)))) for k in ref.files)
     noopmax=max(noopmax,err)
    if r['cell']['repeat']:
     repeats+=1; refcell=dict(r['cell'],repeat=False)
     with np.load(run/(key(refcell)+'.npz'),allow_pickle=False) as ref:
      err=max(float(np.max(np.abs(data[k].astype(float)-ref[k].astype(float)))) for k in ref.files)
     repeatmax=max(repeatmax,err)
   if (n+1)%500==0:print('record audit',scope,n+1,total,flush=True)
  check(noopmax<=1e-4 and repeatmax<=1e-4 and reconstructionmax<=.25 and normmax<=.005,'numeric gates '+scope)
  scope_reports[scope]=dict(completed=total,json_count=len(json_files),npz_count=len(npz_files),noops=noops,repeats=repeats,noop_max_abs=noopmax,repeat_max_abs=repeatmax,reconstruction_max_abs=reconstructionmax,random_norm_max_relative_error=normmax,runtime=runtime)
  all_records[scope]=rows
 save('record_hashes.json',hashes)
 print('Recomputing four endpoints directly from record logits',flush=True)
 rows=all_records['formal']; scales=read(out/'prepared/scales.json')
 summary=read(out/'analysis/formal/SUMMARY.json')
 lookup={(r['base_id'],r['arm'],r['cell']['target'],r['cell']['dose'],r['cell']['random_index']):r for r in rows if r['cell']['operation']=='dose'}
 groups=defaultdict(lambda:defaultdict(list))
 def margin(rec,metric):
  z=rec['logit_diagnostics']['logits']; initial=rec['initial_answer'];target=rec['target_answer']
  a=1. if rec['cell']['target']=='U' else scales[rec['trace_id']]['same']['slope']
  if metric=='broad':return (sum(z[k] for k in 'ABCD' if k!=initial)/3-z[initial])/a
  return (z[target]-sum(z[k] for k in 'ABCD' if k not in (initial,target))/2)/a
 for metric in ['broad','specific']:
  for base in sorted({r['base_id'] for r in rows}):
   g={}
   for c in 'UP':
    for arm in 'VF':
     effects=[]
     for j in [-1,*range(8)]:
      pos=lookup[base,arm,c,1.,j];neg=lookup[base,arm,c,-1.,j]
      effects.append((margin(pos,metric)-margin(neg,metric))/2)
     g[c,arm]=effects[0]-np.mean(effects[1:]);fam=pos['family_id']
   groups['same_'+metric+'_common'][fam].append(((g['P','V']-g['U','V'])+(g['P','F']-g['U','F']))/2)
   groups['same_'+metric+'_interaction'][fam].append((g['P','V']-g['P','F'])-(g['U','V']-g['U','F']))
 recomputed={}
 for name,families in groups.items():
  x=np.array([np.mean(families[f]) for f in sorted(families)]);check(len(x)==12,'families '+name)
  observed=float(x.mean())
  signs=np.array(list(itertools.product([-1.,1.],repeat=12)))
  p=float((np.abs(signs@x/12)>=abs(observed)-1e-12).mean())
  boot=x[np.random.default_rng(20260928).integers(12,size=(20000,12))].mean(1);ci=np.quantile(boot,[.025,.975])
  rec=dict(mean=observed,family_sd=float(x.std(ddof=1)),ci95=ci.tolist(),p=p,family_means={f:float(np.mean(families[f])) for f in sorted(families)})
  recomputed[name]=rec
  ref=summary['primary'][name]
  check(np.allclose([observed,*ci,p],[ref['equal_family_mean'],*ref['descriptive_ci95'],ref['exact_signflip_p']],atol=1e-12,rtol=0),'primary recomputation '+name)
 running=0.
 for i,(name,s) in enumerate(sorted(recomputed.items(),key=lambda a:a[1]['p'])):
  running=max(running,min(1.,(4-i)*s['p']));s['holm_p']=running
  check(abs(running-summary['primary'][name]['holm_four_primary'])<=1e-12,'Holm '+name)
 details=read(out/'analysis/formal/details.json');base_endpoints=read(out/'analysis/formal/base_endpoints.json')
 check(len(details)==3840 and len(base_endpoints)==144,'analysis output counts')
 check({key(r['cell']) for r in details}=={key(r['cell']) for r in rows},'details coverage')
 noops={(r['trace_id'],r['cell']['target']):r for r in rows if r['cell']['operation']=='noop'}
 changes=0
 for r in rows:
  b=noops[r['trace_id'],r['cell']['target']]
  changes+=int(max('ABCD',key=lambda k:r['logit_diagnostics']['logits'][k])!=max('ABCD',key=lambda k:b['logit_diagnostics']['logits'][k]))
 check(changes==summary['argmax_changes'],'argmax count')
 analysis_hashes={p.name:filecheck(p) for p in sorted((out/'analysis/formal').iterdir()) if p.is_file()}
 save('primary_recomputation.json',dict(results=recomputed,unit='12 equally weighted structural families; 24 reused development bases',bootstrap_resamples=20000,seed=20260928,signflips=4096,argmax_changes=changes,scientific_limit=summary['limits']))
 print('Rehashing pinned model and cap assets',flush=True)
 av=read(out/'prepared/asset_verification.json');asset_details=[]
 for n,(path,meta) in enumerate(av['files'].items()):
  p=Path(path);before=p.stat();h=filecheck(p,meta['sha256']);after=p.stat()
  check((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)==(meta['size'],meta['mtime_ns']),'asset metadata '+path)
  asset_details.append(dict(path=path,sha256=h,size=after.st_size,mtime_ns=after.st_mtime_ns))
  print('asset audit',n+1,len(av['files']),p.name,flush=True)
 save('asset_rehash.json',dict(files=asset_details,previous_verified=av['checked']))
 processes=[]
 for pp in Path('/proc').iterdir():
  if not pp.name.isdigit() or int(pp.name)==os.getpid():continue
  try:
   args=(pp/'cmdline').read_bytes().decode().split('\0')
   if any(str(out/'scripts') in a for a in args):
    processes.append(dict(pid=int(pp.name),owner=pwd.getpwuid(pp.stat().st_uid).pw_name,cwd=os.readlink(pp/'cwd'),argv=args))
  except (OSError,UnicodeError):pass
 lockstates={}
 for name in ['campaign.lock','runner.lock']:
  with (out/name).open('r') as f:
   try:
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);lockstates[name]='free';fcntl.flock(f,fcntl.LOCK_UN)
   except BlockingIOError:lockstates[name]='held'
 check(not processes and all(x=='free' for x in lockstates.values()),'remaining managed process/lock')
 campaign=read(out/'explore_outputs/campaign_status.json');budget=read(out/'explore_outputs/gpu_budget.json')
 check(campaign['stage']=='c1_signed_complete_h1_pending' and not budget['active'],'campaign final state')
 runtime_snapshot=dict(processes=processes,locks=lockstates,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True),gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True),disk_free_bytes=shutil.disk_usage(root).free)
 prev=read(out/'explore_outputs/heartbeat_monitor_20261003T095333Z.json')
 log=out/'logs/formal_20260928T133058Z.log'
 final=dict(status='passed' if not errors else 'failed',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),started_utc=datetime.datetime.fromtimestamp(started,datetime.timezone.utc).isoformat(),elapsed_seconds=time.time()-started,errors=errors,protocol_sha256=protocol,prepared_manifest_sha256=ph,scope_reports=scope_reports,original_arrays_verified=len(prep['original_array_sha256']),weights_and_caps_full_sha256_verified=len(asset_details),analysis_hashes=analysis_hashes,primary_recomputation_matches=not any('recomputation' in x or 'Holm' in x for x in errors),campaign=campaign,budget=budget,resources=runtime_snapshot,delta=dict(previous_snapshot='heartbeat_monitor_20261003T095333Z.json',previous_completed=prev['formal_completed'],current_completed=3840,added=3840-prev['formal_completed'],log_bytes=log.stat().st_size,log_delta=log.stat().st_size-prev['log']['bytes']),local_connection_recovery='Terminal tools restored; old SSH session absent; initial direct TCP timeout and old 127.0.0.1:10090 proxy refused. Existing FlClash launched, direct SSH then reachable. Original chat-authorized password supplied only at SSH stdin.',h1_status='Design and precision planning only; no H1 or Jev execution.')
 save('verified_file_hashes.json',checked_files);save('completion_audit.json',final)
 print('AUDIT_RESULT',json.dumps(dict(path=str(dest),status=final['status'],errors=errors,primary=recomputed,resources=runtime_snapshot,delta=final['delta']),ensure_ascii=False),flush=True)
except BaseException as e:
 save('audit_failure.json',dict(type=type(e).__name__,message=str(e),errors=errors))
 raise

