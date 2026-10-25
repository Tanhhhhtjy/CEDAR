from pathlib import Path
import json,hashlib,os,time,sys,tarfile
root=Path('/data4/WuMingrui/TianJinyu/CEDAR');p=root/'arr_next_steps_20261005';e=p/'explore_outputs';dest=e/'closure_20261006'
sys.path.insert(0,str(p/'scripts'));from common import sha,read,save,verify,key,stamp
sys.path.insert(0,str(p/'operations'));from takeover import identity,verify_operations
started=time.time();verify(root,p);verify_operations(p)
r=read(e/'heartbeat_20261005T192918Z.json');prev=read(e/'heartbeat_20261005T185913Z.json')
assert r['scopes']['smoke']['completed']==24 and r['scopes']['B1']['completed']==192
for scope,total in [('smoke',24),('B1',192)]:
    for name,h in prev['scopes'][scope]['verified_file_hashes'].items():assert r['scopes'][scope]['verified_file_hashes'][name]==h
    plan=read(p/f'inputs/{scope}_matrix.json');rows=read(p/f'runs/{scope}/results.json')
    assert len(rows)==total and {key(x['cell']) for x in rows}=={key(c) for c in plan}
    for row in rows:assert row==read(p/f'runs/{scope}/{key(row["cell"])}.json')
print('216_RECORDS_MATCH_PLANNED_MATRIX_AND_RESULTS',flush=True)
av=read(p/'prepared/asset_verification.json');assets={}
for i,(name,meta) in enumerate(av['files'].items(),1):
    actual=sha(name);assert actual==meta['sha256'],name
    assets[name]=dict(sha256=actual,size=Path(name).stat().st_size)
    print('WEIGHT_ASSET_HASH',i,len(av['files']),flush=True)
processes=[]
for d in Path('/proc').iterdir():
    if not d.name.isdigit():continue
    try:
        args=(d/'cmdline').read_bytes().split(b'\0');args=[x.decode(errors='replace') for x in args if x]
        if any('arr_next_steps_20261005' in a for a in args) and any(a.endswith(('/campaign_runtime.py','/run_phase_factorial.py','/analyze_phase_factorial.py','/campaign.py')) for a in args):
            processes.append(identity(int(d.name)))
    except (PermissionError,FileNotFoundError,ProcessLookupError):pass
assert not processes,processes
budget=read(e/'gpu_budget.json');assert not budget['active']
assert r['locks']=={'campaign.lock':'free','runner.lock':'free'}
log=Path(prev['log']['path']);comparison={'previous':'heartbeat_20261005T185913Z.json','B1_delta':10,'all_previous_record_hashes_unchanged':True,'inference_log':str(log),'bytes':log.stat().st_size,'log_delta_bytes':log.stat().st_size-prev['log']['bytes'],'note':'Heartbeat selected the newer empty B1_analysis log at completion; its negative cross-file delta was not inference log shrinkage.'}
audit=dict(status='passed',time_utc=stamp(),duration_seconds=time.time()-started,full_asset_hashes=assets,asset_count=len(assets),record_count=216,array_count=216,all_results_match_records=True,all_frozen_dependencies_verified=True,operations_manifest_verified=True,active_managed_processes=processes,locks=r['locks'],gpu_budget=budget,comparison=comparison,prepared_sha256=sha(p/'prepared/manifest.json'),operations_manifest_sha256=sha(p/'operations/manifest.json'),heartbeat_sha256=sha(e/'heartbeat_20261005T192918Z.json'))
save(dest/'closure_integrity.json',audit)
files=[]
for name in ['analysis','runs','prepared','inputs','scripts','operations','h1','logs']:
    files += [q for q in (p/name).rglob('*') if q.is_file() and '__pycache__' not in q.parts]
files += [q for q in p.glob('*.md') if q.is_file()]
files += [q for q in e.glob('*.json') if 'heartbeat_bundle' not in q.name]
files += [dest/'closure_integrity.json',dest/'verify_closure.py']
files=sorted(set(files))
manifest={str(q.relative_to(p)):dict(sha256=sha(q),size=q.stat().st_size) for q in files}
save(dest/'evidence_manifest.json',manifest)
a=e/'B1_complete_evidence_20261006.tar.gz'
with tarfile.open(a,'w:gz') as t:
    for q in files+[dest/'evidence_manifest.json']:t.add(q,arcname=str(q.relative_to(p)),recursive=False)
h=sha(a);a.with_name(a.name+'.sha256').write_text(h+'  '+a.name+'\n')
print(json.dumps({'status':'passed','file_count':len(files),'archive':str(a),'bytes':a.stat().st_size,'sha256':h,'gpu_wall_seconds':budget['used_seconds']}),flush=True)
