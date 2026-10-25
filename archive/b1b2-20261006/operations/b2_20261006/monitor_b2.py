"""Read-only scientific audit; writes timestamped monitoring evidence only."""
import argparse,fcntl,hashlib,json,os,re,shutil,subprocess,sys,time
from pathlib import Path

def main(root,out):
    sys.path.insert(0,str(out/'scripts'))
    from common import read,save,sha,verify,key,stamp
    sys.path.insert(0,str(out/'operations'))
    from takeover import identity,verify_operations
    sys.path.insert(0,str(out/'operations/b2_20261006'))
    from b2_support import verify_package
    verify_package(out)
    import numpy as np
    started=time.time();prep=verify(root,out);verify_operations(out)
    campaign=read(out/'explore_outputs/campaign_status.json');budget=read(out/'explore_outputs/gpu_budget.json')
    lastfile=out/'explore_outputs/heartbeat_B2_latest.json'
    previous=read(lastfile) if lastfile.exists() else None
    server_previous=read(out/'explore_outputs/monitor_state.json') if (out/'explore_outputs/monitor_state.json').exists() else None
    av=read(out/'prepared/asset_verification.json');assert av['status']=='passed' and av['prepared_sha256']==sha(out/'prepared/manifest.json')
    for name,m in av['files'].items():
        s=Path(name).stat();assert s.st_size==m['size'] and s.st_mtime_ns==m['mtime_ns'],name
    assert budget['cumulative_limit_seconds'] is None and budget['smoke_limit_seconds'] is None
    processes={}
    for role,pid in [('campaign',campaign['pid']),('runner',budget['child_pid'])]:
        info=identity(pid);processes[role]=info
        if info is not None:
            assert info['uid']==os.getuid() and info['cwd']==str(root),role
            expected='operations/b2_20261006/campaign_b2.py' if role=='campaign' else 'scripts/run_phase_factorial.py'
            assert str(out/expected) in info['argv'],(role,info)
            assert info['argv'][info['argv'].index('--out')+1]==str(out)
    locks={}
    for name in ['campaign.lock','runner.lock']:
        with (out/name).open('r') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks[name]='free'
            except BlockingIOError:locks[name]='held'
    scopes={};trace_by_id={r['trace_id']:r for r in read(out/'inputs/development_split.json')}
    for scope,total in [('smoke',24),('B1',192),('B2',192)]:
        run=out/'runs'/scope
        if not run.exists():scopes[scope]=dict(stage='not_started',completed=0,total=total);continue
        plan=read(out/f'inputs/{scope}_matrix.json');plan_by_key={key(c):c for c in plan};fingerprint=read(run/'manifest.json')
        assert fingerprint['prepared_sha256']==sha(out/'prepared/manifest.json')
        assert fingerprint['matrix_sha256']==sha(out/f'inputs/{scope}_matrix.json')
        hashes={};rows=[]
        for p in sorted(run.glob('*.json')):
            if not re.fullmatch('[a-f0-9]{24}',p.stem):continue
            r=read(p);assert p.stem in plan_by_key and r['cell']==plan_by_key[p.stem] and r['fingerprint']==fingerprint
            assert r['technical_pass'] is True and sha(run/r['array_file'])==r['arrays_sha256']
            assert r['model_generate_used'] is True and r['natural_generation'] is False and r['generated_text'] is False
            checks=r['checks'];assert checks['reconstruction_max_abs']<=.25 and checks['hooks_restored']
            assert checks.get('anchor_max_abs',0)<=1e-4 and checks.get('repeat_max_abs',0)<=1e-4
            t=trace_by_id[r['trace_id']];L=len(t['trace_prefix_token_ids']);phase=r['cell']['phase']
            assert set(checks['cap_calls']['calls'].values())=={L+1}
            assert set(checks['cap_calls']['enabled_calls'].values())=={L*int(phase[0])+int(phase[1])}
            b=checks['terminal'];assert b['call_index']==L and b['input_token']==t['trace_prefix_token_ids'][-1] and b['past_cache_length']==t['prompt_tokens']+L-1
            assert r['processor_order']['capture_before_force_all'] and r['processor_order']['capture_call_index']==L
            with np.load(run/r['array_file']) as z:assert all(np.isfinite(z[k]).all() for k in z.files)
            hashes[p.name]=sha(p);hashes[r['array_file']]=r['arrays_sha256'];rows.append(r)
        rs=read(run/'status.json');completion_at_read=len(rows)
        assert completion_at_read<=total
        info=dict(stage=rs['stage'],completed=completion_at_read,total=total,status=rs,verified_file_hashes=hashes,all_records_pass=True,
                  max_anchor_error=max((r['checks'].get('anchor_max_abs',0) for r in rows),default=None),
                  max_repeat_error=max((r['checks'].get('repeat_max_abs',0) for r in rows),default=None),
                  max_reconstruction_error=max((r['checks']['reconstruction_max_abs'] for r in rows),default=None))
        if rs.get('stage')=='running':info['active_cell_seconds']=time.time()-rs['last_cell_started']
        if (run/'gate.json').exists():info['gate']=read(run/'gate.json')
        if rs['stage']=='complete':assert rs['completed']==total and len(read(run/'results.json'))==total
        if previous and scope in previous['scopes']:info['completed_delta_from_previous_heartbeat']=completion_at_read-previous['scopes'][scope]['completed']
        scopes[scope]=info
    log=out/'logs'/('B2_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime(0))+'.log')
    logs=[p for p in (out/'logs').glob('B2_*.log') if p.name[3:11].isdigit() and p.name[11:12]=='T']
    assert logs,'No B2 inference log'
    log=max(logs,key=lambda p:p.stat().st_mtime)
    ls=log.stat();now=time.time()
    result=dict(time=now,time_utc=stamp(),audit_started=started,audit_duration=now-started,status='passed',campaign_status=campaign,processes=processes,locks=locks,
                scopes=scopes,budget=budget,asset_verification=dict(last_full_hash_check=av['checked'],files=len(av['files']),all_asset_file_stats_unchanged=True),
                prepared_sha256=sha(out/'prepared/manifest.json'),operations_manifest_sha256=sha(out/'operations/manifest.json'),
                all_frozen_dependencies_verified=True,log=dict(path=str(log),bytes=ls.st_size,mtime=ls.st_mtime),
                output_free_bytes=shutil.disk_usage(out).free,input_free_bytes=shutil.disk_usage(root).free,
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True),
                gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True),
                prior_failures=[p.name for p in sorted((out/'explore_outputs').glob('*failure*.json'))])
    assert result['output_free_bytes']>20*1024**3 and result['input_free_bytes']>2*1024**3
    if server_previous and server_previous['scope']==budget['scope'] and server_previous.get('log_bytes',0)<=ls.st_size:
        result['delta_from_server_snapshot']=dict(previous_time=server_previous['time'],interval_seconds=now-server_previous['time'],completed=scopes[budget['scope']]['completed']-server_previous['run'].get('completed',0),log_bytes=ls.st_size-server_previous['log_bytes'])
    if previous and previous['budget']['scope']==budget['scope'] and previous['log']['path']==str(log):
        result['delta_log_bytes_from_previous_heartbeat']=ls.st_size-previous['log']['bytes']
    amended=out/'explore_outputs/resource_amendment_final_verification.json'
    if previous is None and amended.exists():
        prior=read(amended);result['delta_from_resource_amendment']=dict(previous_completed=prior['completed'],new_completed=scopes['smoke']['completed'],delta=scopes['smoke']['completed']-prior['completed'],previous_time=prior['time'])
    if campaign['stage'] in ['B2']:
        assert processes['campaign'] is not None and processes['runner'] is not None
        assert locks=={'campaign.lock':'held','runner.lock':'held'}
        rs=scopes[budget['scope']]['status'];assert now-rs['updated']<5400
    path=out/f'explore_outputs/heartbeat_B2_{stamp()}.json';save(path,result);save(lastfile,result)
    print(json.dumps(dict(evidence=str(path),status=result['status'],scope=budget['scope'],completed=scopes[budget['scope']]['completed'],total=scopes[budget['scope']]['total'],delta=result.get('delta_from_server_snapshot'),since_amendment=result.get('delta_from_resource_amendment'),active_cell_seconds=scopes[budget['scope']].get('active_cell_seconds'),output_free_gib=result['output_free_bytes']/1024**3,input_free_gib=result['input_free_bytes']/1024**3,wall_time_limits=None,locks=locks)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();main(a.root,a.out)
