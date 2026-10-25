"""Bounded, resumable server campaign. No credentials or network dependencies."""
import argparse,fcntl,json,os,shutil,subprocess,time,traceback,sys
from pathlib import Path
from common import read,save,sha,stamp,matrix,verify_prepared
PY='/home/WuMingrui/miniconda3/envs/cedar/bin/python'
MAX_GPU_SECONDS=10*24*3600
SMOKE_SECONDS=6*3600

def process(pid):
    p=Path('/proc')/str(pid)
    try:return dict(pid=pid,command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),cwd=os.readlink(p/'cwd'))
    except (FileNotFoundError,PermissionError):return dict(pid=pid,exited=True)

def main(root,out):
    assert root.resolve()==Path('/data4/WuMingrui/TianJinyu/CEDAR')
    assert out.resolve()==root/'c1_signed_20260928'
    (out/'logs').mkdir(parents=True,exist_ok=True);(out/'explore_outputs').mkdir(exist_ok=True)
    lock=open(out/'campaign.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # An orphaned runner from an interrupted controller must be investigated, never duplicated.
    probe=open(out/'runner.lock','a+');fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fcntl.flock(probe,fcntl.LOCK_UN);probe.close()
    runstamp=stamp()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1,2,4,6',HF_HOME=str(root/'cache'),HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONHASHSEED='20260928')
    def status(stage,**more):
        v=dict(stage=stage,updated=time.time(),pid=os.getpid(),**more)
        save(out/'explore_outputs/campaign_status.json',v)
        with (out/'explore_outputs/campaign_history.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
        print(json.dumps(v),flush=True)
    def command(cmd,label):
        path=out/f'logs/{label}_{runstamp}.log'
        with path.open('a') as f:subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    args=['--root',str(root),'--out',str(out)]
    status('cpu_tests');command([PY,str(out/'scripts/test_c1_core.py')],'cpu_tests')
    command([str(root/'.venv-analysis/bin/python'),str(out/'scripts/analysis_core.py')],'analysis_tests')
    status('preparing_frozen_provenance');command([PY,str(out/'scripts/prepare_signed.py'),*args],'prepare')
    status('checking_all_planned_patch_norms');command([PY,str(out/'scripts/preflight_signed.py'),*args],'patch_preflight')
    status('verifying_assets')
    assets=read(root/'explore_outputs/assets_status.json');assert assets['status']=='ready'
    hashes={}
    candidates=[(Path(assets['model'])/name,meta) for name,meta in assets['files'].items()]
    candidates += [(Path(meta['path']),meta) for meta in assets['assets'].values()]
    for p,meta in candidates:
        before=p.stat();assert before.st_size==meta.get('size',before.st_size)
        digest=sha(p);after=p.stat()
        assert digest==meta['sha256'] and (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),str(p)
        hashes[str(p)]=dict(sha256=digest,size=after.st_size,mtime_ns=after.st_mtime_ns)
        print(json.dumps(dict(asset_verified=p.name,verified=len(hashes),total=len(candidates))),flush=True)
    verification=dict(status='passed',checked=time.time(),files=hashes,model_revision=assets['revision'],
        prepared_sha256=sha(out/'prepared/manifest.json'))
    save(out/f'explore_outputs/asset_verification_{runstamp}.json',verification)
    save(out/'prepared/asset_verification.json',verification)
    bp=out/'explore_outputs/gpu_budget.json'
    budget=read(bp) if bp.exists() else dict(used_seconds=0.,scope_seconds={},active=False)
    if budget['active']:
        # Conservative accounting after controller loss, never silently reset elapsed time.
        lost=max(0.,time.time()-budget['charged_at'])
        budget['used_seconds']+=lost;budget['scope_seconds'][budget['scope']]+=lost
        budget['active']=False;save(bp,budget)
    old=read(root/'runs/formal/results.json')
    baseline={(r['trace_id'],r['condition']):r for r in old}
    ratio=1.
    previous_snapshot=None
    def snapshot(scope,proc,path):
        nonlocal previous_snapshot
        sp=out/f'runs/{scope}/status.json';rs=read(sp) if sp.exists() else {}
        snap=dict(time=time.time(),scope=scope,campaign=process(os.getpid()),runner=process(proc.pid),
            log_bytes=path.stat().st_size,log_mtime=path.stat().st_mtime,disk_free_bytes=shutil.disk_usage(root).free,
            run=rs,weights_verified=verification['checked'],asset_files=len(hashes),
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True),
            gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True))
        if previous_snapshot and previous_snapshot['scope']==scope:
            snap['delta_completed']=rs.get('completed',0)-previous_snapshot['run'].get('completed',0)
            snap['delta_log_bytes']=snap['log_bytes']-previous_snapshot['log_bytes']
            snap['interval_seconds']=snap['time']-previous_snapshot['time']
        with (out/'explore_outputs/monitor_history.jsonl').open('a') as f:f.write(json.dumps(snap)+'\n')
        save(out/'explore_outputs/monitor_state.json',snap)
        save(out/f'explore_outputs/monitor_{stamp()}.json',snap);previous_snapshot=snap
    def stop(proc):
        proc.terminate()
        try:proc.wait(timeout=60)
        except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=30)
    for scope in ['smoke','formal']:
        plan=matrix(old,scope);rundir=out/'runs'/scope
        done={p.stem for p in rundir.glob('*.json') if len(p.stem)==24}
        pending=[c for c in plan if __import__('common').key(c) not in done]
        predicted=sum(baseline[c['trace_id'],{'U':'uncapped','P':'full_p25'}[c['target']]]['seconds'] for c in pending)*ratio
        free=shutil.disk_usage(root).free;storage=len(pending)*2_000_000
        forecast=dict(predicted_seconds=predicted,ratio=ratio,remaining_cells=len(pending),
            storage_upper_bytes=storage,disk_free_bytes=free,used_gpu_seconds=budget['used_seconds'])
        save(out/f'explore_outputs/forecast_{scope}_{runstamp}.json',forecast)
        assert free>20*1024**3 and storage<.2*free,('disk budget',forecast)
        assert predicted+budget['used_seconds']<MAX_GPU_SECONDS,('GPU total budget',forecast)
        if scope=='smoke':assert predicted+budget['scope_seconds'].get(scope,0)<SMOKE_SECONDS,('smoke time budget',forecast)
        if pending:
            status('waiting_for_gpu_memory',scope=scope,forecast=forecast)
            deadline=time.monotonic()+12*3600
            while True:
                rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                memory={int(a):int(b) for a,b in (r.split(',') for r in rows.strip().splitlines())}
                if all(memory[i]>=22528 for i in [1,2,4,6]):break
                if time.monotonic()>deadline:raise RuntimeError('GPU memory wait exceeded 12h')
                time.sleep(30)
        status(scope,forecast=forecast)
        path=out/f'logs/{scope}_{runstamp}.log'
        with path.open('a') as f:
            proc=subprocess.Popen([PY,str(out/'scripts/run_signed.py'),*args,'--scope',scope],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
            budget.update(active=True,scope=scope,charged_at=time.time(),child_pid=proc.pid)
            budget['scope_seconds'].setdefault(scope,0.);save(bp,budget)
            snapshot_at=0.
            try:
                while proc.poll() is None:
                    now=time.time();elapsed=max(0.,now-budget['charged_at'])
                    budget['used_seconds']+=elapsed;budget['scope_seconds'][scope]+=elapsed;budget['charged_at']=now;save(bp,budget)
                    if budget['used_seconds']>MAX_GPU_SECONDS or (scope=='smoke' and budget['scope_seconds'][scope]>SMOKE_SECONDS):
                        raise RuntimeError('Frozen GPU time bound exceeded')
                    if shutil.disk_usage(root).free<20*1024**3:raise RuntimeError('Disk below 20 GiB guard')
                    sp=rundir/'status.json'
                    if sp.exists():
                        rs=read(sp)
                        if rs.get('pid')==proc.pid and now-rs['updated']>90*60:
                            raise RuntimeError('No runner state transition for 90 minutes; affected process stopped for diagnosis')
                    if time.monotonic()>=snapshot_at:snapshot(scope,proc,path);snapshot_at=time.monotonic()+1800
                    time.sleep(15)
                if proc.returncode:raise RuntimeError(f'{scope} failed: exit {proc.returncode}; {path}')
            except BaseException:
                if proc.poll() is None:stop(proc)
                raise
            finally:
                elapsed=max(0.,time.time()-budget['charged_at']);budget['used_seconds']+=elapsed;budget['scope_seconds'][scope]+=elapsed
                budget.update(active=False,charged_at=time.time());save(bp,budget);snapshot(scope,proc,path)
        rs=read(rundir/'results.json')
        if scope=='smoke':
            gate=read(rundir/'gate.json');assert gate['technical_pass'] and gate['completed']==52
            # Throughput forecast uses every smoke cell, regardless of scientific outcome.
            factors=[r['seconds']/baseline[r['trace_id'],r['forward_condition']]['seconds'] for r in rs]
            ratio=max(1.,sum(factors)/len(factors))
            status('smoke_passed',gate=gate,forecast_ratio=ratio)
        else:status('formal_acquisition_complete',completed=len(rs))
    status('formal_analysis');command([str(root/'.venv-analysis/bin/python'),str(out/'scripts/analyze_signed.py'),*args],'analysis')
    status('c1_signed_complete_h1_pending',gpu_wall_seconds=budget['used_seconds'],note='Development C1 completed; independent H1 not run.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:main(a.root,a.out)
    except BlockingIOError:
        print('Campaign or surviving runner lock is occupied; no duplicate started.',file=sys.stderr);sys.exit(78)
    except Exception as e:
        failure=dict(stage='failed',time=time.time(),pid=os.getpid(),error=repr(e),traceback=traceback.format_exc())
        save(a.out/f'explore_outputs/campaign_failure_{stamp()}.json',failure);save(a.out/'explore_outputs/campaign_status.json',failure);raise
