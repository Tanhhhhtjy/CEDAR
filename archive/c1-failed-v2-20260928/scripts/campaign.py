"""Locked bounded C1 campaign. Credentials are never needed by the server job."""
import argparse, fcntl, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
from prepare_c1 import sha, save

PY='/home/WuMingrui/miniconda3/envs/cedar/bin/python'
MAX_GPU_SECONDS=10*24*3600
def main(root,out):
    (out/'logs').mkdir(parents=True,exist_ok=True)
    (out/'explore_outputs').mkdir(exist_ok=True)
    lock=open(out/'campaign.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    def status(stage,**more):
        v=dict(stage=stage,updated=time.time(),pid=os.getpid(),**more)
        save(out/'explore_outputs/campaign_status.json',v)
        with (out/'explore_outputs/campaign_history.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
        print(json.dumps(v),flush=True)
    def command(cmd,label):
        path=out/f'logs/{label}_{stamp}.log'
        with path.open('a') as f:
            subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1,2,4,6',HF_HOME=str(root/'cache'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        MKL_NUM_THREADS='4',PYTHONHASHSEED='20260928')
    status('cpu_tests')
    command([PY,str(out/'scripts/test_c1_core.py')],'cpu_tests')
    status('preparing_existing_state_diagnostics')
    command([str(root/'.venv-analysis/bin/python'),str(out/'scripts/prepare_c1.py'),
        '--root',str(root),'--out',str(out)],'prepare')
    status('checking_all_planned_patch_norms')
    command([PY,str(out/'scripts/preflight_patches.py'),'--root',str(root),'--out',str(out)],'patch_preflight')
    status('verifying_assets')
    assets=json.loads((root/'explore_outputs/assets_status.json').read_text())
    assert assets['status']=='ready'
    hashes={}
    for name,meta in assets['files'].items():
        p=Path(assets['model'])/name
        assert p.stat().st_size==meta['size'],str(p)
        assert sha(p)==meta['sha256'],str(p)
        hashes[str(p)]=meta['sha256']
    for name,meta in assets['assets'].items():
        p=Path(meta['path']);assert sha(p)==meta['sha256'],name
        hashes[str(p)]=meta['sha256']
    save(out/'prepared/asset_verification.json',dict(checked=time.time(),files=hashes,
        model_revision=assets['revision']))
    gpu_elapsed=0.
    old=json.loads((root/'runs/formal/results.json').read_text())
    def budget(scope,ratio):
        selected=[r for r in old if r['trace_source']=='e2a_uncapped' and (scope!='dose' or r['arm'] in 'VF')]
        seconds=sum(r['seconds'] for r in selected)*({'main':11,'dose':36}.get(scope,1))*ratio
        free=shutil.disk_usage(root).free
        predicted_bytes={'smoke':44,'main':2112,'dose':3456}[scope]*2_000_000
        assert free>20*1024**3 and predicted_bytes<.2*free,('disk budget',free,predicted_bytes)
        if scope!='smoke':assert seconds+gpu_elapsed<MAX_GPU_SECONDS,('GPU time budget',seconds,gpu_elapsed)
        return dict(predicted_seconds=seconds,storage_upper_bytes=predicted_bytes,disk_free_bytes=free)
    ratio=1.
    for scope in ['smoke','main','dose']:
        forecast=budget(scope,ratio)
        status('waiting_for_gpu_memory',scope=scope,forecast=forecast)
        deadline=time.monotonic()+12*3600
        while True:
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                '--format=csv,noheader,nounits'],text=True)
            free={int(a):int(b) for a,b in (r.split(',') for r in rows.strip().splitlines())}
            if all(free[i]>=22528 for i in [1,2,4,6]):break
            if time.monotonic()>deadline:raise RuntimeError('GPU memory wait exceeded 12h')
            time.sleep(30)
        status(scope,forecast=forecast)
        start=time.monotonic()
        path=out/f'logs/{scope}_{stamp}.log'
        with path.open('a') as f:
            proc=subprocess.Popen([PY,str(out/'scripts/run_c1.py'),'--root',str(root),'--out',str(out),
                '--scope',scope],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
            snapshot_at=0.
            while proc.poll() is None:
                if time.monotonic()-start+gpu_elapsed>MAX_GPU_SECONDS:
                    proc.terminate();proc.wait(timeout=60)
                    raise RuntimeError('campaign exceeded frozen GPU time bound')
                if shutil.disk_usage(root).free<20*1024**3:
                    proc.terminate();proc.wait(timeout=60)
                    raise RuntimeError('disk below 20 GiB guard')
                if time.monotonic()>=snapshot_at:
                    snapshot_at=time.monotonic()+1800
                    snap=dict(time=time.time(),scope=scope,child_pid=proc.pid,
                        log_bytes=path.stat().st_size,disk_free=shutil.disk_usage(root).free)
                    sp=out/f'runs/{scope}/status.json'
                    if sp.exists():snap['run']=json.loads(sp.read_text())
                    snap['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
                    with (out/'explore_outputs/monitor_history.jsonl').open('a') as mf:mf.write(json.dumps(snap)+'\n')
                    save(out/'explore_outputs/monitor_state.json',snap)
                time.sleep(15)
            if proc.returncode:raise RuntimeError(f'{scope} failed: exit {proc.returncode}; {path}')
        gpu_elapsed+=time.monotonic()-start
        rs=json.loads((out/f'runs/{scope}/results.json').read_text())
        if scope=='smoke':
            gate=json.loads((out/'runs/smoke/gate.json').read_text());assert gate['technical_pass']
            baseline={(r['trace_id'],r['forward_condition']):r for r in old}
            factors=[r['seconds']/baseline[r['trace_id'],r['forward_condition']]['seconds'] for r in rs]
            ratio=max(1.,sum(factors)/len(factors))
            status('smoke_passed',gate=gate,forecast_ratio=ratio)
        else:
            status(scope+'_acquisition_complete',completed=len(rs))
            command([str(root/'.venv-analysis/bin/python'),str(out/'scripts/analyze_c1.py'),
                '--root',str(root),'--out',str(out),'--scope',scope],'analysis_'+scope)
    status('c1_complete_h1_pending',gpu_wall_seconds=gpu_elapsed,
        note='Development C1 completed; independent H1 not launched. Review all controls and uncertainty.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    try:main(a.root,a.out)
    except BlockingIOError:
        print('Existing campaign owns the lock; no files changed.',file=sys.stderr)
        sys.exit(78)
    except Exception as e:
        failure=dict(stage='failed',time=time.time(),pid=os.getpid(),error=repr(e),traceback=traceback.format_exc())
        stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
        save(a.out/f'explore_outputs/campaign_failure_{stamp}.json',failure)
        save(a.out/'explore_outputs/campaign_status.json',failure)
        raise

