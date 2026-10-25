"""Authorized B2 only, followed by frozen B2 and pooled analysis."""
import argparse,fcntl,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from common import *
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from takeover import verify_operations
from b2_support import preflight, scope_audit, verify_prior_unchanged, analysis_receipt, verify_package
PY='/home/WuMingrui/miniconda3/envs/cedar/bin/python'
def process(pid):
    p=Path('/proc')/str(pid)
    try:return dict(pid=pid,command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),cwd=os.readlink(p/'cwd'),uid=p.stat().st_uid)
    except FileNotFoundError:return dict(pid=pid,exited=True)
def main(root,out):
    assert root==Path('/data4/WuMingrui/TianJinyu/CEDAR')
    assert out.resolve()==Path('/data1/WuMingrui/TianJinyu/CEDAR/arr_next_steps_20261005')
    verify_operations(out)
    verify(root,out)
    adopted = None
    verify_package(out)
    lock=open(out/'campaign.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if adopted is None:
        with open(out/'runner.lock','a+') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    else:
        adopted.verify_live()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1,2,4,6',HF_HOME=str(root/'cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONHASHSEED='20261005',PYTHONDONTWRITEBYTECODE='1')
    runstamp=stamp();args=['--root',str(root),'--out',str(out)]
    def status(stage,**more):
        s=dict(stage=stage,updated=time.time(),pid=os.getpid(),**more);save(out/'explore_outputs/campaign_status.json',s)
        with (out/'explore_outputs/campaign_history.jsonl').open('a') as f:f.write(json.dumps(s)+'\n')
        print(json.dumps(s),flush=True)
    def command(script,label,extra=None,analysis=False):
        path=out/f'logs/{label}_{runstamp}.log'
        cmd=[str(root/'.venv-analysis/bin/python') if analysis else PY,str(out/'scripts'/script)]+(extra if extra is not None else args)
        with path.open('a') as log:subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    start_dir,preflight_receipt=preflight(root,out,runstamp)
    status('B2_preflight_passed',runtime_limits=None,preflight=str(start_dir/'preflight.json'))
    verify(root,out)
    assert read(out/'explore_outputs/cpu_input_audit.json')['status']=='passed'
    assert read(out/'analysis/A/status.json')['stage']=='offline_analysis_complete_annotations_need_review'
    av=read(out/'prepared/asset_verification.json');assert av['status']=='passed'
    assert av['prepared_sha256']==sha(out/'prepared/manifest.json')
    files=av['files']
    for name,meta in files.items():
        s=Path(name).stat();assert s.st_size==meta['size'] and s.st_mtime_ns==meta['mtime_ns'],name
    # Resume from completed CPU/asset/A work; neither regenerate nor overwrite it.
    bp=out/'explore_outputs/gpu_budget.json';budget=read(bp) if bp.exists() else dict(used_seconds=0.,scope_seconds={},active=False)
    if budget['active']:
        raise RuntimeError('Unclosed prior GPU accounting; verify prior child and preserve evidence before resume')
    budget.update(cumulative_limit_seconds=None,smoke_limit_seconds=None,resource_policy='operations/RESOURCE_AMENDMENT.md');save(bp,budget)
    baseline={(r['trace_id'],r['condition']):r for r in read(root/'runs/formal/results.json')};rows=read(out/'inputs/development_split.json');ratio=1.
    ratio=max(1.,sum(r['seconds']/baseline[r['trace_id'],'full_p25' if r['cell']['phase'][0]=='1' else 'uncapped']['seconds'] for r in read(out/'runs/smoke/results.json'))/24)
    previous=None
    def snapshot(scope,child,log):
        nonlocal previous
        p=out/f'runs/{scope}/status.json';rs=read(p) if p.exists() else {}
        snap=dict(time=time.time(),campaign=process(os.getpid()),runner=process(child.pid),scope=scope,run=rs,log_bytes=log.stat().st_size,log_mtime=log.stat().st_mtime,output_realpath=str(out.resolve()),output_free_bytes=shutil.disk_usage(out).free,input_free_bytes=shutil.disk_usage(root).free,weights_verified=preflight_receipt['time'],asset_count=len(files),budget=budget,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True),gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True))
        if previous and previous['scope']==scope:snap.update(delta_completed=rs.get('completed',0)-previous['run'].get('completed',0),delta_log_bytes=snap['log_bytes']-previous['log_bytes'],interval_seconds=snap['time']-previous['time'])
        save(out/f'explore_outputs/monitor_{stamp()}.json',snap);save(out/'explore_outputs/monitor_state.json',snap);previous=snap
    def charge(scope):
        now=time.time();e=max(0,now-budget['charged_at']);budget['used_seconds']+=e;budget['scope_seconds'][scope]+=e;budget['charged_at']=now;save(bp,budget)
    def stop(child):
        info=process(child.pid)
        if child.poll() is not None:return
        assert info['uid']==os.getuid() and str(out/'scripts/run_phase_factorial.py') in info['command'] and info['cwd']==str(root)
        child.terminate()
        try:child.wait(timeout=45)
        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=15)
    for scope in ['B2']:
        plan=matrix(rows,scope);run=out/'runs'/scope;done={p.stem for p in run.glob('*.json') if len(p.stem)==24};pending=[c for c in plan if key(c) not in done]
        predicted=300+sum(baseline[c['trace_id'],'full_p25' if c['phase'][0]=='1' else 'uncapped']['seconds'] for c in pending)*ratio
        forecast=dict(scope=scope,pending=len(pending),predicted_seconds=predicted,used_seconds=budget['used_seconds'],ratio=ratio,output_free=shutil.disk_usage(out).free,input_free=shutil.disk_usage(root).free)
        save(out/f'explore_outputs/forecast_{scope}_{runstamp}.json',forecast)
        assert forecast['output_free']>20*1024**3 and forecast['input_free']>2*1024**3 and len(pending)*2000000<.2*forecast['output_free'],forecast
        if pending and not (adopted is not None and adopted.scope==scope):
            memory={int(a):int(b) for a,b in (r.split(',') for r in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines())}
            assert all(memory[i]>=22528 for i in [1,2,4,6]),('Selected GPUs not free',memory)
        status(scope,forecast=forecast);log=adopted.log if adopted is not None and adopted.scope==scope else out/f'logs/{scope}_{runstamp}.log'
        with log.open('a') as f:
            if adopted is not None and adopted.scope==scope:
                child=adopted;child.verify_live();child.acknowledge(os.getpid())
            else:
                child=subprocess.Popen([PY,'-u',str(out/'scripts/run_phase_factorial.py'),*args,'--scope',scope],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
            budget.update(active=True,scope=scope,charged_at=time.time(),child_pid=child.pid);budget['scope_seconds'].setdefault(scope,0.);save(bp,budget);next_snapshot=0;launch=time.time()
            try:
                while child.poll() is None:
                    charge(scope);now=time.time()
                    assert shutil.disk_usage(out).free>20*1024**3 and shutil.disk_usage(root).free>2*1024**3,'Disk guard'
                    sp=run/'status.json';rs=read(sp) if sp.exists() else {}
                    transition=rs.get('updated',launch) if rs.get('pid')==child.pid else launch
                    assert now-transition<5400,'No state transition in90min'
                    if now>=next_snapshot:snapshot(scope,child,log);next_snapshot=now+300
                    time.sleep(15)
                if child.returncode:raise RuntimeError(f'{scope} exit {child.returncode}; {log}')
            except BaseException:
                stop(child);raise
            finally:
                charge(scope);budget['active']=False;save(bp,budget);snapshot(scope,child,log)
        recs=read(run/'results.json');assert len(recs)==len(plan)
        if scope=='smoke':
            gate=read(run/'gate.json');assert gate['technical_pass'] and gate['completed']==24
            ratio=max(1.,sum(r['seconds']/baseline[r['trace_id'],'full_p25' if r['cell']['phase'][0]=='1' else 'uncapped']['seconds'] for r in recs)/len(recs))
            status('smoke_passed',gate=gate,forecast_ratio=ratio)
        else:status('B2_acquired',completed=len(recs))
    save(start_dir/'B2_acquisition_audit.json',scope_audit(out,'B2'))
    verify_prior_unchanged(out,preflight_receipt)
    for scope in ['B2','pooled']:
        status(scope+'_analysis')
        dest=out/'analysis'/scope
        if (dest/'SUMMARY.json').exists():
            receipt=analysis_receipt(out,scope)
            prior=out/f'explore_outputs/{scope}_analysis_receipt.json'
            if prior.exists():assert read(prior)==receipt
            status(scope+'_analysis_already_complete')
        else:
            if dest.exists() and any(dest.iterdir()):
                dest.rename(dest.with_name(dest.name+'.preserved_'+str(time.time_ns())))
            command('analyze_phase_factorial.py',scope+'_analysis',args+['--scope',scope],analysis=True)
            receipt=analysis_receipt(out,scope)
        save(out/f'explore_outputs/{scope}_analysis_receipt.json',receipt)
    verify_prior_unchanged(out,preflight_receipt)
    verify(root,out);verify_operations(out);verify_package(out)
    save(start_dir/'completion.json',dict(status='complete',time=time.time(),B2_records=192,B_total_records=384,smoke_records=24,prior_package_unchanged=True,gpu_budget=budget))
    status('B2_and_pooled_complete_C_D_H_not_started',gpu_wall_seconds=budget['used_seconds'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:main(a.root,a.out)
    except BlockingIOError:print('Campaign or runner lock occupied; no duplicate launched');sys.exit(78)
    except Exception as e:
        fail=dict(stage='failed',updated=time.time(),pid=os.getpid(),error=repr(e),traceback=traceback.format_exc());save(a.out/f'explore_outputs/campaign_B2_failure_{stamp()}.json',fail);save(a.out/'explore_outputs/campaign_status.json',fail);raise
