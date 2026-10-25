import argparse,fcntl,os,sys,time,traceback
from pathlib import Path
from common import *
from phase_capper import make_capper,BoundaryCheck
def main(root,out,scope):
    lock=open(out/'runner.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prep=verify(root,out);av=read(out/'prepared/asset_verification.json');assert av['status']=='passed'
    for n,m in av['files'].items():
        s=Path(n).stat();assert s.st_size==m['size'] and s.st_mtime_ns==m['mtime_ns'],n
    sys.path.insert(0,str(root/'scripts'));import run_arr_acquisition as acq
    import numpy as np,torch,transformers
    hist=acq.historical
    assert torch.__version__==prep['original_manifest']['torch'] and transformers.__version__==prep['original_manifest']['transformers']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1,2,4,6' and torch.cuda.device_count()==4
    rows=read(out/'inputs/development_split.json');traces={t['trace_id']:t for t in rows};plan=matrix(rows,scope)
    run=out/'runs'/scope;run.mkdir(parents=True,exist_ok=True)
    fp=dict(prepared_sha256=sha(out/'prepared/manifest.json'),scope=scope,matrix_sha256=sha(out/f'inputs/{scope}_matrix.json'),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    if (run/'manifest.json').exists():assert read(run/'manifest.json')==fp
    else:save(run/'manifest.json',fp)
    if scope!='smoke':assert read(out/'runs/smoke/gate.json')['technical_pass'] is True
    old=read(root/'runs/formal/results.json');indices={t:i for i,t in enumerate(dict.fromkeys(r['trace_id'] for r in old))}
    def state(stage,n,**more):save(run/'status.json',dict(stage=stage,completed=n,total=len(plan),pid=os.getpid(),updated=time.time(),**more))
    done=[]
    for c in plan:
        p=run/(key(c)+'.json')
        if p.exists():
            r=read(p);assert r['cell']==c and r['fingerprint']==fp and r['technical_pass'];assert sha(run/r['array_file'])==r['arrays_sha256'];done.append(r)
    def finish(results):
        save(run/'results.json',results)
        if scope=='smoke':save(run/'gate.json',dict(technical_pass=True,completed=len(results),expected=24,anchor_max_abs=max(r['checks'].get('anchor_max_abs',0) for r in results),repeat_max_abs=max(r['checks'].get('repeat_max_abs',0) for r in results),reconstruction_max_abs=max(r['checks']['reconstruction_max_abs'] for r in results)))
        state('complete',len(results))
    if len(done)==len(plan):finish(done);return
    state('loading_model',len(done));assets=read(root/'explore_outputs/assets_status.json')
    caps,capmeta=hist.e2run.load_p25_assets(root/'cache/hub')
    model,tokenizer=acq.load_qwen_model(acq.QwenModelSpec(repo_id=assets['model']),max_memory={i:'18GiB' for i in range(4)},local_files_only=True)
    assert len(model.model.layers)==64 and all(str(d) not in ('cpu','disk') for d in model.hf_device_map.values())
    assert model.hf_device_map==read(root/'runs/formal/runtime.json')['device_map'],'Device map drift'
    assert model.dtype==torch.bfloat16
    runtime=dict(torch=torch.__version__,transformers=transformers.__version__,device_map=model.hf_device_map,caps=capmeta,pid=os.getpid(),started=time.time(),dtype=str(model.dtype),attention='sdpa',excluded_fit_token_ids=read(root/'runs/formal/runtime.json')['excluded_fit_token_ids'])
    save(run/f'runtime_{stamp()}.json',runtime);save(run/'runtime.json',runtime)
    prompts={p['id']:p for p in acq.readl(root/'CEDAR_SUPPORTING_DATA/data/phase_validity_inputs/answer_inputs96.jsonl')}
    results=[]
    for c in plan:
        rid=key(c);rp=run/(rid+'.json');ap=run/(rid+'.npz')
        if rp.exists():results.append(read(rp));continue
        preserve(ap);preserve(ap.with_suffix('.tmp.npz'))
        t=traces[c['trace_id']];phase=c['phase'];tele={};boundary=BoundaryCheck(t)
        original_capper=hist.ActivationCapper;original_diag=hist.v1run.diagnostics;captured={}
        def capture(z,row,tok):captured['logits']=z.detach().clone();return original_diag(z,row,tok)
        before=hist.v1run.hook_counts(model);start=time.monotonic()
        state('running',len(results),active_cell=c,last_cell_started=time.time())
        hist.ActivationCapper=make_capper(original_capper,t,phase,tele);hist.v1run.diagnostics=capture
        handle=model.register_forward_pre_hook(boundary,with_kwargs=True)
        try:
            with acq.BoundaryObserver(model) as observer:
                rec=hist.run_one(model,tokenizer,transformers,caps,prompts,t,'full_p25')
            with torch.inference_mode():reconstruction=acq.probe_reconstruct(model,observer,captured['logits'])
            values=acq.numpy_state(observer,captured['logits'])
        finally:
            handle.remove();hist.ActivationCapper=original_capper;hist.v1run.diagnostics=original_diag
        assert hist.v1run.hook_counts(model)==before
        assert boundary.calls==len(t['trace_prefix_token_ids'])+1 and boundary.terminal is not None
        assert all(np.isfinite(v).all() for v in values.values()) and reconstruction<=.25
        checks=dict(reconstruction_max_abs=reconstruction,hooks_restored=True,terminal=boundary.terminal,cap_calls=tele)
        if phase in ['00','11']:
            name='uncapped' if phase=='00' else 'full_p25';rel=f'runs/formal/{indices[t["trace_id"]]:04d}_{name}.npz'
            assert sha(root/rel)==prep['original_array_sha256'][rel]
            with np.load(root/rel) as ref:err=max(float(np.max(np.abs(values[k]-ref[k]))) for k in ref.files)
            checks['anchor_max_abs']=err;assert err<=1e-4,('Anchor drift',phase,err)
        if c['repeat']:
            original=dict(c,repeat=False)
            with np.load(run/(key(original)+'.npz')) as ref:err=max(float(np.max(np.abs(values[k]-ref[k]))) for k in ref.files)
            checks['repeat_max_abs']=err;assert err<=1e-4,('Repeat drift',err)
        acq.snapshots(ap,**values)
        rec.update(schema='cedar.phase-factorial.v1',forward_condition='phase_'+phase,condition='phase_'+phase,is_diagonal_replay=phase=='00',cell_id=t['trace_id']+'::phase_'+phase,cell=c,batch=t['batch'],fingerprint=fp,checks=checks,seconds=time.monotonic()-start,array_file=ap.name,arrays_sha256=sha(ap),technical_pass=True,peak_gpu_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(4)])
        save(rp,rec);results.append(rec);state('acquiring',len(results),last_cell=c,last_seconds=rec['seconds'])
        print(json.dumps(dict(completed=len(results),total=len(plan),phase=phase,seconds=rec['seconds'])),flush=True)
    finish(results)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--scope',choices=['smoke','B1','B2'],required=True);a=p.parse_args()
    try:main(a.root,a.out,a.scope)
    except BlockingIOError:sys.exit(78)
    except Exception as e:
        save(a.out/f'explore_outputs/failure_{a.scope}_{stamp()}.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time(),pid=os.getpid()));raise
