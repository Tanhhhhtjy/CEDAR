"""Cache-faithful signed-dose C1. All output is isolated; release gates are assertions."""
import argparse, fcntl, hashlib, json, os, sys, time, traceback
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from c1_core import make_patch, hook_inventory, PromptPatch
from common import sha, save, read, preserve, COND, key, matrix, verify_prepared

def main(root,out,scope):
    sys.path.insert(0,str(root/'scripts'))
    import run_arr_acquisition as acq
    historical=acq.historical
    import transformers
    run=out/'runs'/scope;run.mkdir(parents=True,exist_ok=True)
    lock=open(out/'runner.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prep=verify_prepared(root,out)
    pf=read(out/'prepared/patch_preflight.json')
    assert pf['status']=='passed' and pf['prepared_sha256']==sha(out/'prepared/manifest.json')
    av=read(out/'prepared/asset_verification.json')
    assert av['status']=='passed' and av['prepared_sha256']==sha(out/'prepared/manifest.json')
    for name,meta in av['files'].items():
        st=Path(name).stat()
        assert st.st_size==meta['size'] and st.st_mtime_ns==meta['mtime_ns'],name
    original_records=json.loads((root/'runs/formal/results.json').read_text())
    traces={t['trace_id']:t for t in acq.readl(root/'inputs/frozen_traces192.jsonl')}
    byrow={(r['trace_id'],r['condition']):r for r in original_records}
    trace_idx={t:i for i,t in enumerate(dict.fromkeys(r['trace_id'] for r in original_records))}
    directions=json.loads((out/'prepared/directions.json').read_text())
    plan=matrix(original_records,scope)
    fingerprint=dict(prepared_sha256=sha(out/'prepared/manifest.json'),scope=scope,
        plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    mp=run/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==fingerprint,'mixed provenance resume'
    else:save(mp,fingerprint)
    save(run/'matrix.json',plan)
    if scope!='smoke':
        assert json.loads((out/'runs/smoke/gate.json').read_text())['technical_pass'] is True
    def state(stage,n,**extra):
        save(run/'status.json',dict(stage=stage,completed=n,total=len(plan),
            pid=os.getpid(),updated=time.time(),**extra))
    @lru_cache(maxsize=6)
    def original_arrays(t,c):
        p=root/'runs/formal'/f'{trace_idx[t]:04d}_{COND[c]}.npz'
        assert sha(p)==prep['original_array_sha256'][str(p.relative_to(root))]
        with np.load(p) as f:return {k:f[k] for k in f.files}
    @lru_cache(maxsize=12)
    def direction(fam):
        rec=directions[fam+'_U']
        assert fam not in rec['training_families']
        p=out/rec['file'];assert sha(p)==rec['sha256']
        with np.load(p) as z:return dict(d=z['direction'],random=z['random_directions'],sigma=rec['sigma'])
    existing=[]
    for c in plan:
        p=run/(key(c)+'.json')
        if p.exists():
            r=json.loads(p.read_text())
            assert r['cell']==c and r['fingerprint']==fingerprint
            assert sha(run/r['array_file'])==r['arrays_sha256']
            assert r['technical_pass'] is True
            existing.append(r)
    if len(existing)==len(plan):
        save(run/'results.json',existing)
        if scope=='smoke':
            preserve(run/'gate.json')
            save(run/'gate.json',dict(technical_pass=True,completed=len(existing),expected=len(plan),
                noop_max_abs=max(r['checks'].get('baseline_all_arrays_max_abs',0) for r in existing),
                repeat_max_abs=max(r['checks'].get('repeat_all_arrays_max_abs',0) for r in existing),
                reconstruction_max_abs=max(r['checks']['reconstruction_max_abs'] for r in existing),
                norm_relative_max=max(r['patch']['random_norm_relative_error'] or 0 for r in existing)))
        state('complete',len(plan),resumed_complete=True);return
    assets=json.loads((root/'explore_outputs/assets_status.json').read_text())
    assert assets['status']=='ready'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1,2,4,6'
    assert torch.cuda.device_count()==4
    versions=dict(torch=torch.__version__,transformers=transformers.__version__)
    for k,v in versions.items():assert v==prep['original_manifest'][k],(k,v)
    state('loading_model',len(existing))
    caps,capmeta=historical.e2run.load_p25_assets(root/'cache/hub')
    model,tokenizer=acq.load_qwen_model(acq.QwenModelSpec(repo_id=assets['model']),
        max_memory={i:'18GiB' for i in range(4)},local_files_only=True)
    assert len(model.model.layers)==64
    assert all(str(d) not in ('cpu','disk') for d in model.hf_device_map.values())
    prompts={r['id']:r for r in acq.readl(root/'CEDAR_SUPPORTING_DATA/data/phase_validity_inputs/answer_inputs96.jsonl')}
    save(run/'runtime.json',dict(**versions,device_map=model.hf_device_map,caps=capmeta,
        gpu_names=[torch.cuda.get_device_name(i) for i in range(4)],
        started=time.time(),pid=os.getpid(),precision='bfloat16',attention='sdpa'))
    results=[]
    for cell in plan:
        rid=key(cell);rp=run/(rid+'.json');ap=run/(rid+'.npz')
        if rp.exists():
            results.append(json.loads(rp.read_text()));continue
        preserve(ap)  # Retain orphan arrays from an interrupted write.
        preserve(ap.with_suffix('.tmp.npz'))
        t=traces[cell['trace_id']];target=cell['target'];donor='U' if target=='P' else 'P'
        row=byrow[(t['trace_id'],COND[target])]
        arrays=original_arrays(t['trace_id'],target);donor_arrays=original_arrays(t['trace_id'],donor)
        d=direction(t['family_id']);j=cell['random_index']
        new,telemetry=make_patch(arrays['prompt_layer_53'],donor_arrays['prompt_layer_53'],
            d['d'],d['sigma'],cell['operation'],cell['dose'],d['random'][j] if j>=0 else None)
        hook=PromptPatch(model,arrays['prompt_layer_53'],new,row['prompt_tokens'],len(t['trace_prefix_token_ids']))
        before_hooks=hook_inventory(model)
        original_diagnostics=historical.v1run.diagnostics
        captured={}
        def capture_logits(z,row,tok):
            captured['logits']=z.detach().clone()
            return original_diagnostics(z,row,tok)
        historical.v1run.diagnostics=capture_logits
        start=time.monotonic()
        state('running',len(results),active_cell=cell,last_cell_started=time.time())
        try:
            with hook:
                with acq.BoundaryObserver(model) as observer:
                    rec=historical.run_one(model,tokenizer,transformers,caps,prompts,t,COND[target])
            with torch.inference_mode():
                reconstruction=acq.probe_reconstruct(model,observer,captured['logits'])
            values=acq.numpy_state(observer,captured['logits'])
        except Exception as exc:
            save(out/f'explore_outputs/failed_cell_{scope}_{rid}_{time.time_ns()}.json',dict(
                cell=cell,error=repr(exc),patch=telemetry,hook_calls=hook.calls,applied=hook.applied,
                hooks_restored=before_hooks==hook_inventory(model),time=time.time()))
            raise
        finally:
            historical.v1run.diagnostics=original_diagnostics
        assert before_hooks==hook_inventory(model),'patch or observer hook leaked'
        assert all(np.isfinite(v).all() for v in values.values())
        assert reconstruction<=.25,reconstruction
        checks=dict(hook_calls=hook.calls,expected_calls=len(t['trace_prefix_token_ids'])+1,
            applied=hook.applied,hook_inventory_restored=True,reconstruction_max_abs=reconstruction)
        if cell['operation']=='noop':
            error=max(float(np.max(np.abs(values[k]-arrays[k]))) for k in arrays)
            checks['baseline_all_arrays_max_abs']=error
            assert error<=1e-4,('no-op drift',error)
        if cell['repeat']:
            original_cell=dict(cell,repeat=False);basefile=run/(key(original_cell)+'.npz')
            with np.load(basefile) as ref:
                error=max(float(np.max(np.abs(values[k]-ref[k]))) for k in values)
            checks['repeat_all_arrays_max_abs']=error
            assert error<=1e-4,('repeat drift',error)
        values.update(patch_before=hook.before,patch_after=hook.after,
            donor_prompt=donor_arrays['prompt_layer_53'],validity_direction=d['d'],
            applied_direction=(telemetry['random_direction_sign']*d['random'][j]) if j>=0 else d['d'])
        acq.snapshots(ap,**values)
        rec.update(schema='cedar.c1.signed.v1',cell=cell,fingerprint=fingerprint,
            arrays_sha256=sha(ap),array_file=ap.name,patch=telemetry,checks=checks,
            seconds=time.monotonic()-start,technical_pass=True,
            peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(4)])
        save(rp,rec);results.append(rec)
        state('acquiring',len(results),last_seconds=rec['seconds'],last_cell=cell)
        # Never log scientific outcomes to the progress channel.
        print(json.dumps(dict(completed=len(results),total=len(plan),seconds=rec['seconds'],
            operation=cell['operation'],target=target)),flush=True)
    assert len(results)==len(plan) and all(r['technical_pass'] for r in results)
    save(run/'results.json',results)
    if scope=='smoke':
        gate=dict(technical_pass=True,completed=len(results),expected=len(plan),
            noop_max_abs=max(r['checks'].get('baseline_all_arrays_max_abs',0) for r in results),
            repeat_max_abs=max(r['checks'].get('repeat_all_arrays_max_abs',0) for r in results),
            reconstruction_max_abs=max(r['checks']['reconstruction_max_abs'] for r in results),
            norm_relative_max=max(r['patch']['random_norm_relative_error'] or 0 for r in results),
            note='Technical release gate only; not independent scientific evidence.')
        preserve(run/'gate.json');save(run/'gate.json',gate)
    state('complete',len(results))
    print(json.dumps(dict(scope=scope,stage='complete',completed=len(results))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--scope',choices=['smoke','formal'],required=True)
    a=p.parse_args()
    try:main(a.root,a.out,a.scope)
    except BlockingIOError:
        print('Existing C1 runner owns the lock; no files changed.',file=sys.stderr)
        sys.exit(78)
    except Exception as e:
        stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
        failure=dict(stage='failed',error=repr(e),traceback=traceback.format_exc(),updated=time.time(),pid=os.getpid())
        save(a.out/f'explore_outputs/failure_{a.scope}_{stamp}.json',failure)
        preserve(a.out/f'runs/{a.scope}/failure.json')
        save(a.out/f'runs/{a.scope}/failure.json',failure)
        raise

