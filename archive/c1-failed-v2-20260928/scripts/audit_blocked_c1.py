"""Read-only scientific audit; writes only new timestamped evidence files."""
import collections, datetime, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from c1_core import unit, rms, realized

ROOT=Path('/data4/WuMingrui/TianJinyu/CEDAR')
OUT=ROOT/'c1_20260928_v2'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')

def main():
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    records=read(ROOT/'runs/formal/results.json')
    idx={t:i for i,t in enumerate(dict.fromkeys(r['trace_id'] for r in records))}
    rows={r['trace_id']:r for r in records}
    pre=read(OUT/'explore_outputs/patch_preflight.json')
    dirs=read(OUT/'prepared/directions.json')
    failures=[]
    for failure in pre['failures']:
        c=failure['cell'];r=rows[c['trace_id']];target=c['target'];donor='P' if target=='U' else 'U'
        cached={s:np.load(ROOT/f"runs/formal/{idx[c['trace_id']]:04d}_{n}.npz")['prompt_layer_53']
                for s,n in [('U','uncapped'),('P','full_p25')]}
        spec=dirs[r['family_id']+'_U']
        with np.load(OUT/spec['file']) as z:
            d=unit(z['direction']);rnd=unit(z['random_directions'][c['random_index']])
        t=cached[target].astype(float);hd=cached[donor].astype(float)
        amp=rms(t)*float((hd/rms(hd)-t/rms(t))@d)
        h=torch.tensor(t,dtype=torch.bfloat16)
        true,tn=realized(h,torch.tensor(d,dtype=torch.float32),amp)
        rd=torch.tensor(rnd,dtype=torch.float32)
        low=0.;high=max(tn*2,1e-6)
        for _ in range(32):
            if realized(h,rd,high)[1]>=tn:break
            high*=2
        for _ in range(80):
            mid=(low+high)/2
            if realized(h,rd,mid)[1]<tn:low=mid
            else:high=mid
        ln=realized(h,rd,low)[1];hn=realized(h,rd,high)[1]
        assert ln<=tn<=hn,(ln,tn,hn)
        nearest=min(abs(ln-tn),abs(hn-tn))/tn
        assert nearest>.005
        failures.append(dict(cell=c,family=r['family_id'],base=r['base_id'],arm=r['arm'],
            requested_true_amplitude=amp,true_norm=tn,neighbor_below=ln,neighbor_above=hn,
            minimum_relative_error=nearest,float32_scalar_below=float(np.float32(low)),
            float32_scalar_above=float(np.float32(high)),original_error=failure['error']))
    old=read(ROOT/'runs/c1_smoke/results.json');v1=old['results']
    orig={(r['trace_id'],r['condition']):r for r in records}
    noop=[]
    for r in v1:
        if r['control']=='noop':
            a=orig[r['trace_id'],r['target_forward_condition']]['logit_diagnostics']['logits']
            b=r['logit_diagnostics']['logits']
            noop.append(max(abs(a[k]-b[k]) for k in 'ABCD'))
    pairs=[]
    lookup={(r['trace_id'],r['transfer'],r['control']):r for r in v1}
    for r in v1:
        if r['control']=='true':
            rr=lookup[r['trace_id'],r['transfer'],'random']
            pairs.append(dict(trace_id=r['trace_id'],transfer=r['transfer'],true_norm=r['patch_norm'],
                random_norm=rr['patch_norm']))
    # Limit inventory to same-uid experiment processes belonging to this CEDAR directory.
    processes=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            argv=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            cwd=str(p.joinpath('cwd').resolve())
            if cwd==str(ROOT) and any(s in argv for s in ['campaign.py','run_c1.py','run_c1_component_smoke.py','run_arr_acquisition.py','fetch_assets.py','preflight_patches.py']):
                processes.append(dict(pid=int(p.name),cwd=cwd,command=argv))
        except (OSError,PermissionError,ProcessLookupError):pass
    assets=read(ROOT/'explore_outputs/assets_status.json')
    sizes=[]
    for name,m in assets['files'].items():
        p=Path(assets['model'])/name
        sizes.append(dict(file=name,exists=p.exists(),size_matches=p.exists() and p.stat().st_size==m['size'],sha256_recorded=m['sha256']))
    history=[json.loads(x) for x in (OUT/'explore_outputs/campaign_history.jsonl').read_text().splitlines() if x]
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
    main_seconds=sum(r['seconds'] for r in records if r['trace_source']=='e2a_uncapped')*11
    dose_seconds=sum(r['seconds'] for r in records if r['trace_source']=='e2a_uncapped' and r['arm'] in 'VF')*36
    v1audit=dict(completed=len(v1),family=old['family'],option_logits_finite=all(np.isfinite(list(r['logit_diagnostics']['logits'].values())).all() for r in v1),
        modified_positions=sorted(set(r['modified_positions'] for r in v1)),hook_calls_range=[min(r['hook_calls'] for r in v1),max(r['hook_calls'] for r in v1)],
        cap_hook_counts_equal=all(r['hook_counts_before']==r['hook_counts_after'] for r in v1),noop_count=len(noop),noop_option_max_abs=max(noop),
        random_norm_pairs=pairs,original_gate=read(ROOT/'runs/c1_smoke/gate.json'),limited_reaudit=read(ROOT/'runs/c1_smoke/gate_reaudited.json'),
        eligible_for_v2_release=False,eligible_for_formal_scientific_inference=False,
        limitations=['Raw-space donor-dependent direction, unlike primary unit-RMS P1.','No full-vocabulary readout or repeated forward.','No full custom-hook inventory verification.','Realized BF16 norms were not matched.'])
    diag=read(OUT/'explore_outputs/prerequisite_diagnostics.json')
    report=dict(timestamp_utc=stamp,status='blocked_before_v2_gpu',
        protocol_sha256=sha(ROOT/'PROTOCOL.md'),v2_protocol_sha256=sha(OUT/'PROTOCOL.md'),
        original_stage=read(ROOT/'explore_outputs/campaign_status.json'),v2_stage=read(OUT/'explore_outputs/campaign_status.json'),
        preflight_scopes=pre['scopes'],failures_confirmed=failures,failed_families=sorted(set(x['family'] for x in failures)),
        failed_arms=dict(collections.Counter(x['arm'] for x in failures)),prerequisite_status=diag['status'],
        reconstruction_errors=diag['p1_score_reconstruction_max_errors'],original_arrays_verified=len(read(OUT/'prepared/manifest.json')['original_array_sha256']),
        prompt_source_duplicate_max_error=diag['prompt_duplicate_max_error'],v1=v1audit,
        task_processes=processes,gpu=gpu,disk_free_bytes=shutil.disk_usage(ROOT).free,
        asset_status=assets['status'],asset_files=sizes,asset_hashes_recomputed_this_audit=False,
        forecast_from_historical_seconds=dict(main=main_seconds,dose=dose_seconds,
            interpretation='Wall seconds on the same four-GPU layout; preliminary, not a promised completion time.'),
        history=history,log_files=[dict(name=p.name,bytes=p.stat().st_size,mtime=p.stat().st_mtime) for p in sorted((OUT/'logs').glob('*.log'))],
        v2_gpu_records=sum(len(list((OUT/'runs'/s).glob('*.npz'))) for s in ['smoke','main','dose']))
    dest=OUT/f'explore_outputs/blocked_audit_{stamp}.json';save(dest,report)
    print(json.dumps(dict(report=str(dest),failures=len(failures),families=report['failed_families'],
        arms=report['failed_arms'],processes=processes,disk_gib=report['disk_free_bytes']/1024**3,
        error_range=[min(x['minimum_relative_error'] for x in failures),max(x['minimum_relative_error'] for x in failures)],
        forecast_hours={k:v/3600 for k,v in report['forecast_from_historical_seconds'].items() if isinstance(v,(int,float))})),flush=True)

if __name__=='__main__':main()
