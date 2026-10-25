"""Check all planned perturbations on saved states before any model forward."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from c1_core import make_patch
from run_c1 import matrix
from prepare_c1 import save
def main(root,out):
    records=json.loads((root/'runs/formal/results.json').read_text())
    idx={t:i for i,t in enumerate(dict.fromkeys(r['trace_id'] for r in records))}
    rows={r['trace_id']:r for r in records}
    specs=json.loads((out/'prepared/directions.json').read_text())
    matrices={scope:matrix(records,scope) for scope in ['smoke','main','dose']}
    report=dict(status='running',checked=0,failures=[],scopes={},updated=time.time())
    for scope,cells in matrices.items():
        last=None;failures=[];maxerr=0.;zero=0;norms=[]
        for cell in cells:
            trace=cell['trace_id'];fam=rows[trace]['family_id']
            if last!=trace:
                last=trace
                cache={c:np.load(root/'runs/formal'/f"{idx[trace]:04d}_{name}.npz")['prompt_layer_53']
                    for c,name in [('U','uncapped'),('P','full_p25')]}
                spec=specs[fam+'_U'];vectors=np.load(out/spec['file'])
            target=cell['target'];donor='P' if target=='U' else 'U';j=cell['random_index']
            try:
                _,s=make_patch(cache[target],cache[donor],vectors['direction'],spec['sigma'],
                    cell['operation'],cell['dose'],vectors['random_directions'][j] if j>=0 else None)
                maxerr=max(maxerr,s['random_norm_relative_error'] or 0)
                norms.append(s['realized_norm'])
                zero+=s['realized_norm']==0 and cell['operation'] not in ['noop']
            except RuntimeError as e:
                failures.append(dict(cell=cell,error=str(e)))
            report['checked']+=1
        report['failures']+=failures
        report['scopes'][scope]=dict(cells=len(cells),failures=len(failures),
            maximum_passed_relative_error=maxerr,non_noop_zero_norm=int(zero),
            norm_min=min(norms),norm_max=max(norms))
        save(out/'explore_outputs/patch_preflight.json',report)
        print(json.dumps(dict(scope=scope,**report['scopes'][scope])),flush=True)
    report.update(status='passed' if not report['failures'] else 'blocked_quantized_norm_matching',updated=time.time())
    save(out/'explore_outputs/patch_preflight.json',report)
    if report['failures']:raise RuntimeError('Matching infeasible for fixed random vectors; no GPU execution authorized by this gate.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root,a.out)

