"""Check actual signed patches for the frozen matrix before GPU loading."""
import argparse,time
from pathlib import Path
import numpy as np
from common import read,save,sha,stamp,matrix,verify_prepared,COND
from c1_core import make_patch
def main(root,out):
    prep=verify_prepared(root,out);rows=read(root/'runs/formal/results.json')
    idx={t:i for i,t in enumerate(dict.fromkeys(r['trace_id'] for r in rows))};byid={r['trace_id']:r for r in rows}
    specs=read(out/'prepared/directions.json');scopes={};failures=[];norms=[]
    for scope in ['smoke','formal']:
        plan=matrix(rows,scope);last=None;passed=0;maxerr=0.
        for cell in plan:
            t=cell['trace_id'];fam=byid[t]['family_id']
            if t!=last:
                last=t;cache={c:np.load(root/f'runs/formal/{idx[t]:04d}_{COND[c]}.npz')['prompt_layer_53'] for c in 'UP'}
                spec=specs[fam+'_U']
                with np.load(out/spec['file']) as z:direction=z['direction'];random=z['random_directions']
            c=cell['target'];j=cell['random_index'];donor='U' if c=='P' else 'P'
            try:
                y,s=make_patch(cache[c],cache[donor],direction,spec['sigma'],cell['operation'],cell['dose'],random[j] if j>=0 else None)
                assert np.isfinite(y).all() and np.isfinite(s['realized_norm'])
                if cell['operation']=='dose':assert s['realized_norm']>0
                maxerr=max(maxerr,s['random_norm_relative_error'] or 0.);passed+=1
                norms.append(dict(scope=scope,cell=cell,patch=s))
            except Exception as exc:failures.append(dict(scope=scope,cell=cell,error=repr(exc)))
        scopes[scope]=dict(total=len(plan),passed=passed,max_relative_error=maxerr)
        print(scopes[scope],flush=True)
    report=dict(status='passed' if not failures else 'failed',time=time.time(),scopes=scopes,failures=failures,
        prepared_sha256=sha(out/'prepared/manifest.json'),patches=norms)
    save(out/f'explore_outputs/preflight_{stamp()}.json',report)
    save(out/'prepared/patch_preflight.json',report)
    assert not failures,failures[:2]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root,a.out)
