"""Read-only audit of historical inputs; freeze the new development matrix."""
import argparse,sys,time
from pathlib import Path
from common import *
def main(root,out):
    old=root/'c1_signed_20260928/prepared/manifest.json';p=read(old)
    for rel,h in p['dependencies'].items():assert sha(root/rel)==h,rel
    for rel,h in p['original_array_sha256'].items():assert sha(root/rel)==h,rel
    rows=split(readl(root/'inputs/frozen_traces192.jsonl'))
    save(out/'inputs/development_split.json',rows)
    for scope in ['smoke','B1','B2']:save(out/f'inputs/{scope}_matrix.json',matrix(rows,scope))
    for L in [1,2,754,1500]:
        for c in ['00','10','01','11']:
            seq=[enabled(i,L,c) for i in range(L+1)]
            assert sum(seq)==L*int(c[0])+int(c[1])
            assert seq[-1]==bool(int(c[1])) and seq[:-1]==[bool(int(c[0]))]*L
    deps=dict(p['dependencies']);deps.update(p['original_array_sha256'])
    for rel in ['c1_signed_20260928/prepared/manifest.json','c1_signed_20260928/prepared/directions.json','c1_signed_20260928/prepared/asset_verification.json','runs/formal/results.json','runs/formal/runtime.json','runs/formal/manifest.json','runs/formal/norm_weight.npy','explore_outputs/assets_status.json']:
        deps[rel]=sha(root/rel)
    for d in read(root/'c1_signed_20260928/prepared/directions.json').values():
        rel='c1_signed_20260928/'+d['file'];deps[rel]=sha(root/rel)
    files={str(p.relative_to(out)):sha(p) for folder in ['scripts','inputs'] for p in (out/folder).glob('*') if p.is_file()}
    for n in ['EXPERIMENT_PLAN.md','B_PROTOCOL.md','FEEDBACK_20261005.txt','ORIGINAL_PLAN.md']:files[n]=sha(out/n)
    manifest=dict(schema='cedar.phase-factorial.v1',files=files,dependencies=deps,created=time.time(),original_manifest=read(root/'runs/formal/manifest.json'),original_array_sha256=p['original_array_sha256'])
    dest=out/'prepared/manifest.json'
    if dest.exists():
        prior=read(dest)
        assert prior['files']==files and prior['dependencies']==deps,'Prepared provenance drift'
    else:save(dest,manifest)
    save(out/'explore_outputs/cpu_input_audit.json',dict(status='passed',time=time.time(),traces=192,primary=96,smoke=24,B1=192,B2=192,structure_groups=8,old_arrays=len(p['original_array_sha256']),prepared_sha256=sha(dest)))
    print('Input/provenance/matrix checks passed',flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--out',type=Path,required=True);v=a.parse_args();main(v.root,v.out)
