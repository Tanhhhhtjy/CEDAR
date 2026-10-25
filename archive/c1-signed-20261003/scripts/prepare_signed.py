"""Reuse previously reproduced P1 directions exactly; freeze new execution provenance."""
import argparse,shutil,time
from pathlib import Path
from common import read,save,sha,verify_prepared

def main(root,out):
    if (out/'prepared/manifest.json').exists():
        verify_prepared(root,out);print('Existing prepared artifacts verified, not regenerated.',flush=True);return
    source=root/'c1_20260928_v2';old=read(source/'prepared/manifest.json')
    assert old['status']=='ready' and max(old['p1_reconstruction'].values())<=1e-6
    original=read(root/'runs/formal/manifest.json');assert original==old['original_manifest']
    assert sha(root/'runs/formal/results.json')==old['records_sha256']
    for field,rel in [('protocol_sha256','PROTOCOL.md'),('source_runner_sha256','CEDAR_SUPPORTING_DATA/code/scripts/run_e2b_fixed_trace_cached.py'),('trace_sha256','inputs/frozen_traces192.jsonl'),('supporting_archive_sha256','CEDAR_SUPPORTING_DATA.zip')]:
        assert sha(root/rel)==original[field],rel
    for rel,h in old['original_array_sha256'].items():assert sha(root/rel)==h,rel
    prepared=out/'prepared';prepared.mkdir(parents=True,exist_ok=True)
    for name in ['scales','directions']:
        src=source/f'prepared/{name}.json';assert sha(src)==old[name+'_sha256']
        dst=prepared/(name+'.json')
        if dst.exists():assert sha(dst)==sha(src)
        else:shutil.copy2(src,dst)
    specs=read(prepared/'directions.json')
    assert len(specs)==24
    for name,rec in specs.items():
        fam=name.rsplit('_',1)[0]
        assert fam not in rec['training_families'] and len(rec['training_families'])==11
        assert len(rec['training_trace_ids'])==44
        src=source/rec['file'];assert sha(src)==rec['sha256']
        dst=out/rec['file'];dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert sha(dst)==rec['sha256']
        else:shutil.copy2(src,dst)
    scales=read(prepared/'scales.json')
    assert all(v[k]['slope']>0 for v in scales.values() for k in ['same','family'])
    dependencies=['PROTOCOL.md','scripts/run_arr_acquisition.py','scripts/analyze_s1.py',
        'inputs/frozen_traces192.jsonl','CEDAR_SUPPORTING_DATA/data/phase_validity_inputs/answer_inputs96.jsonl',
        'analysis-output/formal/s1/token_splits.json','analysis-output/formal/s1/S1_DETAILS.json',
        'analysis-output/formal/p1/P1_SUMMARY.json','explore_outputs/assets_status.json']
    # Freeze every Python module in the immutable historical execution package.
    dependencies += [str(p.relative_to(root)) for p in (root/'CEDAR_SUPPORTING_DATA/code').rglob('*.py')]
    frozen={str(p.relative_to(out)):sha(p) for p in [out/'PROTOCOL.md',*sorted((out/'scripts').glob('*.py'))]}
    save(prepared/'manifest.json',dict(status='ready',created=time.time(),source=str(source),
        original_manifest=original,original_array_sha256=old['original_array_sha256'],
        p1_reconstruction=old['p1_reconstruction'],source_prepared_sha256=sha(source/'prepared/manifest.json'),
        directions_sha256=sha(prepared/'directions.json'),scales_sha256=sha(prepared/'scales.json'),
        records_sha256=old['records_sha256'],files=frozen,dependencies={r:sha(root/r) for r in dependencies}))
    verify_prepared(root,out)
    print('Prepared: 384 arrays, 24 unchanged P1 directions, scales and code fingerprints verified.',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root,a.out)
