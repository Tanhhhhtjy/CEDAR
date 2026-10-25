"""Operational guards for the authorized frozen B2 package; no scientific changes."""
import fcntl
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from common import read, save, sha, key, matrix, stamp

def verify_package(out):
    manifest = read(out / 'operations/b2_20261006/manifest.json')
    assert manifest['prepared_sha256'] == sha(out / 'prepared/manifest.json')
    for name, digest in manifest['files'].items():
        assert sha(out / name) == digest, name

def scope_audit(out, scope, complete=True):
    import numpy as np
    run = out / 'runs' / scope
    rows = read(out / 'inputs/development_split.json')
    plan = matrix(rows, scope)
    assert plan == read(out / f'inputs/{scope}_matrix.json')
    expected = {key(c): c for c in plan}
    fp = dict(prepared_sha256=sha(out/'prepared/manifest.json'), scope=scope,
              matrix_sha256=sha(out/f'inputs/{scope}_matrix.json'), cuda_visible_devices='1,2,4,6')
    assert read(run/'manifest.json') == fp
    traces = {r['trace_id']: r for r in rows}
    hashes = {}
    records = {}
    for path in run.glob('*.json'):
        if len(path.stem) != 24 or any(c not in '0123456789abcdef' for c in path.stem):
            continue
        r = read(path)
        assert path.stem in expected and r['cell'] == expected[path.stem]
        assert r['fingerprint'] == fp and r['technical_pass'] is True
        assert r['model_generate_used'] and r['natural_generation'] is False
        assert r['generated_text'] is False
        array = run/r['array_file']
        assert sha(array) == r['arrays_sha256']
        with np.load(array) as z:
            assert all(np.isfinite(z[k]).all() for k in z.files)
        c = r['checks'];t = traces[r['trace_id']];length = len(t['trace_prefix_token_ids']);phase=r['cell']['phase']
        assert c['hooks_restored'] and c['reconstruction_max_abs'] <= .25
        assert c.get('anchor_max_abs', 0) <= 1e-4 and c.get('repeat_max_abs', 0) <= 1e-4
        assert set(c['cap_calls']['calls'].values()) == {length+1}
        assert set(c['cap_calls']['enabled_calls'].values()) == {length*int(phase[0])+int(phase[1])}
        terminal = c['terminal']
        assert terminal['call_index'] == length and terminal['input_token'] == t['trace_prefix_token_ids'][-1]
        assert terminal['past_cache_length'] == t['prompt_tokens']+length-1
        assert r['processor_order']['capture_before_force_all']
        hashes[str(path.relative_to(out))] = sha(path)
        hashes[str(array.relative_to(out))] = r['arrays_sha256']
        records[path.stem] = r
    if complete:
        assert set(records) == set(expected)
        result = read(run/'results.json')
        assert len(result) == len(plan) and {key(r['cell']):r for r in result} == records
        assert read(run/'status.json')['stage'] == 'complete'
    return dict(scope=scope, completed=len(records), expected=len(plan), hashes=hashes)

def preflight(root, out, runstamp):
    verify_package(out)
    target = out/f'explore_outputs/B2_start_{runstamp}'
    target.mkdir(parents=True, exist_ok=True)
    for name in ['campaign_status.json','gpu_budget.json','monitor_state.json']:
        path=out/'explore_outputs'/name
        if path.exists():
            (target/f'prior_{name}').write_bytes(path.read_bytes())
    audits={s:scope_audit(out,s) for s in ['smoke','B1']}
    assert read(out/'runs/smoke/gate.json')['technical_pass'] is True
    frozen={}
    for directory in ['runs/smoke','runs/B1','analysis/B1']:
        for p in (out/directory).rglob('*'):
            if p.is_file():frozen[str(p.relative_to(out))]=sha(p)
    av=read(out/'prepared/asset_verification.json')
    assets={}
    for i,(name,meta) in enumerate(av['files'].items(),1):
        p=Path(name);s=p.stat()
        assert s.st_size==meta['size'] and s.st_mtime_ns==meta['mtime_ns'],name
        digest=sha(p);assert digest==meta['sha256'],name
        assets[name]=digest
        print(json.dumps(dict(stage='B2_asset_hash',completed=i,total=len(av['files']))),flush=True)
    record=dict(status='passed',time=time.time(),campaign_pid=os.getpid(),campaign_uid=os.getuid(),
                campaign_cwd=os.getcwd(),prior_scopes=audits,asset_hashes=assets,
                prior_package_file_hashes=frozen,prepared_sha256=sha(out/'prepared/manifest.json'),
                B2_matrix_sha256=sha(out/'inputs/B2_matrix.json'))
    save(target/'preflight.json',record)
    return target,record

def verify_prior_unchanged(out, receipt):
    for name,digest in receipt['prior_package_file_hashes'].items():
        assert sha(out/name)==digest,('Prior completed package changed',name)

def analysis_receipt(out, scope):
    dest=out/'analysis'/scope
    summary=read(dest/'SUMMARY.json')
    assert summary['scope']==scope and summary['bases']==(24 if scope=='pooled' else 12)
    expected='complete_development_analysis' if scope=='pooled' else 'descriptive_interim_no_primary_tests'
    assert summary['status']==expected
    tested={k for k,v in summary['summaries'].items() if 'primary_test' in v}
    wanted={f'raw_{m}_{c}' for m in ['broad','specific'] for c in ['EH','ET','I']} if scope=='pooled' else set()
    assert tested==wanted
    for name in ['details.json','base_endpoints.json','frozen_scales.json','SUMMARY.json','REPORT.md']:
        assert (dest/name).is_file()
    return dict(scope=scope,record_hashes={b:sha(out/f'runs/{b}/results.json') for b in (['B1','B2'] if scope=='pooled' else ['B2'])},
                outputs={str(p.relative_to(out)):sha(p) for p in dest.iterdir() if p.is_file()})
