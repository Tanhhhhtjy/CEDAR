"""Produce an auditable operational revision; original frozen files stay intact."""
from pathlib import Path
import hashlib,json,difflib,ast
p=Path(__file__).resolve().parents[1]
original=(p/'scripts/campaign.py').read_text(encoding='utf-8')
source=original
def replace(before,after):
    global source
    assert source.count(before)==1,before[:80]
    source=source.replace(before,after)
replace('from common import *',"sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))\nfrom common import *\nfrom takeover import prepare_takeover, AdoptedProcess, verify_operations")
replace('def main(root,out):','def main(root,out,takeover=None):')
replace("    lock=open(out/'campaign.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n    with open(out/'runner.lock','a+') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)","    verify_operations(out)\n    verify(root,out)\n    adopted = prepare_takeover(root,out,takeover) if takeover else None\n    lock=open(out/'campaign.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n    if adopted is None:\n        with open(out/'runner.lock','a+') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\n    else:\n        adopted.verify_live()")
start=source.index("    status('cpu_tests');")
end=source.index("    bp=out/'explore_outputs/gpu_budget.json';",start)
source=source[:start]+'''    status('resource_amendment_active',runtime_limits=None,adopted_runner=adopted.pid if adopted else None)
    verify(root,out)
    assert read(out/'explore_outputs/cpu_input_audit.json')['status']=='passed'
    assert read(out/'analysis/A/status.json')['stage']=='offline_analysis_complete_annotations_need_review'
    av=read(out/'prepared/asset_verification.json');assert av['status']=='passed'
    assert av['prepared_sha256']==sha(out/'prepared/manifest.json')
    files=av['files']
    for name,meta in files.items():
        s=Path(name).stat();assert s.st_size==meta['size'] and s.st_mtime_ns==meta['mtime_ns'],name
    # Resume from completed CPU/asset/A work; neither regenerate nor overwrite it.
'''+source[end:]
replace("    baseline={(r['trace_id'],r['condition']):r", "    budget.update(cumulative_limit_seconds=None,smoke_limit_seconds=None,resource_policy='operations/RESOURCE_AMENDMENT.md');save(bp,budget)\n    baseline={(r['trace_id'],r['condition']):r")
replace("        assert budget['used_seconds']+predicted<43200,('12h budget forecast',forecast)\n",'')
replace('        if pending:\n            memory=',"        if pending and not (adopted is not None and adopted.scope==scope):\n            memory=")
replace("        status(scope,forecast=forecast);log=out/f'logs/{scope}_{runstamp}.log'", "        status(scope,forecast=forecast);log=adopted.log if adopted is not None and adopted.scope==scope else out/f'logs/{scope}_{runstamp}.log'")
replace("            child=subprocess.Popen([PY,'-u',str(out/'scripts/run_phase_factorial.py'),*args,'--scope',scope],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)","            if adopted is not None and adopted.scope==scope:\n                child=adopted;child.verify_live();child.acknowledge(os.getpid())\n            else:\n                child=subprocess.Popen([PY,'-u',str(out/'scripts/run_phase_factorial.py'),*args,'--scope',scope],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)")
replace("                    assert budget['used_seconds']<43200,'12h cumulative GPU budget exceeded'\n",'')
replace("                    assert scope!='smoke' or budget['scope_seconds'][scope]<10800,'3h smoke budget exceeded'\n",'')
replace("p.add_argument('--out',type=Path,required=True);a=p.parse_args()", "p.add_argument('--out',type=Path,required=True);p.add_argument('--takeover',type=Path);a=p.parse_args()")
replace('try:main(a.root,a.out)','try:main(a.root,a.out,a.takeover)')
ast.parse(source)
assert '43200' not in source and '10800' not in source
assert "assert now-transition<5400" in source and "20*1024**3" in source and "2*1024**3" in source
assert "['smoke','B1']" in source
(p/'operations/campaign_runtime.py').write_text(source,encoding='utf-8')
(p/'operations/controller_changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile='frozen scripts/campaign.py',tofile='operations/campaign_runtime.py')),encoding='utf-8')
print('Runtime limits removed; disk, stall, scope and scientific gates retained.')
