"""One-shot ARR job: verified assets -> smoke -> acquisition -> S1/P1 analyses."""
from pathlib import Path
import json
import os
import subprocess
import time
import traceback
import fcntl

ROOT=Path(__file__).resolve().parents[1]
PY='/home/WuMingrui/miniconda3/envs/cedar/bin/python'
ANALYSIS=str(ROOT/'.venv-analysis/bin/python')
STATUS=ROOT/'explore_outputs/campaign_status.json'


def status(stage,**kw):
    value=dict(stage=stage,updated=time.time(),pid=os.getpid(),**kw)
    tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(STATUS)
    print(json.dumps(value),flush=True)


def launch(label,command,env=None,timeout=7200):
    status(label,command=command)
    with (ROOT/'logs'/f'{label}.log').open('a') as log:
        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)


def main():
    assert json.loads((ROOT/'explore_outputs/CPU_ENGINEERING_TEST.json').read_text())['passed']
    assert (ROOT/'explore_outputs/CPU_ANALYSIS_TEST.json').is_file()
    assert json.loads((ROOT/'explore_outputs/CPU_ANALYSIS_TEST.json').read_text())['passed']
    assert json.loads((ROOT/'explore_outputs/CPU_INPUT_VALIDATION.json').read_text())['passed']
    status('waiting_for_verified_weights')
    deadline=time.monotonic()+24*3600
    while True:
        assets=json.loads((ROOT/'explore_outputs/assets_status.json').read_text())
        if assets['status']=='ready':break
        if assets['status']=='failed':raise RuntimeError('Asset acquisition failed; inspect assets_status.json')
        if time.monotonic()>deadline:raise TimeoutError('Asset wait exceeded 24 hours')
        time.sleep(30)
    gpus=[1,2,4,6]
    status('waiting_for_gpu_memory',physical_gpus=gpus,minimum_free_mib=22528)
    deadline=time.monotonic()+12*3600
    while True:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        free={int(line.split(',')[0]):int(line.split(',')[1]) for line in raw.splitlines()}
        if all(free[i]>=22528 for i in gpus):break
        if time.monotonic()>deadline:raise TimeoutError('GPU memory wait exceeded 12 hours')
        time.sleep(30)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    launch('smoke',[PY,'scripts/run_arr_acquisition.py','--scope','smoke'],env,4*3600)
    launch('smoke_analysis',[ANALYSIS,'scripts/analyze_s1.py','--scope','smoke'],env)
    gate=json.loads((ROOT/'runs/smoke/gate.json').read_text())
    assert gate['technical_pass']
    records=json.loads((ROOT/'runs/smoke/results.json').read_text())
    measured=sum(r['seconds'] for r in records)/sum(r['teacher_forced_input_tokens'] for r in records)
    traces=[json.loads(line) for line in (ROOT/'inputs/frozen_traces192.jsonl').read_text().splitlines()]
    estimate=measured*2*sum(t['teacher_forced_input_tokens'] for t in traces)
    status('smoke_passed',gate=gate,rough_formal_seconds=estimate,
        note='Runtime projection is approximate and depends on other shared-GPU jobs. Historical identity is assessed separately.')
    launch('formal',[PY,'scripts/run_arr_acquisition.py','--scope','formal'],env,48*3600)
    launch('s1_analysis',[ANALYSIS,'scripts/analyze_s1.py','--scope','formal'],env,4*3600)
    launch('p1_analysis',[ANALYSIS,'scripts/analyze_p1.py'],env,4*3600)
    status('s1_p1_complete_c1_h1_pending',
        note='S1/P1 acquisition and analysis completed. Review scale residuals and probe specificity before causal patches and independent structural confirmation. No mechanism claim is automatic.')


if __name__=='__main__':
    lock=(ROOT/'explore_outputs/campaign.lock').open('a')
    # Prevent accidental duplicate acquisition when a launch command is repeated.
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:main()
    except Exception as e:
        status('failed',error=repr(e),traceback=traceback.format_exc())
        raise
