"""Exercise the real controller-to-orphan takeover on tiny CPU-only test processes."""
import os,sys,time,json,subprocess,fcntl
from pathlib import Path
from takeover import identity,same_process,prepare_takeover,save,sha

def main():
    base=Path(__file__).resolve().parent/'test_outputs'/str(time.time_ns())
    root=base/'root';out=root/'case';(out/'scripts').mkdir(parents=True)
    (out/'prepared').mkdir();save(out/'prepared/manifest.json',{})
    (out/'explore_outputs').mkdir();(out/'runs/smoke').mkdir(parents=True);(out/'logs').mkdir()
    runner='''import sys,time,os,fcntl,json
from pathlib import Path
p=Path(sys.argv[sys.argv.index('--out')+1])
f=(p/'runner.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
while not (p/'finish').exists():time.sleep(.05)
(p/'runs/smoke/status.json').write_text(json.dumps(dict(pid=os.getpid(),stage='complete',completed=24,total=24)))
'''
    supervisor='''import sys,time,os,fcntl,json,subprocess
from pathlib import Path
p=Path(sys.argv[sys.argv.index('--out')+1])
f=(p/'campaign.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
r=subprocess.Popen([sys.executable,str(p/'scripts/run_phase_factorial.py'),'--root',str(p.parent),'--out',str(p),'--scope','smoke'],cwd=p.parent)
(p/'ready.json').write_text(json.dumps(dict(controller=os.getpid(),runner=r.pid)))
while True:time.sleep(.05)
'''
    (out/'scripts/run_phase_factorial.py').write_text(runner);(out/'scripts/campaign.py').write_text(supervisor)
    log=out/'logs/smoke.log';log.write_text('')
    controller=subprocess.Popen([sys.executable,str(out/'scripts/campaign.py'),'--root',str(root),'--out',str(out)],cwd=root)
    adopted=None;runner_pid=None
    try:
        deadline=time.monotonic()+10
        while not (out/'ready.json').exists():
            assert time.monotonic()<deadline;time.sleep(.05)
        r=json.loads((out/'ready.json').read_text());runner_pid=r['runner']
        while not (out/'runner.lock').exists():
            assert time.monotonic()<deadline;time.sleep(.05)
        target=identity(runner_pid);assert target is not None
        fake=dict(target,start_ticks=target['start_ticks']+1);assert not same_process(target,fake)
        save(out/'explore_outputs/campaign_status.json',dict(pid=controller.pid,stage='smoke'))
        save(out/'explore_outputs/gpu_budget.json',dict(used_seconds=123.,active=True))
        spec=dict(authorization='user_remove_runtime_budget_20261005',scope='smoke',prepared_sha256=sha(out/'prepared/manifest.json'),controller=identity(controller.pid),runner=target,completed_hashes={},original_controller_sha256=sha(out/'scripts/campaign.py'),log=str(log))
        save(out/'takeover.json',spec)
        adopted=prepare_takeover(root,out,out/'takeover.json')
        controller.wait(timeout=5);assert controller.returncode==-15
        assert adopted.poll() is None and same_process(identity(runner_pid),target)
        with (out/'campaign.lock').open('r') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (out/'finish').write_text('ok');assert adopted.wait(timeout=5)==0
        assert json.loads((out/'explore_outputs/gpu_budget.json').read_text())['used_seconds']==123.
        save(base/'result.json',dict(status='passed',old_controller_terminated_only=True,same_runner_survived=True,completion_detected=True,starttime_mismatch_rejected=True,budget_preserved=True))
        print('CPU takeover test passed:',base)
    finally:
        if controller.poll() is None:controller.terminate();controller.wait(timeout=5)
        if adopted and adopted.poll() is None:adopted.terminate();adopted.wait(timeout=5)
        elif runner_pid and identity(runner_pid):
            # This PID came only from the test supervisor above; never touch a scientific runner.
            i=identity(runner_pid)
            if i and str(out/'scripts/run_phase_factorial.py') in i['argv']:os.kill(runner_pid,15)
if __name__=='__main__':main()
