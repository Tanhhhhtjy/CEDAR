"""Replace only an identified supervisor; preserve and adopt its scientific runner."""
import os,signal,time,json,fcntl,subprocess,hashlib,re
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(pid):
    p=Path('/proc')/str(pid)
    try:
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,start_ticks=int(stat[19]),state=stat[0],uid=p.stat().st_uid,cwd=os.readlink(p/'cwd'),argv=(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0'))
    except (FileNotFoundError,ProcessLookupError):return None
def same_process(actual,expected):
    return actual is not None and actual['state']!='Z' and all(actual[k]==expected[k] for k in ['pid','start_ticks','uid','cwd','argv'])
def records(out,scope):
    r=out/'runs'/scope;hashes={}
    for p in r.glob('*.json'):
        if not re.fullmatch('[a-f0-9]{24}',p.stem):continue
        v=read(p);assert v['technical_pass'] and sha(r/v['array_file'])==v['arrays_sha256'],p
        hashes[str(p.relative_to(out))]=sha(p);hashes[str((r/v['array_file']).relative_to(out))]=v['arrays_sha256']
    return hashes
def verify_operations(out):
    m=read(out/'operations/manifest.json')
    for rel,h in m['files'].items():assert sha(out/rel)==h,rel
    assert sha(out/'prepared/manifest.json')==m['prepared_sha256']

class AdoptedProcess:
    def __init__(self,root,out,spec):
        self.root,self.out,self.spec=root,out,spec;self.pid=spec['runner']['pid'];self.scope=spec['scope'];self.log=Path(spec['log']);self.returncode=None
    def verify_live(self):
        assert same_process(identity(self.pid),self.spec['runner']),'Runner identity changed'
        with (self.out/'runner.lock').open('r') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return
            raise RuntimeError('Expected live runner lock is free')
    def acknowledge(self,controller_pid):
        self.verify_live()
        prior=self.spec['completed_hashes']
        assert all(sha(self.out/n)==h for n,h in prior.items())
        save(self.out/'explore_outputs/resource_takeover_verified.json',dict(time=time.time(),controller=identity(controller_pid),same_runner=identity(self.pid),preserved_files=len(prior),preserved_completed=len(prior)//2,current_record_files=len(records(self.out,self.scope)),old_controller=self.spec['controller'],cumulative_budget_limit=None,smoke_budget_limit=None,operations_manifest_sha256=sha(self.out/'operations/manifest.json')))
    def poll(self):
        now=identity(self.pid)
        if same_process(now,self.spec['runner']):return None
        if now is not None and now['state']!='Z':raise RuntimeError('Runner PID identity changed; refusing to signal')
        status=read(self.out/f'runs/{self.scope}/status.json')
        # Exit status is not available for adopted nonchild PIDs. Require durable completion;
        # the controller then independently checks complete results and the smoke gate.
        self.returncode=0 if status.get('pid')==self.pid and status.get('stage')=='complete' and status.get('completed')==status.get('total') else 1
        return self.returncode
    def terminate(self):
        self.verify_live();os.kill(self.pid,signal.SIGTERM)
    def kill(self):
        self.verify_live();os.kill(self.pid,signal.SIGKILL)
    def wait(self,timeout):
        deadline=time.monotonic()+timeout
        while self.poll() is None:
            if time.monotonic()>=deadline:raise subprocess.TimeoutExpired('adopted runner',timeout)
            time.sleep(.2)
        return self.returncode

def prepare_takeover(root,out,path):
    spec=read(path);assert spec['authorization']=='user_remove_runtime_budget_20261005'
    assert spec['scope'] in ['smoke','B1']
    assert spec['prepared_sha256']==sha(out/'prepared/manifest.json')
    for kind,name in [('controller','campaign.py'),('runner','run_phase_factorial.py')]:
        target=spec[kind];actual=identity(target['pid']);assert same_process(actual,target),kind
        assert actual['uid']==os.getuid() and actual['cwd']==str(root)
        assert str(out/'scripts'/name) in actual['argv']
        assert actual['argv'][actual['argv'].index('--out')+1]==str(out)
    assert spec['runner']['argv'][spec['runner']['argv'].index('--scope')+1]==spec['scope']
    assert all(sha(out/n)==h for n,h in spec['completed_hashes'].items())
    assert sha(out/'scripts/campaign.py')==spec['original_controller_sha256']
    with (out/'campaign.lock').open('r') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:raise RuntimeError('Expected old campaign lock is free')
    state=read(out/'explore_outputs/campaign_status.json');assert state['pid']==spec['controller']['pid'] and state['stage']==spec['scope']
    adopted=AdoptedProcess(root,out,spec);adopted.verify_live()
    # Only the supervisor receives SIGTERM; no process group signal, no runner signal.
    save(out/'explore_outputs/resource_takeover_before.json',dict(time=time.time(),spec=spec,status=state,budget=read(out/'explore_outputs/gpu_budget.json')))
    os.kill(spec['controller']['pid'],signal.SIGTERM)
    deadline=time.monotonic()+15
    while same_process(identity(spec['controller']['pid']),spec['controller']):
        assert time.monotonic()<deadline,'Old controller has not exited'
        time.sleep(.1)
    adopted.verify_live()
    return adopted
