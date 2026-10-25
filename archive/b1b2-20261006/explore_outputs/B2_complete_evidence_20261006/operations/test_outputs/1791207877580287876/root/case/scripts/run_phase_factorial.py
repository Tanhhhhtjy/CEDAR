import sys,time,os,fcntl,json
from pathlib import Path
p=Path(sys.argv[sys.argv.index('--out')+1])
f=(p/'runner.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
while not (p/'finish').exists():time.sleep(.05)
(p/'runs/smoke/status.json').write_text(json.dumps(dict(pid=os.getpid(),stage='complete',completed=24,total=24)))
