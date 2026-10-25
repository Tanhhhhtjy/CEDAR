import sys,time,os,fcntl,json,subprocess
from pathlib import Path
p=Path(sys.argv[sys.argv.index('--out')+1])
f=(p/'campaign.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
r=subprocess.Popen([sys.executable,str(p/'scripts/run_phase_factorial.py'),'--root',str(p.parent),'--out',str(p),'--scope','smoke'],cwd=p.parent)
(p/'ready.json').write_text(json.dumps(dict(controller=os.getpid(),runner=r.pid)))
while True:time.sleep(.05)
