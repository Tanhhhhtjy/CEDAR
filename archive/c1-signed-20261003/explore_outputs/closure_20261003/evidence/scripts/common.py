"""Pure I/O and provenance helpers (no ML environment dependency)."""
import hashlib,json,time
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)
def stamp():return time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
def preserve(p):
    p=Path(p)
    if p.exists():p.rename(p.with_name(p.name+'.preserved_'+str(time.time_ns())))
def key(cell):return hashlib.sha256(json.dumps(cell,sort_keys=True).encode()).hexdigest()[:24]
COND={'U':'uncapped','P':'full_p25'}

def matrix(rows,scope):
    assert scope in ['smoke','formal']
    primary=sorted([r for r in rows if r['condition']=='uncapped' and r['trace_source']=='e2a_uncapped'],key=lambda r:r['source_id'])
    assert len(primary)==96
    if scope=='smoke':
        base=min(r['base_id'] for r in primary)
        primary=[r for r in primary if r['base_id']==base]
        assert len(primary)==4 and {r['arm'] for r in primary}==set('NRVF')
    cells=[]
    def add(r,c,op,j=-1,dose=0.,repeat=False):
        cells.append(dict(trace_id=r['trace_id'],target=c,operation=op,random_index=j,dose=float(dose),repeat=repeat))
    for r in primary:
        for c in 'PU':
            add(r,c,'noop');add(r,c,'replace')
            for dose in [-1.,1.]:
                for j in [-1,*range(1 if scope=='smoke' else 8)]:add(r,c,'dose',j,dose)
    if scope=='smoke':
        r=next(r for r in primary if r['arm']=='V')
        for c in 'PU':
            add(r,c,'noop',repeat=True);add(r,c,'dose',dose=1.,repeat=True)
    assert len(cells)=={'smoke':52,'formal':3840}[scope]
    assert len({key(c) for c in cells})==len(cells)
    return cells

def verify_prepared(root,out):
    prep=read(out/'prepared/manifest.json')
    for rel,h in prep['files'].items():assert sha(out/rel)==h,rel
    for rel,h in prep['dependencies'].items():assert sha(root/rel)==h,rel
    for name in ['scales','directions']:
        assert sha(out/f'prepared/{name}.json')==prep[name+'_sha256'],name
    assert sha(root/'runs/formal/results.json')==prep['records_sha256']
    for rec in read(out/'prepared/directions.json').values():assert sha(out/rec['file'])==rec['sha256']
    return prep
