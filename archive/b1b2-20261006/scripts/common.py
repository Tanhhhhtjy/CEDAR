import hashlib,json,time
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def readl(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x]
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)
def stamp():return time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
def key(c):return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:24]
def preserve(p):
    p=Path(p)
    if p.exists():p.rename(p.with_name(p.name+'.preserved_'+str(time.time_ns())))
def split(traces):
    rows=[t for t in traces if t['source_condition']=='uncapped'];assert len(rows)==96
    families=sorted({t['family_id'] for t in rows});assert len(families)==12
    out=[]
    for f in families:
        bases=sorted({t['base_id'] for t in rows if t['family_id']==f});assert len(bases)==2
        for i,b in enumerate(bases):
            group=[t for t in rows if t['base_id']==b];assert len(group)==4 and {t['arm'] for t in group}==set('NRVF')
            for t in sorted(group,key=lambda t:t['arm']):out.append(dict(t,batch='B1' if i==0 else 'B2'))
    assert len({t['structure_class'] for t in out})==8
    return out
def matrix(rows,scope):
    chosen=[t for t in rows if t['batch']==scope] if scope!='smoke' else [t for t in rows if t['base_id']==min(x['base_id'] for x in rows)]
    cells=[dict(trace_id=t['trace_id'],phase=c,repeat=False) for t in chosen for c in ['00','10','01','11']]
    if scope=='smoke':cells += [dict(trace_id=t['trace_id'],phase=c,repeat=True) for t in chosen for c in ['00','11']]
    assert len(cells)==(24 if scope=='smoke' else 192) and len({key(c) for c in cells})==len(cells)
    return cells
def enabled(call,L,phase):
    assert 0<=call<=L and L>0 and phase in ['00','10','01','11']
    return phase[1 if call==L else 0]=='1'
def verify(root,out):
    p=read(out/'prepared/manifest.json')
    for n,h in p['files'].items():assert sha(out/n)==h,n
    for n,h in p['dependencies'].items():assert sha(root/n)==h,n
    return p
