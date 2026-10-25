from pathlib import Path
import json,csv,hashlib,tarfile,io,collections,shutil,datetime,gzip
from concurrent.futures import ThreadPoolExecutor,as_completed
B=Path('D:/CEDAR/repository_maintenance_20261025');R=B/'repository';P=B/'packages'
rows=json.loads((B/'selection_final.json').read_text(encoding='utf-8'))
assert not json.loads((B/'supplement_secret_flags.json').read_text(encoding='utf-8'))
groups=collections.defaultdict(list)
for r in rows:groups[r['package']].append(r)
def prefix(package):
 return {'research-notes':'research/notes','research-history':'research/history','historical-manuscript':'research/manuscripts'}.get(package,'archive/'+package)
for d in [P,R/'data_releases/manifests',R/'data_releases/audits']:d.mkdir(parents=True,exist_ok=True)
fields=['path','bytes','sha256','source','already_in_historical_branch','line_ending_equivalent_in_historical_branch','repository_preview']
def build(item):
 name,items=item
 mf=R/'data_releases/manifests'/f'{name}.csv'
 with mf.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(items)
 out=P/f'{name}.tar.gz';preview=0
 with tarfile.open(out,'w:gz',compresslevel=6,format=tarfile.PAX_FORMAT) as tar:
  for r in items:
   p=Path(r['source']);b=p.read_bytes()
   if hashlib.sha256(b).hexdigest()!=r['sha256']:raise ValueError('Source changed: '+str(p))
   info=tarfile.TarInfo(name+'/'+r['path']);info.size=len(b);info.mtime=1792886400;info.mode=0o644
   tar.addfile(info,io.BytesIO(b))
   if r['repository_preview']:
    dest=R/prefix(name)/r['path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b);preview+=1
  b=mf.read_bytes();info=tarfile.TarInfo(name+'/ARCHIVE_FILE_MANIFEST.csv');info.size=len(b);info.mtime=1792886400;tar.addfile(info,io.BytesIO(b))
 expected={name+'/'+r['path']:r for r in items};checked=0
 with tarfile.open(out,'r:gz') as tar:
  for m in tar:
   if m.name.endswith('/ARCHIVE_FILE_MANIFEST.csv'):continue
   r=expected.pop(m.name);h=hashlib.sha256()
   for chunk in iter(lambda:tar.extractfile(m).read(0),b''):pass
   stream=tar.extractfile(m)
   while chunk:=stream.read(2**20):h.update(chunk)
   if h.hexdigest()!=r['sha256'] or m.size!=r['bytes']:raise ValueError(m.name)
   checked+=1
 assert not expected
 h=hashlib.sha256()
 with out.open('rb') as f:
  while b:=f.read(2**20):h.update(b)
 rec={'package':name,'asset':out.name,'bytes':out.stat().st_size,'sha256':h.hexdigest(),'files':len(items),'source_bytes':sum(r['bytes'] for r in items),'verified_members':checked,'repository_preview_files':preview,'manifest':'manifests/'+mf.name}
 print(json.dumps({'completed':name,'files':len(items),'archive_mib':round(rec['bytes']/2**20,2),'verified':checked}),flush=True)
 return rec
results=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 for f in as_completed([pool.submit(build,item) for item in groups.items()]):results.append(f.result())
results.sort(key=lambda r:r['package'])
(R/'data_releases/ASSETS.json').write_text(json.dumps({'tag':'archive-20261025','repository':'Tanhhhhtjy/CEDAR','assets':results},indent=2)+'\n',encoding='utf-8')
(P/'SHA256SUMS').write_text(''.join(r['sha256']+'  '+r['asset']+'\n' for r in results),encoding='utf-8')
shutil.copyfile(P/'SHA256SUMS',R/'data_releases/SHA256SUMS')
for name in ['comparison_audit.json','additional_comparison.csv']:
 shutil.copyfile(B/name,R/'data_releases/audits'/name)
audit={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selected_files':len(rows),'unique_content_hashes':len({r['sha256'] for r in rows}),'archives':len(results),'archive_bytes':sum(r['bytes'] for r in results),'source_bytes':sum(r['bytes'] for r in rows),'preview_files':sum(r['repository_preview_files'] for r in results),'source_hashes_verified_before_packaging':True,'all_archive_members_verified_after_packaging':True,'unresolved_secret_flags':0,'thirdparty_downloaded_reference_copies_omitted':12,'excludes_raw_conversation_and_access_material':True}
(R/'data_releases/audits/packaging_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit,indent=2),flush=True)
