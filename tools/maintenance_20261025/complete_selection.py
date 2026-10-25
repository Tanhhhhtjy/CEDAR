from pathlib import Path
import csv,json,re,hashlib,subprocess,collections,tarfile
from concurrent.futures import ThreadPoolExecutor
B=Path('D:/CEDAR/repository_maintenance_20261025');R=B/'repository'
rows=json.loads((B/'selection.json').read_text(encoding='utf-8'))
removed=[r for r in rows if '/H_JEV_design_20261006/references/' in r['source']]
rows=[r for r in rows if r not in removed]
known={r['sha256']:r for r in rows};sources={r['source'] for r in rows}
refs=subprocess.check_output(['git','for-each-ref','--format=%(refname)','refs/remotes/origin'],cwd=R,text=True).splitlines()
oids=set()
for ref in refs:
 for entry in subprocess.check_output(['git','ls-tree','-rz',ref],cwd=R).split(b'\0'):
  if entry:
   meta,name=entry.split(b'\t',1);mode,typ,oid=meta.split()
   if typ==b'blob':oids.add(oid.decode())
groups={'arr_c1_signed_20260928','arr_next_steps_20261005','CEDAR','CEDAR-prover-generator-feasibility','CEDAR-prover-generator-hardening-v1-1','CEDAR-prover-generator-hardening-v1-2','CEDAR-rg1p-autodl-amend','_worktrees','_wt_e2b_tip','figures','image','outputs'}
inv=list(csv.DictReader((B/'local_inventory.csv').open(encoding='utf-8')))
candidates=[];omit=[]
for x in inv:
 root=x['root'];rel=x['path'];p=Path(root)/rel;n=p.name.lower()
 if p.as_posix() in sources:continue
 if root=='D:/CEDAR' and rel.split('/')[0] not in groups:continue
 if root=='D:/CEDAR_RG1P_TRANSFER':continue
 if n.endswith(('.tar','.tar.gz','.tgz','.zip','.bundle','.partial','.pyc','.pyo')) or n in {'.git','known_hosts'} or n.startswith('.env') or '/__pycache__/' in p.as_posix():continue
 if any(v in p.as_posix().lower() for v in ['http_connect_proxy','monitor_access','automation_','pause_automation','ssh_access','conversation_log','对话全记录','h_jev_design_20261006/references','muse_jlens_research']):continue
 if n.endswith(('.lock','.pid')):continue
 candidates.append((x,p))
def inspect(item):
 x,p=item;b=p.read_bytes();sha=hashlib.sha256(b).hexdigest()
 if sha in known:return x,p,sha,'duplicate',None
 norm=b.replace(b'\r\n',b'\n');oid=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();noid=hashlib.sha1(b'blob '+str(len(norm)).encode()+b'\0'+norm).hexdigest()
 if oid in oids or noid in oids:return x,p,sha,'in_remote_history',None
 return x,p,sha,'new',b
aliases=[];added=[];flags=[];counts=collections.Counter()
pat=re.compile(r'''-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----|\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-[A-Za-z0-9_-]{32,}|AKIA[A-Z0-9]{16})\b|(?i:\b(?:password|passwd|pwd)\b\s*[=:]\s*["']([^"'\r\n]{4,})["'])|(?:密码|口令)\s*[：:=]\s*([^\s，。；;]{4,})''')
print(json.dumps({'additional_candidates':len(candidates)}),flush=True)
with ThreadPoolExecutor(max_workers=8) as pool:
 for i,(x,p,sha,status,b) in enumerate(pool.map(inspect,candidates)):
  if status=='new' and sha in known:status='duplicate'
  if status=='new':
   rel=Path(x['root']).name+'/'+x['path'];preview=p.suffix.lower() in {'.md','.bib','.tex'} and len(b)<2**20
   rec={'package':'supplementary-local-files','path':rel,'bytes':len(b),'sha256':sha,'source':p.as_posix(),'git_blob':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),'already_in_historical_branch':False,'line_ending_equivalent_in_historical_branch':False,'repository_preview':preview}
   added.append(rec);known[sha]=rec
   if p.suffix.lower() in {'.md','.txt','.py','.json','.jsonl','.csv','.yaml','.yml','.toml','.sh','.ps1','.js','.tex','.bib','.log','.out','.tsv','.html'}:
    text=b.decode('utf-8',errors='replace')
    for m in pat.finditer(text):flags.append({'source':p.as_posix(),'line':text.count('\n',0,m.start())+1})
  counts[status]+=1
  target=known.get(sha,{})
  aliases.append({'source':p.as_posix(),'bytes':int(x['bytes']),'sha256':sha,'disposition':status,'archive_package':target.get('package',''),'archive_path':target.get('path','')})
  if (i+1)%20000==0:print(json.dumps({'compared':i+1}),flush=True)
rows+=added
(B/'selection_final.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
(B/'supplement_secret_flags.json').write_text(json.dumps(flags,ensure_ascii=False,indent=2),encoding='utf-8')
with (B/'additional_comparison.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(aliases[0]));w.writeheader();w.writerows(aliases)
manifest={r['path']:r for r in rows if r['package']=='rg1p-full-20260728'}
verified=0;differences=[]
with tarfile.open('D:/CEDAR_RG1P_TRANSFER/rg1p_route_a_artifacts.tar') as t:
 for m in t:
  if not m.isfile():continue
  data=t.extractfile(m).read();rel=m.name.removeprefix('./');r=manifest.get(rel)
  if r and hashlib.sha256(data).hexdigest()==r['sha256']:verified+=1
  else:differences.append(rel)
audit={'additional_compared':len(candidates),'dispositions':dict(counts),'unique_supplement_files':len(added),'unique_supplement_bytes':sum(r['bytes'] for r in added),'excluded_thirdparty_reference_copies':len(removed),'unresolved_secret_flags':len(flags),'transfer_tar_files_verified':verified,'transfer_tar_differences':differences}
(B/'comparison_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(audit,ensure_ascii=False,indent=2))
