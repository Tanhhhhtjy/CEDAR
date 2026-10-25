"""Read-only source inventory; copies only into the dated maintenance workspace."""
from pathlib import Path
import os,re,csv,json,hashlib,subprocess,shutil,collections,datetime
from concurrent.futures import ThreadPoolExecutor
BASE=Path('D:/CEDAR/repository_maintenance_20261025');REPO=BASE/'repository'
OUT=BASE/'packages';OUT.mkdir(exist_ok=True)
SELECTION={};OMITTED=[]
IGNORE={'.git','.venv','venv','node_modules','__pycache__','.pytest_cache','.ruff_cache'}
ARCHIVE_EXT=('.tar','.tar.gz','.tgz','.zip','.bundle','.partial')
def excluded(p):
 n=p.name.lower()
 if any(x in IGNORE for x in p.parts):return 'environment_or_cache'
 if n in {'known_hosts','.git','.env'} or n.startswith('.env.'):return 'access_or_environment_configuration'
 if n.endswith(ARCHIVE_EXT):return 'archive_duplicate_or_transfer_container'
 if any(x in n for x in ['http_connect_proxy','monitor_access','automation_before','automation_paused','pause_automation','ssh_access','conversation_log','对话全记录']):return 'access_configuration_or_private_conversation'
 if n.endswith('.lock') or n.endswith('.pid'):return 'runtime_lock_or_pid'
 return None
def add(package,p,rel):
 p=Path(p);reason=excluded(p)
 if reason:OMITTED.append({'source':p.as_posix(),'reason':reason});return
 if p.is_file():
  key=(package,str(rel).replace('\\','/'))
  if key in SELECTION and SELECTION[key]!=p:raise ValueError(key)
  SELECTION[key]=p
def tree(package,root,prefix='',filterfn=None):
 root=Path(root)
 for parent,dirs,files in os.walk(root):
  dirs[:]=[x for x in dirs if x not in IGNORE]
  for name in files:
   p=Path(parent)/name;rel=p.relative_to(root).as_posix()
   if filterfn is None or filterfn(p,rel):add(package,p,Path(prefix)/rel)

tree('historical-server-20260922','D:/CEDAR/server_snapshot_20260922')
tree('rg1p-full-20260728','D:/CEDAR_RG1P_DELIVERY_REPO/artifacts/m2/rg1p_stagea_autodl_route_a','artifacts/m2/rg1p_stagea_autodl_route_a')
tree('supporting-data','D:/CEDAR/CEDAR_SUPPORTING_DATA')
tree('s1p1-20260928','D:/CEDAR/arr_revision_20260927')
tree('c1-failed-v2-20260928','D:/CEDAR/arr_c1_20260928')
root=Path('D:/CEDAR/arr_c1_signed_20260928')
tree('c1-signed-20261003',root,filterfn=lambda p,r:not r.startswith('explore_outputs/') or r.startswith(('explore_outputs/closure_20261003/','explore_outputs/deep_analysis_20261003/')) or p.suffix.lower()=='.md')
root=Path('D:/CEDAR/arr_next_steps_20261005')
keep=('B2_complete_evidence_20261006/','B2_closure_20261006/','closure_20261006/','H_JEV_design_20261006/')
tree('b1b2-20261006',root,filterfn=lambda p,r:not r.startswith('explore_outputs/') or r[len('explore_outputs/'):].startswith(keep) or p.suffix.lower()=='.md')
tree('cross-model-additional','D:/CEDAR/outputs/xm_records','xm_records')
tree('cross-model-additional','D:/CEDAR/LLama_data','LLama_data')
tree('research-notes','D:/CEDAR/research_notes',filterfn=lambda p,r:p.suffix in {'.md','.json','.py'} and 'FULLTEXT_CHECKS' not in p.name)
tree('historical-manuscript','D:/CEDAR/CEDAR_overleaf_AAAI27')
tree('historical-manuscript','D:/CEDAR/evolution2/manuscripts','evolution2/manuscripts')
tree('historical-manuscript','D:/CEDAR/evolution/source','evolution/source')
tree('research-history','D:/CEDAR/outputs/bib','bibliography')
add('research-history','D:/CEDAR/ARR_RESULTS_INDEX.md','ARR_RESULTS_INDEX.md')
source=Path('C:/Users/legion/Desktop/paper_proposal')
for p in source.glob('*.md'):
 if p.stat().st_size>5 and ('CEDAR' in p.name or p.name.startswith(('核心思想','基线判别','激活钳制','理论工具','两周日程','评审意见','认识论世界模型','实验路径','项目对齐','项目任务书','AAAI_project','background_','cedar-aaai','SECTION3_'))):add('research-history',p,p.name)

raw=subprocess.check_output(['git','ls-tree','-rz','origin/agent/cross-model-transport-delivery-v1'],cwd=REPO)
REMOTE=collections.defaultdict(list)
for entry in raw.split(b'\0'):
 if entry:
  meta,name=entry.split(b'\t',1);mode,typ,oid=meta.split()
  if typ==b'blob':REMOTE[oid.decode()].append(name.decode())
patterns={
 'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
 'github_token':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
 'api_token':re.compile(r'\b(?:sk-[A-Za-z0-9_-]{32,}|AKIA[A-Z0-9]{16})\b'),
 'literal_password':re.compile(r'''(?i)\b(?:password|passwd|pwd)\b\s*[=:]\s*["']([^"'\r\n]{4,})["']'''),
 'chinese_password':re.compile(r'(?:密码|口令)\s*[：:=]\s*([^\s，。；;]{4,})'),
 'url_credentials':re.compile(r'https?://[^\s/:@]+:[^\s/@]+@'),
}
MANIFEST=[];flags=[];totals=collections.defaultdict(lambda:collections.Counter())
def inspect(item):
 (package,rel),p=item;local_flags=[]
 b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();oid=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
 if p.suffix.lower() in {'.md','.txt','.py','.json','.jsonl','.csv','.yaml','.yml','.toml','.sh','.ps1','.js','.tex','.bib','.log','.out','.tsv','.html'} or p.name in {'README','Dockerfile'}:
  text=b.decode('utf-8',errors='replace')
  for rule,pat in patterns.items():
   for m in pat.finditer(text):
    # Values are deliberately never logged. Review flags by source and line only.
    local_flags.append({'package':package,'path':rel,'source':p.as_posix(),'rule':rule,'line':text.count('\n',0,m.start())+1})
 dataarea=any(x in rel.split('/') for x in {'records','hidden_states','runs','artifacts','data','datasets','prepared','inputs','logs','xm_records','LLama_data'})
 if package in {'research-notes','research-history','historical-manuscript'}:preview=not p.suffix.lower() in {'.pdf','.png','.jpg','.jpeg','.svg'} or len(b)<2**20
 elif package in {'historical-server-20260922','rg1p-full-20260728','cross-model-additional'}:preview=False
 else:preview=(not dataarea and len(b)<2**20 and p.suffix.lower() in {'.md','.py','.csv','.json','.txt','.yml','.yaml','.toml','.png','.svg','.sh','.ps1'}) or (p.suffix.lower()=='.md' and len(b)<2**20)
 normalized=b.replace(b'\r\n',b'\n');normalized_oid=hashlib.sha1(b'blob '+str(len(normalized)).encode()+b'\0'+normalized).hexdigest()
 rec={'package':package,'path':rel,'bytes':len(b),'sha256':sha,'source':p.as_posix(),'git_blob':oid,'already_in_historical_branch':bool(REMOTE.get(oid)),'line_ending_equivalent_in_historical_branch':bool(REMOTE.get(normalized_oid)),'repository_preview':preview}
 return rec,local_flags
print(json.dumps({'selected_for_inspection':len(SELECTION)}),flush=True)
with ThreadPoolExecutor(max_workers=8) as pool:
 for i,(rec,found) in enumerate(pool.map(inspect,sorted(SELECTION.items()))):
  flags.extend(found);MANIFEST.append(rec);package=rec['package'];totals[package]['files']+=1;totals[package]['bytes']+=rec['bytes'];totals[package]['matched_historical_blobs']+=rec['already_in_historical_branch'];totals[package]['matched_after_line_endings']+=rec['line_ending_equivalent_in_historical_branch'];totals[package]['preview_files']+=rec['repository_preview']
  if (i+1)%2500==0:print(json.dumps({'inspected':i+1}),flush=True)
(BASE/'selection.json').write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2),encoding='utf-8')
(BASE/'excluded_files.json').write_text(json.dumps(OMITTED,ensure_ascii=False,indent=2),encoding='utf-8')
(BASE/'secret_scan_flags.json').write_text(json.dumps(flags,ensure_ascii=False,indent=2),encoding='utf-8')
(BASE/'selection_summary.json').write_text(json.dumps(dict(totals),indent=2),encoding='utf-8')
print(json.dumps({'selected':len(MANIFEST),'packages':dict(totals),'flag_count':len(flags),'flag_files':list({(x['package'],x['path']) for x in flags})},ensure_ascii=False,indent=2))
