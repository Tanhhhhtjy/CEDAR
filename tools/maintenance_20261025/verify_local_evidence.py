from pathlib import Path
import json,re,hashlib,csv,collections
B=Path('D:/CEDAR/repository_maintenance_20261025');R=B/'repository';out=R/'data_releases/audits';out.mkdir(parents=True,exist_ok=True)
selection=json.loads((B/'selection_final.json').read_text(encoding='utf-8'));hashes={r['sha256']:r for r in selection};missing=[];available=[];report={};errors=[]
root=Path('D:/CEDAR/arr_next_steps_20261005/explore_outputs/B2_complete_evidence_20261006/runs')
for batch,expected in [('smoke',24),('B1',192),('B2',192)]:
 records=[p for p in (root/batch).glob('*.json') if re.fullmatch('[0-9a-f]{24}',p.stem)];verified=0;cells=set()
 for p in records:
  d=json.loads(p.read_text(encoding='utf-8'));npz=p.parent/d['array_file'];actual=hashlib.sha256(npz.read_bytes()).hexdigest()
  if actual!=d['arrays_sha256']:errors.append('array_hash:'+batch+'/'+p.name)
  else:verified+=1
  cells.add(d['cell_id'])
  if d.get('technical_pass') is not True:errors.append('technical_pass:'+batch+'/'+p.name)
 if len(records)!=expected:errors.append('count:'+batch)
 report[batch]={'record_count':len(records),'unique_cells':len(cells),'array_hashes_verified':verified}
root=Path('D:/CEDAR/arr_c1_signed_20260928/explore_outputs/closure_20261003/evidence/runs')
for batch,expected in [('smoke',52),('formal',3840)]:
 records=[p for p in (root/batch).glob('*.json') if re.fullmatch('[0-9a-f]{24}',p.stem)];cells=set();local=0
 for p in records:
  d=json.loads(p.read_text(encoding='utf-8'));cells.add(d['cell_id']);sha=d['arrays_sha256']
  if sha in hashes:local+=1
  else:missing.append({'experiment':'signed-dose-C1','batch':batch,'cell_id':d['cell_id'],'expected_sha256':sha,'remote_relative_path':'c1_signed_20260928/runs/'+batch+'/'+d['array_file'],'local_status':'not_present_in_selected_local_files'})
 if len(records)!=expected:errors.append('count:C1/'+batch)
 report['C1_'+batch]={'record_count':len(records),'unique_cells':len(cells),'raw_arrays_present_locally':local,'raw_arrays_remote_only':len(records)-local}
root=Path('D:/CEDAR/arr_revision_20260927/explore_outputs/batch_evidence_20260928/runs')
for batch,expected in [('smoke',12),('formal',384)]:
 records=json.loads((root/batch/'results.json').read_text(encoding='utf-8'));local=0
 for index,d in enumerate(records):
  sha=d['arrays_sha256']
  if sha in hashes:
   local+=1;match=hashes[sha];available.append({'experiment':'S1-P1','batch':batch,'cell_id':d['cell_id'],'sha256':sha,'archive_package':match['package'],'archive_path':match['path']})
  else:
   conditions=['uncapped','full_p25','uncapped_repeat'] if batch=='smoke' else ['uncapped','full_p25']
   condition=conditions[index%len(conditions)];assert d['condition']==('uncapped' if condition=='uncapped_repeat' else condition) or d['condition']==condition
   name=f'{index//len(conditions):04d}_{condition}.npz'
   missing.append({'experiment':'S1-P1','batch':batch,'cell_id':d['cell_id'],'expected_sha256':sha,'remote_relative_path':'runs/'+batch+'/'+name,'local_status':'not_present_in_selected_local_files'})
 if len(records)!=expected:errors.append('count:S1P1/'+batch)
 report['S1P1_'+batch]={'aggregate_records':len(records),'raw_arrays_present_locally':local,'raw_arrays_remote_only':len(records)-local}
root=Path('D:/CEDAR/CEDAR_SUPPORTING_DATA');matched=0;mismatches=[]
for d in csv.DictReader((root/'MANIFEST.csv').open(encoding='utf-8')):
 p=root/d['path']
 if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==d['sha256']:matched+=1
 else:mismatches.append(d['path'])
report['supporting_package_original_manifest']={'matched':matched,'mismatches':mismatches}
report['scope']='Local record counts and hash availability verified on 2026-10-25. Historical remote audits retained; no remote recheck, model run, or scientific effect recomputation performed.'
report['errors']=errors
(out/'local_record_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (out/'remote_only_array_inventory.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(missing[0]));w.writeheader();w.writerows(missing)
with (out/'available_array_aliases.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(available[0]));w.writeheader();w.writerows(available)
print(json.dumps(report,ensure_ascii=False,indent=2));assert not errors
