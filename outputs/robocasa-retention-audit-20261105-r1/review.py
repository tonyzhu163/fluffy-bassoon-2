"""Refine read-only candidates to explicit closed-run decisions and cold proofs."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib,importlib.util,json,os,stat,time
from pathlib import Path
B=Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-retention-audit-20261105-r1')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def save(p,j):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(j,indent=2)+'\n');t.replace(p)
spec=importlib.util.spec_from_file_location('audit',B/'audit.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
initial=read(B/'AUDIT.json');save(B/'AUDIT-INITIAL.json',initial)
closed_evaluations={'robocasa-flow-sde-checkpoint-selection-20261104-r1','robocasa-flow-sde-endpoint-test512-20261104-r1','robocasa-endpoint512-acceleration-20261104-r1'}
visual_screens={'robocasa-interface-screen-20261102-r1','robocasa-interface-screen-20261102-r2','robocasa-interface-screen-20261102-r3','robocasa-rollout-capacity-1632-20261103-r1','robocasa-rollout-capacity-3248-20261103-r1','robocasa-rollout-batch-capacity-20261102-r3'}
proofs=[]; inventories={}; extracts=[]; samples=[]
for e in initial['sources']:
 if e['status']!='audited_hot_source':continue
 assert not e['errors'] and not e['protected']
 inv=read(e['inventory']);inventories[e['source']]=inv;root=Path(e['source']);name=root.name
 for row in inv['rows']:
  if not stat.S_ISREG(row[1]):continue
  p=root/row[0]
  if p.name=='extracted-full_weights.pt':
   ready=p.parent/'READY.json'
   if ready.exists():
    meta=read(ready);cp=meta['checkpoint'];storage=meta['storage'];assert meta['private_copy']
   else:
    assert row[0]=='staged-models/0/extracted-full_weights.pt'
    cp=next(r for r in read(root/'PREPARED-R1.json')['checkpoints'] if r['staged_path']==str(p));storage=cp['storage'];assert cp['private_copy']
   assert storage['storage']=='verified_cold_member'
   marker=Path(cp['marker']);assert sha(marker)==cp['marker_sha256']
   coldpath=marker.parent/'COLD-STORAGE.json';cold=read(coldpath)
   assert sha(coldpath)==storage['cold_receipt_sha256']
   assert cold['tar_compare'] and cold['full_BOS_readback']
   assert cold['sha256']==storage['original_archive_sha256'] and cold['archive']==storage['archive']
   archive=Path(cold['archive']);assert archive.stat().st_size==cold['archive_bytes']
   assert cold['source_inventory']['actor/model_state_dict/full_weights.pt'][0]==row[2]==cp['bytes']
   assert storage['extracted_model_sha256']==cp['sha256']
   extracts.append((p,cp,storage,coldpath,cold['archive_bytes']))

def verify_extract(item):
 p,cp,storage,coldpath,archive_bytes=item
 assert sha(p)==cp['sha256']
 return dict(path=str(p),sha256=cp['sha256'],bytes=cp['bytes'],recovery_archive=storage['archive'],archive_bytes=archive_bytes,
             archive_sha256=storage['original_archive_sha256'],cold_receipt=str(coldpath),cold_receipt_sha256=storage['cold_receipt_sha256'],
             member=storage['extracted_member'],prior_full_archive_readback=True,current_private_copy_full_hash=True)
with ThreadPoolExecutor(max_workers=4) as pool:proofs=list(pool.map(verify_extract,extracts))
verified={r['path'] for r in proofs}; assert len(verified)==11
summary=[];types=Counter();retained_code=[]
for e in initial['sources']:
 if e['status']!='audited_hot_source':summary.append(e);continue
 root=Path(e['source']);inv=inventories[e['source']];rows=inv['rows'];name=root.name
 # Preserve two complete camera triplets and two initial snapshots in each historical screen directory.
 representatives=set();byparent=defaultdict(list);initials=defaultdict(list)
 for r in rows:
  if stat.S_ISREG(r[1]) and Path(r[0]).suffix.lower()=='.mp4':byparent[str(Path(r[0]).parent)].append(r[0])
  if stat.S_ISREG(r[1]) and Path(r[0]).name.endswith('-initial.pt'):initials[str(Path(r[0]).parent)].append(r[0])
 for parent,values in byparent.items():representatives.update(sorted(values)[:6])
 sampled_initials={v for values in initials.values() for v in sorted(values)[:2]}
 for r in rows:
  p=Path(r[0]);parts=p.parts;reason=audit.bulk_reason(p,r[2]) if stat.S_ISREG(r[1]) else None;delete=False
  if reason=='regenerable run-local compiler cache':delete=True
  elif str(root/p) in verified:delete=True
  elif name in visual_screens and p.suffix.lower()=='.mp4' and r[0] not in representatives:delete=True
  elif name in visual_screens and p.name.endswith('-initial.pt') and 'videos' in parts:
   delete=r[0] not in sampled_initials
  elif name in closed_evaluations and 'environment' in parts and not any(v in ('bank-00','bank-000') for v in parts):
   delete=(p.name.endswith('-initial.pt') or (p.name.endswith('.pt.gz') and 'chunk' in p.name))
  if delete:
   assert reason, str(root/p);r[7:]=['bulk',reason];types[reason]+=r[4]*512
  else:
   r[7:]=['keep',None]
   if stat.S_ISREG(r[1]) and p.suffix in ('.py','.sh') and r[2]<2<<20:
    retained_code.append(dict(path=str(root/p),sha256=sha(root/p)))
   if r[0] in representatives and name in visual_screens:samples.append(str(root/p))
 save(e['inventory'],inv)
 bulk=[r for r in rows if r[7]=='bulk'];keep=[r for r in rows if r[7]=='keep' and stat.S_ISREG(r[1])]
 e.update(inventory_sha256=sha(e['inventory']),bulk_files=len(bulk),bulk_allocated_bytes=sum(r[4]*512 for r in bulk),bulk_logical_bytes=sum(r[2] for r in bulk),retained_files=len(keep),retained_allocated_bytes=sum(r[4]*512 for r in keep),recoverable_one_off_files=0,reviewed=True)
 summary.append(e)
result=dict(initial,sources=summary,bulk_files=sum(e.get('bulk_files',0) for e in summary),bulk_allocated_bytes=sum(e.get('bulk_allocated_bytes',0) for e in summary),retained_allocated_bytes=sum(e.get('retained_allocated_bytes',0) for e in summary),reviewed=True)
save(B/'AUDIT.json',result);save(B/'RECOVERY-PROOFS.json',dict(complete=True,files=proofs));save(B/'RETAINED-CODE.json',dict(files=retained_code));save(B/'REPRESENTATIVE-VIDEOS.json',dict(paths=samples))
review=dict(bulk_prune_review_complete=True,audit_sha256=sha(B/'AUDIT.json'),plan_sha256=sha(B/'PLAN.json'),classifier_sha256=sha(B/'audit.py'),
 reasons_allocated_bytes=dict(types),private_model_copies_full_hash_verified=len(proofs),retained_code_files=len(retained_code),representative_videos=len(samples),
 decisions='Only run-local compiler caches, 11 hash-verified recoverable extracted models, redundant interface/capacity screen videos/initial snapshots, and closed evaluation environment chunks/initial snapshots outside first banks. Preserve all code/config/log/result, canonical reset snapshots, sampler records, first raw banks, all Flora training/calibration videos, all model/recovery tensors and numerical diagnostic tensors.',checked_shanghai=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
save(B/'REVIEW.json',review);print(json.dumps(review,indent=2),flush=True)
