"""Reconcile every reviewed decision with receipts and retained evidence."""
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import hashlib,importlib.util,json,os,stat,time
from pathlib import Path
B=Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-retention-audit-20261105-r1')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,j):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(j,indent=2)+'\n');t.replace(p)
spec=importlib.util.spec_from_file_location('prune',B/'prune.py');prune=importlib.util.module_from_spec(spec);spec.loader.exec_module(prune)
p=read(B/'PRUNE-SUMMARY.json');a=read(B/'AUDIT.json');r=read(B/'REVIEW.json');assert p['finished'] and not p['preserved'] and not p['skipped_sources']
assert sha(B/'AUDIT.json')==r['audit_sha256'] and sha(B/'PLAN.json')==r['plan_sha256'] and sha(B/'audit.py')==r['classifier_sha256']
assert sha(p['receipt_file'])==p['receipt_sha256'];expected={};retained=[];absent=[];sources=[]
for e in a['sources']:
 if e['status']!='audited_hot_source':continue
 assert sha(e['inventory'])==e['inventory_sha256'];inv=read(e['inventory']);groups=defaultdict(list)
 for row in inv['rows']:
  if row[7]=='bulk':groups[str(Path(row[0]).parent)].append(row);absent.append(str(Path(e['source'])/row[0]))
  elif not stat.S_ISDIR(row[1]):retained.append((e['source'],row))
 for parent,rows in groups.items():expected[(e['source'],parent)]=(len(rows),sum(v[2] for v in rows),sum(v[4]*512 for v in rows))
 sources.append(dict(source=e['source'],deleted_files=e['bulk_files'],deleted_allocated_payload_bytes=e['bulk_allocated_bytes'],retained_regular_files=e['retained_files'],retained_regular_allocated_bytes=e['retained_allocated_bytes']))
dependency_checks=[]
for path in read(B/'DEPENDENCY-REVIEW.json')['referenced_selected_input_paths']:
 matched=[(root,row) for root,row in retained if str(Path(root)/row[0])==path]
 assert path not in absent, ('referenced input selected for deletion',path)
 if matched:
  assert Path(path).exists();dependency_checks.append(dict(path=path,status='retained inventoried dependency'))
 else:
  dependency_checks.append(dict(path=path,status='not an inventoried artifact; source/pinned metadata reference preserved'))
seen=set();counts=[0,0,0]
for line in Path(p['receipt_file']).read_text().splitlines():
 x=json.loads(line);key=(x['source'],x['parent']);assert key not in seen and key in expected and not x['preserved'];seen.add(key)
 values=(x['files'],x['logical_bytes'],x['allocated_payload_bytes']);assert values==expected[key]
 counts=[v+w for v,w in zip(counts,values)]
assert seen==set(expected) and len(seen)==p['directory_groups_completed']==p['directory_groups']
assert counts==[p['files_deleted'],p['deleted_logical_bytes'],p['deleted_allocated_payload_bytes']]
assert counts[0]==a['bulk_files'] and counts[2]==a['bulk_allocated_bytes']
def check_absent(path):
 try:Path(path).lstat()
 except FileNotFoundError:return 1
 raise AssertionError(('reviewed deletion still exists',path))
def check_kept(item):
 root,row=item;assert prune.identity_matches((Path(root)/row[0]).lstat(),row),(root,row[0]);return 1
with ThreadPoolExecutor(max_workers=24) as pool:
 removed_checked=sum(pool.map(check_absent,absent));kept_checked=sum(pool.map(check_kept,retained))
code=read(B/'RETAINED-CODE.json')['files']
for x in code:assert sha(x['path'])==x['sha256']
proofs=read(B/'RECOVERY-PROOFS.json')['files']
for x in proofs:
 assert sha(x['cold_receipt'])==x['cold_receipt_sha256'];assert Path(x['recovery_archive']).stat().st_size==x['archive_bytes'];assert not Path(x['path']).exists()
vids=read(B/'REPRESENTATIVE-VIDEOS.json')['paths']
for v in vids:assert Path(v).is_file()
fs=os.statvfs('/mowenjie');free=fs.f_bavail*fs.f_frsize
manifest=dict(complete=True,checked_shanghai=time.strftime('%Y-%m-%dT%H:%M:%S%z'),selected_paths=len(a['sources']),existing_roots=len(sources),absent_paths=sum(e['status']=='absent_no_scan' for e in a['sources']),
 retained_dependency_checks=dependency_checks,deleted_files=counts[0],deleted_logical_bytes=counts[1],deleted_allocated_payload_bytes=counts[2],deletion_groups_verified=len(seen),deleted_paths_independently_absent=removed_checked,retained_records_independently_identity_checked=kept_checked,
 retained_source_code_sha256_checked=len(code),retained_representative_videos=len(vids),recoverable_private_model_copies= len(proofs),cold_recovery_receipts_reconciled=len(proofs),reasons_allocated_bytes=r['reasons_allocated_bytes'],
 filesystem_free_before=p['filesystem_free_before'],filesystem_free_after=free,filesystem_free_delta=free-p['filesystem_free_before'],concurrent_writes_mean_free_delta_is_not_attributable_bytes=True,
 retained_regular_files=sum(e['retained_regular_files'] for e in sources),retained_regular_allocated_bytes=sum(e['retained_regular_allocated_bytes'] for e in sources),no_active_job_folder_scan=True,GPU_jobs_launched=0,errors=0,skipped_sources=0,
 scope='Historical closed roots only. Protected live/new source/evaluation/held roots and shared teacher/calibration data were not refreshed or mutated.',sources=sources)
save(B/'MANIFEST.json',manifest)
print(json.dumps({k:v for k,v in manifest.items() if k!='sources'},indent=2),flush=True)
