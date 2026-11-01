"""Bounded R2 savedmetadata/hash check; no arrays,models or oldforest."""
from pathlib import Path
import hashlib,json,time
root=Path('/mowenjie/fluffy-bassoon-2/outputs/libero40-teacher-assessment-20261101-r2');data=root/'data';old=Path('/mowenjie/fluffy-bassoon-2/outputs/libero40-teacher-assessment-20261101-r1')
def read(p):return json.loads(Path(p).read_text())
def ref(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),sha256=h,bytes=p.stat().st_size)
clock=read(old/'clock/worker-time.json');r=dict(format='all40_assessment_R2_first_progress_v1',checked_unix=time.time(),job_id='job-bc2ooyztqsfh',execution_source='9d29a805cae2c129ffc088af4eed81b3a38d2769',original_source='7ef8a62058773201e85c8b2ec26bf3847ff15026',clock=clock,probe=read(root/'admission/gpu-probe.json'),probe_ref=ref(root/'admission/gpu-probe.json'),ticket_ref=ref(root/'admission/ticket.json'),old_recovery_actual=ref(old/'data/assessment-recovery.pkl'),old_recovery_expected='c96dc68b18f95f136550eb824fa5bdf7fcad23950257500f446d28e3a1b94b8c',remaining_planned_roles=158,failed_ordinal_replayed=False,all40_bulk_payload_rehash_repeated=False,limits='Only current JSON plus firstnew162 receipt/trace/RNG/tworeset-evidence bytehashes. No pickle/tensor/array/model deserialize, oldjournal/input forest or job control.')
for name in ['migration-receipt.json','ABORTED-161-RESET-GUARD.json','progress.json','results-summary.json','motor-binding.json','motor-final.json']:
 p=data/name
 if p.exists():r[name]=read(p);r[name+'_ref']=ref(p)
p=data/'episode-162/receipt.json'
if p.exists():
 receipt=read(p);r['first_receipt']=receipt;r['first_receipt_ref']=ref(p);r['first_receipt_payloads']=[]
 for key in ['trace','RNG','input_feature_audit','reset_comparison_audit']:
  actual=ref(receipt[key]['path']);actual['matches']=actual['sha256']==receipt[key]['sha256'];r['first_receipt_payloads'].append(dict(field=key,**actual))
if (data/'progress.json').exists():
 value=r['progress.json']['recovery'];r['current_recovery_actual']=ref(value['path']);r['current_recovery_matches_last_closed_progress']=r['current_recovery_actual']['sha256']==value['sha256']
if (root/'exit-code.txt').exists():r['exit_code']=(root/'exit-code.txt').read_text().strip()
if (data/'index.json').exists():
 value=read(data/'index.json');r['index']={k:v for k,v in value.items() if k!='receipts'};r['index_receipts']=len(value['receipts']);r['index_ref']=ref(data/'index.json')
console=root/'console.log'
if console.exists():
 with console.open('rb') as f:f.seek(max(0,console.stat().st_size-3000));r['console_tail']=f.read().decode(errors='replace')
print(json.dumps(r,indent=2))
