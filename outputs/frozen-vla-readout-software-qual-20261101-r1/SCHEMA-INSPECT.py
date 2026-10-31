"""Three bound cache rows, allocated zero-GPU CPU only; no model or rewrites."""
import argparse,hashlib,json,socket,sys,time
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--inputs-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
assert sys.platform=='linux' and socket.gethostname().startswith('job-')
assert torch.cuda.device_count()==0 and not torch.cuda.is_initialized()
def guard():
 if time.time()>=a.deadline:raise TimeoutError('Original readoutCPU/schema deadline')
def sha(path):
 guard()
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(ref):
 assert sha(ref['path'])==ref['sha256'];return json.loads(Path(ref['path']).read_text())
assert sha(a.inputs)==a.inputs_sha256
inputs=json.loads(a.inputs.read_text());request=read(inputs['request']);bank=read(request['bank']);old=read(request['cache_receipts'][0]);new=read(inputs['new04_receipt'])
assert sha(inputs['sealer_source']['path'])==inputs['sealer_source']['sha256']
assert len(old['records'])==77 and len(new['records'])==13
selected=[('first_paid11',old['records'][0]),('first_new04',new['records'][0]),('last_new04',new['records'][-1])]
def meta(t):
 t=t.detach().cpu().contiguous();raw=t.view(torch.uint8).numpy().tobytes()
 return dict(shape=list(t.shape),dtype=str(t.dtype),sha256=hashlib.sha256(str(t.dtype).encode()+str(tuple(t.shape)).encode()+raw).hexdigest(),elements=t.numel())
rows=[]
for label,row in selected:
 guard();assert sha(row['path'])==row['sha256'];saved=torch.load(row['path'],map_location='cpu',weights_only=False)
 context=row['context'];entry=next(e for e in bank['entries'] if e['id']==context['teacher_id'])
 wanted=dict(teacher_id=entry['id'],split=entry['split'],producer_result=entry['producer_result'],producer_commit=entry['producer_commit'],profile=entry['profile'],record_id=entry['record']['spec']['id'],episode_manifest_ref=entry['record']['episode_manifest_ref'],original_source=entry['original_source'])
 submitted=saved['submitted_plan_mask'];executed=saved['native_executed_mask'];data=saved['data']
 checks=dict(format_correct=saved['format']=='robotwin_captured_plan_cache_v1',context_exact=saved['context']==wanted,submitted_shape_32=submitted.shape==(32,),executed_shape_32=executed.shape==(32,),submitted_bool=submitted.dtype==torch.bool,executed_bool=executed.dtype==torch.bool,submitted_prefix=torch.equal(submitted,torch.arange(32)<int(submitted.sum())),executed_prefix=torch.equal(executed,torch.arange(32)<int(executed.sum())),executed_subset_submitted=not bool((executed & ~submitted).any()),executed_count_matches_receipt=int(executed.sum())==row['native_actual_actions'],submitted_count_matches_receipt=int(submitted.sum())==row['valid_actions'],data_valid_matches_submitted=torch.equal(data['valid'],submitted[None]),teacher_matches_full_proposal=torch.equal(data['teacher14'],saved['full_known_proposed_plan14'][None]),data_keys_exact=set(data)=={'state','features','cache','mask','target32','teacher14','base14','valid'},features_shape_1x2062=data['features'].shape==(1,2062),state_shape_1x14=data['state'].shape==(1,14),base_shape_1x32x14=data['base14'].shape==(1,32,14),target_shape_1x32x32=data['target32'].shape==(1,32,32))
 canonical=read(saved['canonical_chunk'])
 item=dict(selection=label,teacher_id=context['teacher_id'],anchor=row['anchor'],payload_ref=dict(path=row['path'],sha256=row['sha256']),saved_context=saved['context'],expected_context=wanted,predicates=checks,failed_predicates=[k for k,v in checks.items() if not v],fields={k:meta(data[k]) for k in ('state','features','base14','target32','teacher14','valid')},full_known_proposed_plan14=meta(saved['full_known_proposed_plan14']),submitted_mask=meta(submitted),executed_mask=meta(executed),submitted_count=int(submitted.sum()),executed_count=int(executed.sum()),receipt_valid_actions=row['valid_actions'],receipt_native_actual_actions=row['native_actual_actions'],canonical_native_actual_actions=canonical['native_result']['executed_primitive_steps'],canonical_chunk=saved['canonical_chunk'])
 rows.append(item);guard()
result=dict(format='three_row_cache_schema_inspection_v1',hostname=socket.gethostname(),inputs_sha256=a.inputs_sha256,sealer_source=inputs['sealer_source'],request=inputs['request'],new04_receipt=inputs['new04_receipt'],rows=rows,maximum_rows=3,model_or_weights_loaded=False,simulator_or_actions=0,cache_arrays_transferred=False,payloads_rewritten=False,sealer_guard_relaxed=False,seal_PASS_claim=False,scope='Only three specified savedCPU cache rows; false predicates are observations, not full-cache admission')
a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
