"""Bounded saved-JSON/hash read; no models, arrays or journal traversal."""
from pathlib import Path
import hashlib,json,time
root=Path('/mowenjie/fluffy-bassoon-2/outputs/libero40-teacher-assessment-20261101-r1')
def ref(p):
 p=Path(p)
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':digest,'bytes':p.stat().st_size}
def read(p):return json.loads(Path(p).read_text())
clock=read(root/'clock/worker-time.json');attempt=Path(clock['attempts'][-1]['path']);data=root/'data'
r={'format':'all40_teacher_assessment_first_progress_read_v1','checked_unix':time.time(),'job_id':'job-bmolej07bhsg','source':'7ef8a62058773201e85c8b2ec26bf3847ff15026','clock':clock,'probe_ref':ref(attempt/'gpu-probe.json'),'ticket_ref':ref(attempt/'ticket.json'),'status':'startup_or_active','limits':'Only current JSON, original binding/provenance and first episode JSON/recovery bytes; no payload/checkpoint deserialization, arrays, journal scan or model/test/physics calls.'}
if (data/'loaded-actor-bindings.json').exists():
 b=read(data/'loaded-actor-bindings.json');r['actor_binding_ref']=ref(data/'loaded-actor-bindings.json');r['actor_binding_count']=len(b['actors']);r['actor_task_keys']=[x['task_key'] for x in b['actors']];r['all40_source_motor_rng_present']=all(x['original_source_motor_rng_present'] for x in b['actors']);r['actor_updates']={x['task_key']:x['updates'] for x in b['actors']};r['no_initial_actor_slots_repurposed']=b['initial_roles_omitted']==160
if (data/'source-provenance.json').exists():r['source_provenance']=read(data/'source-provenance.json');r['source_provenance_ref']=ref(data/'source-provenance.json')
if (data/'progress.json').exists():
 r['progress']=read(data/'progress.json');recovery=r['progress']['recovery'];r['full_recovery_actual']=ref(recovery['path']);r['full_recovery_matches']=r['full_recovery_actual']['sha256']==recovery['sha256']
for n in ['qualification.json','receipt.json']:
 p=data/'episode-000'/n
 if p.exists():r['first_'+n.removesuffix('.json')]=read(p);r['first_'+n.removesuffix('.json')+'_ref']=ref(p)
if (data/'results-summary.json').exists():r['summary']=read(data/'results-summary.json')
if (root/'exit-code.txt').exists():r['exit_code']=(root/'exit-code.txt').read_text().strip();r['status']='terminal'
if (data/'index.json').exists():r['index_ref']=ref(data/'index.json');r['index']=read(data/'index.json')
if (data/'motor-final.json').exists():r['motor_final']=read(data/'motor-final.json');r['motor_final_ref']=ref(data/'motor-final.json')
print(json.dumps(r,indent=2))
