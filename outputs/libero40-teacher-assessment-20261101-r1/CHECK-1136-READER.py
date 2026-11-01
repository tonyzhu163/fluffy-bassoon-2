"""One due saved-JSON/hash check. No input forest, models or array decoding."""
import hashlib,json,math,shutil,subprocess,time
from pathlib import Path
root=Path('/mowenjie/fluffy-bassoon-2/outputs/libero40-teacher-assessment-20261101-r1');data=root/'data'
def read(p):return json.loads(Path(p).read_text())
def ref(p):
 p=Path(p)
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),sha256=digest,bytes=p.stat().st_size)
clock=read(root/'clock/worker-time.json');attempt=clock['attempts'][-1];checked=time.time()
progress=read(data/'progress.json');cursor=progress['cursor'];summary=read(data/'results-summary.json')
rows=[];start=(cursor//4-1)*4
if start>=0:
 for i in range(start,start+4):
  p=data/f'episode-{i:03d}/receipt.json';rows.append(dict(ordinal=i,ref=ref(p),receipt=read(p)))
reference=rows[0]['receipt'] if rows else None
identity_fields=['initial_state','scene_seed','episode_random_seed','initial_observation_sha256','initial_state_vector_sha256','source_actor','source_updates','frozen_learning_state']
paired=reference is not None and all(all(x['receipt'][k]==reference[k] for k in identity_fields) for x in rows)
latest_path=data/f'episode-{cursor-1:03d}/receipt.json';latest=read(latest_path)
# Hash only latest retained physical trace and RNG, never deserialize.
payload_refs=[]
for key in ['trace','RNG']:
 actual=ref(latest[key]['path']);actual['matches']=actual['sha256']==latest[key]['sha256'];payload_refs.append(actual)
# The producer may already have atomically replaced the current recovery for
# the next active episode; do not relabel that normal race as corruption.
actual_recovery=ref(progress['recovery']['path']);recovery_matches=actual_recovery['sha256']==progress['recovery']['sha256']
settled=sum(x['charged_seconds'] for x in clock['attempts'] if x['status']=='finished')
active_elapsed=checked-attempt['started_unix'] if attempt['status']=='started' else None
storage=None
try:
 result=subprocess.run(['du','-sb',str(root)],capture_output=True,text=True,timeout=8,check=True)
 storage={'actual_phase_output_bytes':int(result.stdout.split()[0]),'scope':'One output-size metadata walk,8second limit; no input forest/hash traversal'}
except (subprocess.TimeoutExpired,subprocess.CalledProcessError,ValueError) as error:storage={'actual_phase_output_bytes':None,'limit':type(error).__name__}
console=Path(attempt['path'])/'console.log'
with console.open('rb') as f:f.seek(max(0,console.stat().st_size-8192));tail=f.read().decode(errors='replace')
r=dict(format='all40_assessment_due_light_v1',job_id='job-bmolej07bhsg',source_commit='7ef8a62058773201e85c8b2ec26bf3847ff15026',checked_unix=checked,current_progress=progress,summary=summary,clock=clock,settled_GPU_seconds=settled,active_elapsed_seconds=active_elapsed,conservative_observed_GPU_seconds=settled+math.ceil(active_elapsed)+60 if active_elapsed is not None else settled,maximum_GPU_seconds=10800,maximum_scheduled_actions=131200,maximum_scheduled_episodes=320,maximum_output_bytes=32<<30,storage=storage,free_PFS_bytes=shutil.disk_usage(root).free,newest_complete_case_four_rows=rows,newest_four_role_initials_and_frozen_state_equal=paired,queried_source_optimizer_updates_zero=all(x['receipt']['optimizer_updates']==0 for x in rows),latest_receipt=latest,latest_receipt_ref=ref(latest_path),newest_trace_RNG_bytes=payload_refs,current_recovery_actual=actual_recovery,current_recovery_matches_last_closed_progress=recovery_matches,recovery_concurrency_limit='One current rolling recovery hash; it may already encode the next active episode. No recovery deserialization.',qualification='First plannednative passeddelta0/feature0,0extraactions; immutable0797ed0f proof carried,not rehashed',all40_actual_binding='Immutablee7f860f6... proof carried from0797ed0f,not rehashed',console_tail=tail,actual_exit_code=(root/'exit-code.txt').read_text().strip() if (root/'exit-code.txt').exists() else None,next_owned_check='2026-11-01 11:56 Asia/Shanghai',verification_limits='Current producer counters plus atmost4newcompletecase JSON receipts, newest trace/RNG bytehashes and rolling recovery bytes. No model/tensor/array/checkpoint deserialization, old40actor payload rehash, journal forest or job control.')
if (data/'index.json').exists():
 index=read(data/'index.json');r['terminal_index_ref']=ref(data/'index.json');r['terminal_index']={k:v for k,v in index.items() if k!='receipts'};r['terminal_receipt_count']=len(index['receipts'])
if (data/'motor-final.json').exists():r['motor_final']=read(data/'motor-final.json');r['motor_final_ref']=ref(data/'motor-final.json')
print(json.dumps(r,indent=2))
