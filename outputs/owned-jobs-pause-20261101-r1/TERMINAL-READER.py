import csv,hashlib,json,math,os,shutil,time
from pathlib import Path
base=Path('/mowenjie/fluffy-bassoon-2/outputs');out=base/'owned-jobs-pause-20261101-r1'
def ref(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest())
def cp(src,dest):
 dest=out/dest;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);return ref(dest)
def small(p):return json.loads(Path(p).read_text())
r=dict(format='owned_jobs_pause_terminal_v1',checked_unix=time.time(),authority_message='01a19572-3598-7cb1-a201-a619b5f12660',new_launches_blocked=True,new_fixed_bank_cpu_allocated=False)
root=base/'dsrl-pi05-headroom-20261101-s0-r1';case=root/'experiment/case-learn-44';state=small(case/'state.json');index=small(case/'index.json');model=state['model']
cp(case/'state.json','primary/final-state.json');cp(case/'index.json','primary/final-index.json');clock=small(root/'worker-time.json');cp(root/'worker-time.json','primary/worker-time-after-stop.json')
final=out/'primary/final-closed-recovery.pkl'
if not final.exists():os.link(case/model['path'],final)
actual=ref(final)
assert actual['sha256']==model['sha256']
rows=[];names=sorted((case/'physical-journal').glob('*.json'))
next_ep=f"training-{state['training']['episodes']:06d}"
for p in reversed(names[-44:]):
 j=small(p)
 if j.get('episode_id')!=next_ep:break
 rows.append(dict(record=ref(p),metadata=j))
suffix=sum(j['metadata'].get('actual_steps',j['metadata'].get('maximum_steps',0)) if j['metadata'].get('returned') else j['metadata'].get('maximum_steps',0) for j in rows)
last=state['training']['receipts'][-1]
closed_delta=sum(state['training'][k]-index['training'][k] for k in ('primitive_steps','settling_steps'))
a=clock['attempts'][-1];ended=1793509143
rt=sorted((root/'experiment/runtime-attempts').iterdir())[-1]
motor={}
for n in ('motor-before.json','motor-after.json','motor-final.json'):
 p=rt/n
 if p.exists():motor[n]=dict(ref=ref(p),data=small(p));cp(p,'primary/'+n)
r['primary']=dict(job_id='job-2qn8dpw3yvlb',scheduler_status='ManualTermination',scheduler_ended='2026-11-01 12:59:03 Asia/Shanghai',scheduler_duration_seconds=12703,
 execution_source='d2369fb94a024315c7ef41d7021bebe171c8b515',scientific_source='b6655659d8bff2c6b5d14eaf05d63f4647176d4f',
 stable_closed_recovery=actual,model=model,full_checkpoint_matches=True,phase=state['phase'],next_threshold=state['next_threshold'],active_panel=state['active_panel'],
 training={k:state['training'][k] for k in ('episodes','chunks','primitive_steps','settling_steps')},train_successes=sum(bool(x['success']) for x in state['training']['receipts']),train_failures=sum(not x['success'] for x in state['training']['receipts']),
 state=ref(out/'primary/final-state.json'),index=ref(out/'primary/final-index.json'),last_losses_finite=all(math.isfinite(v) for v in last['update_metrics'].values() if isinstance(v,(float,int))),
 last_closed_training_receipt=last,index_cursor_lagging_state=index['training']['episodes']!=state['training']['episodes'],completed_panels=[dict(kind=p['kind'],threshold=p['threshold_updates'],actual_updates=p['actual_optimizer_updates'],queries=len(p['receipts']),successes=sum(bool(x['success']) for x in p['receipts'])) for p in state['panels']],
 closed_index_whole_phase_actions=index['total_action_charge'],closed_index_costs=index['costs'],pending_episode_id=next_ep,interrupted_suffix_records=rows,interrupted_suffix_conservative_actions=suffix,
 closed_state_action_delta_after_index=closed_delta,closed_state_whole_phase_actions=index['total_action_charge']+closed_delta,conservative_total_including_suffix=index['total_action_charge']+closed_delta+suffix,index_output_bytes=index['output_bytes'],clock_attempt=a,
 producer_exit_file_emitted=(Path(a['path'])/'exit-code.txt').exists(),root_exit_file_is_previous_attempt=True,root_exit_file_content=(root/'exit-code.txt').read_text(),
 canonical_clock_unsettled=a['status']=='started',owner_conservative_current_attempt_gpu_seconds=math.ceil(ended-a['started_unix'])+a['scheduler_allowance_seconds'],
 prior_settled_gpu_seconds=sum(x['charged_seconds'] for x in clock['attempts'] if x['status']=='finished'),motor_evidence=motor,
 resume_limit='Full closed learner/RNG checkpoint preserved. Interrupted physical suffix is immutable and charged, not a complete episode or exact mid-physics state; no automatic resume/replay authorized.',
 full_checkpoint_fields_source='OfficialSAC.save: agent serialized actor/critic/target/temp including TrainState optimizer; agent_rng,replay,updates,sampler,NumPy/Python RNG,contract,boundary,training progress+motor_rng. Source schema proof; no checkpoint deserialization in this closure.',
 cost_verification_limit='Closed index producer counters plus at most44 newest same-episode journal metadata; earlier forest not independently replayed. Storage is last producer index snapshot, not a new full-directory measurement.')
child=base/'libero40-teacher-assessment-20261101-r2';data=child/'data';idx=small(data/'index.json');clockroot=base/'libero40-teacher-assessment-20261101-r1/clock';aclock=small(clockroot/'worker-time.json')
files={}
for n in ('index.json','progress.json','results-summary.json','case-results.csv','motor-final.json','assessment-recovery.pkl'):
 files[n]=cp(data/n,'assessment/final-'+n)
files['worker-time.json']=cp(clockroot/'worker-time.json','assessment/worker-time-after-completion.json');files['exit-code.txt']=cp(child/'exit-code.txt','assessment/exit-code.txt')
counts={};cs=list(csv.DictReader((data/'case-results.csv').open()))
for x in cs:counts[x['status']]=counts.get(x['status'],0)+1
r['assessment']=dict(job_id='job-bc2ooyztqsfh',scheduler_status='Succeeded',scheduler_ended='2026-11-01 12:57:11 Asia/Shanghai',scheduler_duration_seconds=1251,stopped_by_owner=False,exit_code=int((child/'exit-code.txt').read_text()),
 execution_source='9d29a805cae2c129ffc088af4eed81b3a38d2769',original_clock_source='7ef8a62058773201e85c8b2ec26bf3847ff15026',completed=idx['completed'],cursor=idx['cursor'],valid_completed_episodes=idx['valid_completed_episodes'],aborted=idx['aborted'],all_planned_roles_valid=idx['all_planned_roles_valid'],costs=idx['costs'],
 recovery=files['assessment-recovery.pkl'],recovery_matches=files['assessment-recovery.pkl']['sha256']==idx['recovery']['sha256'],files=files,results_summary=small(data/'results-summary.json'),all_csv_status_counts=counts,csv_rows=len(cs),
 motor_final=small(data/'motor-final.json'),optimizer_updates=idx['optimizer_updates'],source_pretraining_admitted=False,
 clock_attempts=aclock['attempts'],total_settled_gpu_seconds=sum(x['charged_seconds'] for x in aclock['attempts']),
 verification_limit='Saved final index/CSV summaries, final motor guard, clock and recovery bytes. Individual319 receipt/RPC trees and payload tensors not rescanned/deserialized; qualification/bindings were verified on prior actual allocation receipts.')
(out/'TERMINAL-SNAPSHOT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'primary':{k:r['primary'][k] for k in ['model','training','closed_index_whole_phase_actions','closed_state_action_delta_after_index','closed_state_whole_phase_actions','interrupted_suffix_conservative_actions','conservative_total_including_suffix','canonical_clock_unsettled','owner_conservative_current_attempt_gpu_seconds','prior_settled_gpu_seconds','producer_exit_file_emitted']},'assessment':{k:r['assessment'][k] for k in ['completed','valid_completed_episodes','costs','results_summary','total_settled_gpu_seconds','recovery_matches']}},indent=2))
