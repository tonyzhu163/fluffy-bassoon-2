"""Final allocated-Linux CPU cost reconciliation of the closed Stage0 campaign."""
from datetime import datetime,timezone,timedelta
import hashlib,json,time
from pathlib import Path

ROOT=Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-interface-screen-20261102-r3')
TASKS=('OpenDrawer','PickPlaceDrawerToCounter','CoffeeSetupMug','TurnOffStove')
BASE=ROOT.parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))


def journals(run,expected=None):
    rows=[];phases={}
    for path in sorted((run/'actions').glob('rank-*/*.json')):
        if expected is not None:assert sha(path)==expected[str(path)]
        value=json.loads(path.read_text());totals={}
        for call in value['calls']:
            phase=call['phase'];row=phases.setdefault(phase,dict(calls=0,controls=0,unresolved=0))
            row['calls']+=1;row['controls']+=call.get('executed_or_conservative',call['requested']);row['unresolved']+=not call['resolved']
        rows.append(ref(path))
    if expected is not None:assert len(rows)==len(expected)
    return dict(journal_refs=rows,phases=phases,all_in_controls=sum(r['controls'] for r in phases.values()),unresolved_calls=sum(r['unresolved'] for r in phases.values()))


prior=json.loads((ROOT/'PRIOR-COSTS.json').read_text())['prior_attempts']
attempts=[];calibration=0;physical_draws=0
for old in prior:
    directory=BASE/old['run'];expected={r['path']:r['sha256'] for r in old['journal_refs']}
    journal=journals(directory,expected);assert journal['all_in_controls']==old['all_in_controls'] and journal['unresolved_calls']==old['unresolved_calls']
    count=0
    for task in TASKS:
        q=json.loads((directory/task/'INTERFACE-QUALIFICATION.json').read_text())
        for row in q['rows']:
            lengths={len(values) for values in row['scales'].values()};assert len(lengths)==1
            count+=next(iter(lengths))*(2+len(row['scales']))
    calibration+=count;physical_draws+=journal['phases']['policy_controls']['calls']
    attempts.append(dict(**old,journal_audit=journal,calibration_motor_draws=count,
        physical_chunk_motor_draws=journal['phases']['policy_controls']['calls']))

journal=journals(ROOT);assert journal['unresolved_calls']==0
clocks=[];records=0;reset_controls=0;physics_ticks=0;native_draws=0;head_draws=0;selector_scores=0;fit=[]
for rank,task in enumerate(TASKS):
    directory=ROOT/task;clock=ROOT/f'job/node-{rank}/clock.json';terminal=ROOT/f'job/node-{rank}/worker-terminal.json'
    c=json.loads(clock.read_text());assert c['exit_code']==json.loads(terminal.read_text())['exit_code']==0
    assert c['observed_gpu_seconds']==c['finished_unix']-c['started_unix']
    clocks.append(dict(**c,rank=rank,clock=ref(clock),terminal=ref(terminal)))
    final=json.loads((directory/'RESULT.json').read_text());assert final['completed'] and final['PPO_updates']==0
    for result_path in directory.glob('*/RESULT.json'):
        result=json.loads(result_path.read_text());n=result['new_episodes'];records+=n
        native_draws+=result['new_inference_draws']
        if result['arm']=='direct':head_draws+=n*12
        if result['arm']=='select':selector_scores+=n*12*8
        for row in result['rows']:
            if row.get('source_run')!='robocasa-interface-screen-20261102-r3':continue
            truth=json.loads(Path(row['truth']['path']).read_text())
            reset_controls+=truth['native_counts']['reset_controls'];physics_ticks+=truth['native_counts']['physics_ticks']
    f=json.loads((directory/'DIRECT-FIT.json').read_text());fit.append(dict(task=task,optimizer_updates=f['updates'],selected_update=f['selected_update'],minibatch=128,validation_checks=len(f['log']),fit=ref(directory/'DIRECT-FIT.json')))
    q=json.loads((directory/'INTERFACE-QUALIFICATION.json').read_text())
    for row in q['rows']:calibration+=len(next(iter(row['scales'].values())))*(2+len(row['scales']))
assert records==3915 and journal['phases']['policy_controls']['calls']==records*12
assert journal['all_in_controls']==records*600+reset_controls==2349000
assert native_draws==53700 and head_draws==2208 and selector_scores==7680
physical_draws+=journal['phases']['policy_controls']['calls']
created=datetime(2026,11,2,18,7,14,tzinfo=timezone(timedelta(hours=8)))
closed=datetime(2026,11,2,20,8,6,tzinfo=timezone(timedelta(hours=8)))
charge=4*int((closed-created).total_seconds());assert charge==29008
prior_charge=sum(a['conservative_GPU_seconds'] for a in attempts)
total=prior_charge+charge;assert total==32652 and total<43200
new=dict(run=ROOT.name,job_id='job-8ppjn68jxa9e',scheduler_status='Succeeded',scheduler_created_Shanghai=created.isoformat(),scheduler_closed_Shanghai=closed.isoformat(),
    scheduler_evidence='scheduler-terminal.jpg',all_four_instances_terminal=True,process_clocks=clocks,process_GPU_seconds=sum(c['observed_gpu_seconds'] for c in clocks),
    conservative_four_GPU_creation_to_scheduler_close_seconds=charge,completed_episodes=records,complete_policy_controls=records*600,
    reset_controls=reset_controls,physics_ticks=physics_ticks,journal_audit=journal,physical_chunk_motor_draws=records*12,
    motor_draws_including_selector_candidates=native_draws,direct_head_rollout_predictions=head_draws,selector_critic_scores=selector_scores,supervised_direct_fits=fit)
result=dict(completed=True,checked_unix=time.time(),prior_attempts=attempts,current_attempt=new,
    campaign_conservative_GPU_seconds=total,campaign_conservative_GPU_hours=total/3600,authorized_GPU_seconds=43200,unused_authorized_GPU_seconds=43200-total,
    campaign_complete_episodes=349+records,campaign_complete_policy_controls=209400+records*600,
    campaign_all_in_controls=sum(a['all_in_controls'] for a in attempts)+journal['all_in_controls'],campaign_unresolved_calls=16,
    calibration_motor_draws=calibration,physical_chunk_motor_draws=physical_draws,
    campaign_native_motor_draws_including_calibration_and_selector_candidates=physical_draws+calibration+7*80*12,
    PPO_updates=0,limitations=[
        'Conservative charge retains all four GPUs through complete scheduler closure, including startup, queue and waiting, even where individual worker processes exited earlier.',
        'Nine old unresolved policy calls and seven old unresolved constructor-reset calls remain conservatively charged. Their pre-dispatch native draws follow the immutable call ordering.',
        'Inference counts denote per-observation native motor draws, not CUDA kernel calls or isolated forward latency. Bank wall times include simulation and reset overhead.',
        'Direct-head supervised fitting and qualification are charged inside the same GPU allocations; optimizer updates are not PPO updates.',
        'Pre-existing S8 PPO histories/teacher weights are reused references, with no new S8 collection or training. Their original costs are inherited and separate from this new Stage0 allowance.',
        'All post-job audits run on allocated Linux CPU with GPUs hidden; no subsequent GPU inference or fresh allocation.'
    ])
with (ROOT/'TERMINAL-ACCOUNTING.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['prior_attempts','current_attempt','limitations']},indent=2))
