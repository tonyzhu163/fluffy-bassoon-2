"""Narrow diagnosed capacity-R1 successor admission; no scheduler/ledger open."""
from copy import deepcopy
import json
import math
from pathlib import Path

from .robotwin_demo_capacity import file_sha, bound_json
from .robotwin_reactive_capacity import CONFIG, build_plan, LEDGER_COUNTERS
from .robotwin_replay_diagnosis import PRODUCER

FORMAT='original_a1_event_aligned_capacity_v1'
RESULT_SHA='e9035c734ff5f7aa688d93e5a476794e452f1b19c2dedc813e5af262f0acd05b'
RECOVERY_SHA='c59c91799938c928bdd683a6ce681cfa3f9853fe0c829c05fe6657fe30af5d5b'
DIAGNOSIS_SHA='3c393bb2c2a69647db5fffb22d970e7a57fdbfb227310200248164abf2069e8d'
TIP='4e2ad8344fcc0343aa7d500ea445a61f4524bfa52118bd5cc3c4e582f35435b9'
FAILURE=dict(type='RuntimeError',message='Fixed published teacher trajectory did not execute successfully; diagnose before fitting')


def validate_allocation(allocation,*,now,run_id):
    if (allocation.get('mode')!=FORMAT or allocation.get('prior_a1_gpu_seconds')!=2036
            or allocation.get('prior_phase_gpu_seconds')!=2036
            or type(allocation.get('reserved_gpu_seconds')) is not int
            or not 1<=allocation['reserved_gpu_seconds']<=5164
            or allocation.get('capacity_action_bound')!=10000
            or allocation.get('capacity_fit_config')!=CONFIG or allocation.get('run_id')!=run_id):
        raise ValueError('Exact diagnosed A1 remainder2036+<=5164 seconds and10000actions required')
    start=allocation.get('scheduler_start_unix');end=allocation.get('driver_deadline_unix')
    if (any(type(v) not in (int,float) or not math.isfinite(v) for v in (start,end))
            or not start<=now<end<=start+allocation['reserved_gpu_seconds']-60):
        raise ValueError('Actual allocation clock includes startup/cleanup; no fresh grant/reset')


def validate_migration(proof,prior,allocation,*,source,run_id):
    """All immutable evidence first, before any canonical ledger constructor."""
    if (proof.get('format')!='original_a1_event_replay_migration_v1'
            or proof.get('source_commit_after')!=source or proof.get('run_id')!=run_id
            or proof['result']['sha256']!=RESULT_SHA or proof['recovery']['sha256']!=RECOVERY_SHA
            or proof['diagnosis']['sha256']!=DIAGNOSIS_SHA
            or bound_json(proof['result'])!=prior):
        raise ValueError('Only the exact independently diagnosed failed R1 can migrate')
    diagnosis=bound_json(proof['diagnosis']);terminal=bound_json(proof['terminal'])
    closeout=bound_json(proof['allocation_closeout'])
    if (diagnosis.get('status')!='audit_complete' or diagnosis.get('producer')!=PRODUCER
            or prior.get('source_commit')!=PRODUCER or prior.get('status')!='blocked_failure'
            or prior.get('stage')!='teacher_executability' or prior.get('failure')!=FAILURE
            or prior.get('pending') is not None or prior.get('cleanup_error') is not None
            or [r['primitive_steps'] for r in prior['records']]!=[400,104,104,400]
            or [r['success'] for r in prior['records']]!=[False,True,True,False]
            or [(r['id'],r['success'],r['actions']) for r in prior['teacher_execution']]!=
               [('phone-00',True,104),('phone-01',True,104),('phone-02',False,400)]):
        raise ValueError('Failure/case/cost history changed; no blind successor')
    if (terminal.get('format')!='original_a1_capacity_terminal_receipt_v1'
            or terminal.get('job_id')!='job-471ns3j0v2n4' or terminal.get('source_commit')!=PRODUCER
            or terminal.get('scheduler_status')!='Failed' or terminal.get('exit_code')!=1
            or terminal.get('charged_gpu_seconds')!=266
            or terminal.get('cumulative_a1_gpu_seconds')!=2036
            or terminal.get('full_recovery')!=dict(proof['recovery'],bytes=41381)
            or terminal.get('no_cache_or_fit') is not True
            or closeout.get('source_commit')!=PRODUCER or closeout.get('charged_gpu_seconds')!=266
            or closeout.get('exit_code')!=1 or closeout.get('error') is not None
            or closeout['attempt']['status']!='finished' or closeout['attempt']['slice_overrun']):
        raise ValueError('Actual terminal closure/full recovery/retained266GPU-s required')
    recovery=Path(proof['recovery']['path'])
    if (not recovery.is_absolute() or recovery.is_symlink() or recovery.stat().st_size!=41381
            or file_sha(recovery)!=RECOVERY_SHA):raise ValueError('Full predecessor recovery bytes differ')
    old=prior['actions'];ledger=allocation['action_ledger']
    root=Path(ledger['root']).resolve(strict=True)
    if (root!=Path(prior['action_ledger']['root']).resolve(strict=True)
            or ledger!=prior['action_ledger'] or root.parent!=Path(allocation['phase_output_root']).resolve(strict=True)
            or old['actual_actions']!=18599 or old['charged_actions']!=18599
            or old['pending_actions'] or old['uncertain_actions'] or old['unresolved_decisions']
            or old['journal_events']!=1222 or old['journal_sha256']!=TIP or run_id in old['runs']
            or sorted(p.name for p in (root/'events').glob('*.json'))!=[f'{i:010d}.json' for i in range(1222)]):
        raise ValueError('Exact unchanged canonical root/headers/closed18599cost/fresh identity required')
    first=json.loads((root/'events/0000000000.json').read_text())
    tip=json.loads((root/'events/0000001221.json').read_text())
    if first['header']!=old['header'] or tip['sha256']!=TIP:raise ValueError('Canonical header/tip differs before open')
    return dict(diagnosis=proof['diagnosis'],terminal=proof['terminal'],recovery=proof['recovery'],
        prior_actions=18599,prior_a1_gpu_seconds=2036,old_failure_retained=True)


def validate_recovery(saved,prior):
    if (saved.get('state')!=prior or saved.get('fits')!={} or saved.get('fit_recovery_refs')!={}
            or saved.get('control',{}).get('mode')!='action'
            or saved['control'].get('elapsed_steps')!=400
            or saved['control'].get('motor_rng') is None
            or set(saved.get('global_rng',{}))!={'python','numpy','torch','cuda'}):
        raise ValueError('Exact closed failed recovery with zero fits and full RNG/control required')


def build_event_plan(original_records,demos):
    base=build_plan(original_records,demos)
    rows={r['teacher_id']:r for r in base['teacher_replays']}
    control=deepcopy(rows['phone-02'])
    control.update(id='event-control-fixed-phone-02',stage='event_fixed_control',method='event_fixed_control')
    aligned=[]
    order=['phone-02','phone-00','phone-01']+[f'phone-{i:02d}' for i in range(3,12)]
    for identity in order:
        spec=deepcopy(rows[identity]);spec.update(id='event-teacher-'+identity,
            stage='event_aligned_teacher_execution',method='event_aligned_teacher')
        aligned.append(spec)
    development=deepcopy(base['development'])
    for row in development:row['id']=row['id'].replace('capacity-dev-','event-dev-')
    return dict(format=FORMAT,legacy_control=control,teacher_replays=aligned,development=development,
        maximum_complete_episodes=25,maximum_new_actions=10000,existing_A1_actions=18599,
        existing_A1_gpu_seconds=2036,maximum_new_GPU_seconds=5164,fit_config=deepcopy(CONFIG),
        teacher_order=order,whole_split={r['id']:r['split'] for r in demos['episodes']},
        cache_role='only12newalignedsuccesses; exclude new fixed02 control and every old rollout',
        supervision='expert submitted-plan imitation, native executed/cost mask separate',
        native_executor='unchanged H32/TOPP/compression/gripper semantics, native400horizon',
        scope='source-only reactive capacity diagnosis, not RL/AD/DPT/ICRL; no class rejection or causal rescue claim')
