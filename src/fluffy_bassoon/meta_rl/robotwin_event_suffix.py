"""One diagnostic attempt of the original unrun phone05--11 event bank.

Uses the be247 native H32 controller/episode loop unchanged. Complete failures
are retained and do not stop suffix coverage. No cache, fitter or source/student
admission is available here; all12 fixed TRAIN/VAL identities remain in the plan.
"""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter, _atomic_json
from fluffy_bassoon.rl.robotwin_noise import RobotwinNoiseEnv, load_benchmark_episode_limits
from fluffy_bassoon.rl.robotwin_steering import validate_interface
from .checkpoint import rng_state, restore_rng
from .warmstart import atomic_save
from .robotwin_demo_capacity import bound_json, file_sha, read_episode, validate_demo_manifest
from .robotwin_event_capacity import parser as original_parser
from .robotwin_event_admission import build_event_plan
from .robotwin_event_replay import EventEpisodeControl
from .robotwin_event_schedule import build_schedule
from .robotwin_reactive_capacity import CONFIG, TASK, LIMIT, validate_capacity_locked
from .robotwin_reactive_fit import make_heads
from .robotwin_steering_screen import (
    read_bound, validate_admission, verify_source_conventions, run_complete_episode, qualify_real_motor,
)

FORMAT = 'original_a1_event_fixed_suffix_diagnostic_v1'
PRODUCER = 'be247676c1217d9e046091c52cac2b7d0f2d3014'
RESULT_SHA = 'ba775b4fcbb3029120927ecd59461fabb1cccd7e95326f8d9f3a6f731ea77d3f'
RECOVERY_SHA = '0403db7c2bd0c4e7566cfdbb5a6404409a0d45a0eb85df4c0972a2bbcd58a153'
TERMINAL_SHA = 'a352462ced9ce556498b1871d76fdeacc0a6837e22fe54d36a01f203a5db1bef'
CLOSEOUT_SHA = '6a2822cc524e2042cab646996d5879834a2723d81b832597cdb9cfb107dd469f'
TIP = '9c0b1dccb663b52f569c64098b8e2d47f70911226ad18d129fc7375032eb9df3'
SUFFIX = [f'phone-{i:02d}' for i in range(5, 12)]
EXPECTATION = dict(target='7/7 original suffix teachers executable',
    prediction='uncertain: fixed04 already failed; no universal-success prior',
    interpretation='Failure misses the bank target, not by itself statistical surprise or a causal diagnosis')
FAILURE = dict(type='RuntimeError', message='Fixed aligned teacher failed; preserve all demos and stop before fitting')


def validate_allocation(allocation, *, now, run_id):
    start, end = allocation.get('scheduler_start_unix'), allocation.get('driver_deadline_unix')
    seconds = allocation.get('reserved_gpu_seconds')
    if (allocation.get('mode') != FORMAT or allocation.get('run_id') != run_id
            or allocation.get('prior_a1_gpu_seconds') != 2334
            or allocation.get('prior_phase_gpu_seconds') != 2334
            or type(seconds) is not int or not 1 <= seconds <= 1800
            or allocation.get('capacity_action_bound') != 2800
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end))
            or not start <= now < end <= start + seconds - 60):
        raise ValueError('One <=1800 all-in GPU-second /2800-action suffix, retained2334 costs, finite actual clock required')


def fixed_suffix(prior, demos):
    plan = prior['plan']
    if (plan['teacher_order'] != ['phone-02','phone-00','phone-01'] + [f'phone-{i:02d}' for i in range(3,12)]
            or plan['whole_split'] != {r['id']:r['split'] for r in demos['episodes']}
            or len(plan['teacher_replays']) != 12):
        raise ValueError('Keep the entire original12 /8TRAIN4VAL bank')
    specs = deepcopy(plan['teacher_replays'][5:])
    if [s['teacher_id'] for s in specs] != SUFFIX or len({s['id'] for s in specs}) != 7:
        raise ValueError('Only original05--11, exactly once in order')
    for spec, row in zip(specs, demos['episodes'][5:]):
        if (spec['teacher_id'] != row['id'] or spec['reset_seed'] != row['reset_seed']
                or spec['expected_teacher_seed'] != row['reset_seed']
                or spec['paired_instruction'] != [row['instruction']]
                or spec['motor_seed'] != 20261201 + row['canonical_index']
                or spec['id'] != 'event-teacher-' + row['id']
                or spec['stage'] != 'event_aligned_teacher_execution'
                or spec['method'] != 'event_aligned_teacher' or spec['arm'] != 'action'
                or spec['coefficients'] != [0.]*4 or spec['task'] != TASK or spec['horizon'] != LIMIT):
            raise ValueError('Original teacher file identity/seed/instruction/motor/schedule contract changed')
    return specs


def validate_migration(proof, prior, terminal, allocation, *, source, run_id):
    if (proof.get('format') != FORMAT or proof.get('source_commit_after') != source
            or proof.get('run_id') != run_id or proof['result']['sha256'] != RESULT_SHA
            or proof['recovery']['sha256'] != RECOVERY_SHA or proof['terminal']['sha256'] != TERMINAL_SHA
            or proof['allocation_closeout']['sha256'] != CLOSEOUT_SHA
            or bound_json(proof['result']) != prior or bound_json(proof['terminal']) != terminal):
        raise ValueError('Exact ended be247 result/terminal/full recovery/closeout migration required')
    closeout = bound_json(proof['allocation_closeout'])
    attempt = closeout['attempt']
    if (prior.get('source_commit') != PRODUCER or prior.get('status') != 'blocked_failure'
            or prior.get('stage') != 'teacher_executability' or prior.get('failure') != FAILURE
            or prior.get('pending') is not None or prior.get('cleanup_error') is not None
            or terminal.get('job_id') != 'job-2rkimbvypowd' or terminal.get('source_commit') != PRODUCER
            or terminal.get('scheduler_status') != 'Failed' or terminal.get('exit_code') != 1
            or terminal.get('cumulative_a1_gpu_seconds') != 2334 or terminal.get('charged_gpu_seconds') != 298
            or terminal.get('unrun_aligned_teacher_suffix') != SUFFIX
            or terminal.get('cache_created') is not False or terminal.get('all_twelve_teacher_gate_passed') is not False
            or terminal.get('full_recovery') != dict(proof['recovery'], bytes=223589, opaque_no_deserialization=True)
            or closeout.get('source_commit') != PRODUCER or closeout.get('charged_gpu_seconds') != 298
            or closeout.get('exit_code') != 1 or closeout.get('error') is not None
            or attempt.get('status') != 'finished' or attempt.get('slice_overrun') is not False):
        raise ValueError('Preserve failed04 and actual settled298/2334 costs; no retry/filter/fit')
    expected = [prior['plan']['legacy_control'], *prior['plan']['teacher_replays'][:5]]
    rows = prior['records']
    if (len(rows) != 6 or [r['primitive_steps'] for r in rows] != [400,193,193,193,193,400]
            or [r['success'] for r in rows] != [False,True,True,True,True,False]
            or any(not r['complete'] or {k:v for k,v in r['spec'].items() if k != 'teacher_initial_state'} != spec
                   for r,spec in zip(rows,expected))
            or len(prior.get('closed_control_recoveries',[])) != 6):
        raise ValueError('Exact six complete predecessor episodes, including failed04, required')
    ledger, old = allocation['action_ledger'], prior['actions']
    root = Path(ledger['root']).resolve(strict=True)
    recovery = Path(proof['recovery']['path'])
    if (ledger != prior['action_ledger'] or root != Path(prior['action_ledger']['root']).resolve(strict=True)
            or root.parent != Path(allocation['phase_output_root']).resolve(strict=True)
            or old['actual_actions'] != 20171 or old['charged_actions'] != 20171
            or old['pending_actions'] or old['uncertain_actions'] or old['unresolved_decisions']
            or old['journal_events'] != 1331 or old['journal_sha256'] != TIP or run_id in old['runs']
            or sorted(p.name for p in (root/'events').glob('*.json')) != [f'{i:010d}.json' for i in range(1331)]
            or json.loads((root/'events/0000000000.json').read_text())['header'] != old['header']
            or json.loads((root/'events/0000001330.json').read_text())['sha256'] != TIP
            or not recovery.is_absolute() or recovery.is_symlink() or recovery.stat().st_size != 223589
            or file_sha(recovery) != RECOVERY_SHA):
        raise ValueError('Exact canonical root/header/tip/closed20171costs/fresh identity/full recovery required')


def validate_recovery(saved, prior):
    control = saved.get('control') or {}
    if (saved.get('state') != prior or saved.get('fits') != {} or saved.get('fit_recovery_refs') != {}
            or control != prior['closed_control_recoveries'][-1] or control.get('mode') != 'action'
            or control.get('elapsed_steps') != 400 or control.get('motor_rng') is None
            or control.get('source_cursor',{}).get('closed') is not True
            or set(saved.get('global_rng',{})) != {'python','numpy','torch','cuda'}
            or saved['global_rng']['cuda'] is None):
        raise ValueError('Exact complete closed04 control/private/global RNG recovery with zero fits required')


def restore_closed_control(control, saved, targets):
    """Restore saved control at its CLOSED boundary, never mid-physics resume."""
    if control.active:raise ValueError('Cannot restore into active physics')
    control.select_teacher(targets, saved['teacher_reference'], aligned=True)
    if control.schedule != saved['source_schedule']:raise ValueError('Predecessor04 source schedule changed')
    control.set_coefficients(saved['mode'], [0.]*4)
    control.env.elapsed_steps = saved['elapsed_steps']
    control.motor_rng = np.random.default_rng()
    control.motor_rng.bit_generator.state = deepcopy(saved['motor_rng'])
    control.cursor = deepcopy(saved['source_cursor'])
    control.last_audit = deepcopy(saved['last_native_rng_audit'])
    control.replay.chunk_index = control.cursor['chunk_index']
    if control.snapshot() != saved:raise ValueError('Closed full-control restoration differs')


def run_fixed_suffix(specs, episode):
    """Continue complete negative outcomes; runtime/integrity exceptions stop."""
    if [s['teacher_id'] for s in specs] != SUFFIX:raise ValueError('No retry, filter or reordered suffix')
    for index, spec in enumerate(specs):
        episode(deepcopy(spec), qualify=(index == 0))


def main(argv=None):
    parser = original_parser()
    parser.add_argument('--continuation-manifest', type=Path, required=True)
    parser.add_argument('--continuation-manifest-sha256', required=True)
    args = parser.parse_args(argv)
    if sys.platform != 'linux':raise RuntimeError('Allocated Linux only')
    source = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if source != args.source_commit or subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Require clean exact Git-deployed successor')
    manifest = read_bound(args.continuation_manifest,args.continuation_manifest_sha256)
    scope = read_bound(args.scope,args.scope_sha256)
    allocation = read_bound(args.allocation,args.allocation_sha256)
    interface = validate_interface(read_bound(args.interface,args.interface_sha256))
    validate_admission(scope,allocation,scope_sha=args.scope_sha256,source=source,output=args.output,
                       hostname=socket.gethostname(),interface=interface)
    validate_allocation(allocation,now=time.time(),run_id=args.run_id)
    if (file_sha(args.model_path/'model.safetensors') != interface['motor_sha256']
            or file_sha(args.norm_stats) != interface['normalization_sha256']
            or interface['joint_convention_sha256'] != args.joint_conventions_sha256):
        raise ValueError('Exact original motor/norm/conventions required')
    verify_source_conventions(read_bound(args.joint_conventions,args.joint_conventions_sha256),interface,
        dict(robotwin=args.robotwin_root,rlinf=args.rlinf_root))
    limits = load_benchmark_episode_limits(args.robotwin_root,[TASK])
    probe = json.loads(args.gpu_probe.read_text())
    if (limits[TASK] != LIMIT or probe.get('status') != 'passed' or probe.get('hostname') != socket.gethostname()
            or len(probe.get('devices',[])) != 1):raise ValueError('Actual one-GPU probe/native400 required')
    demos = validate_demo_manifest(read_bound(args.demos,args.demos_sha256))
    if (demos['motor_sha256'] != interface['motor_sha256'] or demos['normalization_sha256'] != interface['normalization_sha256']):
        raise ValueError('Demo motor/norm mismatch')
    paths = demos['actual_whole_files']
    if set(paths) != {r['id'] for r in demos['episodes']}:raise ValueError('Retain all12 source files')
    for ref in paths.values():
        path = Path(ref['path'])
        if not path.is_absolute() or path.is_symlink() or file_sha(path) != ref['sha256']:raise ValueError('Demo file changed')
    for ref in bound_json(demos['alignment_qualification'])['source_files']:
        path = args.robotwin_root/ref['path']
        if not path.resolve().is_relative_to(args.robotwin_root.resolve()) or file_sha(path) != ref['sha256']:
            raise ValueError('Pinned author conventions changed')
    original = read_bound(args.original_result,args.original_result_sha256)
    if (args.original_result_sha256 != 'aece67fdfd33af52653dd74587424450571288a26e03be18a0f1d99721fe3538'
            or original['scope_sha256'] != args.scope_sha256 or original['interface_sha256'] != args.interface_sha256):
        raise ValueError('Bind original accepted scope/interface')
    prior = read_bound(args.predecessor_result,args.predecessor_result_sha256)
    proof = read_bound(args.migration,args.migration_sha256)
    terminal = bound_json(proof['terminal'])
    if (args.predecessor_result_sha256 != RESULT_SHA or args.resume_sha256 != RECOVERY_SHA
            or Path(proof['result']['path']).resolve() != args.predecessor_result.resolve()
            or Path(proof['recovery']['path']).resolve() != args.resume.resolve()
            or prior['demos_sha256'] != args.demos_sha256
            or prior['plan'] != build_event_plan(original['records'],demos)):
        raise ValueError('Immutable original plan/demo/result/recovery binding differs')
    validate_migration(proof,prior,terminal,allocation,source=source,run_id=args.run_id)
    specs = fixed_suffix(prior,demos)
    expected_manifest = dict(format=FORMAT,producer=PRODUCER,predecessor_result_sha256=RESULT_SHA,
        predecessor_recovery_sha256=RECOVERY_SHA,predecessor_terminal_sha256=TERMINAL_SHA,
        original_teacher_ids=prior['plan']['teacher_order'],whole_split=prior['plan']['whole_split'],
        suffix_specs=specs,maximum_complete_episodes=7,maximum_new_actions=2800,maximum_gpu_seconds=1800,
        cache_fit_source_student_admission=False,continue_complete_failures=True,expected_outcomes=EXPECTATION)
    if manifest != expected_manifest:raise ValueError('Exact fixed-suffix manifest required')
    saved = torch.load(args.resume,map_location='cpu',weights_only=False)
    validate_recovery(saved,prior)
    # All12 existing schedules are checked; none is removed or modified.
    schedules = {}
    for row in demos['episodes']:
        _,targets = read_episode(paths[row['id']]['path'],row,demos['schema'])
        schedules[row['id']] = build_schedule(targets)
    if not all(s['admissible'] for s in schedules.values()):raise ValueError('Original full12 schedule bank changed')
    args.output.mkdir(parents=True,exist_ok=False)
    _atomic_json(args.output/'plan.json',dict(expected_manifest,source_commit=source,
        demo_manifest_sha256=args.demos_sha256,allocation_sha256=args.allocation_sha256,
        interface_sha256=args.interface_sha256,migration_sha256=args.migration_sha256))
    _atomic_json(args.output/'schedule-preflight.json',dict(schedules=schedules,demos_sha256=args.demos_sha256))
    state = dict(format=FORMAT,status='running',stage='suffix_coverage',source_commit=source,
        plan=expected_manifest,records=[],pending=None,actions=deepcopy(prior['actions']),
        action_ledger=deepcopy(allocation['action_ledger']),predecessor_result=proof['result'],
        predecessor_recovery=proof['recovery'],predecessor_terminal=proof['terminal'],
        predecessor_records=deepcopy(prior['records']),predecessor_control_recovery=deepcopy(saved['control']),
        chunk_audits=[],physical_reset_audits=[],teacher_execution=[],closed_control_recoveries=[],
        cache_created=False,fit_updates=0,source_student_admitted=False,cleanup_error=None)
    control = None;env = None
    def guard():
        if time.time()+60 >= allocation['driver_deadline_unix']:raise TimeoutError('Finite suffix clock')
    def storage():
        guard();total=0
        for directory,dirs,files in os.walk(allocation['phase_output_root'],followlinks=False):
            if any((Path(directory)/n).is_symlink() for n in (*dirs,*files)):raise RuntimeError('No storage symlinks')
            total += sum((Path(directory)/n).stat().st_size for n in files)
        if total+(64<<20) > 64<<30:raise RuntimeError('Original64GiB aggregate storage cap')
        return total
    def persist():
        atomic_save(args.output/'recovery.pt',dict(state=deepcopy(state),fits={},fit_recovery_refs={},
            control=None if control is None else control.snapshot(),global_rng=rng_state()))
    rows = {r['id']:r for r in demos['episodes']}
    ledger = allocation['action_ledger']
    try:
        with CampaignBudget(ledger['root'],campaign_id=ledger['campaign_id'],total_actions=300000,
            source_caps=ledger['source_caps'],phase_caps=ledger['phase_caps'],contract=ledger['contract'],
            run_id=args.run_id,source_commit=source) as budget:
            try:
                validate_capacity_locked(budget,prior)
                if budget.remaining(phase='a1',method='event_aligned_teacher') < 2800:raise ValueError('Full suffix capacity required')
                storage()
                env = RobotwinNoiseEnv(model_path=args.model_path,norm_stats=args.norm_stats,
                    robotwin_root=args.robotwin_root,rlinf_root=args.rlinf_root,output_dir=args.output/'runtime',
                    action_horizon=32,num_steps=5,task_episode_limits=limits)
                heads = make_heads(env.backend,interface,CONFIG)
                writer = RobotwinCorpusWriter(args.output/'corpus',dict(role=FORMAT,source_commit=source,
                    motor_sha256=interface['motor_sha256'],normalization_sha256=interface['normalization_sha256'],
                    dataset_sha256=args.demos_sha256,scope_sha256=args.scope_sha256))
                control = EventEpisodeControl(env,heads,writer,interface)
                frozen = env.frozen_fingerprint()
                if frozen != prior['frozen_motor_sha256']:raise ValueError('Original frozen motor changed')
                _,old_targets = read_episode(paths['phone-04']['path'],rows['phone-04'],demos['schema'])
                restore_closed_control(control,saved['control'],old_targets)
                restore_rng(saved['global_rng']);persist()
                def episode(spec,*,qualify):
                    if allocation['driver_deadline_unix']-time.time() < 240:raise TimeoutError('Full native episode clock admission')
                    storage()
                    if len(state['records']) >= 7:raise RuntimeError('Seven attempts only')
                    row = rows[spec['teacher_id']]
                    initial,targets = read_episode(paths[row['id']]['path'],row,demos['schema'])
                    spec = dict(spec,teacher_initial_state=initial[0].tolist())
                    control.select_teacher(targets,dict(paths[row['id']],alignment=demos['alignment_qualification']),aligned=True)
                    state['pending']=deepcopy(spec);persist()
                    def before(current,actual_spec):
                        guard()
                        state['physical_reset_audits'].append(current.publish_reset_audit(args.output/'physical-audits'))
                        raw=current.env.backend._raw_obs['states']
                        actual=(raw.detach().cpu().numpy() if isinstance(raw,torch.Tensor) else np.asarray(raw))[0]
                        if (current.last_reset_info['actual_seed'] != spec['expected_teacher_seed']
                                or not np.allclose(actual,initial[0],rtol=0,atol=1e-5)):
                            raise ValueError('Exact original teacher seed/initial drive targets required')
                        if qualify:state['qualification']=qualify_real_motor(current,actual_spec,args.output)
                    def on_chunk(_):
                        guard();state['chunk_audits'].append(control.publish_chunk_audit(args.output/'chunk-audits'))
                        if budget.snapshot()['charged_actions']-20171 > 2800:raise RuntimeError('Suffix action cap')
                        storage()
                    receipt = run_complete_episode(control,spec,budget,run_id=args.run_id,phase='a1',before_step=before,on_chunk=on_chunk)
                    if qualify:
                        if receipt['primitive_steps'] <= 0:raise RuntimeError('Qualification needs its charged05 episode')
                        state['qualification'].update(status='passed',actual_actions=receipt['primitive_steps'],complete_real_episode=spec['id'])
                        _atomic_json(args.output/'gpu-qualification.json',state['qualification'],replace=True)
                    small={k:v for k,v in receipt.items() if k!='control_recovery'}
                    state['records'].append(small);state['closed_control_recoveries'].append(receipt['control_recovery'])
                    state['teacher_execution'].append(dict(id=row['id'],success=receipt['success'],actions=receipt['primitive_steps'],
                        source=paths[row['id']],cursor=deepcopy(control.cursor),diagnostic_only=True))
                    state['pending']=None;state['actions']=budget.snapshot()
                    _atomic_json(args.output/(spec['id']+'.json'),small);persist()
                    return receipt
                run_fixed_suffix(specs,episode)
                state.update(status='complete_diagnostic_coverage',stage='complete',
                    outcome_comparison=dict(target_successes=7,actual_successes=sum(r['success'] for r in state['records']),
                        attempted_cases=[r['spec']['teacher_id'] for r in state['records']],
                        failed_cases=[r['spec']['teacher_id'] for r in state['records'] if not r['success']],
                        expected_outcomes=EXPECTATION,causal_diagnosis='not established; preserve physical/command audits'),
                    actual_new_actions=budget.snapshot()['actual_actions']-20171,
                    frozen_after=env.frozen_fingerprint(),storage_bytes=storage(),all_twelve_teacher_gate_passed=False)
                if state['frozen_after'] != frozen:raise RuntimeError('Final frozen motor changed')
            finally:
                state['actions']=budget.snapshot()
    except BaseException as error:
        state.update(status='blocked_failure',failure=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        full_control = None if control is None else control.snapshot()
        global_rng = rng_state()
        try:
            if env is not None:env.close()
        except BaseException as error:
            state.update(status='blocked_failure',cleanup_error=dict(type=type(error).__name__,message=str(error)));raise
        finally:
            atomic_save(args.output/'recovery.pt',dict(state=deepcopy(state),fits={},fit_recovery_refs={},control=full_control,global_rng=global_rng))
            _atomic_json(args.output/'result.json',state,replace=True)


if __name__ == '__main__':main()
