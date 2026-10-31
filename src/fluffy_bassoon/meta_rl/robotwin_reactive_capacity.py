"""Single admitted original-A1 reactive capacity diagnostic; never schedules.

Requires actual stage0/archive alignment qualification and exact original
campaign admission. The existing scalar runtime, raw writer and episode loop
retain every physical command/mask; offline fitting is not an RL history.
"""
import argparse
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

from fluffy_bassoon.rl.robotwin_noise import RobotwinNoiseEnv,load_benchmark_episode_limits
from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter,_atomic_json
from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
from fluffy_bassoon.rl.robotwin_steering import validate_interface
from .robotwin_steering_screen import (
    read_bound,validate_admission,verify_source_conventions,run_complete_episode,
    paired_reset_fields,qualify_real_motor,
)
from .robotwin_demo_capacity import validate_demo_manifest,file_sha,bound_json,read_episode
from .robotwin_reactive_fit import make_heads,cache_demos,fit_reactive_head,ReactiveEpisodeControl
from .warmstart import atomic_save
from .checkpoint import rng_state


FORMAT='original_a1_full_reactive_capacity_v1'
TASK='place_phone_stand'
LIMIT=400
CONFIG=dict(seed=20261031,rank=4,kv_scale=.1,gripper_bound=0.,
            updates=100,global_batch=2,learning_rate=1e-3,validate_every=25)

LEDGER_COUNTERS=('actual_actions','charged_actions','pending_actions','uncertain_actions',
    'by_phase','by_method','decision_count','receipts','unresolved_decisions')


def validate_capacity_predecessor(original,allocation,*,run_id):
    """Read-only BEFORE opening anything that can create/register a ledger."""
    ledger=allocation['action_ledger'];prior=original['action_ledger'];old=original['actions']
    try:
        root=Path(ledger['root']).resolve(strict=True)
        original_root=Path(prior['root']).resolve(strict=True)
        phase_root=Path(allocation['phase_output_root']).resolve(strict=True)
    except OSError as error:
        raise ValueError('Require the existing original canonical ledger, never a replacement root') from error
    header=dict(schema=CampaignBudget.schema,**{k:ledger[k] for k in
        ('campaign_id','total_actions','source_caps','phase_caps','contract')})
    if (root!=original_root or phase_root!=root.parent
            or {k:v for k,v in ledger.items() if k!='root'}!={k:v for k,v in prior.items() if k!='root'}
            or header!=old['header'] or original.get('status')!='complete_restricted_development_search'
            or original.get('pending') is not None or old['actual_actions']!=17591 or old['charged_actions']!=17591
            or old['pending_actions'] or old['uncertain_actions'] or old['unresolved_decisions']
            or sum(r['primitive_steps'] for r in original['records'].values())!=17591
            or run_id in old['runs']):
        raise ValueError('Capacity continuation must bind the original root/header/closed physical counters/fresh run')
    count=old['journal_events'];events=root/'events'
    if (type(count) is not int or count<1 or sorted(p.name for p in events.glob('*.json'))!=
            [f'{i:010d}.json' for i in range(count)]):
        raise ValueError('Canonical journal changed after the bound original tip; diagnose before admission')
    first=json.loads((events/'0000000000.json').read_text())
    tip=json.loads((events/f'{count-1:010d}.json').read_text())
    if first.get('header')!=header or tip.get('sha256')!=old['journal_sha256']:
        raise ValueError('Original canonical header/tip identity differs before ledger open')


def validate_capacity_locked(budget,original):
    """Only this fresh run registration may follow the saved tip under lock."""
    old=original['actions'];current=budget.snapshot();count=old['journal_events']
    tip=json.loads((budget.events/f'{count-1:010d}.json').read_text())
    if (current['journal_events']!=count+1 or tip['sha256']!=old['journal_sha256']
            or current['header']!=old['header'] or any(current[k]!=old[k] for k in LEDGER_COUNTERS)
            or budget.run_id in old['runs'] or current['runs']!=dict(old['runs'],**{budget.run_id:budget.source_commit})):
        raise ValueError('Canonical lock did not preserve exact predecessor costs plus one fresh registration')
    event=json.loads((budget.events/f'{count:010d}.json').read_text())
    if (event.get('kind')!='run' or event.get('run_id')!=budget.run_id
            or event.get('source_commit')!=budget.source_commit
            or event.get('previous_sha256')!=old['journal_sha256']
            or event.get('sha256')!=current['journal_sha256']):
        raise ValueError('Only the exact fresh successor registration is admitted')


def validate_capacity_allocation(allocation,*,now,run_id):
    if (allocation.get('mode')!=FORMAT or allocation.get('prior_a1_gpu_seconds')!=1770
            or allocation.get('reserved_gpu_seconds',0)>5430
            or allocation.get('capacity_action_bound')!=10000
            or allocation.get('capacity_fit_config')!=CONFIG or allocation.get('run_id')!=run_id):
        raise ValueError('Exact original A1 residual/config/fresh identity required; no fresh grant')
    start=allocation.get('scheduler_start_unix');end=allocation.get('driver_deadline_unix')
    if (any(type(v) not in (int,float) or not math.isfinite(v) for v in (start,end))
            or not start<=now<end<=start+allocation['reserved_gpu_seconds']-60):
        raise ValueError('Original actual allocation clock must include startup and cleanup')


def build_plan(original_records,demos):
    """Four old development cases, all three arms; no locked query is opened."""
    bases={r['spec']['case']:r for r in original_records.values()
           if r['spec']['task']==TASK and r['spec']['stage']=='development'}
    if set(bases)!=set(range(8)) or any(not row['complete'] for row in bases.values()):
        raise ValueError('Bind the entire original native phone bank, never a selected failure subset')
    panel=[]
    for case in range(4):
        baseline=bases[case]
        for arm in ('native','kv','action'):
            panel.append(dict(id=f'capacity-dev-{case}-{arm}',task=TASK,case=case,arm=arm,
                method=f'capacity_{arm}',coefficients=[0.]*4,stage='capacity_development',horizon=LIMIT,
                reset_seed=baseline['spec']['reset_seed'],motor_seed=20261101+997*case,
                **paired_reset_fields(baseline)))
    q=dict(panel[0],id='capacity-qualification',case=7,reset_seed=bases[7]['spec']['reset_seed'],
           motor_seed=20261109,stage='capacity_qualification',**paired_reset_fields(bases[7]))
    replay=[]
    for ordinal,episode in enumerate(demos['episodes']):
        replay.append(dict(id=f"capacity-teacher-{episode['id']}",task=TASK,case=ordinal,arm='action',
            method='capacity_teacher',coefficients=[0.]*4,stage='capacity_teacher_execution',horizon=LIMIT,
            reset_seed=episode['reset_seed'],motor_seed=20261201+ordinal,
            expected_teacher_seed=episode['reset_seed'],teacher_id=episode['id'],
            paired_instruction=[episode['instruction']]))
    return dict(format=FORMAT,qualification=q,teacher_replays=replay,development=panel,
        maximum_complete_episodes=25,maximum_new_actions=10000,
        existing_A1_actions=17591,existing_A1_gpu_seconds=1770,
        maximum_new_GPU_seconds=5430,fit_config=deepcopy(CONFIG),
        scope='source-only reactive capacity diagnosis, not RL/AD/DPT/ICRL; no32case panel')


def summarize(records):
    groups={arm:[r for r in records if r['spec']['stage']=='capacity_development' and r['spec']['arm']==arm]
            for arm in ('native','kv','action')}
    if any(len(rows)!=4 for rows in groups.values()): raise ValueError('Tiny dev panel is not complete')
    native={r['spec']['case']:r for r in groups['native']}
    return dict(episodes_per_arm=4,successes={k:sum(r['success'] for r in v) for k,v in groups.items()},
        paired={arm:dict(wins=sum(r['success'] and not native[r['spec']['case']]['success'] for r in groups[arm]),
                          losses=sum(not r['success'] and native[r['spec']['case']]['success'] for r in groups[arm]))
                for arm in ('kv','action')},
        actual_actions={k:sum(r['primitive_steps'] for r in v) for k,v in groups.items()},
        interpretation='four previously opened development cases; descriptive capacity diagnosis only, no heldout/ICRL claim')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scope','allocation','interface','joint-conventions','demos','original-result'):
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha256',required=True)
    for name in ('model-path','norm-stats','robotwin-root','rlinf-root','gpu-probe','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--run-id',required=True)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if sys.platform!='linux':raise RuntimeError('Allocated Linux GPU only; no local models/tests')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if source!=args.source_commit or subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Require exact clean Git-deployed capacity source')
    scope=read_bound(args.scope,args.scope_sha256);allocation=read_bound(args.allocation,args.allocation_sha256)
    interface=validate_interface(read_bound(args.interface,args.interface_sha256))
    validate_admission(scope,allocation,scope_sha=args.scope_sha256,source=source,output=args.output,
                       hostname=socket.gethostname(),interface=interface)
    validate_capacity_allocation(allocation,now=time.time(),run_id=args.run_id)
    if file_sha(args.model_path/'model.safetensors')!=interface['motor_sha256'] or file_sha(args.norm_stats)!=interface['normalization_sha256']:
        raise ValueError('Frozen original motor/normalization bytes differ')
    if interface['joint_convention_sha256']!=args.joint_conventions_sha256:raise ValueError('Unbound decoded conventions')
    verify_source_conventions(read_bound(args.joint_conventions,args.joint_conventions_sha256),interface,
        dict(robotwin=args.robotwin_root,rlinf=args.rlinf_root))
    limits=load_benchmark_episode_limits(args.robotwin_root,[TASK])
    if limits[TASK]!=LIMIT:raise ValueError('Capacity costing requires the actual400-step native task')
    probe=json.loads(args.gpu_probe.read_text())
    if (probe.get('status')!='passed' or probe.get('hostname')!=socket.gethostname() or len(probe.get('devices',[]))!=1):
        raise ValueError('Actual one-GPU CUDA probe required')
    demos=validate_demo_manifest(read_bound(args.demos,args.demos_sha256))
    if (demos['motor_sha256']!=interface['motor_sha256'] or demos['normalization_sha256']!=interface['normalization_sha256']):
        raise ValueError('Demo action transform qualification is for another motor/norm')
    paths=demos['actual_whole_files']
    if set(paths)!={r['id'] for r in demos['episodes']}:raise ValueError('Qualified whole demo files differ')
    for ref in paths.values():
        path=Path(ref['path'])
        if not path.is_absolute() or path.is_symlink() or file_sha(path)!=ref['sha256']:
            raise ValueError('Whole demo identity failed before any teacher execution')
    for ref in bound_json(demos['alignment_qualification'])['source_files']:
        path=args.robotwin_root/ref['path']
        if not path.resolve().is_relative_to(args.robotwin_root.resolve()) or file_sha(path)!=ref['sha256']:
            raise ValueError('Author convention source differs from the actual pinned runtime')
    original=read_bound(args.original_result,args.original_result_sha256)
    if (original.get('scope_sha256')!=args.scope_sha256 or original.get('interface_sha256')!=args.interface_sha256):
        raise ValueError('Original accepted scope/native interface identity must remain exact')
    validate_capacity_predecessor(original,allocation,run_id=args.run_id)
    plan=build_plan(original['records'],demos)
    args.output.mkdir(parents=True,exist_ok=False)
    _atomic_json(args.output/'plan.json',dict(plan,source_commit=source,demo_manifest_sha256=args.demos_sha256,
        allocation_sha256=args.allocation_sha256,interface_sha256=args.interface_sha256))
    state=dict(format=FORMAT,status='running',stage='admission',records=[],pending=None,plan=plan,
        source_commit=source,allocation_sha256=args.allocation_sha256,demos_sha256=args.demos_sha256,
        action_ledger=deepcopy(allocation['action_ledger']),predecessor_journal=deepcopy(original['actions']))
    control=None;fits={}
    def storage():
        total=0
        for directory,dirs,files in os.walk(allocation['phase_output_root'],followlinks=False):
            if any((Path(directory)/n).is_symlink() for n in (*dirs,*files)):raise RuntimeError('Capacity storage rejects symlinks')
            total+=sum((Path(directory)/n).stat().st_size for n in files)
        if total+(64<<20)>64<<30:raise RuntimeError('Original64GiB shared phase storage exhausted')
        return total
    def guard():
        if time.time()+60>=allocation['driver_deadline_unix']:raise TimeoutError('Original capacity allocation deadline')
    def io_guard():
        guard();storage()
    def persist():
        refs={kind:dict(path=str(args.output/kind/'latest.pt'),sha256=file_sha(args.output/kind/'latest.pt'))
              for kind in ('kv','action') if (args.output/kind/'latest.pt').exists()}
        atomic_save(args.output/'recovery.pt',dict(state=deepcopy(state),fits=deepcopy(fits),
            fit_recovery_refs=refs,control=None if control is None else control.snapshot(),global_rng=rng_state()))
    def episode(spec,*,qualify=False):
        if allocation['driver_deadline_unix']-time.time()<240:
            raise TimeoutError('No complete native episode admitted inside remaining original clock')
        io_guard();state['pending']=spec;persist()
        def before(current,actual_spec):
            guard()
            if 'expected_teacher_seed' in spec:
                if current.last_reset_info['actual_seed']!=spec['expected_teacher_seed']:
                    raise ValueError('Demo seed retried/changed; do not claim original teacher replay')
                reference=np.asarray(spec['teacher_initial_state'],dtype=np.float32)
                actual=np.asarray(current.env.backend._raw_obs['states'])[0]
                if not np.allclose(actual,reference,rtol=0,atol=1e-5):
                    raise ValueError('Teacher replay initial drive targets differ from the author data')
            if qualify:
                # This is the existing actual zero/native/gradient/fingerprint
                # gate on the one charged source qualification episode.
                state['qualification']=qualify_real_motor(current,actual_spec,args.output)
        row=run_complete_episode(control,spec,budget,run_id=args.run_id,phase='a1',before_step=before,
                                 on_chunk=lambda _:storage())
        if qualify:
            if row['primitive_steps']<=0:raise RuntimeError('Actual real qualification episode executed no commands')
            state['qualification'].update(status='passed',actual_actions=row['primitive_steps'],complete_real_episode=spec['id'])
            _atomic_json(args.output/'gpu-qualification.json',state['qualification'],replace=True)
        small={k:v for k,v in row.items() if k!='control_recovery'}
        state['records'].append(small);state['pending']=None
        _atomic_json(args.output/(spec['id']+'.json'),small)
        state['actions']=budget.snapshot();persist()
        return row
    ledger=allocation['action_ledger']
    try:
        with CampaignBudget(ledger['root'],campaign_id=ledger['campaign_id'],total_actions=300000,
            source_caps=ledger['source_caps'],phase_caps=ledger['phase_caps'],contract=ledger['contract'],
            run_id=args.run_id,source_commit=source) as budget:
            validate_capacity_locked(budget,original)
            snap=budget.snapshot()
            if (snap['unresolved_decisions'] or snap['actual_actions']!=17591
                    or budget.remaining(phase='a1',method='capacity_native')<10000):
                raise ValueError('Original canonical A1 bank/costs changed; no extra ledger/hold recovery')
            io_guard()
            env=RobotwinNoiseEnv(model_path=args.model_path,norm_stats=args.norm_stats,
                robotwin_root=args.robotwin_root,rlinf_root=args.rlinf_root,output_dir=args.output/'runtime',
                action_horizon=32,num_steps=5,task_episode_limits=limits)
            heads=make_heads(env.backend,interface,CONFIG)
            writer=RobotwinCorpusWriter(args.output/'corpus',dict(role=FORMAT,source_commit=source,
                motor_sha256=interface['motor_sha256'],normalization_sha256=interface['normalization_sha256'],
                dataset_sha256=args.demos_sha256,scope_sha256=args.scope_sha256))
            control=ReactiveEpisodeControl(env,heads,writer,interface)
            frozen=env.frozen_fingerprint();state['frozen_motor_sha256']=frozen
            state['stage']='qualification';episode(plan['qualification'],qualify=True)
            state['stage']='teacher_executability'
            for spec,row in zip(plan['teacher_replays'],demos['episodes']):
                states,targets=read_episode(paths[row['id']]['path'],row,demos['schema'])
                spec=dict(spec,teacher_initial_state=states[0].tolist())
                control.replay.targets=torch.as_tensor(targets,device=env.backend.model.device)
                control.replay.reference=dict(paths[row['id']],alignment=demos['alignment_qualification'])
                receipt=episode(spec)
                state.setdefault('teacher_execution',[]).append(dict(id=row['id'],success=receipt['success'],
                    actions=receipt['primitive_steps'],source=paths[row['id']]))
                if not receipt['success']:
                    raise RuntimeError('Fixed published teacher trajectory did not execute successfully; diagnose before fitting')
            control.replay.targets=None;control.set_coefficients('native',[0.]*4)
            state['teacher_executability_qualified']=True
            state['stage']='offline_cache';persist()
            if control.active:raise ValueError('Never replace an active physical episode with offline cache inputs')
            records=cache_demos(env.backend,demos,paths,args.output/'cache',guard=io_guard,seed=CONFIG['seed'],
                               action_head=heads['action'])
            _atomic_json(args.output/'cache-manifest.json',dict(records=records,whole_demo_split=True))
            contract=dict(source_commit=source,scope_sha256=args.scope_sha256,motor_sha256=interface['motor_sha256'],
                normalization_sha256=interface['normalization_sha256'],demo_manifest_sha256=args.demos_sha256,
                meaning='supervised reactive interface capacity; no learning histories/new RL source')
            for kind in ('kv','action'):
                state['stage']='fit_'+kind;persist()
                fits[kind]=fit_reactive_head(kind,heads[kind],env.backend,records,args.output/kind,
                    guard=guard,config=CONFIG,contract=contract)
                control.references[kind]=fits[kind]['best']
                heads[kind].eval().requires_grad_(False)
            if env.frozen_fingerprint()!=frozen:raise RuntimeError('Frozen motor changed during supervised capacity fits')
            state['stage']='tiny_development_panel';persist()
            for original_spec in plan['development']:
                spec=deepcopy(original_spec)
                if spec['arm'] in fits:spec['policy_version']='capacity:'+fits[spec['arm']]['best']['sha256']
                episode(spec)
            state.update(status='complete',stage='complete',summary=summarize(state['records']),
                actions=budget.snapshot(),frozen_after=env.frozen_fingerprint(),storage_bytes=storage())
            state['runtime_counters']={k:getattr(env.backend,k) for k in
                ('prefix_calls','prefix_seconds','motor_queries','motor_seconds')}
            if state['frozen_after']!=frozen:raise RuntimeError('Final frozen motor fingerprint differs')
    except BaseException as error:
        state.update(status='blocked_failure',failure=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        try:
            if control is not None:control.close()
        finally:
            persist();_atomic_json(args.output/'result.json',state,replace=True)


if __name__=='__main__':main()
