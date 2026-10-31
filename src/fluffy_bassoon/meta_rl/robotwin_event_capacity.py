"""Diagnosed event-aligned successor within the original A1 remainder.

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
from .robotwin_reactive_fit import make_heads,fit_reactive_head
from .warmstart import atomic_save
from .checkpoint import rng_state,restore_rng


from .robotwin_event_admission import (FORMAT,validate_allocation,validate_migration,validate_recovery,build_event_plan)
from .robotwin_reactive_capacity import CONFIG,TASK,LIMIT,validate_capacity_locked,summarize
from .robotwin_event_schedule import build_schedule
from .robotwin_event_replay import EventEpisodeControl,cache_rollouts


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scope','allocation','interface','joint-conventions','demos','original-result','predecessor-result','migration'):
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha256',required=True)
    for name in ('model-path','norm-stats','robotwin-root','rlinf-root','gpu-probe','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume',type=Path,required=True);p.add_argument('--resume-sha256',required=True)
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
    validate_allocation(allocation,now=time.time(),run_id=args.run_id)
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
    if (args.original_result_sha256!='aece67fdfd33af52653dd74587424450571288a26e03be18a0f1d99721fe3538'
            or original.get('scope_sha256')!=args.scope_sha256 or original.get('interface_sha256')!=args.interface_sha256):
        raise ValueError('Original accepted scope/native interface identity must remain exact')
    prior=read_bound(args.predecessor_result,args.predecessor_result_sha256)
    proof=read_bound(args.migration,args.migration_sha256)
    if (Path(proof['recovery']['path']).resolve()!=args.resume.resolve()
            or proof['recovery']['sha256']!=args.resume_sha256
            or proof['result']['sha256']!=args.predecessor_result_sha256
            or prior['demos_sha256']!=args.demos_sha256
            or prior.get('scope_sha256',args.scope_sha256)!=args.scope_sha256
            or prior['action_ledger']!=original['action_ledger']):
        raise ValueError('Migration/recovery/demos/scope/original lineage bindings differ')
    migration=validate_migration(proof,prior,allocation,source=source,run_id=args.run_id)
    saved=torch.load(args.resume,map_location='cpu',weights_only=False)
    validate_recovery(saved,prior)
    # Preflight all12 before constructing a motor or opening the canonical
    # writer. Even rejected schedules get a retained receipt, no demo filtering.
    schedules={}
    for row in demos['episodes']:
        _,targets=read_episode(paths[row['id']]['path'],row,demos['schema'])
        schedules[row['id']]=build_schedule(targets)
    plan=build_event_plan(original['records'],demos)
    args.output.mkdir(parents=True,exist_ok=False)
    _atomic_json(args.output/'plan.json',dict(plan,source_commit=source,demo_manifest_sha256=args.demos_sha256,
        allocation_sha256=args.allocation_sha256,interface_sha256=args.interface_sha256))
    _atomic_json(args.output/'schedule-preflight.json',dict(schedules=schedules,
        all_admissible=all(r['admissible'] for r in schedules.values()),demos_sha256=args.demos_sha256))
    if not all(r['admissible'] for r in schedules.values()):
        raise ValueError('At least one fixed demo cannot issue its full reference within native400; no GPU episode/fit')
    state=dict(format=FORMAT,status='running',stage='admission',records=[],pending=None,plan=plan,
        source_commit=source,allocation_sha256=args.allocation_sha256,demos_sha256=args.demos_sha256,
        action_ledger=deepcopy(allocation['action_ledger']),predecessor_journal=deepcopy(prior['actions']),migration=migration,
        predecessor_control_recovery=deepcopy(saved['control']),predecessor_records=deepcopy(prior['records']),
        old_rollouts_excluded_from_SL=True,chunk_audits=[],physical_reset_audits=[])
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
        if len(state['records'])>=25:raise RuntimeError('Fixed25-episode ceiling reached')
        io_guard();state['pending']=spec;persist()
        def before(current,actual_spec):
            guard()
            state['physical_reset_audits'].append(current.publish_reset_audit(args.output/'physical-audits'))
            if 'expected_teacher_seed' in spec:
                if current.last_reset_info['actual_seed']!=spec['expected_teacher_seed']:
                    raise ValueError('Demo seed retried/changed; do not claim original teacher replay')
                reference=np.asarray(spec['teacher_initial_state'],dtype=np.float32)
                raw_state=current.env.backend._raw_obs['states']
                actual=(raw_state.detach().cpu().numpy() if isinstance(raw_state,torch.Tensor) else np.asarray(raw_state))[0]
                if not np.allclose(actual,reference,rtol=0,atol=1e-5):
                    raise ValueError('Teacher replay initial drive targets differ from the author data')
            if qualify:
                # This is the existing actual zero/native/gradient/fingerprint
                # gate on the one charged source qualification episode.
                state['qualification']=qualify_real_motor(current,actual_spec,args.output)
        def on_chunk(_):
            guard()
            state['chunk_audits'].append(control.publish_chunk_audit(args.output/'chunk-audits'))
            if budget.snapshot()['charged_actions']-18599>10000:raise RuntimeError('New action bound exceeded')
            storage()
        row=run_complete_episode(control,spec,budget,run_id=args.run_id,phase='a1',before_step=before,
                                 on_chunk=on_chunk)
        if qualify:
            if row['primitive_steps']<=0:raise RuntimeError('Actual real qualification episode executed no commands')
            state['qualification'].update(status='passed',actual_actions=row['primitive_steps'],complete_real_episode=spec['id'])
            _atomic_json(args.output/'gpu-qualification.json',state['qualification'],replace=True)
        small={k:v for k,v in row.items() if k!='control_recovery'}
        state['records'].append(small);state['pending']=None
        state['closed_control_recoveries']=state.get('closed_control_recoveries',[])+[row['control_recovery']]
        _atomic_json(args.output/(spec['id']+'.json'),small)
        state['actions']=budget.snapshot();persist()
        return row
    ledger=allocation['action_ledger']
    try:
        with CampaignBudget(ledger['root'],campaign_id=ledger['campaign_id'],total_actions=300000,
            source_caps=ledger['source_caps'],phase_caps=ledger['phase_caps'],contract=ledger['contract'],
            run_id=args.run_id,source_commit=source) as budget:
            validate_capacity_locked(budget,prior)
            snap=budget.snapshot()
            if (snap['unresolved_decisions'] or snap['actual_actions']!=18599
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
            control=EventEpisodeControl(env,heads,writer,interface)
            frozen=env.frozen_fingerprint();state['frozen_motor_sha256']=frozen
            if frozen!=prior['frozen_motor_sha256']:raise ValueError('Actual successor frozen motor differs')
            # Fresh runtime initialization must not silently replace the saved
            # global planner RNG. Each explicitly repeated episode still uses
            # its declared private VLA motor seed; predecessor private state is retained.
            restore_rng(saved['global_rng']);persist()
            rows={row['id']:row for row in demos['episodes']}
            def teacher(spec,*,aligned,qualify=False):
                row=rows[spec['teacher_id']]
                states,targets=read_episode(paths[row['id']]['path'],row,demos['schema'])
                spec=dict(spec,teacher_initial_state=states[0].tolist())
                control.select_teacher(targets,dict(paths[row['id']],alignment=demos['alignment_qualification']),aligned=aligned)
                receipt=episode(spec,qualify=qualify)
                state.setdefault('teacher_execution',[]).append(dict(id=row['id'],success=receipt['success'],
                    actions=receipt['primitive_steps'],source=paths[row['id']],aligned=aligned,
                    cursor=deepcopy(control.cursor),SL_eligible=aligned and receipt['success']))
                persist()
                return receipt
            state['stage']='legacy_control_and_qualification'
            teacher(plan['legacy_control'],aligned=False,qualify=True)
            # Control outcome never selects/changes the fixed aligned schedule.
            state['stage']='teacher_executability'
            aligned_records=[]
            for spec in plan['teacher_replays']:
                receipt=teacher(spec,aligned=True)
                aligned_records.append({k:v for k,v in receipt.items() if k!='control_recovery'})
                if not receipt['success']:
                    raise RuntimeError('Fixed aligned teacher failed; preserve all demos and stop before fitting')
            control.clear_teacher();control.set_coefficients('native',[0.]*4)
            state['teacher_executability_qualified']=True
            state['stage']='offline_canonical_plan_cache';persist()
            if control.active:raise ValueError('Never replace active physics with offline cache inputs')
            records=cache_rollouts(env.backend,demos,paths,aligned_records,args.output/'cache',guard=io_guard,
                deadline=allocation['driver_deadline_unix']-60,action_head=heads['action'])
            _atomic_json(args.output/'cache-manifest.json',dict(records=records,whole_demo_split=True))
            contract=dict(source_commit=source,scope_sha256=args.scope_sha256,motor_sha256=interface['motor_sha256'],
                normalization_sha256=interface['normalization_sha256'],demo_manifest_sha256=args.demos_sha256,
                meaning='expert submitted-plan imitation on actual aligned rollouts; no RL/AD/DPT/ICRL admission',
                migration_sha256=args.migration_sha256,loss_mask='all known submitted plan slots; native executed mask separate')
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
            if len(state['records'])!=25:raise RuntimeError('Fixed control/teacher/development25episode plan incomplete')
            state.update(status='complete',stage='complete',summary=summarize(state['records']),
                actual_new_actions=budget.snapshot()['actual_actions']-18599,
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
