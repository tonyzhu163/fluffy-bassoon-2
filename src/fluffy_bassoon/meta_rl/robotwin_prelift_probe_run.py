"""Two predeclared phone04 counterfactuals; source owner allocates, never here."""
from copy import deepcopy
import argparse
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
from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter,_atomic_json
from fluffy_bassoon.rl.robotwin_noise import RobotwinNoiseEnv,load_benchmark_episode_limits
from fluffy_bassoon.rl.robotwin_steering import validate_interface
from .checkpoint import rng_state,restore_rng
from .warmstart import atomic_save
from .robotwin_demo_capacity import file_sha,bound_json,read_episode,validate_demo_manifest
from .robotwin_reactive_capacity import CONFIG,TASK,validate_capacity_locked
from .robotwin_reactive_fit import make_heads
from .robotwin_steering_screen import read_bound,validate_admission,verify_source_conventions,run_complete_episode,qualify_real_motor,paired_reset_fields
from .robotwin_prelift_probe import FORMAT,CONDITIONS,PreliftControl,probe_schedule,run_pair,preserve_global_rng,compare_pair

RESULT_SHA='ba775b4fcbb3029120927ecd59461fabb1cccd7e95326f8d9f3a6f731ea77d3f'
RECOVERY_SHA='0403db7c2bd0c4e7566cfdbb5a6404409a0d45a0eb85df4c0972a2bbcd58a153'
DIAGNOSIS_SHA='933b3f4452447a47a266d424c5b1bc0cd36c670071635173ea15c2c87ffafd0e'
FROZEN='721a7ed12c70d36e08882b02d8afee709286f6f27e72482570d7ac7451c0ae41'
SUFFIX_RESULT_SHA='73bd6483cd5abf8093fc46d8eb5fb116b745d2010e1e922b9f7d658d4625cea6'
SUFFIX_TERMINAL_SHA='932d8b4402db32b86871a73b7a80afe0cd15b69d34cc5daae4cb92229d936347'
SUFFIX_RECOVERY_SHA='5c0829d3e5ca00427bc589449053c536855b165b5ffd686d6fe26c65985cb883'
SUFFIX_TIP='ed0d9cc4160e4d3ad120a9ec51218982d6bd72a8f9ba2143b02e86264eaae779'


def validate_head_kind(head,reference_sha):
    if (reference_sha!=SUFFIX_RESULT_SHA or head.get('source_commit')!='aefd5563b128cb7ea8e017d4664e1df0ba22a160'
            or head.get('format')!='original_a1_event_fixed_suffix_diagnostic_v1'
            or head.get('status')!='complete_diagnostic_coverage'
            or head.get('predecessor_result',{}).get('sha256')!=RESULT_SHA
            or [r['spec']['teacher_id'] for r in head['records']]!=[f'phone-{i:02d}' for i in range(5,12)]
            or any(not r['complete'] for r in head['records']) or head.get('cache_created') is not False
            or head.get('fit_updates')!=0 or head.get('source_student_admitted') is not False
            or head.get('all_twelve_teacher_gate_passed') is not False or head.get('frozen_after')!=FROZEN):
        raise ValueError('Require actual COMPLETE original05-11 suffix, never running/failed/stale/foreign learning head')
    if head.get('pending') is not None or head.get('cleanup_error') is not None:
        raise ValueError('Pending/unclean accounting predecessor cannot release a new diagnostic')


def validate_accounting(context,head,allocation,*,run_id):
    """Exact settled B head; the old failed04 recovery is evidence only."""
    validate_head_kind(head,context['accounting_result']['sha256'])
    terminal=bound_json(context['accounting_terminal'])
    if (context.get('format')!='robotwin_prelift_actual_execution_context_v1'
            or context['accounting_terminal']['sha256']!=SUFFIX_TERMINAL_SHA
            or bound_json(context['accounting_result'])!=head
            or terminal['source_commit']!=head['source_commit']
            or terminal['cumulative_a1_gpu_seconds']!=allocation['prior_a1_gpu_seconds']
            or context['prior_a1_gpu_seconds']!=allocation['prior_a1_gpu_seconds']
            or context['prior_phase_gpu_seconds']!=allocation['prior_phase_gpu_seconds']
            or type(terminal['charged_gpu_seconds']) is not int or terminal['charged_gpu_seconds']<1
            or terminal['exit_code']!=0 or terminal['cumulative_a1_gpu_seconds']!=2652
            or terminal['canonical_actual_actions']!=21522 or terminal['canonical_charged_actions']!=21522
            or terminal['canonical_journal_events']!=1430 or terminal['canonical_journal_tip']!=SUFFIX_TIP
            or terminal['result']['sha256']!=SUFFIX_RESULT_SHA):
        raise ValueError('Exact current terminal/accounting/cumulative cost refs required, no historical cost rollback')
    ledger=allocation['action_ledger'];old=head['actions'];root=Path(ledger['root']).resolve(strict=True)
    count=old['journal_events']
    if (ledger!=head['action_ledger'] or root.parent!=Path(allocation['phase_output_root']).resolve(strict=True)
            or old['actual_actions']!=21522 or old['charged_actions']!=21522
            or old['pending_actions'] or old['uncertain_actions'] or old['unresolved_decisions']
            or run_id in old['runs'] or type(count) is not int or count!=1430 or old['journal_sha256']!=SUFFIX_TIP
            or sorted(p.name for p in (root/'events').glob('*.json'))!=[f'{i:010d}.json' for i in range(count)]
            or json.loads((root/'events/0000000000.json').read_text())['header']!=old['header']
            or json.loads((root/'events'/f'{count-1:010d}.json').read_text())['sha256']!=old['journal_sha256']):
        raise ValueError('Only latest closed original root/header/tip/costs; no overlap/replacement ledger')


def validate_fork_recovery(saved,head):
    """Use latest CLOSED replay state, not historical04 state as continuation."""
    control=saved.get('control') or {}
    if (saved.get('state')!=head or saved.get('fits')!={} or saved.get('fit_recovery_refs')!={}
            or control!=head['closed_control_recoveries'][-1]
            or control.get('elapsed_steps')!=head['records'][-1]['primitive_steps']
            or control.get('mode')!='action' or control.get('motor_rng') is None
            or control.get('source_cursor',{}).get('closed') is not True
            or set(saved.get('global_rng',{}))!={'python','numpy','torch','cuda'}
            or saved['global_rng']['cuda'] is None):
        raise ValueError('Exact latest closed full-control/private/global RNG recovery with zero fits required')


def specs(profile,row):
    if (profile['format']!=FORMAT or profile['conditions']!=list(CONDITIONS)
            or profile['maximum_complete_episodes']!=2 or profile['maximum_new_actions']!=800
            or profile['maximum_gpu_seconds']!=600 or profile['teacher_id']!='phone-04'
            or profile['bank_candidate_priority']!=['close_endpoint','hold_current']
            or profile['diagnosis']['sha256']!=DIAGNOSIS_SHA
            or profile['historical_result']['sha256']!=RESULT_SHA
            or profile['historical_recovery']['sha256']!=RECOVERY_SHA
            or row['id']!='phone-04' or row['canonical_index']!=4 or row['reset_seed']!=4):
        raise ValueError('Exact original04/two-condition/small bounded profile required')
    return [dict(id='prelift04-'+condition,task=TASK,case=4,arm='action',method='prelift_'+condition,
        condition=condition,coefficients=[0.]*4,stage='prelift_closure_diagnostic',horizon=400,
        teacher_id='phone-04',reset_seed=4,expected_teacher_seed=4,motor_seed=20261205,
        paired_instruction=[row['instruction']]) for condition in CONDITIONS]


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('profile','execution-context','scope','allocation','interface','joint-conventions','demos'):
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha256',required=True)
    for name in ('model-path','norm-stats','robotwin-root','rlinf-root','gpu-probe','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--run-id',required=True)
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    if sys.platform!='linux':raise RuntimeError('Actual allocated Linux GPU only')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if source!=a.source_commit or subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Exact clean deployed source required')
    profile=read_bound(a.profile,a.profile_sha256);context=read_bound(a.execution_context,a.execution_context_sha256)
    scope=read_bound(a.scope,a.scope_sha256);allocation=read_bound(a.allocation,a.allocation_sha256)
    interface=validate_interface(read_bound(a.interface,a.interface_sha256))
    validate_admission(scope,allocation,scope_sha=a.scope_sha256,source=source,output=a.output,hostname=socket.gethostname(),interface=interface)
    seconds=allocation['reserved_gpu_seconds'];start=allocation['scheduler_start_unix'];deadline=allocation['driver_deadline_unix']
    if (allocation.get('mode')!=FORMAT or allocation.get('run_id')!=a.run_id
            or allocation.get('capacity_action_bound')!=800 or type(seconds) is not int or not 1<=seconds<=600
            or any(type(x) not in (int,float) or not math.isfinite(x) for x in (start,deadline))
            or not start<=time.time()<deadline<=start+seconds-60
            or allocation['prior_a1_gpu_seconds']!=2652):raise ValueError('One<=600 all-in GPU-s/800actions on actual clock; retained historical costs')
    if (file_sha(a.model_path/'model.safetensors')!=interface['motor_sha256']
            or file_sha(a.norm_stats)!=interface['normalization_sha256']
            or a.joint_conventions_sha256!=interface['joint_convention_sha256']):raise ValueError('Original frozen motor/norm/interface required')
    verify_source_conventions(read_bound(a.joint_conventions,a.joint_conventions_sha256),interface,dict(robotwin=a.robotwin_root,rlinf=a.rlinf_root))
    limits=load_benchmark_episode_limits(a.robotwin_root,[TASK]);probe=json.loads(a.gpu_probe.read_text())
    if limits[TASK]!=400 or probe.get('status')!='passed' or probe.get('hostname')!=socket.gethostname() or len(probe.get('devices',[]))!=1:
        raise ValueError('Actual one GPU CUDA/native400 probe required')
    demos=validate_demo_manifest(read_bound(a.demos,a.demos_sha256));paths=demos['actual_whole_files']
    if (a.demos_sha256!=profile['demos_sha256'] or set(paths)!={r['id'] for r in demos['episodes']}
            or demos['motor_sha256']!=interface['motor_sha256'] or demos['normalization_sha256']!=interface['normalization_sha256']):
        raise ValueError('Retain original all12/8TRAIN4VAL and exact motor/norm')
    for ref in paths.values():
        path=Path(ref['path'])
        if not path.is_absolute() or path.is_symlink() or file_sha(path)!=ref['sha256']:raise ValueError('Original file changed')
    for ref in bound_json(demos['alignment_qualification'])['source_files']:
        path=a.robotwin_root/ref['path']
        if not path.resolve().is_relative_to(a.robotwin_root.resolve()) or file_sha(path)!=ref['sha256']:
            raise ValueError('Qualified author conventions differ from actual pinned source')
    row=next(r for r in demos['episodes'] if r['id']=='phone-04');plan_specs=specs(profile,row)
    initial,targets=read_episode(paths['phone-04']['path'],row,demos['schema']);plan=probe_schedule(targets)
    if plan!=profile['expected_schedule']:raise ValueError('Predeclared real04 schedule changed')
    diagnosis=bound_json(profile['diagnosis'])
    if diagnosis['status']!='audit_complete' or diagnosis['full_proposed_and_submitted_plans_exact'] is not True:
        raise ValueError('Actual qualified CPU diagnosis must precede new mechanism test')
    head=bound_json(context['accounting_result']);validate_accounting(context,head,allocation,run_id=a.run_id)
    origin_ref=context['accounting_recovery'];origin_path=Path(origin_ref['path'])
    terminal=bound_json(context['accounting_terminal'])
    if (origin_ref['sha256']!=SUFFIX_RECOVERY_SHA or not origin_path.is_absolute() or origin_path.is_symlink()
            or file_sha(origin_path)!=origin_ref['sha256']
            or terminal['full_recovery']['path']!=str(origin_path)
            or terminal['full_recovery']['sha256']!=origin_ref['sha256']
            or terminal['full_recovery']['bytes']!=origin_path.stat().st_size):
        raise ValueError('Exact current terminal-bound full recovery required')
    saved=torch.load(origin_path,map_location='cpu',weights_only=False);validate_fork_recovery(saved,head)
    a.output.mkdir(parents=True,exist_ok=False)
    state=dict(format=FORMAT,status='running',stage='admission',source_commit=source,profile_sha256=a.profile_sha256,
        execution_context_sha256=a.execution_context_sha256,plan=profile,records=[],pending=None,
        action_ledger=deepcopy(allocation['action_ledger']),actions=deepcopy(head['actions']),
        historical_failure_recovery=profile['historical_recovery'],acting_fork_recovery=origin_ref,
        accounting_control_recovery=deepcopy(saved['control']),
        accounting_head=context['accounting_result'],accounting_terminal=context['accounting_terminal'],
        closed_control_recoveries=[],chunk_audits=[],physical_reset_audits=[],post_branch_rng_refs=[],
        cache_created=False,fit_updates=0,all_twelve_teacher_gate_passed=False,source_student_admitted=False)
    _atomic_json(a.output/'plan.json',dict(profile,source_commit=source,execution_context=context))
    control=None;env=None;prior_actions=head['actions']['actual_actions']
    def guard():
        if time.time()+60>=deadline:raise TimeoutError('Finite paired probe deadline')
    def storage():
        guard();size=0
        for directory,dirs,files in os.walk(allocation['phase_output_root'],followlinks=False):
            if any((Path(directory)/n).is_symlink() for n in (*dirs,*files)):raise RuntimeError('No storage symlink')
            size+=sum((Path(directory)/n).stat().st_size for n in files)
        if size+(64<<20)>64<<30:raise RuntimeError('Original aggregate64GiB cap')
        return size
    def persist():
        atomic_save(a.output/'recovery.pt',dict(state=deepcopy(state),fits={},fit_recovery_refs={},
            control=None if control is None else control.snapshot(),global_rng=rng_state()))
    ledger=allocation['action_ledger']
    try:
        with CampaignBudget(ledger['root'],campaign_id=ledger['campaign_id'],total_actions=300000,
                source_caps=ledger['source_caps'],phase_caps=ledger['phase_caps'],contract=ledger['contract'],run_id=a.run_id,source_commit=source) as budget:
            try:
                validate_capacity_locked(budget,head)
                if budget.remaining(phase='a1',method='prelift_hold_current')<800:raise ValueError('Full two-episode800action admission required')
                storage();env=RobotwinNoiseEnv(model_path=a.model_path,norm_stats=a.norm_stats,robotwin_root=a.robotwin_root,
                    rlinf_root=a.rlinf_root,output_dir=a.output/'runtime',action_horizon=32,num_steps=5,task_episode_limits=limits)
                heads=make_heads(env.backend,interface,CONFIG)
                writer=RobotwinCorpusWriter(a.output/'corpus',dict(role=FORMAT,source_commit=source,motor_sha256=interface['motor_sha256'],
                    normalization_sha256=interface['normalization_sha256'],dataset_sha256=a.demos_sha256,scope_sha256=a.scope_sha256))
                control=PreliftControl(env,heads,writer,interface)
                control.probe_profile_ref=dict(path=str(a.profile),sha256=a.profile_sha256)
                if env.frozen_fingerprint()!=FROZEN:raise ValueError('Actual frozen motor signature changed')
                restore_rng(saved['global_rng']);pair_rng=rng_state()
                fork_path=a.output/'paired-global-rng.pt';digest=atomic_save(fork_path,dict(global_rng=pair_rng,origin=origin_ref,
                    convention='Two fresh native resets independently fork same existing four global RNG domains; authoritative final branch is close_endpoint'))
                state['pair_rng_ref']=dict(path=str(fork_path),sha256=digest);persist()
                def episode(spec,*,qualify):
                    if deadline-time.time()<240:raise TimeoutError('Insufficient complete native episode clock')
                    if len(state['records'])>=2:raise RuntimeError('One predeclared pair only')
                    storage();control.select_probe(targets,dict(paths['phone-04'],alignment=demos['alignment_qualification']),spec['condition'])
                    if state['records']:spec=dict(spec,**paired_reset_fields(state['records'][0]))
                    state['pending']=deepcopy(spec);persist()
                    def before(current,actual_spec):
                        guard();state['physical_reset_audits'].append(current.publish_reset_audit(a.output/'physical-audits'))
                        raw=current.env.backend._raw_obs['states'];actual=(raw.detach().cpu().numpy() if isinstance(raw,torch.Tensor) else np.asarray(raw))[0]
                        if current.last_reset_info['actual_seed']!=4 or not np.allclose(actual,initial[0],atol=1e-5,rtol=0):raise ValueError('Original04 seed/initial public drives required')
                        if qualify:
                            with preserve_global_rng():state['qualification']=qualify_real_motor(current,actual_spec,a.output)
                            state['qualification_global_rng_preserved']=True
                    def on_chunk(_):
                        guard();state['chunk_audits'].append(control.publish_chunk_audit(a.output/'chunk-audits'))
                        if budget.snapshot()['charged_actions']-prior_actions>800:raise RuntimeError('Paired800action cap')
                        storage()
                    receipt=run_complete_episode(control,spec,budget,run_id=a.run_id,phase='a1',before_step=before,on_chunk=on_chunk)
                    if qualify:
                        if receipt['primitive_steps']<=0:raise RuntimeError('Qualification needs its charged complete control episode')
                        state['qualification'].update(status='passed',actual_actions=receipt['primitive_steps'],complete_real_episode=spec['id'])
                        _atomic_json(a.output/'gpu-qualification.json',state['qualification'],replace=True)
                    small={k:v for k,v in receipt.items() if k!='control_recovery'}
                    small.update(prelude_signatures=deepcopy(control.prelude),probe_current14=None if control.probe_current14 is None else control.probe_current14.tolist(),
                        probe_target14=None if control.replay.probe_target is None else control.replay.probe_target.detach().cpu().tolist())
                    state['records'].append(small);state['closed_control_recoveries'].append(receipt['control_recovery'])
                    branch=a.output/(spec['condition']+'-post-rng.pt');branch_sha=atomic_save(branch,dict(global_rng=rng_state(),control=receipt['control_recovery']))
                    state['post_branch_rng_refs'].append(dict(path=str(branch),sha256=branch_sha));state['pending']=None
                    state['actions']=budget.snapshot();_atomic_json(a.output/(spec['id']+'.json'),small);persist();return small
                run_pair(plan_specs,episode,pair_rng)
                comparison,candidate=compare_pair(state['records'],profile['bank_candidate_priority'])
                state.update(status='complete_diagnostic_pair',stage='complete',actual_new_actions=budget.snapshot()['actual_actions']-prior_actions,
                    pair_comparison=comparison,bank_candidate_04=candidate,
                    frozen_after=env.frozen_fingerprint(),storage_bytes=storage())
                if state['frozen_after']!=FROZEN:raise RuntimeError('Final frozen motor changed')
            finally:state['actions']=budget.snapshot()
    except BaseException as error:
        state.update(status='blocked_failure',failure=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        closed=None if control is None else control.snapshot();global_rng=rng_state()
        try:
            if env is not None:env.close()
        except BaseException as error:
            state.update(status='blocked_failure',cleanup_error=dict(type=type(error).__name__,message=str(error)));raise
        finally:
            atomic_save(a.output/'recovery.pt',dict(state=deepcopy(state),fits={},fit_recovery_refs={},control=closed,global_rng=global_rng))
            _atomic_json(a.output/'result.json',state,replace=True)


if __name__=='__main__':main()
