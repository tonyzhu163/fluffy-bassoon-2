"""Separate admission for an explicitly warm-started native batch14 phase.

The legacy R3 gate is unchanged. This adapter admits only a supplied terminal
phase, never searches checkpoints and never selects using target outcomes.
"""
import argparse
from dataclasses import asdict
import math
from pathlib import Path

import torch

from .vla_history import file_sha256
from .vla_rl2 import load_rl2_policy
from .vla_source40_batch14 import FORMAT as SOURCE_FORMAT, NATIVE_HORIZONS
from .vla_source40_batch14_warmstart import legacy_fingerprint, load_history, read, verify_inheritance
from .vla_source40_context_probe import policy_signature
from .vla_source40_frozen_validation import source_admission as legacy_admission
from .vla_source40_ppo import PI05_RECEIPT, physical_charge, write_json
from .vla_source40_schedule import SOURCE_SUITES
from .warmstart import checked_load


FORMAT='source40_batch14_frozen_models_v1'


def paths(root):
    root=Path(root).resolve()
    return dict(terminal_index_path=str(root/'experiment/index.json'),
        recovery_path=str(root/'experiment/learner/latest.pt'),
        candidate_path=str(root/'experiment/learner/candidate.pt'),
        source_contract_path=str(root/'experiment/source-contract.json'),
        migration_path=str(root/'experiment/learner/warm-start-migration.json'),
        initial_checkpoint_path=str(root/'experiment/learner/warm-start-initial.pt'),
        worker_time_path=str(root/'worker-time.json'),terminal_exit_path=str(root/'exit-code.txt'))


def terminal_spec(root,mode):
    """Metadata-only explicit supplied root; rejects unfinished/failed phases."""
    names=paths(root)
    if Path(names['terminal_exit_path']).read_text().strip()!='0':
        raise ValueError('Batch14 model must have a successful terminal exit')
    index,clock=read(names['terminal_index_path']),read(names['worker_time_path'])
    attempts=clock.get('attempts',[])
    charged=sum(row.get('charged_seconds',row['reserved_seconds']) for row in attempts)
    contract=read(names['source_contract_path']);summary=index.get('summary',{})
    if (not attempts or any(row.get('status')!='finished' for row in attempts)
            or attempts[-1].get('exit_code')!=0 or charged>clock['worker_seconds']
            or clock['contract'].get('source_commit')!=contract.get('source_commit')
            or clock['contract'].get('run_id')!=Path(root).name
            or clock['contract'].get('memory_mode')!=mode or contract.get('memory_mode')!=mode
            or clock['contract'].get('maximum_total_actions')!=contract.get('maximum_total_actions')
            or clock['contract'].get('warm_start_history_sha256')!=contract.get('warm_start',{}).get('sha256')
            or not contract.get('warm_start',{}).get('sha256')
            or clock['contract'].get('updates')!=contract.get('updates_target')
            or index.get('source_contract')!=contract
            or summary.get('stop_reason') not in ('requested_complete_batch_updates',
                'insufficient_actions_for_another_complete_batch','wall_stop_at_complete_trial_boundary')
            or summary.get('updates',0)<1 or summary.get('pending_complete_trials')!=0
            or index.get('frozen_motor_parameters_unchanged') is not True
            or not index.get('motor_parameters_before_sha256')
            or index['motor_parameters_before_sha256']!=index.get('motor_parameters_after_sha256')):
        raise ValueError('Require terminal exact-source/mode batch14 phase, whole batches and unchanged motor')
    return dict(terminal_root=str(Path(root).resolve()),mode=mode,
        **names,**{key.removesuffix('_path')+'_sha256':file_sha256(path) for key,path in names.items()},
        worker_time_charge_seconds=charged,maximum_worker_seconds=clock['worker_seconds'])


def admit(saved,index,spec,*,mode,feature_contract,tasks,schedule_sha256):
    source=saved['contract']['source'];progress=saved['progress'];contract=saved['contract']
    inherited=verify_inheritance(saved)
    history=load_history(source['warm_start']['path'],source['warm_start']['sha256'],mode,
        maximum_new_actions=source['maximum_total_actions'])
    if history!=inherited['history']:
        raise ValueError('Native phase differs from its admitted immutable source history')
    legacy=inherited['legacy_recovery']
    original=checked_load(history['lineage']['recovery_path'],history['lineage']['recovery_sha256'],weights_only=False)
    if legacy_fingerprint(original)!=legacy_fingerprint(legacy):
        raise ValueError('Inherited leaf fingerprint differs from the immutable original R3 recovery')
    config=legacy_admission(legacy,mode,feature_contract,tasks,schedule_sha256=schedule_sha256)
    native_keys=[f'{suite}:{number:02d}' for number in range(10) for suite in SOURCE_SUITES]
    cycle=saved.get('runtime',{}).get('task_cycle',{});exposure=cycle.get('exposure',{})
    counters=progress['counters'];summary=index['summary']
    if (source.get('format')!=SOURCE_FORMAT or source.get('source_only') is not True
            or source.get('source_data_from_targets') is not False or source.get('source_task_keys')!=native_keys
            or set(tasks)!=set(native_keys) or source.get('schedule_sha256')!=schedule_sha256
            or source.get('motor',{}).get('receipt_sha256')!=PI05_RECEIPT
            or source.get('feature_contract')!=feature_contract or source.get('memory_mode')!=mode
            or source.get('native_horizons')!=NATIVE_HORIZONS
            or source.get('complete_trials_per_update')!=14 or source.get('attempts_per_trial')!=4
            or contract.get('mode')!=mode or contract.get('config')!=asdict(config)
            or contract.get('optimizer')!=dict(name='Adam',learning_rate=1e-4)
            or contract.get('ppo',{}).get('epochs')!=2
            or progress['iteration']!=summary['updates'] or counters!=summary['counters']
            or not 1<=progress['iteration']<=contract['updates_target']
            or progress['completed_trials']!=progress['iteration']*14
            or progress['pending_rollouts'] or progress['pending_receipts'] or progress['plan'] is not None
            or counters['optimizer_steps']!=progress['iteration']*2
            or cycle.get('format')!='source40_ppo_task_cursor_v1' or cycle.get('remaining')!=0
            or cycle.get('cursor')!=progress['completed_trials'] or set(exposure)!=set(native_keys)
            or cycle!=index['committed_task_exposure'] or not saved['runtime'].get('motor_rng')
            or summary.get('full_recovery_sha256')!=spec['recovery_sha256']
            or summary.get('candidate_sha256')!=spec['candidate_sha256']):
        raise ValueError('Batch14 source architecture, whole-batch counters, cursor or artifact binding differs')
    if (sum(row['trials_started'] for row in exposure.values())!=progress['completed_trials']
            or any(row['episodes']!=row['trials_started']*4 for row in exposure.values())
            or sum(row['primitive_steps'] for row in exposure.values())!=counters['environment_steps']
            or sum(row['episodes'] for row in exposure.values())*10!=counters['settling_steps']):
        raise ValueError('New native source exposure lost its actual complete K4 trials')
    expected_steps=legacy['counters']['optimizer_steps']+counters['optimizer_steps']
    steps=[float(row['step']) for row in saved['optimizer']['state'].values() if 'step' in row]
    if not steps or any(not math.isfinite(step) or step!=expected_steps for step in steps):
        raise ValueError('Adam clock does not equal inherited plus new committed SGD steps')
    actual=physical_charge(Path(spec['terminal_root'])/'experiment/execution-journal')
    if (actual!=index['physical_cost'] or not 0<actual['total_action_charge']<=source['maximum_total_actions']
            or history['prior_actions']+actual['total_action_charge']>1_000_000
            or summary.get('cumulative_source_actions')!=history['prior_actions']+actual['total_action_charge']):
        raise ValueError('New physical journals or cumulative source costs differ')
    prior_exposure=legacy['runtime']['task_cycle']['exposure']
    if any(prior_exposure[key]['episodes']+exposure[key]['episodes']<4 for key in native_keys):
        raise ValueError('No actual cumulative all40 source coverage')
    return config,dict(initialization='Legacy R3 warm start; not fresh/random',history=history,
        inherited_updates=legacy['iteration'],inherited_counters=legacy['counters'],
        new_phase_updates=progress['iteration'],new_phase_counters=counters,
        inherited_task_cursor=legacy['runtime']['task_cycle'],new_phase_task_cursor=cycle,
        cumulative_actual_source_tasks=40,new_phase_actual_source_tasks=sum(row['episodes']>0 for row in exposure.values()),
        cumulative_source_actions=history['prior_actions']+actual['total_action_charge'],new_phase_physical_cost=actual,
        new_phase_worker_time_charge_seconds=spec['worker_time_charge_seconds'])


def load_models(manifest_path,expected_sha256,*,motor,tasks,schedule_sha256,output,device):
    """Same callable interface as the legacy loader; either/both modes explicit."""
    if file_sha256(manifest_path)!=expected_sha256:raise ValueError('Batch14 model manifest checksum differs')
    manifest=read(manifest_path)
    if manifest.get('format')!=FORMAT or not manifest.get('models') or not set(manifest['models'])<= {'full','episode'}:
        raise ValueError('Require a fixed explicit native batch14 model manifest')
    models,provenance={},{}
    for mode,spec in manifest['models'].items():
        if spec!=terminal_spec(spec['terminal_root'],mode):raise ValueError('Terminal batch14 model receipts changed')
        saved=checked_load(spec['recovery_path'],spec['recovery_sha256'],weights_only=False)
        config,detail=admit(saved,read(spec['terminal_index_path']),spec,mode=mode,
            feature_contract=motor.feature_encoder.contract,tasks=tasks,schedule_sha256=schedule_sha256)
        policy,metadata=load_rl2_policy(spec['candidate_path'],expected_sha256=spec['candidate_sha256'],device=device)
        if (policy.config!=config or metadata.get('contract')!=saved['contract']
                or metadata.get('full_recovery_sha256')!=spec['recovery_sha256']
                or metadata.get('updates')!=saved['progress']['iteration']
                or metadata.get('counters')!=saved['progress']['counters']
                or any(not torch.equal(value.detach().cpu(),saved['policy'][key]) for key,value in policy.state_dict().items())):
            raise ValueError('Candidate export differs from the admitted complete source recovery')
        migration=read(spec['migration_path'])
        if (migration.get('restore_optimizer_steps')!=0 or migration.get('numerical_state_unchanged') is not True
                or migration.get('old_non_contract_sha256')!=migration.get('new_inherited_non_contract_sha256')
                or migration.get('initial_checkpoint_sha256')!=spec['initial_checkpoint_sha256']):
            raise ValueError('Warm-start migration proof differs')
        models[mode]=policy
        provenance[mode]=dict(policy_path=spec['candidate_path'],policy_sha256=spec['candidate_sha256'],
            recovery_path=spec['recovery_path'],recovery_sha256=spec['recovery_sha256'],
            tensor_sha256=policy_signature(policy),source_commit=saved['contract']['source']['source_commit'],
            reset_identity=saved['contract']['source']['reset_identity'],
            terminal_lineage=spec,source_updates=saved['progress']['iteration'],source_counters=saved['progress']['counters'],
            selection='Fixed terminal source phase; no target checkpoint selection',**detail)
    if file_sha256(manifest_path)!=expected_sha256:raise ValueError('Batch14 model manifest changed during loading')
    return models,provenance


def main(argv=None):
    parser=argparse.ArgumentParser(description='Build a metadata manifest from explicit terminal warm-source phases; no models run')
    parser.add_argument('--full-root',type=Path);parser.add_argument('--episode-root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    models={mode:terminal_spec(root,mode) for mode,root in [('full',args.full_root),('episode',args.episode_root)] if root}
    if not models or args.output.exists():raise ValueError('Require one/both explicit terminal roots and a fresh manifest')
    write_json(args.output,dict(format=FORMAT,models=models,selection='Fixed terminal phases; no target validation selection'))


if __name__=='__main__':main()
