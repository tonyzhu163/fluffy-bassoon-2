"""Explicit legacy-R3→native batch14 phase; no launch or budget creation.

The inherited recovery is retained verbatim. Its policy/Adam/sampler/global and
typed motor RNG initialize a NEW task cycle; all new phase counters start0.
"""
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from .checkpoint import rng_state
from .vla_history import file_sha256
from .vla_source40_ppo import write_json
from .vla_source40_ppo_lineage import terminal_source
from .vla_source40_ppo_migration import fingerprint
from .vla_source40_schedule import SOURCE_SUITES
from .warmstart import atomic_save


FORMAT='source40_batch14_warmstart_history_v1'
MIGRATION='source40_legacy_r3_to_native_batch14_v1'


def read(path):
    return json.loads(Path(path).read_text())


def amount(document,entry):
    if entry.get('slot_mode') is not None:
        mode=entry['slot_mode']
        if (document.get('format')!='source40_frozen_ppo_validation_v1'
                or document.get('completed') is not True or document.get('weights_unchanged') is not True):
            raise ValueError('Source validation cost must be terminal, complete and frozen')
        return sum(cost['total_action_charge'] for slot,cost in zip(document['contract']['slots'],document['slot_costs'],strict=True)
            if slot['mode']==mode)
    value=document
    for key in entry['action_key']:
        value=value[key]
    return value


def time_history(value,lineage):
    rows=value.get('prior_time_history',{})
    clocks=rows.get('audited_clocks',[])
    if not clocks or len({row['id'] for row in clocks})!=len(clocks):
        raise ValueError('Require deduplicated inherited source GPU clock history')
    for row in clocks:
        if file_sha256(row['path'])!=row['sha256']:
            raise ValueError('Inherited source GPU clock changed')
        clock=read(row['path']);attempts=clock.get('attempts',[])
        if (not attempts or any(attempt.get('status')!='finished' for attempt in attempts)
                or sum(attempt['charged_seconds'] for attempt in attempts)!=row['charged_seconds']):
            raise ValueError('Inherited source GPU charge differs from its terminal clock')
    if not any(row['path']==lineage['worker_time_path']
            and row['sha256']==lineage['worker_time_sha256']
            and row['charged_seconds']==lineage['worker_time_charge_seconds'] for row in clocks):
        raise ValueError('Legacy R1/R2/R3 GPU charges were omitted')
    for row in rows.get('unavailable_clocks',[]):
        if (not row.get('reason') or file_sha256(row['evidence_path'])!=row['evidence_sha256']
                or row.get('charged_seconds') is not None):
            raise ValueError('Unavailable historical clocks must remain explicit, not invented as zero')
    return rows


def load_history(path,digest,mode,*,maximum_new_actions):
    if file_sha256(path)!=digest:raise ValueError('Warm-start history checksum differs')
    value=read(path)
    lineage=terminal_source(value['legacy_root'],mode,value['canonical_ppo_ledger'])
    if (value.get('format')!=FORMAT or value.get('mode')!=mode
            or value.get('maximum_cumulative_source_actions')!=1_000_000
            or value.get('legacy_lineage')!=lineage or not value.get('cost_completeness_statement')):
        raise ValueError('Require explicit complete legacy R3 lineage and cumulative source-cost history')
    seen,selectors,roles,total=set(),set(),set(),0
    for entry in value['charges']:
        selector=(entry['artifact'],entry.get('slot_mode'),tuple(entry.get('action_key',[])))
        if entry['id'] in seen or selector in selectors or file_sha256(entry['artifact'])!=entry['sha256']:
            raise ValueError('Repeated cost or changed immutable source cost artifact')
        seen.add(entry['id']);selectors.add(selector);roles.add(entry['role'])
        document=read(entry['artifact']);actual=amount(document,entry)
        if type(actual) is not int or actual<0 or actual!=entry['actions']:
            raise ValueError('Source history charge differs from its actual receipt')
        if entry['role']=='legacy_training':
            if (Path(entry['artifact']).resolve()!=Path(lineage['terminal_index_path']).resolve()
                    or actual!=lineage['cumulative_physical_cost']['total_action_charge']):
                raise ValueError('Legacy training cost lost its original cumulative physical history')
        elif entry['role']=='source_validation':
            if (entry.get('slot_mode')!=mode
                    or document['contract']['models'][mode]['recovery_sha256']!=lineage['recovery_sha256']):
                raise ValueError('Source-validation costs belong to a different frozen legacy model')
        elif entry['role']=='qualification':
            source=document.get('source_contract',{})
            if source.get('source_only') is not True or source.get('source_data_from_targets') is not False:
                raise ValueError('Qualification costs must be declared source-only, not target data')
        else:raise ValueError('Unknown source-cost role')
        total+=actual
    if (roles!={'legacy_training','source_validation','qualification'}
            or sum(row['role']=='legacy_training' for row in value['charges'])!=1
            or value.get('prior_actions')!=total or total+maximum_new_actions>1_000_000):
        raise ValueError('Prior training/validation/qualification plus new phase exceed or omit the1M cumulative cap')
    prior_time=time_history(value,lineage)
    return dict(path=str(Path(path).resolve()),sha256=digest,mode=mode,lineage=lineage,
        charges=value['charges'],prior_actions=total,maximum_cumulative_source_actions=1_000_000,
        cost_completeness_statement=value['cost_completeness_statement'],
        prior_time_history=prior_time)


def deduplicated_actions(histories):
    charges={}
    for history in histories:
        for row in history['charges']:
            if row['id'] in charges and charges[row['id']]!=row:
                raise ValueError('Shared source cost ID has conflicting receipts')
            charges[row['id']]=row
    return sum(row['actions'] for row in charges.values())


def legacy_fingerprint(saved):
    return fingerprint({key:value for key,value in saved.items() if key!='contract'})


def operational(saved,*,legacy=False):
    return dict(policy=saved['policy'],optimizer=saved['optimizer'],
        sampler=saved['environment_rng'] if legacy else saved['sampler'],rng=saved['rng'],
        motor_rng=saved['runtime']['motor_rng'])


def initial_payload(old,contract,history,task_cycle):
    """Pure schema transform; every inherited leaf remains inside legacy_recovery."""
    from .vla_rl2_batch import FORMAT as BATCH_FORMAT
    motor=old.get('runtime',{}).get('motor_rng',{})
    key=np.asarray(motor.get('key_data'))
    if (old.get('version')!=1 or old.get('boundary')!='between_trials'
            or old.get('runtime',{}).get('task_cycle',{}).get('remaining')!=0
            or old['contract']['mode']!=contract['mode'] or old['contract']['config']!=contract['config']
            or old['contract']['optimizer']!=contract['optimizer']
            or old['counters']['optimizer_steps']!=old['iteration']*2
            or motor.get('format')!='jax_motor_key_v1' or motor.get('implementation')!='threefry2x32'
            or key.shape!=(2,) or key.dtype!=np.uint32
            or not {'python','numpy','torch','cuda'}.issubset(old.get('rng',{}))):
        raise ValueError('Legacy recovery must have the exact architecture/Adam/mode and complete source-trial boundary')
    inherited=dict(format=MIGRATION,history=history,legacy_recovery=copy.deepcopy(old),
        legacy_non_contract_sha256=legacy_fingerprint(old),
        initial_operational_sha256=fingerprint(operational(old,legacy=True)))
    progress=dict(iteration=0,completed_trials=0,phase='collect',plan=None,pending_rollouts=[],pending_receipts=[],
        behavior_sha256=None,behavior_state_signature=None,
        counters=dict(environment_steps=0,settling_steps=0,rollout_forward_samples=0,
            update_forward_samples=0,verification_forward_samples=0,optimizer_steps=0),
        completed_batch_statistics=[],update_receipts=[],inheritance=inherited)
    saved=dict(format=BATCH_FORMAT,contract=contract,boundary='complete trial; optimizer idle or whole batch ready',
        policy=copy.deepcopy(old['policy']),optimizer=copy.deepcopy(old['optimizer']),
        sampler=copy.deepcopy(old['environment_rng']),rng=copy.deepcopy(old['rng']),
        runtime=dict(motor_rng=copy.deepcopy(old['runtime']['motor_rng']),task_cycle=copy.deepcopy(task_cycle)),progress=progress)
    if fingerprint(operational(saved))!=inherited['initial_operational_sha256']:
        raise RuntimeError('Schema migration changed an operational learner/RNG leaf')
    return saved


def prepare(output,history,source_contract,config,task_cycle,*,updates,maximum_new_actions,seed,mode):
    """Allocated caller only; old raw data and recoveries are referenced unchanged."""
    from .vla_rl2_batch import training_contract
    from .vla_source40_frozen_validation import source_admission
    output=Path(output)
    lineage=history['lineage']
    if file_sha256(lineage['recovery_path'])!=lineage['recovery_sha256']:
        raise ValueError('Immutable legacy recovery changed before migration')
    old=torch.load(lineage['recovery_path'],map_location='cpu',weights_only=False)
    # This invokes the existing R3 admission; it does not alter that gate.
    expected_tasks=[f'{suite}:{number:02d}' for suite in SOURCE_SUITES for number in range(10)]
    source_admission(old,mode,source_contract['feature_contract'],expected_tasks,
        schedule_sha256=source_contract['schedule_sha256'])
    contract=training_contract(config,source_contract,updates=updates,max_total_actions=maximum_new_actions,
        seed=seed,mode=mode,learning_rate=1e-4,epochs=2)
    checkpoint=output/'latest.pt'
    if checkpoint.exists():
        if not (output/'warm-start-migration.json').is_file():
            raise ValueError('Warm-start checkpoint has no committed migration proof; preserve and diagnose')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved.get('contract')!=contract:raise ValueError('Warm-start phase contract changed')
        verify_inheritance(saved)
        return
    if output.exists():
        raise ValueError('Warm-start output already exists without a complete migration; preserve and diagnose')
    output.mkdir();(output/'traces').mkdir()
    saved=initial_payload(old,contract,history,task_cycle)
    before=legacy_fingerprint(old)
    initial=output/'warm-start-initial.pt'
    initial_sha=atomic_save(initial,saved)
    restored=torch.load(initial,map_location='cpu',weights_only=False)
    after=legacy_fingerprint(restored['progress']['inheritance']['legacy_recovery'])
    if before!=after or file_sha256(lineage['recovery_path'])!=lineage['recovery_sha256']:
        raise RuntimeError('Serialized migration changed an inherited leaf or immutable parent')
    write_json(output/'contract.json',contract)
    write_json(output/'warm-start-migration.json',dict(format=MIGRATION,history=history,
        old_non_contract_sha256=before,new_inherited_non_contract_sha256=after,
        initial_operational_sha256=fingerprint(operational(restored)),numerical_state_unchanged=True,
        restore_optimizer_steps=0,new_phase_updates=0,new_phase_optimizer_steps=0,
        inherited_updates=old['iteration'],inherited_counters=old['counters'],
        new_phase_task_order='Explicit native-horizon interleaved40; old task cursor/exposure retained in legacy_recovery',
        initial_checkpoint_path=str(initial),initial_checkpoint_sha256=initial_sha))
    # Never expose a resumable learner before its immutable migration proof.
    atomic_save(checkpoint,restored)


def verify_inheritance(saved):
    inherited=saved['progress'].get('inheritance',{})
    old=inherited.get('legacy_recovery',{})
    if (inherited.get('format')!=MIGRATION or legacy_fingerprint(old)!=inherited.get('legacy_non_contract_sha256')
            or fingerprint(operational(old,legacy=True))!=inherited.get('initial_operational_sha256')
            or inherited.get('history')!=saved['contract']['source'].get('warm_start')):
        raise ValueError('Warm-start recovery lost an inherited state/counter/RNG/history leaf')
    return inherited


def verify_restoration(saved,policy,optimizer,sampler,runtime,output):
    inherited=verify_inheritance(saved)
    actual=dict(policy=policy.state_dict(),optimizer=optimizer.state_dict(),sampler=sampler.bit_generator.state,
        rng=rng_state(),runtime=runtime)
    expected=fingerprint(dict(**operational(saved),task_cycle=saved['runtime']['task_cycle']))
    after=fingerprint(dict(**operational(actual),task_cycle=actual['runtime']['task_cycle']))
    write_json(Path(output)/f'warm-restore-{time.time_ns()}.json',dict(passed=expected==after,
        expected_operational_sha256=expected,restored_operational_sha256=after,restore_optimizer_steps=0,
        inherited_non_contract_sha256=inherited['legacy_non_contract_sha256'],
        inherited_optimizer_steps=inherited['legacy_recovery']['counters']['optimizer_steps'],
        new_phase_optimizer_steps=saved['progress']['counters']['optimizer_steps']))
    if expected!=after:raise RuntimeError('Warm-start restore changed policy/Adam/sampler/global/motor RNG or task cursor')
