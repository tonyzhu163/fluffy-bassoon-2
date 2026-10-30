"""Explicit, unfunded native-batch14 production continuation preparation.

FULL preserves the qualified native phase's complete recovery/cursor and physical
journal. EPISODE reuses the qualified legacy-R3 warmstart. Neither old phase is
edited, and no model computation is performed by the metadata plan/clock CLI.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

FORMAT = 'source40_native_batch14_production_plan_v1'
MIGRATION = 'source40_native_batch14_production_migration_v1'
TOTAL_SECONDS = 158400  # Proposed44h per learner, not authorization.
PRODUCTION_SECONDS = 154800
FINAL_VALIDATION_SECONDS = 3600
FINAL_VALIDATION_ACTIONS = 30000
MAX_SLICE_SECONDS = 21540


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def native_parent(root, history_path, history_sha256):
    """Only the explicit successful qualified FULL phase, never search/select."""
    root = Path(root).resolve()
    names = {name: root / relative for name, relative in dict(
        index='experiment/index.json', recovery='experiment/learner/latest.pt',
        source_contract='experiment/source-contract.json', clock='worker-time.json', exit='exit-code.txt').items()}
    outer, index, clock = (read(names[key]) for key in ('source_contract', 'index', 'clock'))
    summary = index['summary']
    attempts = clock['attempts']
    if (names['exit'].read_text().strip() != '0' or not attempts
            or any(row.get('status') != 'finished' or row.get('exit_code') != 0 for row in attempts)
            or outer.get('format') != 'openpi_pi05_source40_native_batch14_ppo_v1'
            or outer.get('memory_mode') != 'full' or outer.get('source_only') is not True
            or outer.get('source_data_from_targets') is not False
            or index.get('source_contract') != outer or index.get('frozen_motor_parameters_unchanged') is not True
            or summary.get('updates') != 1 or summary.get('target_updates') != 1
            or summary.get('pending_complete_trials') != 0 or summary.get('complete_source_trials') != 14
            or summary.get('stop_reason') != 'requested_complete_batch_updates'
            or index['committed_task_exposure'].get('cursor') != 14
            or index['committed_task_exposure'].get('remaining') != 0
            or outer['warm_start']['path'] != str(Path(history_path).resolve())
            or outer['warm_start']['sha256'] != history_sha256
            or sha(names['recovery']) != summary['full_recovery_sha256']
            or clock['contract']['source_commit'] != outer['source_commit']):
        raise ValueError('Require the explicit qualified terminal FULL native recovery/cursor')
    actions = index['physical_cost']['total_action_charge']
    if (type(actions) is not int or actions < 1
            or summary.get('cumulative_source_actions') != outer['warm_start']['prior_actions'] + actions):
        raise ValueError('Native parent omitted its existing physical prefix')
    return dict(root=str(root), **{key + '_path': str(path) for key, path in names.items()},
        **{key + '_sha256': sha(path) for key, path in names.items()},
        native_actions=actions, native_updates=1, native_trials=14,
        cumulative_source_actions=summary['cumulative_source_actions'],
        worker_time_charge_seconds=sum(row['charged_seconds'] for row in attempts))


def make_plan(history_path, history_sha256, mode, *, native_root=None):
    """Read existing immutable receipts only; no torch/JAX imports or allocation."""
    history_path = Path(history_path).resolve()
    if sha(history_path) != history_sha256: raise ValueError('Legacy history checksum differs')
    history = read(history_path)
    if (history.get('format') != 'source40_batch14_warmstart_history_v1'
            or history.get('mode') != mode or mode not in ('full', 'episode')
            or history.get('maximum_cumulative_source_actions') != 1000000):
        raise ValueError('Require the matching qualified legacy history')
    # Full admission of legacy migration/validation clocks remains in the
    # existing warmstart loader, on the allocated production runtime.
    seen, total = set(), 0
    for row in history['charges']:
        selector = (row['artifact'], row.get('slot_mode'), tuple(row.get('action_key', [])))
        if row['id'] in seen or selector in seen or sha(row['artifact']) != row['sha256']:
            raise ValueError('Changed or duplicate inherited physical cost receipt')
        seen.update((row['id'], selector))
        document = read(row['artifact'])
        if row.get('slot_mode') is not None:
            actual = sum(cost['total_action_charge'] for slot, cost in zip(
                document['contract']['slots'], document['slot_costs'], strict=True) if slot['mode'] == mode)
        else:
            actual = document
            for key in row['action_key']: actual = actual[key]
        if type(actual) is not int or actual != row['actions']: raise ValueError('Inherited cost receipt differs')
        total += actual
    if total != history['prior_actions']: raise ValueError('Inherited source costs were omitted')
    if (mode == 'full') != (native_root is not None):
        raise ValueError('FULL requires its qualified native parent; EPISODE uses its original R3 directly')
    parent = native_parent(native_root, history_path, history_sha256) if native_root is not None else None
    old_native = 0 if parent is None else parent['native_actions']
    value = dict(format=FORMAT, status='PROPOSAL_NOT_FUNDED', source_only=True,
        source_data_from_targets=False, mode=mode,
        legacy_history_path=str(history_path), legacy_history_sha256=history_sha256,
        legacy_prior_actions=total, native_parent=parent, native_prior_actions=old_native,
        prior_source_resource_actions=total + old_native,
        maximum_cumulative_source_actions=1000000, final_source_validation_actions=FINAL_VALIDATION_ACTIONS,
        maximum_native_stream_actions=1000000-total-FINAL_VALIDATION_ACTIONS,
        maximum_new_actions=1000000-total-old_native-FINAL_VALIDATION_ACTIONS,
        updates_target=100, complete_trials_per_update=14, attempts_per_trial=4,
        seed=2, architecture=dict(width=128,layers=2,heads=4,context_length=300),
        optimizer=dict(learning_rate=1e-4,epochs=2,valid_token_weighting=True),
        maximum_production_gpu_seconds=PRODUCTION_SECONDS,
        final_source_validation_gpu_seconds=FINAL_VALIDATION_SECONDS, maximum_total_gpu_seconds=TOTAL_SECONDS,
        storage_allowance_bytes=64000000000,
        final_source_validation=dict(fixed_goals=8,states=[40,41],controller_episodes=64,native_episodes=16,
            controller_worst_actions=21760,native_worst_actions=5440,nominal_worst_actions=27200,retry_margin=2800,
            shared_native_cost='One unique joint charge ID if actually shared; conservative per-learner reserve otherwise'),
        cap_precedence='Whole14K4 worst-case batch admission and actual cost/time win;100 is a ceiling, not a guarantee',
        case_identity='Actual train states0..39, outcome-independent all40 cycle; scene seeds do not prove new physical states',
        initialization=('Lossless continuation of qualified native phase: iteration1/cursor14 retained'
            if parent else 'Qualified generic EPISODE R3 warmstart: inherited old cursor retained; native phase cursor0'))
    validate_plan(value)
    return value


def validate_plan(value):
    prior, native = value['legacy_prior_actions'], value['native_prior_actions']
    if (value.get('format') != FORMAT or value.get('status') != 'PROPOSAL_NOT_FUNDED'
            or value.get('source_only') is not True or value.get('source_data_from_targets') is not False
            or value.get('mode') not in ('full','episode') or value.get('updates_target') != 100
            or value.get('complete_trials_per_update') != 14 or value.get('attempts_per_trial') != 4
            or value.get('seed') != 2 or value.get('architecture') != dict(width=128,layers=2,heads=4,context_length=300)
            or value.get('optimizer') != dict(learning_rate=1e-4,epochs=2,valid_token_weighting=True)
            or value.get('maximum_cumulative_source_actions') != 1000000
            or value.get('final_source_validation_actions') != FINAL_VALIDATION_ACTIONS
            or value.get('maximum_production_gpu_seconds') != PRODUCTION_SECONDS
            or value.get('final_source_validation_gpu_seconds') != FINAL_VALIDATION_SECONDS
            or value.get('maximum_total_gpu_seconds') != TOTAL_SECONDS
            or any(type(x) is not int or x < 0 for x in (prior,native))
            or value.get('prior_source_resource_actions') != prior+native
            or value.get('maximum_native_stream_actions') != 1000000-prior-FINAL_VALIDATION_ACTIONS
            or value.get('maximum_new_actions') != 1000000-prior-native-FINAL_VALIDATION_ACTIONS
            or value['maximum_new_actions'] <= 0
            or ((value['mode']=='full') != (value.get('native_parent') is not None))
            or (value['mode']=='full' and value['native_parent'].get('native_actions')!=native)
            or value.get('storage_allowance_bytes')!=64000000000
            or (value['mode']=='episode' and native != 0)):
        raise ValueError('Production plan changed cumulative costs, reserve or reviewed batch14 algorithm')
    return value


def load_plan(path, expected_sha256):
    if sha(path) != expected_sha256: raise ValueError('Production plan checksum differs')
    value = validate_plan(read(path))
    if sha(value['legacy_history_path']) != value['legacy_history_sha256']:
        raise ValueError('Immutable legacy history changed')
    if value['native_parent'] is not None and native_parent(value['native_parent']['root'],
            value['legacy_history_path'],value['legacy_history_sha256']) != value['native_parent']:
        raise ValueError('Qualified native parent changed')
    return value


def source_contract(contract, plan_path, plan_sha256, *, actions, updates, mode, seed):
    plan = load_plan(plan_path,plan_sha256)
    if (actions != plan['maximum_native_stream_actions'] or updates != plan['updates_target']
            or mode != plan['mode'] or seed != plan['seed']
            or contract.get('warm_start',{}).get('path') != plan['legacy_history_path']
            or contract.get('warm_start',{}).get('sha256') != plan['legacy_history_sha256']
            or contract['warm_start']['prior_actions'] != plan['legacy_prior_actions']):
        raise ValueError('Production invocation differs from the exact phase/cumulative-source plan')
    return dict(contract, production_continuation=dict(path=str(Path(plan_path).resolve()),sha256=plan_sha256,
        prior_source_resource_actions=plan['prior_source_resource_actions'],native_prior_actions=plan['native_prior_actions'],
        final_source_validation_actions=FINAL_VALIDATION_ACTIONS,maximum_new_actions=plan['maximum_new_actions'],
        initialization=plan['initialization']))


def _copy_files(source, destination):
    """Merge immutable parent receipts beside newly charged setup receipts."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in sorted(source.rglob('*')):
        if not path.is_file(): continue
        relative=path.relative_to(source); target=destination/relative
        original=sha(path)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): shutil.copy2(path,target)
        if sha(target)!=original: raise ValueError('A copied immutable parent receipt differs')
        rows.append(dict(path=str(relative),sha256=original,bytes=path.stat().st_size))
    return rows


def prepare(output, plan_path, plan_sha256, contract, config, task_cycle, *, updates, actions, seed, mode):
    """Allocated runtime: exact non-contract native state, or existing R3 adapter."""
    import torch
    from .vla_rl2_batch import training_contract
    from .vla_source40_batch14_warmstart import prepare as warm_prepare, verify_inheritance, legacy_fingerprint
    from .warmstart import atomic_save
    plan=load_plan(plan_path,plan_sha256)
    source_contract(contract,plan_path,plan_sha256,actions=actions,updates=updates,mode=mode,seed=seed)
    output=Path(output)
    expected=training_contract(config,contract,updates=updates,max_total_actions=actions,
        seed=seed,mode=mode,learning_rate=1e-4,epochs=2)
    if plan['native_parent'] is None:
        warm_prepare(output,contract['warm_start'],contract,config,task_cycle,
            updates=updates,maximum_new_actions=actions,seed=seed,mode=mode)
        return
    parent=plan['native_parent']; proof=output/'production-migration.json'
    if (output/'latest.pt').is_file():
        receipt=read(proof)
        saved=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
        if (saved.get('contract')!=expected or receipt['plan_sha256']!=plan_sha256
                or receipt['parent_recovery_sha256']!=parent['recovery_sha256']
                or sha(output/'production-initial.pt')!=receipt['initial_checkpoint_sha256']
                or receipt['non_contract_sha256_before']!=receipt['non_contract_sha256_after']):
            raise ValueError('Native production resume differs from its lossless migration')
        verify_inheritance(saved)
        return
    if output.exists(): raise ValueError('Incomplete learner publication; preserve and diagnose before retry')
    parent_root=Path(parent['root'])/'experiment'
    old=torch.load(parent['recovery_path'],map_location='cpu',weights_only=False)
    verify_inheritance(old)
    changed_phase_fields={'source_commit','updates_target','maximum_total_actions','production_continuation'}
    old_source={k:v for k,v in old['contract']['source'].items() if k not in changed_phase_fields}
    new_source={k:v for k,v in contract.items() if k not in changed_phase_fields}
    if old_source!=new_source:
        raise ValueError('Production phase changed frozen motor/features/noise/task/method semantics')
    if (old['contract']['config']!=expected['config'] or old['contract']['mode']!=mode
            or old['contract']['optimizer']!=expected['optimizer'] or old['contract']['ppo']!=expected['ppo']
            or old['progress']['iteration']!=parent['native_updates']
            or old['runtime']['task_cycle']['remaining']!=0
            or old['runtime']['task_cycle']['cursor']!=old['progress']['completed_trials']):
        raise ValueError('Native parent architecture/Adam/batch semantics or cursor changed')
    before=legacy_fingerprint(old)
    migrated=copy.deepcopy(old); migrated['contract']=expected
    if legacy_fingerprint(migrated)!=before: raise RuntimeError('Production transition changed numerical state')
    journals={name:_copy_files(parent_root/name,output.parent/name) for name in ('execution-journal','task-resets')}
    staging=output.with_name('learner-production-staging-'+str(time.time_ns()))
    staging.mkdir()
    for name in ('warm-start-migration.json','warm-start-initial.pt','metrics.jsonl'):
        shutil.copy2(parent_root/'learner'/name,staging/name)
    traces=_copy_files(parent_root/'learner/traces',staging/'traces')
    initial=staging/'production-initial.pt'; initial_sha=atomic_save(initial,migrated)
    restored=torch.load(initial,map_location='cpu',weights_only=False)
    after=legacy_fingerprint(restored)
    if before!=after or sha(parent['recovery_path'])!=parent['recovery_sha256']:
        raise RuntimeError('Serialized phase transition changed parent or recovered numerical state')
    write(staging/'contract.json',expected)
    write(staging/'production-migration.json',dict(format=MIGRATION,plan_path=str(Path(plan_path).resolve()),
        plan_sha256=plan_sha256,parent=parent,parent_recovery_sha256=parent['recovery_sha256'],
        non_contract_sha256_before=before,non_contract_sha256_after=after,numerical_state_unchanged=True,
        restore_optimizer_steps=0,initial_checkpoint_sha256=initial_sha,
        retained_native_iteration=old['progress']['iteration'],retained_native_cursor=old['runtime']['task_cycle'],
        pending_complete_trials=len(old['progress']['pending_rollouts']),copied_journals=journals,copied_traces=traces,
        old_raw_paths=[str(parent_root/name) for name in ('raw','raw-rpc')],
        change='Only new immutable source/scientific phase contract; every other recovery leaf preserved'))
    atomic_save(staging/'latest.pt',restored)
    staging.rename(output)


def funding(path, plan, plan_sha256, source, run):
    record=read(path)
    if (record.get('status')!='APPROVED' or not record.get('authorization')
            or record.get('run_id')!=run or record.get('source_commit')!=source
            or record.get('plan_sha256')!=plan_sha256 or record.get('mode')!=plan['mode']
            or record.get('production_gpu_seconds')!=PRODUCTION_SECONDS
            or record.get('final_source_validation_gpu_seconds')!=FINAL_VALIDATION_SECONDS
            or record.get('maximum_total_gpu_seconds')!=TOTAL_SECONDS
            or record.get('maximum_native_stream_actions')!=plan['maximum_native_stream_actions']):
        raise ValueError('No exact approved44h production/final-validation funding record; proposal is not authority')
    return record


def begin(root, plan_path, plan_sha256, funding_path, source, run, schedule, *, seconds, resume, started):
    """Dedicated44h component clock; generic24h worker admission stays unchanged."""
    plan=load_plan(plan_path,plan_sha256); record=funding(funding_path,plan,plan_sha256,source,run)
    if (len(source)!=40 or any(c not in '0123456789abcdef' for c in source)
            or type(seconds)is not int or not 600<=seconds<=MAX_SLICE_SECONDS
            or not math.isfinite(started) or not run or any(not(c.isalnum() or c=='-') for c in run)):
        raise ValueError('Require exact source/run and600..21540-second bounded scheduler slices')
    contract=dict(operation='source40_native_batch14_production',source_commit=source,run_id=run,
        memory_mode=plan['mode'],updates=100,maximum_total_actions=plan['maximum_native_stream_actions'],
        warm_start_history_sha256=plan['legacy_history_sha256'],plan_path=str(Path(plan_path).resolve()),
        plan_sha256=plan_sha256,funding_record_path=str(Path(funding_path).resolve()),funding_record_sha256=sha(funding_path),
        schedule_file_sha256=sha(schedule),maximum_worker_seconds=PRODUCTION_SECONDS,
        maximum_component_seconds=TOTAL_SECONDS,final_source_validation_seconds_reserved=FINAL_VALIDATION_SECONDS)
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    with (root/'worker-time.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=root/'worker-time.json'
        if path.exists():
            clock=read(path)
            if not resume or clock['contract']!=contract or clock['worker_seconds']!=PRODUCTION_SECONDS:
                raise ValueError('Only exact pinned production plan/funding/source may resume')
        else:
            if resume: raise ValueError('Cannot reset a missing production clock')
            clock=dict(format='source40_batch14_production_time_v1',contract=contract,
                worker_seconds=PRODUCTION_SECONDS,attempts=[])
        charged=sum(row.get('charged_seconds',row['reserved_seconds']) for row in clock['attempts'])
        # Allow the last shorter slice, but never pretend the ceiling reset.
        seconds=min(seconds,PRODUCTION_SECONDS-charged-60)
        if seconds<600: raise RuntimeError('Original production time exhausted; final-validation reserve remains separate')
        attempt=root/'attempts'/f'{int(started*1000000)}-{len(clock["attempts"]):04d}'
        attempt.mkdir(parents=True,exist_ok=False)
        row=dict(path=str(attempt),started_unix=started,slice_seconds=seconds,reserved_seconds=seconds+60,
            status='started',scheduler_allowance_seconds=60)
        clock['attempts'].append(row);write(path,clock);write(attempt/'attempt.json',row)
    return attempt,int(started+seconds),seconds


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='operation',required=True)
    make=sub.add_parser('plan');make.add_argument('--legacy-history',type=Path,required=True)
    make.add_argument('--legacy-history-sha256',required=True);make.add_argument('--mode',choices=('full','episode'),required=True)
    make.add_argument('--native-root',type=Path);make.add_argument('--output',type=Path,required=True)
    clock=sub.add_parser('begin')
    for name in ('root','plan','funding','schedule'):clock.add_argument('--'+name,type=Path,required=True)
    for name in ('plan-sha256','source','run'):clock.add_argument('--'+name,required=True)
    clock.add_argument('--slice-seconds',type=int,default=21540);clock.add_argument('--started-unix',type=float,required=True)
    clock.add_argument('--resume',action='store_true')
    args=ap.parse_args(argv)
    if args.operation=='plan':
        if args.output.exists():raise ValueError('Plan output must be a new immutable file')
        value=make_plan(args.legacy_history,args.legacy_history_sha256,args.mode,native_root=args.native_root)
        args.output.parent.mkdir(parents=True,exist_ok=True);write(args.output,value)
        print(json.dumps(dict(path=str(args.output.resolve()),sha256=sha(args.output),status=value['status'])))
    else:
        attempt,stop,seconds=begin(args.root,args.plan,args.plan_sha256,args.funding,args.source,args.run,
            args.schedule,seconds=args.slice_seconds,resume=args.resume,started=args.started_unix)
        print(f'{attempt}\t{stop}\t{seconds}')


if __name__=='__main__':main()
