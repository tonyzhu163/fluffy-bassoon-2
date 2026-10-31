"""Bind the qualified48-query source-SAC diagnostic to the NEW top-up escrow.

Only budget/source-bank admission changes; the qualified query implementation
is reused unchanged. Preparing an alias is not GPU/resource authorization.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

from . import vla_source40_topup as topup
from .vla_source40_campaign import ALL_TASKS, VALIDATION_TASKS


FORMAT = 'source40_topup_final_teacher_competence_alias_v1'
ADMISSION='explicit_plan_retained_topup_parents_v1'
NOISE='32D tied across native H10; warmup unbounded N(0,I), then SAC clipped[-1,1]; no rescaling'


def producer(plan,ledger,key,*,allow_retained_parents=False):
    """Bind one actual terminal producer; prior pins require an exact retained row."""
    slot=key+'/train';selected=next(row for row in plan['slots'] if row['slot']==slot)
    leaf=ledger['leaves'][slot];root=Path(leaf['root']).resolve();exp=root/'experiment'
    retained=not selected['launchable']
    if retained and root!=Path(selected['parent']['root']).resolve():
        raise ValueError('Retained canonical leaf differs from the production plan parent')
    if retained and not allow_retained_parents:
        raise ValueError('Retained prior producer requires explicit retained-parent admission')
    with (root/'launcher.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        index,contract=topup.read(exp/'index.json'),topup.read(exp/'contract.json')
        continuation=contract.get('continuation',{});actual=index.get('continuation',{})
        pin,phase=contract.get('source_commit'),continuation.get('plan_sha256')
        if retained:
            parent=selected['parent'];headroom=selected['maximum_total_actions']-parent['actions']
            expected=dict(source_commit=pin,contract_sha256=topup.sha(exp/'contract.json'),index_sha256=topup.sha(exp/'index.json'),
                recovery_sha256=topup.sha(exp/'source-recovery.pkl'),schedule_sha256=topup.sha(root/'schedule.json'),
                actions=index['total_action_budget_charge'],source_updates=index['source_updates'],chunks=index['chunks'],
                next_case=index['case_index'],episodes=len(index['episode_receipts']))
            if not 0<=headroom<1650 or any(parent.get(name)!=value for name,value in expected.items()):
                raise ValueError('Retained parent proof differs from the explicit terminal producer/cost/cursor')
        elif phase!=plan['plan_sha256'] or pin!=plan['source_commit']:
            raise ValueError('Fresh source producer differs from the latest production source/plan')
        elif (continuation.get('parent')!=selected['parent']
                or leaf.get('parent_root')!=selected['parent']['root']
                or leaf.get('inherited_action_charge')!=selected['parent']['actions']):
            raise ValueError('Fresh producer migration differs from its explicit production parent')
        clock_path=root/'worker-time.json';clock=topup.read(clock_path)
        migration_path=exp/'migration.json';migration=topup.read(migration_path)
        cases=contract.get('cases',[]);cost=index.get('total_action_budget_charge',-1)
        quota_done=index.get('production_quota_exhausted') is True and cost+1640>selected['maximum_total_actions']
        bank_done=index.get('case_index')==len(cases)==512
        if ((root/'exit-code.txt').read_text().strip()!='0' or index.get('completed') is not True
                or index.get('collection_qualified') is not True or index.get('episode_in_case')!=0
                or index.get('slice_completed') is not True or not (quota_done or bank_done)
                or not 0<=cost<=selected['maximum_total_actions'] or leaf.get('maximum_total_actions')!=selected['maximum_total_actions']
                or leaf.get('source_commit')!=pin or leaf.get('plan_sha256')!=phase
                or continuation.get('format')!=topup.FORMAT or actual.get('format')!=topup.FORMAT
                or actual.get('plan_sha256')!=phase or actual.get('parent')!=continuation.get('parent')
                or actual.get('new_optimizer_updates',0)<=0 or actual.get('new_complete_episodes',0)<4
                or actual.get('motor_parameters_unchanged') is not True
                or actual.get('all_new_behavior_postwarmup') is not True
                or migration.get('format')!=topup.FORMAT or migration.get('plan_sha256')!=phase
                or migration.get('parent')!=continuation.get('parent') or len(migration.get('numerical_state_sha256',''))!=64
                or len(pin or '')!=40 or any(char not in '0123456789abcdef' for char in pin)
                or len(phase or '')!=64 or any(char not in '0123456789abcdef' for char in phase)
                or contract.get('source_only') is not True or contract.get('source_data_from_targets') is not False
                or contract.get('partition')!='train' or contract.get('task',{}).get('task_key')!=key
                or index.get('source_only') is not True or index.get('task_key')!=key or index.get('partition')!='train'
                or contract.get('horizon')!=400 or contract.get('chunk')!=10 or contract.get('noise_contract')!=NOISE
                or index.get('schedule_sha256')!=contract.get('schedule_sha256')
                or contract.get('maximum_total_actions')!=selected['maximum_total_actions']
                or contract.get('motor',{}).get('receipt_sha256')!=topup.PI05_RECEIPT
                or contract.get('motor',{}).get('weights_frozen') is not True
                or contract.get('feature_contract',{}).get('checkpoint_receipt_sha256')!=topup.PI05_RECEIPT
                or any(contract.get('source_recipe',{}).get(k)!=v for k,v in [('warmup_chunks',500),('batch_size',256),('utd',20)])
                or contract['source_recipe'].get('gamma_per_primitive')!=.999
                or topup.content_hash(contract)!=index.get('source_contract_sha256')
                or len(cases)!=512 or any(type(case.get('initial_state')) is not int or not 0<=case['initial_state']<40 for case in cases)
                or not clock.get('attempts') or any(row.get('status')!='finished' for row in clock['attempts'])
                or clock['attempts'][-1].get('exit_code')!=0
                or sum(row['charged_seconds'] for row in clock['attempts'])>clock['worker_seconds']
                or clock.get('contract',{}).get('source_commit')!=pin or clock['contract'].get('plan_sha256')!=phase
                or clock['contract'].get('slot')!=slot or clock['contract'].get('run_id')!=root.name):
            raise ValueError('Require an exact canonical quota-complete top-up actor and migration/recipe/motor/clock lineage')
        parent=continuation['parent'];parent_contract_path=Path(parent['experiment'])/'contract.json'
        if topup.sha(parent_contract_path)!=parent['contract_sha256']:
            raise ValueError('Immutable producer ancestor contract differs')
        parent_contract=topup.read(parent_contract_path)
        keys=('task','partition','cases','history_id','source_recipe','learner_seed','horizon','chunk','feature_contract','noise_contract')
        if any(parent_contract[name]!=contract[name] for name in keys):
            raise ValueError('Producer changed inherited task/case/recipe/noise/feature identity')
        parameter_paths=sorted((root/'attempts').glob('*/parameters-after.json'))
        if not parameter_paths:raise ValueError('Producer omitted actual frozen-motor parameter receipt')
        parameter_path=parameter_paths[-1];parameters=topup.read(parameter_path)
        if (parameters.get('unchanged') is not True or not parameters.get('before')
                or parameters['before']!=parameters.get('after')):
            raise ValueError('Actual frozen motor changed during source production')
        publications=[row for row in index.get('publications',[]) if row.get('status')=='paired_ad_dpt']
        if not publications:raise ValueError('Final producer has no paired source publication')
        final=max(publications,key=lambda row:row['through_case']);relative=Path(final['source_checkpoint'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Source checkpoint must stay within its producer')
        checkpoint=exp/relative;candidate=topup.read(exp/'candidate.json')
        recovery_sha=topup.sha(exp/'source-recovery.pkl');checkpoint_sha=topup.sha(checkpoint)
        if (recovery_sha!=index['latest_recovery_sha256'] or checkpoint_sha!=final['source_checkpoint_sha256']
                or final['episodes']!=len(index['episode_receipts']) or final['transitions']!=index['chunks']
                or candidate.get('plan_sha256')!=phase or candidate.get('full_checkpoint')!=str(relative)
                or candidate.get('sha256')!=checkpoint_sha or candidate.get('full_latest_recovery_sha256')!=recovery_sha
                or candidate.get('cumulative_stream_actions')!=cost or candidate.get('teacher_updates')!=index['source_updates']):
            raise ValueError('Current producer full recovery/candidate/publication hashes or clocks differ')
        protocol={name:contract[name] for name in ('source_recipe','noise_contract','feature_contract','horizon','chunk')}
        protocol['motor']={name:contract['motor'][name] for name in ('receipt_sha256','openpi_commit','flow_steps','predicted_horizon','action_dim','weights_frozen')}
        protocol['motor_parameters_sha256']=parameters['before']['sha256']
        proof=dict(slot=slot,root=str(root),admission='explicit_retained_topup_parent' if retained else 'latest_production_phase',
            producer_source_commit=pin,producer_plan_sha256=phase,canonical_leaf=leaf,
            index_sha256=topup.sha(exp/'index.json'),contract_sha256=topup.sha(exp/'contract.json'),
            full_recovery_sha256=recovery_sha,candidate_sha256=topup.sha(exp/'candidate.json'),
            checkpoint_path=str(checkpoint),checkpoint_sha256=checkpoint_sha,migration_sha256=topup.sha(migration_path),
            ancestor_contract_path=str(parent_contract_path),ancestor_contract_sha256=topup.sha(parent_contract_path),
            worker_time_sha256=topup.sha(clock_path),worker_charge_seconds=sum(row['charged_seconds'] for row in clock['attempts']),
            motor_parameters_receipt_path=str(parameter_path),motor_parameters_receipt_sha256=topup.sha(parameter_path),
            motor_parameters_sha256=parameters['before']['sha256'],
            source_updates=index['source_updates'],cumulative_stream_actions=cost,source_schedule_sha256=topup.sha(root/'schedule.json'),
            retained_parent_proof=selected['parent'] if retained else None)
        return proof,protocol


def prepare(plan_path, ledger_path, output, *, run_id, source_commit,allow_retained_parents=False):
    plan, ledger = topup.load_plan(plan_path), topup.read(ledger_path)
    if (plan['mode'] != 'production' or ledger.get('historical_extra_actions') != sum(topup.EXTRAS.values())
            or set(ledger.get('leaves',{}))!=set(topup.expected_slots())
            or len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit)):
        raise ValueError('New final-teacher competence requires the cumulative production source ledger')
    output = Path(output)
    reservations, schedules, parents, producers, protocols = {}, set(), set(), {}, set()
    for key in ALL_TASKS:
        proof,protocol=producer(plan,ledger,key,allow_retained_parents=allow_retained_parents)
        root=Path(proof['root']);producers[key]=proof;protocols.add(topup.content_hash(protocol))
        schedules.add(proof['source_schedule_sha256']);parents.add(str(root.parent))
        reservations[key+'/train']=dict(run_id=root.name,source_commit=proof['producer_source_commit'])
    if len(schedules)!=1 or len(parents)!=1 or len(protocols)!=1:
        raise ValueError('Final source bank must share exact schedule, SAC recipe/noise/features/frozen motor and outputs root')
    output.mkdir(parents=True,exist_ok=False)
    source_ledger = output / 'source-ledger.json'
    topup.write(source_ledger, dict(reservations=reservations))
    first = Path(ledger['leaves'][ALL_TASKS[0] + '/train']['root'])
    schedule = output / 'schedule.json'
    shutil.copy2(first / 'schedule.json', schedule)
    producer_path=output/'producers.json'
    topup.write(producer_path,dict(format=ADMISSION if allow_retained_parents else 'latest_production_only',
        all40_admitted=len(producers),retained_parent_count=sum(row['admission']=='explicit_retained_topup_parent' for row in producers.values()),
        protocol_sha256=next(iter(protocols)),producers=producers))
    alias = dict(format=FORMAT, run_id=run_id, source_commit=source_commit,
        source_plan_source_commit=plan['source_commit'],
        plan_path=str(Path(plan_path).resolve()), plan_sha256=plan['plan_sha256'],
        canonical_topup_ledger=str(Path(ledger_path).resolve()), source_leaves_sha256=topup.content_hash(ledger['leaves']),
        source_ledger=str(source_ledger.resolve()), source_ledger_sha256=topup.sha(source_ledger),
        schedule=str(schedule.resolve()), schedule_sha256=topup.sha(schedule), outputs_root=next(iter(parents)),
        producer_manifest=str(producer_path.resolve()),producer_manifest_sha256=topup.sha(producer_path),
        producer_admission=ADMISSION if allow_retained_parents else 'latest_production_only',
        maximum_total_actions=20000, maximum_worker_seconds=1400, nominal_queries=48,
        query_task_keys=VALIDATION_TASKS,query_initial_states=[40,41],query_methods=['native','mean','sample'],
        nominal_worst_case_actions=19680, optimizer_updates=0, replay_insertion=False,
        teacher_status='Fixed terminal source SAC candidates; competence is measured, no expert or optimality claim',
        resource_status='Unapproved until the owner funds the planned final-check allocation',
        cost_rule='NEW competence actions inside the same1M source cap; historical7540 controls remain charged once')
    topup.write(output / 'alias.json', alias)
    return alias


def load_alias(path, expected_sha256, *, run_id, source_commit):
    if topup.sha(path) != expected_sha256:
        raise ValueError('New competence alias checksum differs')
    value = topup.read(path)
    plan = topup.load_plan(value['plan_path'])
    if (value.get('format') != FORMAT or value.get('run_id') != run_id or value.get('source_commit') != source_commit
            or value.get('maximum_total_actions') != 20000 or value.get('maximum_worker_seconds') != 1400
            or plan['mode'] != 'production' or plan['plan_sha256'] != value['plan_sha256']
            or plan['source_commit'] != value.get('source_plan_source_commit')
            or topup.sha(value['source_ledger']) != value['source_ledger_sha256']
            or topup.sha(value['schedule']) != value['schedule_sha256']):
        raise ValueError('New final-teacher competence changed source/scope/budget')
    ledger_path = Path(value['canonical_topup_ledger'])
    with ledger_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = topup.read(ledger_path)
        if topup.content_hash(ledger['leaves']) != value['source_leaves_sha256']:
            raise ValueError('Source leaves changed after new final-teacher qualification was prepared')
        if 'producer_manifest' in value:
            if (value.get('producer_admission') not in (ADMISSION,'latest_production_only')
                    or value.get('query_task_keys')!=VALIDATION_TASKS or value.get('query_initial_states')!=[40,41]
                    or value.get('query_methods')!=['native','mean','sample']
                    or topup.sha(value['producer_manifest'])!=value['producer_manifest_sha256']):
                raise ValueError('Final producer manifest checksum differs')
            manifest=topup.read(value['producer_manifest'])
            allow=value.get('producer_admission')==ADMISSION
            actual={key:producer(plan,ledger,key,allow_retained_parents=allow)[0] for key in ALL_TASKS}
            if (manifest.get('format')!=value['producer_admission'] or manifest.get('producers')!=actual or manifest.get('all40_admitted')!=40
                    or manifest.get('retained_parent_count')!=sum(row['admission']=='explicit_retained_topup_parent' for row in actual.values())):
                raise ValueError('Final producer lineage changed after alias preparation')
        if sum(row['maximum_total_actions'] for row in ledger['leaves'].values()) + sum(topup.EXTRAS.values()) + 20000 > 1_000_000:
            raise ValueError('New final competence would exceed the cumulative1M source envelope')
        root = Path(value['outputs_root']) / run_id
        record = dict(root=str(root), alias_sha256=expected_sha256, maximum_total_actions=20000,
            plan_sha256=value['plan_sha256'], purpose='NEW final source teacher competence; zero SGD')
        if ledger.get('final_competence') not in (None, record):
            raise ValueError('The source final-competence escrow already has another immutable owner')
        ledger['final_competence'] = record
        topup.write(ledger_path, ledger)
    return value


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == 'run':
        from .vla_source40_sac_diagnostic import main as diagnostic
        return diagnostic(argv[1:], alias_loader=load_alias)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    make = sub.add_parser('prepare')
    for name in ('plan', 'ledger', 'output'):
        make.add_argument('--' + name, type=Path, required=True)
    make.add_argument('--run-id', required=True)
    make.add_argument('--source-commit', required=True)
    make.add_argument('--allow-retained-parents',action='store_true',
        help='Admit only explicit quota-filled top-up parent rows from this production plan')
    start = sub.add_parser('clock-begin')
    start.add_argument('--alias', type=Path, required=True)
    start.add_argument('--alias-sha256', required=True)
    start.add_argument('--output', type=Path, required=True)
    start.add_argument('--source-commit', required=True)
    start.add_argument('--slice-seconds', type=int, default=1340)
    start.add_argument('--started-unix', type=float, required=True)
    start.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.operation == 'prepare':
        value = prepare(args.plan, args.ledger, args.output, run_id=args.run_id, source_commit=args.source_commit,
            allow_retained_parents=args.allow_retained_parents)
        print(json.dumps(value, indent=2))
        return 0
    value = load_alias(args.alias, args.alias_sha256, run_id=args.output.name, source_commit=args.source_commit)
    plan = topup.load_plan(value['plan_path'])
    topup.time_hold(plan, value['canonical_topup_ledger'], args.output, 1400)
    from .vla_source40_ppo_worker import begin
    print(begin(args.output, dict(format=FORMAT, plan_sha256=plan['plan_sha256'],
        source_commit=args.source_commit, run_id=args.output.name, alias_sha256=args.alias_sha256),
        slice_seconds=args.slice_seconds, worker_seconds=1400, resume=args.resume, now=args.started_unix))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
