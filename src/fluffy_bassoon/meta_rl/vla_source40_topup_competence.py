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
from .vla_source40_campaign import ALL_TASKS


FORMAT = 'source40_topup_final_teacher_competence_alias_v1'


def prepare(plan_path, ledger_path, output, *, run_id, source_commit):
    plan, ledger = topup.load_plan(plan_path), topup.read(ledger_path)
    if (plan['mode'] != 'production' or ledger.get('historical_extra_actions') != sum(topup.EXTRAS.values())
            or len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit)):
        raise ValueError('New final-teacher competence requires the cumulative production source ledger')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reservations, schedules, parents = {}, set(), set()
    for key in ALL_TASKS:
        leaf = ledger['leaves'][key + '/train']
        root = Path(leaf['root'])
        with (root / 'launcher.lock').open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            index, contract = topup.read(root / 'experiment/index.json'), topup.read(root / 'experiment/contract.json')
            if ((root / 'exit-code.txt').read_text().strip() != '0' or not index.get('completed')
                    or not index.get('collection_qualified') or index.get('episode_in_case') != 0
                    or contract.get('continuation', {}).get('plan_sha256') != plan['plan_sha256']
                    or contract.get('source_commit') != plan['source_commit']
                    or index.get('continuation', {}).get('new_optimizer_updates', 0) <= 0):
                raise ValueError('All40 current source leaves must be terminal updated top-up candidates, not old16/16 actors')
            schedules.add(topup.sha(root / 'schedule.json'))
            parents.add(str(root.parent))
            reservations[key + '/train'] = dict(run_id=root.name, source_commit=contract['source_commit'])
    if len(schedules) != 1 or len(parents) != 1:
        raise ValueError('Final source bank must share its exact schedule and outputs root')
    source_ledger = output / 'source-ledger.json'
    topup.write(source_ledger, dict(reservations=reservations))
    first = Path(ledger['leaves'][ALL_TASKS[0] + '/train']['root'])
    schedule = output / 'schedule.json'
    shutil.copy2(first / 'schedule.json', schedule)
    alias = dict(format=FORMAT, run_id=run_id, source_commit=source_commit,
        source_plan_source_commit=plan['source_commit'],
        plan_path=str(Path(plan_path).resolve()), plan_sha256=plan['plan_sha256'],
        canonical_topup_ledger=str(Path(ledger_path).resolve()), source_leaves_sha256=topup.content_hash(ledger['leaves']),
        source_ledger=str(source_ledger.resolve()), source_ledger_sha256=topup.sha(source_ledger),
        schedule=str(schedule.resolve()), schedule_sha256=topup.sha(schedule), outputs_root=next(iter(parents)),
        maximum_total_actions=20000, maximum_worker_seconds=1400, nominal_queries=48,
        nominal_worst_case_actions=19680, optimizer_updates=0, replay_insertion=False,
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
        value = prepare(args.plan, args.ledger, args.output, run_id=args.run_id, source_commit=args.source_commit)
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
