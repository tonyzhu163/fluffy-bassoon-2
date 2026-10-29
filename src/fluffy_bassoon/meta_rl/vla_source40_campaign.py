"""Reserve unique source streams under the user-selected shared 1M action cap.

Static planning and ledger operations only; no models or simulator execution.
"""
import argparse
import fcntl
import json
from pathlib import Path

from .vla_source40_schedule import SOURCE_SUITES, content_hash, load_schedule


FORMAT = 'libero40_source_1m_campaign_v1'
ALL_TASKS = [f'{suite}:{number:02d}' for suite in SOURCE_SUITES for number in range(10)]
VALIDATION_TASKS = [f'{suite}:{number:02d}' for suite in SOURCE_SUITES for number in (0, 1)]


def make_campaign(schedule):
    plan = load_schedule(schedule)
    if plan['collection_mode'] != 'physical_budget' or {t['task_key'] for t in plan['tasks']} != set(ALL_TASKS):
        raise ValueError('Require all40 source tasks and joint physical-action reservations')
    caps = {split: plan['partition_budgets'][split]['maximum_total_actions'] for split in ('train', 'validation')}
    if caps != dict(train=9500, validation=12000):
        raise ValueError('Reviewed 1M campaign uses train9500/validation12000 action slots')
    slots = [dict(task_key=task, partition=split, maximum_total_actions=caps[split])
        for split, tasks in (('train', ALL_TASKS), ('validation', VALIDATION_TASKS)) for task in tasks]
    result = dict(format=FORMAT, benchmark='LIBERO', maximum_total_actions=1_000_000,
        source_training_task_keys=ALL_TASKS, source_validation_task_keys=VALIDATION_TASKS,
        source_schedule_sha256=plan['schedule_sha256'],
        prior_qualification_actions=12944, sac_slots=slots,
        sac_maximum_total_actions=490000, ppo_maximum_total_actions=490000,
        unassigned_reserve_actions=21056,
        budget_scope='SAC and fresh PPO source training/validation, settling, qualifications, retries and uncertain intents; targets are separate and not launched by this campaign',
        controls='AD/DPT full/static reuse the same SAC source corpus; source interactions counted once. Any separately collected PPO reset control must share the PPO reservation.',
        case_rule='All40 training goals; train states0..39 and fixed8 source-local validation goals on states40..49. No target90/Plus/Pro trajectories.',
        source_status='Runtime qualified; informative source improvement and frozen-controller benefit remain unestablished',
        ppo_status='Fresh on-policy source collection required; shared all40 policy orchestration must be qualified before launch',
        reserve_rule='No automatic cap expansion or reassignment. Journaled attempts retain their original reservation on resume.',
        resource_status='Bounded pilot jobs first; total GPU-hour projection updated from actual source and PPO throughput')
    result['campaign_sha256'] = content_hash(result)
    return result


def load_campaign(path):
    data = json.loads(Path(path).read_text())
    if (data.get('format') != FORMAT or data.get('maximum_total_actions') != 1_000_000
            or data.get('campaign_sha256') != content_hash({k: v for k, v in data.items() if k != 'campaign_sha256'})
            or data.get('source_training_task_keys') != ALL_TASKS
            or data.get('source_validation_task_keys') != VALIDATION_TASKS):
        raise ValueError('Campaign identity or task/action contract differs')
    slots = data['sac_slots']
    expected = {(task, split, cap) for split, tasks, cap in
        [('train', ALL_TASKS, 9500), ('validation', VALIDATION_TASKS, 12000)] for task in tasks}
    if (len(slots) != len(expected) or {(row['task_key'], row['partition'], row['maximum_total_actions'])
            for row in slots} != expected):
        raise ValueError('Missing, duplicate or changed source reservation')
    sac = sum(row['maximum_total_actions'] for row in slots) + data['prior_qualification_actions']
    if sac > data['sac_maximum_total_actions'] or sac + data['ppo_maximum_total_actions'] + data['unassigned_reserve_actions'] != data['maximum_total_actions']:
        raise ValueError('Aggregate source reservations exceed or omit the shared action cap')
    return data


def reserve(campaign, schedule, ledger, *, task, partition, run, commit, resume=False):
    value, plan = load_campaign(campaign), load_schedule(schedule)
    if plan['schedule_sha256'] != value['source_schedule_sha256']:
        raise ValueError('Campaign schedule differs')
    slots = [row for row in value['sac_slots'] if row['task_key'] == task and row['partition'] == partition]
    if len(slots) != 1 or slots[0]['maximum_total_actions'] != plan['partition_budgets'][partition]['maximum_total_actions']:
        raise ValueError('Task/partition has no matching campaign action reservation')
    if not run or not all(c.isalnum() or c == '-' for c in run) or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Require a plain run identity and exact commit')
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    record = dict(slots[0], run_id=run, source_commit=commit,
        campaign_sha256=value['campaign_sha256'], source_schedule_sha256=plan['schedule_sha256'])
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(ledger.read_text()) if ledger.exists() else dict(campaign_sha256=value['campaign_sha256'], reservations={})
        if data['campaign_sha256'] != value['campaign_sha256']:
            raise ValueError('Ledger belongs to another campaign')
        key = task + '/' + partition
        previous = data['reservations'].get(key)
        if previous is not None:
            if not resume or previous != record:
                raise ValueError('Slot already reserved; only its exact pinned run may resume')
        elif resume:
            raise ValueError('Cannot resume an unreserved source stream')
        else:
            if any(row['run_id'] == run for row in data['reservations'].values()):
                raise ValueError('Run identity already owns another source stream')
            data['reservations'][key] = record
            temporary = ledger.with_suffix('.tmp')
            temporary.write_text(json.dumps(data, indent=2) + '\n')
            temporary.replace(ledger)
    return record


def reserve_ppo(campaign, ledger, *, run, commit, action_cap, resume=False):
    value = load_campaign(campaign)
    if type(action_cap) is not int or action_cap < 1650 or len(commit) != 40:
        raise ValueError('Require an explicit complete-trial PPO reservation and commit')
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    record = dict(run_id=run, source_commit=commit, maximum_total_actions=action_cap,
                  campaign_sha256=value['campaign_sha256'])
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(ledger.read_text()) if ledger.exists() else dict(campaign_sha256=value['campaign_sha256'], reservations={})
        if data['campaign_sha256'] != value['campaign_sha256']:
            raise ValueError('PPO ledger belongs to another campaign')
        previous = data['reservations'].get(run)
        if previous is not None:
            if not resume or previous != record:
                raise ValueError('PPO run already reserved; exact pinned resume required')
        elif resume:
            raise ValueError('PPO resume has no reservation')
        else:
            if sum(row['maximum_total_actions'] for row in data['reservations'].values()) + action_cap > value['ppo_maximum_total_actions']:
                raise ValueError('Fresh PPO reservation would exceed its shared benchmark cap')
            data['reservations'][run] = record
            temporary = ledger.with_suffix('.tmp')
            temporary.write_text(json.dumps(data, indent=2) + '\n')
            temporary.replace(ledger)
    return record


def reserve_diagnostic(campaign, ledger, *, run, commit, action_cap, resume=False):
    """Escrow a bounded evaluation inside the unassigned benchmark allowance.

    Evaluation traces remain separate from source pretraining. This does not
    change any existing SAC/PPO slot or permit another source collection run.
    """
    value = load_campaign(campaign)
    if (type(action_cap) is not int or action_cap < 410
            or not run or not all(c.isalnum() or c == '-' for c in run)
            or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit)):
        raise ValueError('Require an explicit diagnostic action reservation and exact identity')
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    record = dict(run_id=run, source_commit=commit, maximum_total_actions=action_cap,
        campaign_sha256=value['campaign_sha256'], pool='unassigned_reserve_actions',
        purpose='controller_subset_transfer_evaluation', source_pretraining_data=False,
        total_benchmark_cap_unchanged=True)
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(ledger.read_text()) if ledger.exists() else dict(
            campaign_sha256=value['campaign_sha256'], pool='unassigned_reserve_actions', reservations={})
        if data.get('campaign_sha256') != value['campaign_sha256'] or data.get('pool') != 'unassigned_reserve_actions':
            raise ValueError('Diagnostic ledger belongs to another campaign or budget pool')
        previous = data['reservations'].get(run)
        if previous is not None:
            if not resume or previous != record:
                raise ValueError('Diagnostic already reserved; exact pinned resume required')
        elif resume:
            raise ValueError('Diagnostic resume has no reservation')
        else:
            if sum(row['maximum_total_actions'] for row in data['reservations'].values()) + action_cap > value['unassigned_reserve_actions']:
                raise ValueError('Diagnostic would exceed the shared benchmark reserve')
            data['reservations'][run] = record
            temporary = ledger.with_suffix('.tmp')
            temporary.write_text(json.dumps(data, indent=2) + '\n')
            temporary.replace(ledger)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='operation', required=True)
    make = sub.add_parser('make')
    make.add_argument('--schedule', type=Path, required=True)
    make.add_argument('--output', type=Path, required=True)
    take = sub.add_parser('reserve')
    for name in ('campaign', 'schedule', 'ledger'):
        take.add_argument('--' + name, type=Path, required=True)
    for name in ('task', 'partition', 'run', 'commit'):
        take.add_argument('--' + name, required=True)
    take.add_argument('--resume', action='store_true')
    ppo = sub.add_parser('reserve-ppo')
    ppo.add_argument('--campaign', type=Path, required=True)
    ppo.add_argument('--ledger', type=Path, required=True)
    ppo.add_argument('--run', required=True)
    ppo.add_argument('--commit', required=True)
    ppo.add_argument('--action-cap', type=int, required=True)
    ppo.add_argument('--resume', action='store_true')
    diagnostic = sub.add_parser('reserve-diagnostic')
    diagnostic.add_argument('--campaign', type=Path, required=True)
    diagnostic.add_argument('--ledger', type=Path, required=True)
    diagnostic.add_argument('--run', required=True)
    diagnostic.add_argument('--commit', required=True)
    diagnostic.add_argument('--action-cap', type=int, required=True)
    diagnostic.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if args.operation == 'make':
        data = make_campaign(args.schedule)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(json.dumps(data, indent=2) + '\n')
    elif args.operation == 'reserve':
        data = reserve(args.campaign, args.schedule, args.ledger, task=args.task,
            partition=args.partition, run=args.run, commit=args.commit, resume=args.resume)
    elif args.operation == 'reserve-ppo':
        data = reserve_ppo(args.campaign, args.ledger, run=args.run, commit=args.commit,
                           action_cap=args.action_cap, resume=args.resume)
    else:
        data = reserve_diagnostic(args.campaign, args.ledger, run=args.run, commit=args.commit,
                                 action_cap=args.action_cap, resume=args.resume)
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
