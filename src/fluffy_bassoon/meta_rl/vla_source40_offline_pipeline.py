"""Canonical all40/fixed8 corpus assembly and existing milestone-trainer entry.

No slot is filtered, replaced or relaunched. Inputs come only from the exact
campaign ledger; incomplete/raw-only slots remain missing inputs, not AD labels.
"""
import argparse
import json
from pathlib import Path
import re
import socket
import subprocess
import sys

from .vla_history import file_sha256
from .vla_source40_schedule import content_hash
from .vla_source40_training import assemble, load_assembly, load_campaign


FORMAT = 'openpi_pi05_all40_offline_inputs_v1'


def canonical_inputs(project_root, campaign_path, campaign_sha256, ledger_path, ledger_sha256):
    campaign = load_campaign(campaign_path, campaign_sha256)
    definition = json.loads(Path(campaign_path).read_text())
    if file_sha256(ledger_path) != ledger_sha256:
        raise ValueError('Explicit completed source-ledger file SHA256 differs')
    ledger = json.loads(Path(ledger_path).read_text())
    slots = definition['sac_slots']
    keys = {row['task_key'] + '/' + row['partition'] for row in slots}
    if (ledger.get('campaign_sha256') != campaign['campaign_sha256']
            or set(ledger.get('reservations', {})) != keys):
        raise ValueError('Require every canonical all40 train/fixed8 validation reservation exactly once')
    groups, blocked, seen_runs = {'train': [], 'validation': []}, [], set()
    for slot in slots:
        key = slot['task_key'] + '/' + slot['partition']
        record = ledger['reservations'][key]
        run = record.get('run_id', '')
        if (not re.fullmatch(r'[a-zA-Z0-9-]+', run) or run in seen_runs
                or not re.fullmatch(r'[0-9a-f]{40}', record.get('source_commit', ''))
                or any(record.get(name) != value for name, value in slot.items())
                or record.get('campaign_sha256') != campaign['campaign_sha256']
                or record.get('source_schedule_sha256') != definition['source_schedule_sha256']):
            raise ValueError('Canonical slot reservation changed identity, cap, source or schedule')
        seen_runs.add(run)
        directory = Path(project_root).resolve() / 'outputs' / run
        path, contract_path = directory / 'experiment/index.json', directory / 'experiment/contract.json'
        if not path.is_file() or not contract_path.is_file():
            blocked.append(dict(slot=key, run_id=run, reason='missing source index/contract'))
            continue
        index, contract = json.loads(path.read_text()), json.loads(contract_path.read_text())
        if (index.get('completed') is not True or index.get('collection_qualified') is not True
                or index.get('source_updates', 0) <= 0):
            blocked.append(dict(slot=key, run_id=run, reason='incomplete or raw-only; no completed learning history'))
            continue
        exit_path = directory / 'exit-code.txt'
        if not exit_path.is_file() or exit_path.read_text().strip() != '0':
            blocked.append(dict(slot=key, run_id=run, reason='collector has no successful terminal exit receipt'))
            continue
        if (contract.get('source_commit') != record['source_commit']
                or contract.get('schedule_sha256') != record['source_schedule_sha256']
                or contract.get('task', {}).get('task_key') != slot['task_key']
                or contract.get('partition') != slot['partition']
                or contract.get('maximum_total_actions') != slot['maximum_total_actions']
                or index.get('source_contract_sha256') != content_hash(contract)):
            raise ValueError('Completed source history differs from its exact canonical reservation')
        groups[slot['partition']].append(dict(slot=key, task_key=slot['task_key'], run_id=run,
            source_commit=record['source_commit'], index_path=str(path), index_sha256=file_sha256(path)))
    if blocked:
        raise ValueError('All48 source histories are required without filtering: ' + json.dumps(blocked))
    return campaign, groups, definition.get('prior_qualification_actions', 0)


def assemble_canonical(project_root, campaign_path, campaign_sha256, ledger_path, ledger_sha256, output):
    campaign, groups, prior = canonical_inputs(project_root, campaign_path, campaign_sha256, ledger_path, ledger_sha256)
    if type(prior) is not int or prior < 0:
        raise ValueError('Campaign prior qualification cost must remain explicit')
    result = assemble([Path(row['index_path']) for row in groups['train']],
        [Path(row['index_path']) for row in groups['validation']], output,
        campaign_path=campaign_path, campaign_sha256=campaign_sha256)
    output = Path(output).resolve()
    load_assembly(output / 'assembly.json')
    if any(file_sha256(row['index_path']) != row['index_sha256'] for rows in groups.values() for row in rows):
        raise ValueError('A supposedly terminal source index changed during assembly')
    selected = sum(row['charged_total_actions'] for row in result['collection_costs_count_once'].values())
    value = dict(format=FORMAT, completed=True, source_only=True, small_pilot=False,
        campaign=campaign, ledger_path=str(Path(ledger_path).resolve()), ledger_file_sha256=ledger_sha256,
        source_indices=groups, assembly_path=str(output / 'assembly.json'),
        assembly_file_sha256=file_sha256(output / 'assembly.json'), assembly_sha256=result['assembly_sha256'],
        collection_costs_count_once=result['collection_costs_count_once'], selected_source_actions=selected,
        prior_qualification_actions_separate=prior, selected_plus_prior_source_actions=selected + prior,
        cost_scope='Source histories and separately spent prior source qualification only; PPO/target expenses remain in their campaign ledgers',
        additional_simulator_actions=0,
        teacher_status='Actual evolving source SAC candidates; no expert or improvement claim')
    value['content_sha256'] = content_hash(value)
    (output / 'all40-inputs.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    return value


def milestone_command(manifest_path, expected_sha256, *, output, source_commit, hostname, resume=False):
    manifest_path = Path(manifest_path).resolve()
    if file_sha256(manifest_path) != expected_sha256:
        raise ValueError('All40 input-manifest file checksum differs')
    value = json.loads(manifest_path.read_text())
    if (value.get('format') != FORMAT or value.get('completed') is not True or value.get('small_pilot') is not False
            or value.get('content_sha256') != content_hash({key: item for key, item in value.items() if key != 'content_sha256'})
            or file_sha256(value['assembly_path']) != value['assembly_file_sha256']):
        raise ValueError('Require the completed immutable all40/fixed8 assembly manifest')
    assembly = load_assembly(value['assembly_path'])
    if assembly['assembly_sha256'] != value['assembly_sha256']:
        raise ValueError('All40 source assembly content hash differs')
    command = [sys.executable, '-m', 'fluffy_bassoon.meta_rl.vla_source40_subset_training', '--all40-source']
    for partition in ('train', 'validation'):
        rows = value['source_indices'][partition]
        if ({(row['index_path'], row['index_sha256']) for row in rows}
                != {(row['index_path'], row['index_sha256']) for row in assembly['selected_sources'][partition]}
                or any(file_sha256(row['index_path']) != row['index_sha256'] for row in rows)):
            raise ValueError('All40 manifest changed source exposure or an original index')
        command += ['--' + partition + '-index'] + [row['index_path'] for row in rows]
    command += ['--assembly', value['assembly_path'], '--assembly-file-sha256', value['assembly_file_sha256'],
        '--output', str(output), '--source-commit', source_commit, '--expected-hostname', hostname,
        '--scales', '25', '100', '500', '--seed', '0', '--max-controller-seconds', '600', '--max-total-seconds', '2700']
    if resume:
        command.append('--resume')
    return command


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='operation', required=True)
    prepare = sub.add_parser('assemble')
    for name in ('project-root', 'campaign', 'ledger', 'output'):
        prepare.add_argument('--' + name, type=Path, required=True)
    for name in ('campaign-sha256', 'ledger-sha256'):
        prepare.add_argument('--' + name, required=True)
    train = sub.add_parser('train')
    for name in ('manifest', 'output'):
        train.add_argument('--' + name, type=Path, required=True)
    for name in ('manifest-sha256', 'source-commit', 'expected-hostname'):
        train.add_argument('--' + name, required=True)
    train.add_argument('--resume', action='store_true')
    args = ap.parse_args(argv)
    if args.operation == 'assemble':
        value = assemble_canonical(args.project_root, args.campaign, args.campaign_sha256,
            args.ledger, args.ledger_sha256, args.output)
        print(json.dumps(value, allow_nan=False))
    else:
        if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname:
            raise ValueError('Train only in the explicitly identified allocated Linux resource')
        command = milestone_command(args.manifest, args.manifest_sha256, output=args.output,
            source_commit=args.source_commit, hostname=args.expected_hostname, resume=args.resume)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
