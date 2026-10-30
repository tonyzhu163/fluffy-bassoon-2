"""Freeze one complete source prefix per canonical top-up leaf, then fit.

Live indices may advance. Only committed immutable paired publications enter
the snapshot. Neither this module nor its manifests reserve compute/resources.
"""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from .vla_history import file_sha256, load_dataset
from .vla_source40_campaign import ALL_TASKS, VALIDATION_TASKS
from .vla_source40_corpus import PI05_RECEIPT, physical_costs
from .vla_source40_schedule import content_hash
from .vla_source40_topup import COMPETENCE_ACTIONS, EXTRAS, expected_slots, load_plan


KIND = 'source40_topup_complete_prefix_admission_v1'
SNAPSHOT = 'source40_immutable_complete_prefix_v1'
INPUTS = 'source40_topup_offline_inputs_v1'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def member(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError('A committed source artifact must stay inside its immutable history')
    result = (Path(root) / relative).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError('Source artifact escapes its history')
    return result


def copy_checked(source, target, expected=None):
    digest = file_sha256(source)
    if expected is not None and digest != expected:
        raise ValueError('Committed source checkpoint/dataset checksum differs')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if file_sha256(target) != digest or file_sha256(source) != digest:
        raise ValueError('An immutable committed source artifact changed while copying')
    return digest


def prefix_source(leaf, parent):
    """Availability-only fallback, never substitute around a corrupt publication."""
    candidates = list(dict.fromkeys([leaf['root'], leaf.get('parent_root', parent['root']), parent['root']]))
    for candidate in candidates:
        root = Path(candidate).resolve() / 'experiment'
        path = root / 'index.json'
        if not path.is_file():
            continue
        raw = path.read_bytes()
        index = json.loads(raw)
        paired = [row for row in index.get('publications', []) if row.get('status') == 'paired_ad_dpt']
        if paired:
            if len({row['through_case'] for row in paired}) != len(paired):
                raise ValueError('A canonical source leaf published duplicate prefix versions')
            return root, index, raw, max(paired, key=lambda row: row['through_case'])
    raise ValueError('Canonical source stream has no committed learning prefix; do not filter it out')


def cost_capture(leaf, parent, target):
    """Copy small physical receipts; active unfinished RPCs remain conservative."""
    choices = list(dict.fromkeys([leaf['root'], leaf.get('parent_root', parent['root']), parent['root']]))
    journal = None
    for root in choices:
        possible = Path(root) / 'experiment/raw/decision-journal'
        if possible.is_dir() and any(possible.glob('reset-*.json')):
            journal = possible
            break
    if journal is None:
        raise ValueError('Cumulative source cost needs its actual primitive/reset receipts')
    target.mkdir(parents=True)
    files = [path for pattern in ('decision-*.json', 'intent-*.json', 'reset-*.json') for path in journal.glob(pattern)]
    receipts = []
    for source in files:
        # Each collector JSON is atomically replaced. Capture one byte version;
        # a later intent completion may legitimately change the live receipt.
        raw = source.read_bytes()
        (target / source.name).write_bytes(raw)
        receipts.append(dict(path=source.name, sha256=file_sha256(target / source.name), raw_json=raw.decode('utf-8')))
    primitive = physical_costs(target)
    resets = [read(path) for path in target.glob('reset-*.json')]
    known_settle = sum(row['settling_steps'] for row in resets if row['returned'])
    uncertain_settle = sum(row['settling_steps'] for row in resets if not row['returned'])
    charge = primitive['primitive_step_budget_charge'] + known_settle + uncertain_settle
    floor = leaf.get('inherited_action_charge', parent['actions'])
    if not floor <= charge <= leaf['maximum_total_actions']:
        raise ValueError('Captured current source leaf cost lost its inherited floor or exceeds quota')
    return dict(source_journal=str(journal), receipts=receipts, physical_cost_accounting=primitive,
        actual_primitive_steps=primitive['known_primitive_steps'], actual_settling_steps=known_settle,
        settling_step_budget_charge=known_settle + uncertain_settle,
        uncertain_settling_steps_ceiling=uncertain_settle, total_action_budget_charge=charge,
        maximum_total_actions=leaf['maximum_total_actions'], inherited_action_floor=floor)


def verify_captured_cost(costs):
    """Verify the portable compact receipt archive without scanning live files."""
    rows = {}
    for receipt in costs['receipts']:
        raw = receipt['raw_json'].encode('utf-8')
        if hashlib.sha256(raw).hexdigest() != receipt['sha256'] or receipt['path'] in rows:
            raise ValueError('Compact source physical receipt bytes changed or duplicated')
        rows[receipt['path']] = json.loads(raw)
    decisions = {name: row for name, row in rows.items() if name.startswith('decision-')}
    known = sum(row['primitive_steps'] for row in decisions.values())
    uncertain = sum(row['maximum_primitive_steps'] for name, row in rows.items()
        if name.startswith('intent-') and 'decision-' + name[7:] not in decisions)
    resets = [row for name, row in rows.items() if name.startswith('reset-')]
    known_settle = sum(row['settling_steps'] for row in resets if row['returned'])
    uncertain_settle = sum(row['settling_steps'] for row in resets if not row['returned'])
    expected = dict(known_primitive_steps=known, uncertain_primitive_steps_ceiling=uncertain,
        primitive_step_budget_charge=known + uncertain)
    if (costs['physical_cost_accounting'] != expected or costs['actual_primitive_steps'] != known
            or costs['actual_settling_steps'] != known_settle or costs['uncertain_settling_steps_ceiling'] != uncertain_settle
            or costs['settling_step_budget_charge'] != known_settle + uncertain_settle
            or costs['total_action_budget_charge'] != known + uncertain + known_settle + uncertain_settle
            or not costs['inherited_action_floor'] <= costs['total_action_budget_charge'] <= costs['maximum_total_actions']):
        raise ValueError('Compact source physical receipt counts do not support cumulative cost')
    return costs


def freeze_source(slot, leaf, parent, output):
    root, original, original_raw, publication = prefix_source(leaf, parent)
    output.mkdir(parents=True)
    (output / 'origin-index.json').write_bytes(original_raw)
    objective_paths = {key: member(root, publication['datasets'][key]['path']) for key in ('ad', 'dpt')}
    manifests = {key: read(path.with_suffix('.json')) for key, path in objective_paths.items()}
    producer = manifests['ad']['metadata']['source_contract']
    if producer != manifests['dpt']['metadata'].get('source_contract'):
        raise ValueError('AD/DPT publication producer contracts differ')
    task, partition = slot.split('/')
    if (original.get('task_key') != task or original.get('partition') != partition
            or producer.get('task', {}).get('task_key') != task or producer.get('partition') != partition
            or producer.get('source_only') is not True or producer.get('source_data_from_targets') is not False
            or producer.get('motor', {}).get('receipt_sha256') != PI05_RECEIPT
            or producer.get('horizon') != 400 or producer.get('chunk') != 10):
        raise ValueError('Committed prefix is not the matching canonical source40 task/partition/motor')
    count, episodes, chunks = publication['case_count'], publication['episodes'], publication['transitions']
    cases = producer['cases'][:count]
    receipts = original.get('episode_receipts', [])[:episodes]
    if (not cases or count != publication['through_case'] + 1 or len(cases) != count
            or any(case['episodes'] != 4 for case in cases) or episodes != count * 4
            or len(receipts) != episodes or sum(row['chunks'] for row in receipts) != chunks):
        raise ValueError('Snapshot must cover an actual complete chronological K4 prefix')
    teacher_updates = manifests['dpt']['metadata'].get('source_updates')
    if type(teacher_updates) is not int or teacher_updates < 1:
        raise ValueError('Selected DPT actor needs its actual published optimizer-clock identity')
    for key, source in objective_paths.items():
        target = output / publication['datasets'][key]['path']
        copy_checked(source, target, publication['datasets'][key]['sha256'])
        copy_checked(source.with_suffix('.json'), target.with_suffix('.json'))
    copy_checked(member(root, publication['source_checkpoint']), output / publication['source_checkpoint'],
        publication['source_checkpoint_sha256'])
    write(output / 'contract.json', producer)  # Producer bytes/provenance, not a fabricated quota contract.
    costs = cost_capture(leaf, parent, output / 'physical-cost-journal')
    if sum(row.get('primitive_steps', 0) for row in receipts) > costs['actual_primitive_steps']:
        raise ValueError('Cumulative cost capture omits selected prefix primitives')
    write(output / 'cost-capture.json', costs)
    index = copy.deepcopy(original)
    index.pop('continuation', None)  # The live-run receipt remains in origin-index.json.
    index.update(completed=False, collection_qualified=True, smoke_qualified=None,
        source_contract_sha256=content_hash(producer), collection_mode='physical_budget',
        source_updates=teacher_updates, chunks=chunks, case_index=count, episode_in_case=0,
        primitive_steps=sum(row.get('primitive_steps', 0) for row in receipts),
        committed_learning_history_primitive_steps=sum(row.get('primitive_steps', 0) for row in receipts),
        episode_receipts=receipts, publications=[publication],
        case_coverage=dict(available_cases=len(producer['cases']), started_cases=count, completed_cases=count,
            completed_episodes=episodes, all_started_cases_complete=True,
            candidate_bank_exhausted=count == len(producer['cases'])),
        all_started_cases_complete=True, stopping_reason='immutable_complete_prefix',
        physical_cost_accounting=costs['physical_cost_accounting'],
        actual_primitive_steps=costs['actual_primitive_steps'], actual_settling_steps=costs['actual_settling_steps'],
        settling_step_budget_charge=costs['settling_step_budget_charge'],
        uncertain_settling_steps_ceiling=costs['uncertain_settling_steps_ceiling'],
        total_action_budget_charge=costs['total_action_budget_charge'], maximum_total_actions=costs['maximum_total_actions'],
        unused_action_reservation=costs['maximum_total_actions'] - costs['total_action_budget_charge'])
    proof = dict(format=SNAPSHOT, slot=slot, current_canonical_leaf=leaf,
        selected_origin=str(root), origin_index_sha256=file_sha256(output / 'origin-index.json'),
        producer_contract_sha256=content_hash(producer), publication_sha256=content_hash(publication),
        cost_capture_sha256=file_sha256(output / 'cost-capture.json'),
        selected_teacher_updates=teacher_updates, live_source_updates_at_capture=original.get('source_updates'),
        data_scope='One actual complete committed prefix; source run and512-case bank may continue',
        teacher_cost='Conservatively retain entire captured current-leaf charge, including any later unused interactions; no DPT cost discount')
    proof['content_sha256'] = content_hash(proof)
    index['complete_prefix_snapshot'] = proof
    write(output / 'index.json', index)
    return dict(slot=slot, index_path=str((output / 'index.json').resolve()), index_sha256=file_sha256(output / 'index.json'),
        current_leaf=leaf, producer_source_commit=producer['source_commit'], selected_cases=count,
        selected_episodes=episodes, selected_chunks=chunks, selected_teacher_updates=teacher_updates,
        charged_total_actions=costs['total_action_budget_charge'], proof_sha256=proof['content_sha256'])


def capture(plan_path, plan_sha256, ledger_path, output):
    if file_sha256(plan_path) != plan_sha256:
        raise ValueError('Explicit top-up plan file SHA256 differs')
    plan = load_plan(plan_path)
    ledger_path, output = Path(ledger_path), Path(output).resolve()
    with ledger_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        ledger_raw = ledger_path.read_bytes()
    ledger = json.loads(ledger_raw)
    if (set(ledger.get('leaves', {})) != set(expected_slots())
            or ledger.get('historical_extra_actions') != sum(EXTRAS.values())
            or sum(row['maximum_total_actions'] for row in ledger['leaves'].values()) + sum(EXTRAS.values()) + COMPETENCE_ACTIONS > 1_000_000):
        raise ValueError('Require the canonical all40/fixed8 top-up ledger within cumulative1M')
    output.mkdir(parents=True, exist_ok=False)
    copy_checked(plan_path, output / 'plan.json', plan_sha256)
    (output / 'ledger-asof.json').write_bytes(ledger_raw)
    by_slot = {row['slot']: row for row in plan['slots']}
    groups = dict(train=[], validation=[])
    for slot in expected_slots():
        leaf, parent = ledger['leaves'][slot], by_slot[slot]['parent']
        if leaf['root'] != parent['root'] and leaf.get('parent_root') != parent['root']:
            raise ValueError('Current canonical leaf does not descend from the prepared source parent')
        row = freeze_source(slot, leaf, parent,
            output / 'sources' / slot.replace(':', '-').replace('/', '-'))
        groups[slot.split('/')[1]].append(row)
    final_actions = 0
    if ledger.get('final_competence') is not None:
        from .vla_source40_sac_diagnostic import charged_actions
        final_actions = charged_actions(Path(ledger['final_competence']['root']) / 'experiment')
    captured_source = sum(row['charged_total_actions'] for rows in groups.values() for row in rows)
    total = captured_source + sum(EXTRAS.values()) + final_actions
    if total > 1_000_000:
        raise ValueError('Captured source generation/qualification/competence costs exceed1M')
    value = dict(format=KIND, source_only=True, source_data_from_targets=False, small_pilot=False,
        plan_path=str(output / 'plan.json'), plan_file_sha256=file_sha256(output / 'plan.json'),
        plan_content_sha256=plan['plan_sha256'], ledger_snapshot_path=str(output / 'ledger-asof.json'),
        ledger_snapshot_sha256=file_sha256(output / 'ledger-asof.json'), canonical_ledger_path=str(ledger_path.resolve()),
        source_indices=groups, captured_source_actions=captured_source,
        historical_extra_costs_counted_once=EXTRAS, final_teacher_competence_actions_asof=final_actions,
        cumulative_source_action_charge=total, maximum_total_actions=1_000_000,
        source_training_task_keys=sorted(ALL_TASKS), source_validation_task_keys=sorted(VALIDATION_TASKS),
        capture_finished_unix=time.time(), additional_simulator_actions=0,
        labels='Unchanged chronological AD arrays; unchanged DPT labels from each selected committed actor, charged conservatively at full current-leaf cost',
        immutable_dataset_generation=True, other_collectors_may_continue=True)
    value['content_sha256'] = content_hash(value)
    write(output / 'admission.json', value)
    return value


def load_admission(path, expected_sha256, *, verify_cost_archives=True):
    path = Path(path).resolve()
    if expected_sha256 is None or file_sha256(path) != expected_sha256:
        raise ValueError('Complete-prefix top-up admission needs its explicit immutable file SHA256')
    value = read(path)
    if (value.get('format') != KIND or value.get('content_sha256') != content_hash({k: v for k, v in value.items() if k != 'content_sha256'})
            or value.get('source_only') is not True or value.get('source_data_from_targets') is not False
            or value.get('source_training_task_keys') != sorted(ALL_TASKS)
            or value.get('source_validation_task_keys') != sorted(VALIDATION_TASKS)
            or value.get('maximum_total_actions') != 1_000_000 or value.get('cumulative_source_action_charge', 1_000_001) > 1_000_000
            or value.get('historical_extra_costs_counted_once') != EXTRAS
            or file_sha256(value['plan_path']) != value['plan_file_sha256']
            or load_plan(value['plan_path'])['plan_sha256'] != value['plan_content_sha256']
            or file_sha256(value['ledger_snapshot_path']) != value['ledger_snapshot_sha256']):
        raise ValueError('Complete-prefix top-up identity, source coverage or cumulative cost changed')
    indices = [row for rows in value['source_indices'].values() for row in rows]
    ledger = read(value['ledger_snapshot_path'])
    if (set(ledger.get('leaves', {})) != set(expected_slots()) or ledger.get('historical_extra_actions') != sum(EXTRAS.values())
            or sum(row['maximum_total_actions'] for row in ledger['leaves'].values()) + sum(EXTRAS.values()) + COMPETENCE_ACTIONS > 1_000_000):
        raise ValueError('Captured canonical top-up ledger no longer proves cumulative1M')
    if (len(indices) != 48 or {row['slot'] for row in indices} != set(expected_slots())
            or len({row['index_path'] for row in indices}) != 48
            or (verify_cost_archives and any(file_sha256(row['index_path']) != row['index_sha256'] for row in indices))):
        raise ValueError('Require one immutable complete prefix per canonical all40/fixed8 stream')
    for row in indices if verify_cost_archives else []:
        source = Path(row['index_path']).parent
        index = read(source / 'index.json')
        proof = index['complete_prefix_snapshot']
        if (file_sha256(source / 'origin-index.json') != proof['origin_index_sha256']
                or file_sha256(source / 'cost-capture.json') != proof['cost_capture_sha256']):
            raise ValueError('Snapshot origin/cumulative cost proof changed')
        costs = verify_captured_cost(read(source / 'cost-capture.json'))
        if costs['total_action_budget_charge'] != row['charged_total_actions']:
            raise ValueError('Snapshot admission changed cumulative source cost')
    if (sum(row['charged_total_actions'] for row in indices) != value['captured_source_actions']
            or value['cumulative_source_action_charge'] != value['captured_source_actions'] + sum(EXTRAS.values()) + value['final_teacher_competence_actions_asof']
            or not 0 <= value['final_teacher_competence_actions_asof'] <= COMPETENCE_ACTIONS):
        raise ValueError('Top-up snapshot source/overhead/final-competence costs are not counted once')
    return dict(kind=KIND, path=str(path), file_sha256=expected_sha256, admission_content_sha256=value['content_sha256'],
        maximum_total_actions=1_000_000, source_training_task_keys=sorted(ALL_TASKS),
        source_validation_task_keys=sorted(VALIDATION_TASKS),
        cumulative_source_action_charge=value['cumulative_source_action_charge'],
        historical_extra_costs_counted_once=EXTRAS,
        final_teacher_competence_actions_asof=value['final_teacher_competence_actions_asof'])


def verify_snapshot(path, partition, admission):
    if load_admission(admission['path'], admission['file_sha256'], verify_cost_archives=False) != admission:
        raise ValueError('Top-up snapshot admission descriptor changed')
    value = read(admission['path'])
    path = Path(path).resolve()
    selected = [row for row in value['source_indices'][partition] if Path(row['index_path']).resolve() == path]
    if len(selected) != 1 or file_sha256(path) != selected[0]['index_sha256']:
        raise ValueError('Snapshot is not the selected unique canonical source stream')
    index = read(path)
    proof = index.get('complete_prefix_snapshot', {})
    leaf = read(value['ledger_snapshot_path'])['leaves'][selected[0]['slot']]
    if (proof.get('format') != SNAPSHOT or proof.get('slot') != selected[0]['slot']
            or proof.get('current_canonical_leaf') != leaf or selected[0].get('current_leaf') != leaf
            or index.get('maximum_total_actions') != leaf['maximum_total_actions']
            or proof.get('content_sha256') != content_hash({k: v for k, v in proof.items() if k != 'content_sha256'})
            or proof['content_sha256'] != selected[0]['proof_sha256']
            or file_sha256(path.parent / 'origin-index.json') != proof['origin_index_sha256']
            or file_sha256(path.parent / 'cost-capture.json') != proof['cost_capture_sha256']
            or content_hash(read(path.parent / 'contract.json')) != proof['producer_contract_sha256']
            or len(index.get('publications', [])) != 1
            or content_hash(index['publications'][0]) != proof['publication_sha256']):
        raise ValueError('Immutable complete-prefix source proof changed')
    costs = read(path.parent / 'cost-capture.json')
    captured = physical_costs(path.parent / 'physical-cost-journal')
    resets = [read(item) for item in (path.parent / 'physical-cost-journal').glob('reset-*.json')]
    known_settle = sum(row['settling_steps'] for row in resets if row['returned'])
    uncertain_settle = sum(row['settling_steps'] for row in resets if not row['returned'])
    if (captured != costs['physical_cost_accounting'] or index['physical_cost_accounting'] != captured
            or known_settle != costs['actual_settling_steps'] or index['actual_settling_steps'] != known_settle
            or uncertain_settle != costs['uncertain_settling_steps_ceiling']
            or index['total_action_budget_charge'] != captured['primitive_step_budget_charge'] + known_settle + uncertain_settle
            or index['total_action_budget_charge'] != selected[0]['charged_total_actions']
            or any(file_sha256(path.parent / 'physical-cost-journal' / row['path']) != row['sha256'] for row in costs['receipts'])):
        raise ValueError('Snapshot physical receipts do not support their cumulative source cost')
    return proof


def assemble_topup(admission_path, admission_sha256, output):
    descriptor = load_admission(admission_path, admission_sha256)
    value = read(admission_path)
    from .vla_source40_training import assemble, load_assembly
    groups = value['source_indices']
    result = assemble([Path(row['index_path']) for row in groups['train']],
        [Path(row['index_path']) for row in groups['validation']], output,
        topup_admission_path=admission_path, topup_admission_sha256=admission_sha256)
    output = Path(output).resolve()
    load_assembly(output / 'assembly.json')
    manifest = dict(format=INPUTS, completed=True, small_pilot=False, source_only=True,
        admission=descriptor, source_indices=groups, assembly_path=str(output / 'assembly.json'),
        assembly_file_sha256=file_sha256(output / 'assembly.json'), assembly_sha256=result['assembly_sha256'],
        cumulative_source_action_charge=value['cumulative_source_action_charge'], additional_simulator_actions=0)
    manifest['content_sha256'] = content_hash(manifest)
    write(output / 'topup-inputs.json', manifest)
    return manifest


def milestone_command(path, expected_sha256, *, output, source_commit, hostname, resume=False):
    if file_sha256(path) != expected_sha256:
        raise ValueError('Top-up input-manifest checksum differs')
    value = read(path)
    from .vla_source40_training import load_assembly
    if (value.get('format') != INPUTS or value.get('content_sha256') != content_hash({k: v for k, v in value.items() if k != 'content_sha256'})
            or value.get('completed') is not True or value.get('small_pilot') is not False
            or load_admission(value['admission']['path'], value['admission']['file_sha256']) != value['admission']
            or file_sha256(value['assembly_path']) != value['assembly_file_sha256']):
        raise ValueError('Require the immutable all40/fixed8 top-up dataset generation')
    assembly = load_assembly(value['assembly_path'])
    if assembly['assembly_sha256'] != value['assembly_sha256']:
        raise ValueError('Top-up assembly content identity changed')
    command = [sys.executable, '-m', 'fluffy_bassoon.meta_rl.vla_source40_subset_training', '--all40-source']
    for split in ('train', 'validation'):
        rows = value['source_indices'][split]
        if ({(r['index_path'], r['index_sha256']) for r in rows}
                != {(r['index_path'], r['index_sha256']) for r in assembly['selected_sources'][split]}):
            raise ValueError('Top-up manifest changed selected snapshot exposure')
        command += ['--' + split + '-index'] + [row['index_path'] for row in rows]
    command += ['--assembly', value['assembly_path'], '--assembly-file-sha256', value['assembly_file_sha256'],
        '--output', str(output), '--source-commit', source_commit, '--expected-hostname', hostname,
        '--scales', '25', '100', '500', '--seed', '0', '--max-controller-seconds', '600', '--max-total-seconds', '2700']
    if resume:
        command.append('--resume')
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    snap = sub.add_parser('capture')
    for name in ('plan', 'ledger', 'output'):
        snap.add_argument('--' + name, type=Path, required=True)
    snap.add_argument('--plan-sha256', required=True)
    assemble = sub.add_parser('assemble')
    assemble.add_argument('--admission', type=Path, required=True)
    assemble.add_argument('--admission-sha256', required=True)
    assemble.add_argument('--output', type=Path, required=True)
    train = sub.add_parser('train')
    for name in ('manifest', 'output'):
        train.add_argument('--' + name, type=Path, required=True)
    for name in ('manifest-sha256', 'source-commit', 'expected-hostname'):
        train.add_argument('--' + name, required=True)
    train.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.operation == 'capture':
        value = capture(args.plan, args.plan_sha256, args.ledger, args.output)
    elif args.operation == 'assemble':
        value = assemble_topup(args.admission, args.admission_sha256, args.output)
    else:
        if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname:
            raise ValueError('Student training requires the explicitly allocated Linux host')
        subprocess.run(milestone_command(args.manifest, args.manifest_sha256, output=args.output,
            source_commit=args.source_commit, hostname=args.expected_hostname, resume=args.resume), check=True)
        return
    print(json.dumps(value, allow_nan=False))


if __name__ == '__main__':
    main()
