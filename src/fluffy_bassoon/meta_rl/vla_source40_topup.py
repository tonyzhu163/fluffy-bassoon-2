"""Explicit, source-only SAC ancestry migrations under one cumulative1M ledger.

Planning is metadata-only. Migration/collection requires allocated Linux.
Creating a plan or action ledger supplies no GPU/resource authorization.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import shutil
import socket
import subprocess
import sys
import time

import numpy as np

from .vla_source40_campaign import ALL_TASKS, VALIDATION_TASKS
from .vla_source40_corpus import PI05_RECEIPT, physical_action_charge, save_motor_rng
from .vla_source40_schedule import content_hash, load_schedule, partition_budget


FORMAT = 'openpi_pi05_source40_cumulative_sac_topup_v1'
CAPS = dict(train=21800, validation=12000)
COMPETENCE_ACTIONS = 20000
EXTRAS = dict(technical_source_qualification=12944, failed_typed_key_settling=10,
    frozen_source_sac_controls=7540, collector_speed_qualification=759)
MUTABLE = {'contract.json', 'index.json', 'progress.json', 'source-recovery.pkl', 'failure.json',
    'migration.json', 'migration-initial-recovery.pkl', 'motor-binding.json', 'motor-qualification.json',
    'restored-learner.json', 'motor-restore.json', 'parameters-before.json', 'parameters-after.json', 'candidate.json'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def expected_slots():
    # Interleaving is fixed before outcomes. Independent streams retain their
    # original whole-case cursors, so this is stream admission, not bank exhaustion.
    train = [f'{suite}:{number:02d}/train' for number in range(10)
        for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')]
    return train + [key + '/validation' for key in VALIDATION_TASKS]


def parent_record(root, slot):
    root = Path(root).resolve()
    experiment = root / 'experiment'
    with (experiment / 'run.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        contract, index = read(experiment / 'contract.json'), read(experiment / 'index.json')
        task, partition = slot.split('/')
        cost = physical_action_charge(experiment / 'raw/decision-journal')
        if ((root / 'exit-code.txt').read_text().strip() != '0'
                or index.get('completed') is not True or not index.get('collection_qualified')
                or contract.get('source_only') is not True or contract.get('source_data_from_targets') is not False
                or contract.get('task', {}).get('task_key') != task or contract.get('partition') != partition
                or contract.get('horizon') != 400 or contract.get('chunk') != 10
                or contract.get('motor', {}).get('receipt_sha256') != PI05_RECEIPT
                or contract.get('source_recipe', {}).get('warmup_chunks') != 500
                or contract['source_recipe'].get('batch_size') != 256 or contract['source_recipe'].get('utd') != 20
                or index.get('total_action_budget_charge') != cost or index.get('source_updates', 0) <= 0
                or len(contract.get('cases', [])) != 512 or index.get('episode_in_case') != 0
                or len(index.get('episode_receipts', [])) % 4):
            raise ValueError('Require the exact terminal whole-case source-only SAC parent, recipe and physical charge')
        recovery = experiment / 'source-recovery.pkl'
        if sha(recovery) != index['latest_recovery_sha256']:
            raise ValueError('Parent latest full recovery checksum differs from its terminal index')
        return dict(slot=slot, root=str(root), experiment=str(experiment), actions=cost,
            source_updates=index['source_updates'], chunks=index['chunks'], next_case=index['case_index'],
            episodes=len(index['episode_receipts']), remaining_candidate_cases=len(contract['cases']) - index['case_index'],
            contract_sha256=sha(experiment / 'contract.json'), index_sha256=sha(experiment / 'index.json'),
            recovery_sha256=sha(recovery), schedule_sha256=sha(root / 'schedule.json'),
            source_commit=contract['source_commit'])


def cost_evidence(project_root, evidence_root):
    """Deduplicate source40 overhead; no target/PPO/RLinf130 trajectories enter."""
    root, evidence = Path(project_root), Path(evidence_root)
    technical_path = evidence / 'libero40-source-qualification-audit-20261029-r1/receipts.json'
    technical = read(technical_path)
    if sum(row['primitive_steps'] + row['settling_steps'] for row in technical['partitions']) != EXTRAS['technical_source_qualification']:
        raise ValueError('Technical source qualification charge changed')
    failed = root / 'outputs/libero40-pi05-source-smoke-20261029-s00-train-r1/experiment/raw/decision-journal'
    if physical_action_charge(failed) != EXTRAS['failed_typed_key_settling']:
        raise ValueError('Failed typed-key qualification charge changed')
    controls = root / 'outputs/libero40-source-sac-competence-20261030-r2/experiment/index.json'
    if read(controls).get('total_action_charge') != EXTRAS['frozen_source_sac_controls'] or not read(controls).get('completed'):
        raise ValueError('Frozen source-SAC controls charge changed')
    ledger = evidence / 'libero40-bulk-approved-20261030-r1/compute-ledger.json'
    speed = read(ledger)['components']['collector-speed-qualification']
    if speed.get('physical_actions_spent') != EXTRAS['collector_speed_qualification'] or not speed['status'].startswith('terminal'):
        raise ValueError('Collector-speed source qualification charge changed')
    return dict(actions=EXTRAS, total=sum(EXTRAS.values()), technical_receipt_sha256=sha(technical_path),
        failed_reset_files=[dict(path=str(p), sha256=sha(p)) for p in failed.glob('reset-*.json')],
        controls_index_sha256=sha(controls), speed_component=speed,
        speed_component_sha256=content_hash(speed), compute_ledger_snapshot_sha256=sha(ledger),
        classification='Source SAC overhead counted ONCE; speed alias is not a second759; no inherited PPO22753 or target actions',
        failed_competence_r1='No reset/action receipts;0 actions, prior78 GPU-seconds remains separate compute cost')


def make_plan(receipts, *, project_root, evidence_root, source_commit, mode='production', overrides=None):
    if mode not in ('production', 'qualification') or len(source_commit) != 40:
        raise ValueError('Require exact source and an explicit unapproved preparation mode')
    rows = read(receipts)['rows']
    by_slot = {row['slot']: row for row in rows}
    if set(by_slot) != set(expected_slots()) or len(rows) != 48:
        raise ValueError('Top-up requires all40 original train learners and the fixed8 independent validation learners')
    overrides = overrides or {}
    if not set(overrides).issubset(by_slot) or any(not key.endswith('/train') for key in overrides):
        raise ValueError('Parent overrides cannot add target/source goals or partitions')
    slots = []
    for key in expected_slots():
        parent = parent_record(overrides.get(key, Path(project_root) / 'outputs' / by_slot[key]['run_id']), key)
        cap = CAPS['train'] if key.endswith('/train') else parent['actions']
        if mode == 'qualification':
            cap = parent['actions'] + 5000 if key == 'libero_spatial:00/train' else parent['actions']
        if parent['actions'] > cap or cap > CAPS[key.split('/')[1]]:
            raise ValueError('An existing source charge exceeds its proposed cumulative slot cap')
        slots.append(dict(slot=key, parent=parent, maximum_total_actions=cap,
            maximum_new_actions=cap - parent['actions'], launchable=cap - parent['actions'] >= 1650,
            admission='New complete-case suffix' if cap - parent['actions'] >= 1650 else
                'Retain existing source/validation leaf; quota cannot fit qualification plus another complete K4'))
    extras = cost_evidence(project_root, evidence_root)
    total = extras['total'] + sum(row['maximum_total_actions'] for row in slots) + COMPETENCE_ACTIONS
    if total > 1_000_000:
        raise ValueError('Cumulative source quotas plus all historical qualifications/controls exceed1M')
    value = dict(format=FORMAT, source_commit=source_commit, motor_receipt_sha256=PI05_RECEIPT,
        source_only=True, source_data_from_targets=False, maximum_total_actions=1_000_000,
        mode=mode, slots=slots, historical_extra_costs=extras,
        historical_extra_cost_group_sha256=content_hash(EXTRAS),
        inherited_action_charge=extras['total'] + sum(row['parent']['actions'] for row in slots),
        maximum_stream_action_charge=sum(row['maximum_total_actions'] for row in slots),
        final_teacher_competence=dict(maximum_total_actions=COMPETENCE_ACTIONS, maximum_gpu_seconds=1400,
            source_only=True, optimizer_updates=0, queries=48, task_keys=VALIDATION_TASKS,
            methods=['native', 'mean', 'sample'], states=[40, 41], funded=False),
        maximum_charged_including_overhead=total, unassigned_actions=1_000_000 - total,
        new_case_admitted_slots=[row['slot'] for row in slots if row['launchable']],
        retained_only_slots=[row['slot'] for row in slots if not row['launchable']],
        resource_status='PROPOSAL ONLY: no GPU-hours or physical reservation granted by preparation',
        storage=dict(maximum_new_bytes=120_000_000_000, per_stream_bytes=3_000_000_000,
            shared_free_floor_bytes=500_000_000_000, whole_case_atomic_headroom_bytes=1_000_000_000),
        qualification=dict(maximum_new_actions=5000, maximum_gpu_seconds=1000, funded=False),
        case_rule='Original512 candidate cases/cursors/IDs retained; whole K4; report unused candidates, no bank-exhaustion claim',
        retention='Atomic latest full state every episode; original16-case sparse full milestones plus final; all old bytes preserved',
        labels='AD actual chronological noises/clocks; DPT final source-only actor relabels whole inherited+new history and retains full teacher cost')
    value['plan_sha256'] = content_hash(value)
    if sum(row['launchable'] for row in slots) * value['storage']['per_stream_bytes'] > value['storage']['maximum_new_bytes']:
        raise ValueError('Launchable new stream storage exceeds the proposed120GB cap')
    return value


def load_plan(path):
    value = read(path)
    if (value.get('format') != FORMAT or value.get('plan_sha256') != content_hash({k: v for k, v in value.items() if k != 'plan_sha256'})
            or [row['slot'] for row in value.get('slots', [])] != expected_slots()
            or value.get('historical_extra_costs', {}).get('actions') != EXTRAS
            or value.get('source_data_from_targets') is not False or value.get('source_only') is not True
            or value.get('maximum_total_actions') != 1_000_000 or value.get('motor_receipt_sha256') != PI05_RECEIPT
            or value.get('mode') not in ('production', 'qualification')
            or any(row['parent']['slot'] != row['slot'] or not row['parent']['actions'] <= row['maximum_total_actions'] <= CAPS[row['slot'].split('/')[1]]
                for row in value['slots'])
            or any(row['maximum_total_actions'] != row['parent']['actions'] for row in value['slots'] if row['slot'].endswith('/validation'))
            or value.get('final_teacher_competence', {}).get('maximum_total_actions') != COMPETENCE_ACTIONS
            or value['maximum_charged_including_overhead'] != sum(EXTRAS.values()) + sum(row['maximum_total_actions'] for row in value['slots']) + COMPETENCE_ACTIONS
            or value['maximum_charged_including_overhead'] > 1_000_000):
        raise ValueError('Source top-up identity, all40/fixed8 slots or cumulative charge changed')
    return value


def claim(plan, slot, *, output, ledger):
    """Unique current leaf per slot; ancestry replaces a leaf, never adds its cost twice."""
    selected = next(row for row in plan['slots'] if row['slot'] == slot)
    if not selected['launchable']:
        raise ValueError('This preparation mode admits no new actions for that stream')
    ledger, output = Path(ledger), Path(output).resolve()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = read(ledger) if ledger.exists() else dict(format=FORMAT, historical_extra_actions=sum(EXTRAS.values()),
            maximum_total_actions=1_000_000, leaves={row['slot']: dict(root=row['parent']['root'],
                maximum_total_actions=row['parent']['actions']) for row in plan['slots']}, history=[], time_holds={})
        if data.get('format') != FORMAT or data.get('historical_extra_actions') != sum(EXTRAS.values()) or set(data['leaves']) != set(expected_slots()):
            raise ValueError('The canonical cumulative source ledger differs')
        previous = data['leaves'][slot]
        current = dict(root=str(output), maximum_total_actions=selected['maximum_total_actions'],
            plan_sha256=plan['plan_sha256'], source_commit=plan['source_commit'],
            inherited_action_charge=selected['parent']['actions'], parent_root=selected['parent']['root'])
        if previous != current:
            if previous['root'] != selected['parent']['root']:
                raise ValueError('Another leaf owns this source stream; continuation must import that terminal leaf')
            data['history'].append(dict(slot=slot, parent=previous, child=current))
            data['leaves'][slot] = current
        if sum(row['maximum_total_actions'] for row in data['leaves'].values()) + sum(EXTRAS.values()) + COMPETENCE_ACTIONS > 1_000_000:
            raise ValueError('Concurrent source leaves exceed the canonical1M source cap')
        write(ledger, data)
    return selected


def status(ledger):
    """Compact conservative charges, not a raw-corpus re-audit or completion guess."""
    value = read(ledger)
    if value.get('format') != FORMAT or set(value['leaves']) != set(expected_slots()):
        raise ValueError('Require the canonical all40/fixed8 cumulative source ledger')
    rows = []
    for key in expected_slots():
        leaf = value['leaves'][key]
        root = Path(leaf['root'])
        index_path = root / 'experiment/index.json'
        index = read(index_path) if index_path.exists() else {}
        journal = root / 'experiment/raw/decision-journal'
        charge = max(leaf.get('inherited_action_charge', leaf['maximum_total_actions']),
            physical_action_charge(journal))
        rows.append(dict(slot=key, root=str(root), actions=charge, maximum_total_actions=leaf['maximum_total_actions'],
            source_updates=index.get('source_updates'), complete_cases=index.get('case_coverage', {}).get('completed_cases'),
            next_case=index.get('case_index'), remaining_candidate_cases=None if 'case_index' not in index else 512-index['case_index'],
            has_new_migration=bool(index.get('continuation')), bank_exhausted=index.get('case_index') == 512))
    final = value.get('final_competence')
    final_actions = 0
    if final is not None:
        from .vla_source40_sac_diagnostic import charged_actions
        final_actions = charged_actions(Path(final['root']) / 'experiment')
    total = sum(EXTRAS.values()) + sum(row['actions'] for row in rows) + final_actions
    return dict(format=FORMAT, historical_extra_actions=sum(EXTRAS.values()), total_action_charge=total,
        maximum_total_actions=1_000_000, within_ceiling=total <= 1_000_000, rows=rows,
        new_final_teacher_competence_actions=final_actions, final_teacher_competence_ceiling=COMPETENCE_ACTIONS,
        interpretation='All40 source leaves retained/admitted; actual cases and unused512-bank suffix reported explicitly')


def time_hold(plan, ledger, output, worker_seconds, *, actual=None):
    """All attempts/setup/retries share the proposed24h new SAC compute ceiling."""
    ledger, output = Path(ledger), Path(output).resolve()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = read(ledger) if ledger.exists() else dict(format=FORMAT, historical_extra_actions=sum(EXTRAS.values()),
            maximum_total_actions=1_000_000, leaves={row['slot']: dict(root=row['parent']['root'],
                maximum_total_actions=row['parent']['actions']) for row in plan['slots']}, history=[], time_holds={})
        if data['format'] != FORMAT or data['historical_extra_actions'] != sum(EXTRAS.values()):
            raise ValueError('Worker compute holds belong to another source campaign')
        holds = data.setdefault('time_holds', {})
        key = str(output)
        previous = holds.get(key)
        if previous is not None and previous['worker_seconds'] != worker_seconds:
            raise ValueError('A source worker cannot reset its cumulative compute ceiling')
        holds[key] = dict(worker_seconds=worker_seconds, plan_sha256=plan['plan_sha256'],
            charged_or_reserved_seconds=worker_seconds if actual is None else actual)
        cap = 1000 if plan['mode'] == 'qualification' else 86400
        if actual is None and sum(row['charged_or_reserved_seconds'] for row in holds.values()) > cap:
            raise ValueError('New worker would exceed proposed cumulative qualification/24h source compute hold')
        write(ledger, data)  # Actual overtime is retained rather than truncated.


def materialize_schedule(parent, path, cap):
    original = load_schedule(Path(parent['root']) / 'schedule.json')
    split = parent['slot'].split('/')[1]
    value = copy.deepcopy(original)
    count = len(value['tasks'][0]['partitions'][split])
    value['partition_budgets'][split] = partition_budget(count, 4, collection_mode='physical_budget',
        target_chunks=None, publish_every_cases=value['publish_every_cases'], total_action_budget=cap)
    budgets = value['partition_budgets'].values()
    value.update(maximum_chunks_per_task_partition=max(row['maximum_chunks'] for row in budgets),
        maximum_primitive_steps_per_task_partition=max(row['maximum_primitive_steps'] for row in budgets),
        planned_maximum_chunks=len(value['tasks']) * sum(row['maximum_chunks'] for row in value['partition_budgets'].values()),
        planned_maximum_primitive_steps=len(value['tasks']) * sum(row['maximum_primitive_steps'] for row in value['partition_budgets'].values()),
        planned_maximum_settling_steps=len(value['tasks']) * sum(row['maximum_settling_steps'] for row in value['partition_budgets'].values()),
        planned_maximum_total_actions=len(value['tasks']) * sum(row['maximum_total_actions'] for row in value['partition_budgets'].values()),
        planned_storage_allowance_bytes=len(value['tasks']) * sum(row['storage_allowance']['total_bytes_with_ten_percent_reserve'] for row in value['partition_budgets'].values()))
    value['schedule_sha256'] = content_hash({k: v for k, v in value.items() if k != 'schedule_sha256'})
    if Path(path).exists() and read(path) != value:
        raise ValueError('Materialized top-up schedule changed on resume')
    write(path, value)
    return load_schedule(path)


class Continuation:
    def __init__(self, plan, selected, output, stop_unix):
        self.plan, self.selected, self.parent = plan, selected, selected['parent']
        self.root, self.output = Path(output), Path(output) / 'experiment'
        self.stop_unix = stop_unix
        self.attempt = self.root / 'attempts' / str(time.time_ns())
        self.attempt.mkdir(parents=True)
        self.motor_before = None
        self.stop_reason = None

    def prepare(self):
        from .vla_libero import structure_digest
        source, target = Path(self.parent['experiment']), self.output
        actual = parent_record(self.parent['root'], self.selected['slot'])
        if actual != self.parent:
            raise ValueError('Terminal parent changed after metadata preparation')
        storage = self.plan.get('storage')
        if storage is not None:
            free = os.statvfs(self.root)
            if free.f_bavail * free.f_frsize < storage['shared_free_floor_bytes'] + storage['whole_case_atomic_headroom_bytes']:
                raise RuntimeError('Shared PFS capacity floor cannot fit a new migration/atomic recovery')
            probe = self.attempt / 'write-probe'
            with probe.open('xb') as stream:
                stream.write(bytes(4096))
                stream.flush()
                os.fsync(stream.fileno())
            probe.unlink()
        if (target / 'migration.json').exists():
            migration = read(target / 'migration.json')
            if migration['plan_sha256'] != self.plan['plan_sha256'] or migration['parent'] != self.parent:
                raise ValueError('Resume belongs to another source ancestry/plan')
            for row in migration['copied_parent_files']:
                if sha(source / row['path']) != row['sha256'] or sha(target / row['preserved_as']) != row['sha256']:
                    raise ValueError('An immutable parent or preserved source raw/recovery artifact changed')
            return
        if target.exists():
            raise ValueError('New source migration requires a fresh immutable experiment output')
        temporary = target.with_name('.experiment-preparing-' + str(time.time_ns()))
        shutil.copytree(source, temporary)
        archive = Path('ancestry') / self.parent['recovery_sha256']
        (temporary / archive).mkdir(parents=True)
        copied = []
        for file in source.rglob('*'):
            if not file.is_file() or file.name == 'run.lock':
                continue
            relative = file.relative_to(source)
            if sha(file) != sha(temporary / relative):
                raise ValueError('Copied source raw/recovery/milestone bytes changed')
            preserved = archive / relative if len(relative.parts) == 1 and file.name in MUTABLE else relative
            if preserved != relative:
                (temporary / relative).rename(temporary / preserved)
            copied.append(dict(path=str(relative), preserved_as=str(preserved), sha256=sha(file), bytes=file.stat().st_size))
        with (temporary / archive / 'source-recovery.pkl').open('rb') as stream:
            saved = pickle.load(stream)
        if (saved['contract'] != read(source / 'contract.json') or saved['boundary'] != 'complete physical episode'
                or saved['progress']['episode_in_case'] != 0 or saved['updates'] != self.parent['source_updates']
                or len(saved['replay']) != self.parent['chunks'] or saved['updates'] <= 0 or len(saved['replay']) <= 500):
            raise ValueError('Source migration requires complete-case postwarmup full SAC/replay/RNG state')
        write(temporary / 'migration.json', dict(format=FORMAT, plan_sha256=self.plan['plan_sha256'],
            parent=self.parent, copied_parent_files=copied, archive=str(archive),
            numerical_state_sha256=structure_digest({k: v for k, v in saved.items() if k != 'contract'})))
        temporary.rename(target)

    def bind_contract(self, value):
        from .vla_libero import structure_digest
        migration = read(self.output / 'migration.json')
        archive = self.output / migration['archive']
        old = read(archive / 'contract.json')
        for key in ('task', 'partition', 'cases', 'history_id', 'source_recipe', 'learner_seed', 'horizon', 'chunk', 'feature_contract', 'noise_contract'):
            if old[key] != value[key]:
                raise ValueError('Top-up changed original cases, recipe, public features or teacher clock identity: ' + key)
        value['continuation'] = dict(format=FORMAT, plan_sha256=self.plan['plan_sha256'], parent=self.parent,
            numerical_state_unchanged=True, source_cost='All current leaf journals plus21253 overhead; inherited actions counted once',
            shared_source_cost_group_sha256=content_hash(EXTRAS), shared_source_cost_actions=sum(EXTRAS.values()),
            stage='Restored updated SAC throughout every new episode; no new Gaussian warmup',
            labels='Actual AD noises/clocks; final source actor DPT labels include inherited+new history and full cumulative teacher cost')
        latest = self.output / 'source-recovery.pkl'
        if latest.exists():
            with latest.open('rb') as stream:
                current = pickle.load(stream)
            if current['contract'] != value:
                raise ValueError('Existing top-up continuation scientific contract changed')
            if not (self.output / 'contract.json').exists():
                write(self.output / 'contract.json', value)
            elif read(self.output / 'contract.json') != value:
                raise ValueError('Top-up latest/JSON contracts differ')
            return value
        with (archive / 'source-recovery.pkl').open('rb') as stream:
            saved = pickle.load(stream)
        before = structure_digest({k: v for k, v in saved.items() if k != 'contract'})
        saved['contract'] = value
        temporary = latest.with_suffix('.migration-partial')
        if temporary.exists():
            temporary.rename(temporary.with_name(temporary.name + '-uncommitted-' + str(time.time_ns())))
        with temporary.open('xb') as stream:
            pickle.dump(saved, stream, protocol=5)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(latest)
        shutil.copy2(latest, self.output / 'migration-initial-recovery.pkl')
        with latest.open('rb') as stream:
            after = structure_digest({k: v for k, v in pickle.load(stream).items() if k != 'contract'})
        if before != after or before != migration['numerical_state_sha256']:
            raise ValueError('Source migration changed a learner/optimizer/replay/sampler/RNG/progress leaf')
        write(self.output / 'contract.json', value)
        write(self.attempt / 'migration-fingerprint.json', dict(before=before, after=after, exact=True,
            numerical_state_changed=False, metadata_changes_only=True))
        return value

    def make_learner(self, root, seed):
        from .vla_libero import OfficialSAC, structure_digest
        self.dsrl_root, self.learner_seed = root, seed
        outer = self
        class VerifiedSourceSAC(OfficialSAC):
            def update(self, batch_size, sampler):
                result = super().update(batch_size, sampler)
                if not result or any(not math.isfinite(float(value)) for value in result.values()):
                    raise FloatingPointError('Nonfinite or absent real postwarmup SAC update metrics')
                self.last_update_metrics = result
                return result

            def restore(self, path, sampler, contract):
                from flax import serialization
                with Path(path).open('rb') as stream:
                    saved = pickle.load(stream)
                progress = super().restore(path, sampler, contract)
                expected = dict(agent=serialization.msgpack_restore(saved['agent']), agent_rng=saved['agent_rng'],
                    replay=saved['replay'], updates=saved['updates'], sampler=saved['sampler'],
                    numpy_rng=saved['numpy_rng'], python_rng=saved['python_rng'])
                actual = dict(agent=serialization.to_state_dict(self.agent._save_dict), agent_rng=np.asarray(self.agent._rng),
                    replay=self.replay, updates=self.updates, sampler=sampler.bit_generator.state,
                    numpy_rng=np.random.get_state(), python_rng=random.getstate())
                before, after = structure_digest(expected), structure_digest(actual)
                write(outer.attempt / 'restored-learner.json', dict(expected_sha256=before, actual_sha256=after,
                    passed=before == after, replay_rows=len(self.replay), updates=self.updates, optimizer_steps_during_restore=0))
                if before != after or self.updates <= 0 or len(self.replay) <= 500:
                    raise RuntimeError('Allocated source SAC restore changed a complete numeric state leaf or reset warmup')
                outer.expected_motor = copy.deepcopy(progress['motor_rng'])
                return progress
        return VerifiedSourceSAC(root, seed)

    def verify_motor_restore(self, motor):
        from .vla_libero import structure_digest
        from .vla_motor_replay import parameter_signature
        before, after = structure_digest(self.expected_motor), structure_digest(save_motor_rng(motor.policy._rng))
        write(self.attempt / 'motor-restore.json', dict(expected_sha256=before, actual_sha256=after, passed=before == after))
        if before != after:
            raise RuntimeError('Typed motor RNG changed during source continuation restore')
        self.motor_before = parameter_signature(motor.policy._model)
        write(self.attempt / 'parameters-before.json', self.motor_before)

    def require_physical_state(self, result):
        if any(not isinstance(result.get(key), str) or len(result[key]) != 64
                for key in ('initial_state_vector_sha256', 'initial_simulator_state_sha256')):
            raise RuntimeError('New source episodes require measured nominal and actual settled-state hashes')

    def validate_episode(self, summary, provenance):
        self.require_physical_state(summary)
        if (not provenance or any(row['source_stage'] != 'sac_online' for row in provenance)
                or len({row['source_updates_before'] for row in provenance}) != 1):
            raise RuntimeError('A restored source episode changed behavior stage/weights or restarted Gaussian warmup')

    def wall_stop(self):
        if time.time() + 120 >= self.stop_unix:
            self.stop_reason = 'caller_cumulative_time_deadline'
            return True
        storage = self.plan.get('storage')
        if storage is not None:
            free = os.statvfs(self.root)
            occupied = sum(path.stat().st_size for path in self.root.rglob('*') if path.is_file())
            if (free.f_bavail * free.f_frsize < storage['shared_free_floor_bytes'] + storage['whole_case_atomic_headroom_bytes']
                    or occupied + storage['whole_case_atomic_headroom_bytes'] > storage['per_stream_bytes']):
                self.stop_reason = 'storage_capacity_or_atomic_case_headroom'
                return True
        return False

    def finish(self, index, learner, sampler, motor):
        from .vla_motor_replay import parameter_signature
        after = parameter_signature(motor.policy._model)
        if after != self.motor_before:
            raise RuntimeError('Frozen source motor parameter bytes changed')
        write(self.attempt / 'parameters-after.json', dict(before=self.motor_before, after=after, unchanged=True))
        new_actions = index['total_action_budget_charge'] - self.parent['actions']
        advanced = learner.updates > self.parent['source_updates']
        new_episodes = len(index['episode_receipts']) - self.parent['episodes']
        remaining = len(index['case_coverage'].get('actual_initial_state_ids', []))
        index.update(continuation=dict(format=FORMAT, parent=self.parent, plan_sha256=self.plan['plan_sha256'],
            new_action_charge=new_actions, inherited_action_charge=self.parent['actions'],
            new_complete_episodes=new_episodes, new_optimizer_updates=learner.updates - self.parent['source_updates'],
            actual_distinct_nominal_state_ids=remaining, remaining_candidate_cases=512 - index['case_index'],
            actual_distinct_settled_states_measured=sum(1 for value in {row.get('initial_simulator_state_sha256')
                for row in index['episode_receipts']} if value is not None),
            episodes_without_settled_state_measurement=sum(row.get('initial_simulator_state_sha256') is None
                for row in index['episode_receipts']),
            all40_coverage='This stream is one admitted all40/fixed8 leaf, not exhaustion of the512-case bank',
            motor_parameters_unchanged=True, all_new_behavior_postwarmup=True),
            slice_completed=index['episode_in_case'] == 0, stopped_at_wall_boundary=self.wall_stop(),
            continuation_stop_reason=self.stop_reason,
            production_quota_exhausted=index['total_action_budget_charge'] + 1640 > self.selected['maximum_total_actions'])
        if not 0 <= new_actions <= self.selected['maximum_new_actions']:
            raise RuntimeError('Cumulative leaf exceeded its new-action quota')
        if self.plan['mode'] == 'qualification' and (not advanced or new_episodes < 4 or not index['collection_qualified']):
            raise RuntimeError('Bounded qualification did not complete K4/update/export/full recovery; no scale admission')
        if self.plan['mode'] == 'qualification':
            audit = copy.copy(self)
            audit.attempt = self.attempt / 'post-update-roundtrip'
            audit.attempt.mkdir()
            independent = audit.make_learner(self.dsrl_root, self.learner_seed)
            recovered = independent.restore(self.output / 'source-recovery.pkl', np.random.default_rng(),
                read(self.output / 'contract.json'))
            from .vla_libero import structure_digest
            if (independent.updates != learner.updates or len(independent.replay) != len(learner.replay)
                    or structure_digest(recovered['motor_rng']) != structure_digest(save_motor_rng(motor.policy._rng))):
                raise RuntimeError('Post-update independent full recovery or typed motor RNG round-trip changed')
            index['continuation']['post_update_independent_full_restore_passed'] = True
        latest = index['publications'][-1]
        write(self.output / 'candidate.json', dict(status='Unvalidated source SAC candidate; not an expert',
            full_checkpoint=latest['source_checkpoint'], sha256=latest['source_checkpoint_sha256'],
            full_latest_recovery_sha256=index['latest_recovery_sha256'], cumulative_stream_actions=index['total_action_budget_charge'],
            historical_extra_actions_counted_once=sum(EXTRAS.values()), plan_sha256=self.plan['plan_sha256'],
            historical_extra_cost_group_sha256=content_hash(EXTRAS),
            datasets=latest['datasets'], teacher_updates=learner.updates,
            latest_new_update_metrics=getattr(learner, 'last_update_metrics', None)))
        return index


def run(args):
    if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname:
        raise ValueError('Source migration/model computation requires the actual allocated Linux host')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    code = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=code, text=True).strip()):
        raise ValueError('Require the exact clean reviewed source-topup worktree')
    plan = load_plan(args.plan)
    if args.source_commit != plan['source_commit'] or not math.isfinite(args.stop_unix) or args.stop_unix <= time.time() + 120:
        raise ValueError('Require exact prepared source and live caller-ledger deadline')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'experiment').exists() and not args.resume:
        raise ValueError('Existing source top-up requires explicit exact-source --resume')
    with (args.output / 'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected = claim(plan, args.slot, output=args.output, ledger=args.ledger)
        continuation = Continuation(plan, selected, args.output, args.stop_unix)
        continuation.prepare()
        schedule_path = args.output / 'schedule.json'
        schedule = materialize_schedule(selected['parent'], schedule_path, selected['maximum_total_actions'])
        split = args.slot.split('/')[1]
        from .vla_source40_corpus import main as collect
        result = collect(['--schedule', str(schedule_path), '--output', str(args.output / 'experiment'),
            '--checkpoint', str(args.checkpoint), '--openpi-root', str(args.openpi_root), '--dsrl-root', str(args.dsrl_root),
            '--task-key', args.slot.split('/')[0], '--partition', split, '--source-commit', args.source_commit,
            '--max-primitive-steps', str(schedule['partition_budgets'][split]['maximum_primitive_steps']),
            '--max-total-actions', str(selected['maximum_total_actions']), '--port', str(args.port), '--resume'],
            continuation=continuation)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    make = commands.add_parser('plan')
    for name in ('receipts', 'project-root', 'evidence-root', 'output'):
        make.add_argument('--' + name, type=Path, required=True)
    make.add_argument('--source-commit', required=True)
    make.add_argument('--mode', choices=('qualification', 'production'), required=True)
    make.add_argument('--overrides', type=Path)
    clock = commands.add_parser('clock-begin')
    for name in ('plan', 'output'):
        clock.add_argument('--' + name, type=Path, required=True)
    clock.add_argument('--slot', required=True)
    clock.add_argument('--ledger', type=Path, required=True)
    clock.add_argument('--source-commit', required=True)
    clock.add_argument('--slice-seconds', type=int, required=True)
    clock.add_argument('--worker-seconds', type=int, required=True)
    clock.add_argument('--started-unix', type=float, required=True)
    clock.add_argument('--resume', action='store_true')
    end = commands.add_parser('clock-finish')
    end.add_argument('--output', type=Path, required=True)
    end.add_argument('--attempt', type=Path, required=True)
    end.add_argument('--exit-code', type=int, required=True)
    end.add_argument('--plan', type=Path, required=True)
    end.add_argument('--ledger', type=Path, required=True)
    inspect = commands.add_parser('status')
    inspect.add_argument('--ledger', type=Path, required=True)
    execute = commands.add_parser('run')
    for name in ('plan', 'ledger', 'output', 'checkpoint', 'openpi-root', 'dsrl-root'):
        execute.add_argument('--' + name, type=Path, required=True)
    for name in ('slot', 'source-commit', 'expected-hostname'):
        execute.add_argument('--' + name, required=True)
    execute.add_argument('--stop-unix', type=float, required=True)
    execute.add_argument('--port', type=int, default=8765)
    execute.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.operation == 'status':
        print(json.dumps(status(args.ledger), indent=2))
        return 0
    if args.operation == 'clock-finish':
        from .vla_source40_ppo_worker import finish
        plan_value = load_plan(args.plan)
        before = read(args.output / 'worker-time.json')
        if before['contract']['plan_sha256'] != plan_value['plan_sha256']:
            raise ValueError('Cannot close a source worker under another prepared plan')
        row = finish(args.output, args.attempt, args.exit_code)
        clock_value = read(args.output / 'worker-time.json')
        charge = sum(item.get('charged_seconds', item['reserved_seconds']) for item in clock_value['attempts'])
        time_hold(plan_value, args.ledger, args.output, clock_value['worker_seconds'], actual=charge)
        print(json.dumps(row))
        return 0
    if args.operation == 'clock-begin':
        from .vla_source40_ppo_worker import begin
        value = load_plan(args.plan)
        if args.source_commit != value['source_commit'] or args.slot not in expected_slots():
            raise ValueError('Worker clock differs from its prepared source/slot')
        if value['mode'] == 'qualification' and not 60 < args.worker_seconds <= 1000:
            raise ValueError('Qualification worker must fit its remaining shared1000-second ceiling')
        if not next(row for row in value['slots'] if row['slot'] == args.slot)['launchable']:
            raise ValueError('This preparation has no launchable quota for the requested stream')
        contract = dict(format=FORMAT, plan_sha256=value['plan_sha256'], slot=args.slot,
            source_commit=args.source_commit, run_id=args.output.name,
            purpose='Source-only SAC top-up; no resource approval granted by this receipt')
        time_hold(value, args.ledger, args.output, args.worker_seconds)
        print(begin(args.output, contract, slice_seconds=args.slice_seconds,
            worker_seconds=args.worker_seconds, resume=args.resume, now=args.started_unix))
        return 0
    if args.operation == 'plan':
        value = make_plan(args.receipts, project_root=args.project_root, evidence_root=args.evidence_root,
            source_commit=args.source_commit, mode=args.mode, overrides=None if args.overrides is None else read(args.overrides))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise ValueError('Prepared plans are immutable; use a fresh filename')
        write(args.output, value)
        print(json.dumps(dict(plan_sha256=value['plan_sha256'], inherited=value['inherited_action_charge'],
            maximum=value['maximum_charged_including_overhead'], unassigned=value['unassigned_actions'])))
        return 0
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
