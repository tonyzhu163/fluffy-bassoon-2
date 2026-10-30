"""Lossless, explicit target-online DSRL continuation on development tasks0/40.

Each worker owns one case and at most100k NEW actions. The terminal six-case
parent remains immutable; its original milestone prefix, experience and full
learner/RNG state are copied exactly. No budget is reserved by this module.
"""
import argparse
import fcntl
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

from . import vla_target_online_dsrl as onset
from . import vla_target_online_dsrl_six as six
from .vla_libero import structure_digest
from .vla_source40_ppo_worker import begin
from .vla_source40_schedule import content_hash


FORMAT = 'openpi_pi05_target_online_dsrl_development_continuation_v1'
PARENT_SOURCE = 'ba4674fafbe6d8f816979b61aa4ccd38e2912eff'
TASKS = (0, 40)
EPISODES = (0, 1, 4, 8, 13, 16, 40, 80, 160, 240)
MILESTONES = tuple((count, count * 410) for count in EPISODES)
INHERITED_CASE_ACTIONS, PARENT_CAMPAIGN_ACTIONS, NEW_ACTION_CAP = 9030, 35558, 100000
CASE_CAP = INHERITED_CASE_ACTIONS + NEW_ACTION_CAP


def read(path):
    return json.loads(Path(path).read_text())


def terminal_parent(root, task):
    root = Path(root).resolve()
    if task not in TASKS:
        raise ValueError('Only the explicitly selected development tasks0/40 may continue')
    with (root / 'launcher.lock').open('r') as lock:
        # Independent case workers may read a terminal parent concurrently;
        # its launcher holds an exclusive writer lock while active.
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        clock_path, index_path = root / 'worker-time.json', root / 'experiment/index.json'
        clock, index = read(clock_path), read(index_path)
        source = root / 'experiment' / f'case-{task:02d}'
        case_index, case_contract = read(source / 'index.json'), read(source / 'contract.json')
        charge = sum(row.get('charged_seconds', row['reserved_seconds']) for row in clock['attempts'])
        if ((root / 'exit-code.txt').read_text().strip() != '0' or index.get('completed') is not True
                or index.get('format') != six.FORMAT or index.get('total_action_charge') != PARENT_CAMPAIGN_ACTIONS
                or clock.get('worker_seconds') != 7200 or charge > 7200
                or not clock['attempts'] or clock['attempts'][-1].get('exit_code') != 0
                or any(row.get('status') != 'finished' or type(row.get('charged_seconds')) is not int
                       or row['charged_seconds'] < 0 for row in clock['attempts'])
                or index.get('contract', {}).get('source_commit') != PARENT_SOURCE
                or clock.get('contract', {}).get('source_commit') != PARENT_SOURCE
                or clock['contract'].get('run_id') != root.name
                or case_index.get('total_action_charge') != INHERITED_CASE_ACTIONS
                or case_index.get('training_chunks') != 640 or case_index.get('target_optimizer_updates') != 3200
                or case_index.get('case') != onset.case(task)):
            raise ValueError('Require the exact terminal six-case parent and its complete16-episode task recovery')
        six.validate_completed(case_index, case_contract, source)
        return dict(root=str(root), source_case=str(source), task=task,
            parent_source_commit=PARENT_SOURCE, parent_index_sha256=onset.sha256(index_path),
            parent_worker_time_sha256=onset.sha256(clock_path), parent_worker_charge_seconds=charge,
            parent_campaign_action_charge=PARENT_CAMPAIGN_ACTIONS, inherited_case_action_charge=INHERITED_CASE_ACTIONS,
            parent_case_index_sha256=onset.sha256(source / 'index.json'),
            parent_case_contract_sha256=onset.sha256(source / 'contract.json'),
            parent_recovery_sha256=case_index['latest_recovery_sha256'])


def contract(source, task, parent, worker_seconds):
    if (len(source) != 40 or any(char not in '0123456789abcdef' for char in source)
            or task not in TASKS or worker_seconds != 6000):
        raise ValueError('Require an exact source and explicit6000-second per-worker ceiling')
    return dict(format=FORMAT, source_commit=source, task=task, case=onset.case(task), parent=parent,
        milestones=[dict(training_episodes=episodes, training_action_ceiling=cost) for episodes, cost in MILESTONES],
        original_milestone_prefix=list(EPISODES[:6]), existing_query16='Reused immutable parent query, not rerun',
        maximum_new_actions=NEW_ACTION_CAP, inherited_case_actions=INHERITED_CASE_ACTIONS,
        maximum_cumulative_case_actions=CASE_CAP, maximum_worker_seconds=worker_seconds,
        recipe=six.RECIPE, learner_seed=0, target_only=True, source_assembly_admission=False,
        source_checkpoint_reused=False, target_checkpoint_reused=True, prefix_features_are_actor_input=False,
        prefix_features_recorded=True, task_selection='Outcome-based development diagnosis after failing onset queries; not held-out confirmatory selection',
        unique_campaign_cost='Parent35558 counted ONCE + each worker(cumulative_case_actions-9030); do not add copied prefixes again',
        time_cost='Parent six-run actual GPU charge counted ONCE + all new worker attempts; no parent budget is automatically extended',
        stopping='Whole episode plus published optimizer burst; query after40/80/160/240 if cap/time permits; exact recovery if earlier stop',
        interpretation='Longer development learning curve, not converged DSRL or a matched reproduction of the original paper')


def case_contract(global_contract, parent_contract):
    value = dict(parent_contract, source_commit=global_contract['source_commit'],
        contract_sha256=content_hash(global_contract), maximum_total_actions=CASE_CAP,
        continuation=dict(parent=global_contract['parent'], original_case_contract_sha256=content_hash(parent_contract),
            numerical_state_unchanged=True, original_milestone_prefix=list(EPISODES[:6]), maximum_new_actions=NEW_ACTION_CAP))
    return value


def state_fingerprint(saved):
    return structure_digest({key: value for key, value in saved.items() if key != 'contract'})


class VerifiedSAC(onset.OfficialSAC):
    """Audit the real allocated full restore before any subsequent gradient step."""
    def __init__(self, root, audit_directory):
        super().__init__(root, 0)
        self.audit_directory = Path(audit_directory)

    def restore(self, path, sampler, source_contract):
        from flax import serialization
        with Path(path).open('rb') as stream:
            saved = pickle.load(stream)
        progress = super().restore(path, sampler, source_contract)
        expected = dict(agent=serialization.msgpack_restore(saved['agent']), agent_rng=saved['agent_rng'],
            replay=saved['replay'], updates=saved['updates'], sampler=saved['sampler'],
            numpy_rng=saved['numpy_rng'], python_rng=saved['python_rng'])
        actual = dict(agent=serialization.to_state_dict(self.agent._save_dict), agent_rng=np.asarray(self.agent._rng),
            replay=self.replay, updates=self.updates, sampler=sampler.bit_generator.state,
            numpy_rng=np.random.get_state(), python_rng=random.getstate())
        before, after = structure_digest(expected), structure_digest(actual)
        onset.write_json(self.audit_directory/'restored-learner.json', dict(passed=before==after,
            checkpoint_sha256=onset.sha256(path), expected_full_state_sha256=before, restored_full_state_sha256=after,
            learner_updates=self.updates, replay_rows=len(self.replay), optimizer_steps_during_restore=0,
            motor_rng_in_checkpoint=progress.get('motor_rng',{}).get('format')))
        if before != after:
            raise RuntimeError('Allocated continuation restore changed a learner/replay/optimizer/sampler/global RNG leaf')
        return progress


def migrate(parent, output, global_contract):
    """Called ONLY by the allocated driver; copies all parent bytes before editing metadata."""
    output = Path(output)
    source = Path(parent['source_case'])
    target = output / f'case-{parent["task"]:02d}'
    expected = case_contract(global_contract, read(source / 'contract.json'))
    if target.exists():
        migration = read(target / 'migration.json')
        if (read(target / 'contract.json') != expected or migration.get('global_contract_sha256') != content_hash(global_contract)
                or migration.get('parent') != parent
                or onset.sha256(target / 'migration-initial-recovery.pkl') != migration['initial_recovery_sha256']):
            raise ValueError('Explicit continuation migration changed on exact-source resume')
        for item in migration['copied_parent_files']:
            if (onset.sha256(target / item['preserved_as']) != item['sha256']
                    or onset.sha256(source / item['path']) != item['sha256']):
                raise ValueError('Immutable parent or copied raw/checkpoint/journal evidence changed')
        return target, migration
    temporary = output / f'.case-{parent["task"]:02d}-preparing-{time.time_ns()}'
    shutil.copytree(source, temporary)
    copied = []
    for path in sorted(source.rglob('*')):
        if path.is_file():
            relative = path.relative_to(source)
            digest = onset.sha256(path)
            if onset.sha256(temporary / relative) != digest:
                raise ValueError('Parent raw/checkpoint/journal copy is not byte-identical')
            preserved = 'parent-' + str(relative) if str(relative) in ('contract.json','index.json','target-recovery.pkl') else str(relative)
            copied.append(dict(path=str(relative), preserved_as=preserved, sha256=digest, bytes=path.stat().st_size))
    if onset.charged(temporary) != INHERITED_CASE_ACTIONS:
        raise ValueError('Copied case journals lost parent physical/query/qualification costs')
    with (temporary / 'target-recovery.pkl').open('rb') as stream:
        saved = pickle.load(stream)
    if (saved.get('contract') != read(source / 'contract.json')
            or saved.get('progress', {}).get('training_episodes') != 16
            or saved['progress'].get('next_milestone') != 6):
        raise ValueError('Parent continuation must start from the complete16-episode RNG/cursor boundary')
    before = state_fingerprint(saved)
    for name in ('contract.json', 'index.json', 'target-recovery.pkl'):
        (temporary / name).rename(temporary / ('parent-' + name))
    saved['contract'] = expected
    with (temporary / 'target-recovery.pkl').open('wb') as stream:
        pickle.dump(saved, stream, protocol=5)
    shutil.copy2(temporary / 'target-recovery.pkl', temporary / 'migration-initial-recovery.pkl')
    with (temporary / 'target-recovery.pkl').open('rb') as stream:
        restored = pickle.load(stream)
    after = state_fingerprint(restored)
    if before != after or onset.sha256(source / 'target-recovery.pkl') != parent['parent_recovery_sha256']:
        raise ValueError('Migration changed a learner/replay/optimizer/RNG/progress leaf or the immutable parent')
    onset.write_json(temporary / 'contract.json', expected)
    migration = dict(format=FORMAT + '_migration', global_contract_sha256=content_hash(global_contract), parent=parent,
        numerical_state_unchanged=True, state_sha256_before=before, state_sha256_after=after,
        initial_recovery_sha256=onset.sha256(temporary / 'migration-initial-recovery.pkl'),
        copied_parent_files=copied, source_contract_changes_only=True,
        source_contract_note='Historical milestone recoveries remain untouched; only latest recovery contract/source/action limits change')
    onset.write_json(temporary / 'migration.json', migration)
    temporary.rename(target)
    return target, migration


def run(args):
    if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname:
        raise ValueError('Target continuation migration/model computation requires allocated Linux')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed continuation commit')
    parent = terminal_parent(args.parent_root, args.task)
    science = contract(args.source_commit, args.task, parent, args.worker_seconds)
    remaining = args.stop_unix - time.time()
    if not math.isfinite(remaining) or not 0 < remaining <= args.worker_seconds:
        raise ValueError('Require an explicit live invocation stop inside this cumulative worker hold')
    deadline, started = time.monotonic() + remaining, time.monotonic()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume:
        if read(args.output / 'contract.json') != science:
            raise ValueError('Continuation resume changed its exact contract')
    else:
        onset.write_json(args.output / 'contract.json', science)
    output, migration = migrate(parent, args.output, science)
    attempt = args.output / 'attempts' / str(time.time_ns())
    attempt.mkdir(parents=True)
    original = args.output
    args.output, args.model_config, args.feature_interface = attempt, 'pi05_libero', 'vla-prefix'
    motor = onset.FrozenOpenPI(args)
    args.output = original
    parent_features = read(Path(parent['root']) / 'experiment/recorded-feature-contract.json')
    if (motor.metadata['receipt_sha256'] != onset.PI05_RECEIPT or motor.horizon != 10
            or motor.feature_encoder.contract != parent_features):
        raise ValueError('Continuation changed the frozen motor or recorded prefix interface')
    server = onset.TargetServer(task_id=args.task, python=args.simulator_python,
        pythonpath=args.simulator_pythonpath, output=attempt / 'simulator', port=args.port, libero_root=args.libero_root)
    try:
        if (server.metadata.get('suite') != 'libero_90' or server.metadata.get('task_id') != args.task
                or server.metadata.get('horizon') != 400 or server.metadata.get('settle_steps') != 10
                or server.metadata.get('libero_commit') != onset.LIBERO_COMMIT
                or server.metadata.get('fresh_environment_on_reset') is not True):
            raise ValueError('Continuation simulator changed its task/horizon/native fresh-reset contract')
        journal = onset.JournaledSimulator(server, output, args.output, case_cap=CASE_CAP, total_cap=CASE_CAP)
        if not journal.reserve_episode():
            raise RuntimeError('Remaining continuation cap cannot qualify and finish another query/episode')
        initial = journal.reset(0, 123)
        if initial['initial_state_vector_sha256'] != read(Path(parent['source_case'])/'index.json')['nominal_state_sha256']:
            raise ValueError('Continuation qualification changed the parent nominal physical case')
        zeros = np.zeros((10, 32), np.float32)
        a, b = motor.infer(initial['observation'], server.metadata['instruction'], zeros), motor.infer(initial['observation'], server.metadata['instruction'], zeros)
        if not np.allclose(a, b, rtol=0, atol=1e-5):
            raise RuntimeError('Continuation motor same-noise qualification failed')
        onset.write_json(attempt / 'qualification.json', dict(max_delta=float(np.max(np.abs(a-b))), settling_actions=10,
            feature_parity=motor.feature_encoder.qualify(initial['observation'], server.metadata['instruction'])))
        learner, sampler = VerifiedSAC(args.dsrl_root, attempt), np.random.default_rng(0)
        result = onset.run_case(server, motor, learner, sampler, target=onset.case(args.task), output=output, root=args.output,
            contract=case_contract(science, read(Path(parent['source_case']) / 'contract.json')), resume=True,
            # Existing run_case reserves90s per episode. The additional180s
            # covers fresh JIT/update compilation and final growing checkpoints.
            deadline=deadline-180, milestones=MILESTONES, case_cap=CASE_CAP, total_cap=CASE_CAP, record_features=True)
    finally:
        server.close()
    cumulative = onset.joint_charge(args.output)
    onset.write_json(args.output / 'index.json', dict(format=FORMAT, contract=science, completed=result['completed'],
        case=result, migration=migration, cumulative_case_actions=cumulative,
        inherited_case_actions=INHERITED_CASE_ACTIONS, new_action_charge=cumulative-INHERITED_CASE_ACTIONS,
        parent_campaign_action_charge=PARENT_CAMPAIGN_ACTIONS,
        unique_campaign_charge_if_this_worker_only=PARENT_CAMPAIGN_ACTIONS+cumulative-INHERITED_CASE_ACTIONS,
        unique_two_worker_formula='Parent35558 ONCE + new_action_charge(worker0) + new_action_charge(worker40)',
        elapsed_seconds=time.monotonic()-started, target_only=True, source_assembly_admission=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    start = sub.add_parser('clock-begin')
    execute = sub.add_parser('run')
    for selected in (start, execute):
        for name in ('parent-root', 'output'):
            selected.add_argument('--'+name, type=Path, required=True)
        selected.add_argument('--source-commit', required=True)
        selected.add_argument('--task', type=int, choices=TASKS, required=True)
        selected.add_argument('--worker-seconds', type=int, choices=(6000,), required=True)
        selected.add_argument('--resume', action='store_true')
    start.add_argument('--run-id', required=True)
    start.add_argument('--started-unix', type=float, required=True)
    for name in ('checkpoint','openpi-root','dsrl-root','libero-root','simulator-python'):
        execute.add_argument('--'+name, type=Path, required=True)
    for name in ('expected-hostname','simulator-pythonpath'):
        execute.add_argument('--'+name, required=True)
    execute.add_argument('--stop-unix', type=float, required=True)
    execute.add_argument('--port', type=int, default=8780)
    args = parser.parse_args(argv)
    if args.operation == 'run':
        run(args)
    else:
        parent = terminal_parent(args.parent_root,args.task)
        science = contract(args.source_commit,args.task,parent,args.worker_seconds)
        worker_contract = dict(operation=FORMAT, source_commit=args.source_commit, run_id=args.run_id,
            task=args.task, scientific_contract_sha256=content_hash(science), parent=parent,
            maximum_new_actions=NEW_ACTION_CAP, maximum_worker_seconds=args.worker_seconds)
        clock_path = args.output/'worker-time.json'
        charge = (sum(row.get('charged_seconds',row['reserved_seconds']) for row in read(clock_path)['attempts'])
            if clock_path.exists() else 0)
        seconds = args.worker_seconds-charge-60
        attempt = begin(args.output,worker_contract,slice_seconds=seconds,worker_seconds=args.worker_seconds,
            resume=args.resume,now=args.started_unix)
        print(f'{attempt}\t{math.floor(args.started_unix+seconds)}\t{seconds}')


if __name__ == '__main__':
    main()
