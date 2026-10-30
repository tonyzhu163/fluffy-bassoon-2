"""Frozen competence queries for eight canonical PER-TASK source SAC actors.

Native full-H Gaussian, deterministic SAC mean, and stochastic SAC sample are
queried on the same two source-local states. No replay insertion or SGD occurs.
An explicit alias must bind all costs to the terminal PPO validation's unused
original56k/4h envelope; this module makes no resource reservation.
"""
import argparse
from dataclasses import asdict, dataclass
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from .checkpoint import restore_rng, rng_state
from .vla_history import file_sha256
from .vla_libero import FrozenOpenPI, OfficialSAC, sac_observation, write_json
from .vla_source40_corpus import PI05_RECEIPT, restore_motor_rng, save_motor_rng
from .vla_source40_evaluate import CostLedger, _save_recovery, rollout_episode
from .vla_source40_frozen_validation import invocation_deadline
from .vla_source40_multitask_ppo import OwnedSimulator, source_tasks
from .vla_source40_memory_replication import CaptureMotor, CaptureSimulator, raw_record
from .vla_source40_schedule import SOURCE_SUITES, content_hash, load_schedule
from .vla_source40_subset_evaluate import FrozenSAC
from .vla_reset_identity import RESET_IDENTITY, comparison


FORMAT = 'source40_per_task_sac_competence_v1'
ALIAS_FORMAT = 'source40_validation_remainder_alias_v1'
METHODS = ('native', 'mean', 'sample')
MAX_ACTIONS, MAX_SECONDS = 20000, 3600
KEYS = [f'{suite}:{task:02d}' for suite in SOURCE_SUITES for task in (0, 1)]


def read(path):
    return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class SourceCase:
    benchmark: str
    case_id: str
    suite: str
    task_id: int
    initial_state: int
    scene_seed: int


def query_plan():
    rows = []
    for key in KEYS:
        suite, number = key.split(':')
        for trial in range(2):
            case = SourceCase('source40', f'{key}/source-validation/state-{40 + trial}',
                suite, int(number), 40 + trial,
                int(content_hash(dict(task=key, validation_trial=trial, seed=2))[:8], 16) % (2**31 - 1))
            for method in METHODS:
                rows.append(dict(task_key=key, case=asdict(case), method=method))
    return rows


def load_alias(path, expected_sha256, *, run_id, source_commit):
    if file_sha256(path) != expected_sha256:
        raise ValueError('Source validation remainder alias checksum differs')
    value = read(path)
    parent = Path(value.get('parent_output_root', ''))
    index_path, clock_path = parent / 'experiment/index.json', parent / 'worker-time.json'
    index, clock = read(index_path), read(clock_path)
    ledger = read(value['canonical_ppo_ledger_path'])
    reservation = ledger.get('reservations', {}).get(parent.name, {})
    charged_seconds = sum(row.get('charged_seconds', row['reserved_seconds']) for row in clock['attempts'])
    parent_charge = index.get('total_action_charge')
    if (value.get('format') != ALIAS_FORMAT or value.get('run_id') != run_id
            or value.get('source_commit') != source_commit or value.get('maximum_total_actions') != MAX_ACTIONS
            or type(value.get('maximum_worker_seconds')) is not int
            or not 600 < value['maximum_worker_seconds'] <= MAX_SECONDS
            or value.get('parent_terminal_index_sha256') != file_sha256(index_path)
            or value.get('parent_worker_time_sha256') != file_sha256(clock_path)
            or value.get('parent_total_action_charge') != parent_charge
            or value.get('parent_worker_charge_seconds') != charged_seconds
            or (parent / 'exit-code.txt').read_text().strip() != '0'
            or index.get('format') != 'source40_frozen_ppo_validation_v1'
            or index.get('completed') is not True or index.get('weights_unchanged') is not True
            or index.get('optimizer_updates') != 0 or clock.get('worker_seconds') != 14400
            or clock.get('contract', {}).get('operation') != 'frozen_source_ppo_validation'
            or clock['contract'].get('run_id') != parent.name
            or clock['contract'].get('source_commit') != index.get('contract', {}).get('source_commit')
            or clock['contract'].get('maximum_total_actions') != 56000
            or not clock['attempts'] or clock['attempts'][-1].get('exit_code') != 0
            or any(row.get('status') != 'finished' or type(row.get('charged_seconds')) is not int
                   or row['charged_seconds'] < 0 for row in clock['attempts'])
            or reservation.get('maximum_total_actions') != 56000
            or reservation.get('source_commit') != index.get('contract', {}).get('source_commit')
            or reservation.get('campaign_sha256') != ledger.get('campaign_sha256')
            or run_id in ledger.get('reservations', {})
            or type(parent_charge) is not int or parent_charge + MAX_ACTIONS > 56000
            or charged_seconds + value['maximum_worker_seconds'] > 14400
            or value.get('original_action_ceiling') != 56000 or value.get('original_worker_seconds') != 14400):
        raise ValueError('Require a terminal successful PPO validation and an explicit unused original56k/4h remainder; no duplicate reservation')
    return value


def source_bank(ledger_path, outputs_root, schedule):
    """Select only each fixed goal's canonical TRAIN final publication."""
    ledger, tasks = read(ledger_path), {row['task_key']: row for row in source_tasks(schedule)}
    result = {}
    for key in KEYS:
        reservation = ledger['reservations'][key + '/train']
        root = Path(outputs_root) / reservation['run_id'] / 'experiment'
        index_path, contract_path = root / 'index.json', root / 'contract.json'
        index, contract = read(index_path), read(contract_path)
        publications = [row for row in index.get('publications', []) if row.get('status') == 'paired_ad_dpt']
        if not publications:
            raise ValueError('Canonical source actor has no committed publication')
        final = max(publications, key=lambda row: row['through_case'])
        relative = Path(final['source_checkpoint'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Canonical source checkpoint must stay inside its immutable stream')
        checkpoint = root / relative
        task = contract.get('task', {})
        if (index.get('completed') is not True or index.get('collection_qualified') is not True
                or index.get('source_only') is not True or index.get('partition') != 'train'
                or index.get('task_key') != key or contract.get('partition') != 'train'
                or contract.get('source_only') is not True or contract.get('source_data_from_targets') is not False
                or contract.get('source_commit') != reservation.get('source_commit')
                or contract.get('motor', {}).get('receipt_sha256') != PI05_RECEIPT
                or contract.get('horizon') != 400 or contract.get('chunk') != 10
                or task.get('task_key') != key or task.get('bddl_sha256') != tasks[key]['bddl_sha256']
                or content_hash(contract) != index.get('source_contract_sha256')
                or index.get('schedule_sha256') != schedule['schedule_sha256']
                or file_sha256(checkpoint) != final.get('source_checkpoint_sha256')
                or final.get('episodes') != len(index.get('episode_receipts', []))
                or final.get('episodes', 0) < 4 or final['episodes'] % 4
                or final.get('transitions') != index.get('chunks')
                or index.get('source_updates', 0) < 1
                or any(row.get('initial_state', 40) >= 40 for row in index.get('episode_receipts', []))):
            raise ValueError('Require the matching canonical per-task TRAIN actor, complete source-only history and frozen motor')
        result[key] = dict(task=tasks[key], index_path=str(index_path), index_sha256=file_sha256(index_path),
            contract_path=str(contract_path), contract_sha256=file_sha256(contract_path),
            checkpoint_path=str(checkpoint), checkpoint_sha256=final['source_checkpoint_sha256'],
            source_updates=index['source_updates'], source_episodes=len(index['episode_receipts']),
            through_case=final['through_case'],
            source_train_action_charge=index['total_action_budget_charge'],
            canonical_reservation=reservation, feature_contract=contract['feature_contract'])
    return result


class FrozenTeacher(FrozenSAC):
    def __init__(self, learner, *, deterministic, output=None):
        super().__init__(learner)
        self.deterministic = deterministic
        self.output = None if output is None else Path(output)

    def act(self, observation):
        if self.pending:
            raise RuntimeError('Consume the frozen teacher outcome before another query action')
        inputs = sac_observation(observation, batched=True)
        if inputs['pixels'].shape != (1, 64, 64, 3, 1) or inputs['state'].shape != (1, 8, 1):
            raise ValueError('Frozen teacher requires canonical pixel64/proprio8 inputs')
        self.pending = True
        action = self.learner.action(observation, deterministic=self.deterministic)
        if self.output is not None:
            raw_record(self.output / f'{time.time_ns()}-teacher-proposal.pkl.gz', dict(
                observation=observation, sac_observation=inputs, proposed_noise=np.asarray(action).copy(),
                deterministic=self.deterministic, source_updates=self.learner.updates))
        return action


class QuerySimulator(CaptureSimulator):
    def reset(self, state, seed):
        self.initial_reset = super().reset(state, seed)
        return self.initial_reset


def charged_actions(output):
    total = 0
    for path in Path(output).glob('query-*/physical-journal'):
        costs = CostLedger(path, max_primitive_steps=MAX_ACTIONS, max_settling_steps=MAX_ACTIONS).costs()
        total += sum(row[name] for row in costs.values() for name in
            ('primitive_steps', 'uncertain_primitive_steps_ceiling', 'settling_steps', 'uncertain_settling_steps_ceiling'))
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'alias', 'schedule', 'source-ledger', 'outputs-root', 'checkpoint', 'openpi-root', 'dsrl-root', 'simulator-python'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('alias-sha256', 'source-commit', 'expected-hostname', 'simulator-pythonpath'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--invocation-stop-unix', type=float, required=True)
    parser.add_argument('--port', type=int, default=8773)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname:
        raise ValueError('Frozen teacher computation requires the explicitly allocated Linux host')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed diagnostic commit')
    alias = load_alias(args.alias, args.alias_sha256, run_id=args.output.parent.name, source_commit=args.source_commit)
    deadline = invocation_deadline(alias['maximum_worker_seconds'], args.invocation_stop_unix)
    started = time.monotonic()
    schedule = load_schedule(args.schedule)
    bank = source_bank(args.source_ledger, args.outputs_root, schedule)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = args.output / 'attempts' / str(time.time_ns())
    attempt.mkdir(parents=True)
    motor = FrozenOpenPI(SimpleNamespace(**dict(vars(args), output=attempt,
        model_config='pi05_libero', feature_interface='vla-prefix')))
    if motor.metadata['receipt_sha256'] != PI05_RECEIPT or motor.horizon != 10:
        raise ValueError('Require the same qualified frozen OpenPI40 motor')
    if any(spec['feature_contract'] != motor.feature_encoder.contract for spec in bank.values()):
        raise ValueError('Canonical source bank and current frozen motor feature contracts differ')
    contract = dict(format=FORMAT, source_commit=args.source_commit, alias_path=str(args.alias),
        alias_sha256=args.alias_sha256, source_bank=bank, queries=query_plan(), maximum_total_actions=MAX_ACTIONS,
        maximum_worker_seconds=alias['maximum_worker_seconds'], horizon=400, settle_steps=10,
        optimizer_updates=0, replay_insertion=False, source_pretraining_data=False,
        reset_identity=RESET_IDENTITY,
        methods=dict(native='Independent unbounded Gaussian[10,32]',
            mean='Corresponding per-task actor.eval_actions; clip32D coordinate[-1,1], repeat10slots',
            sample='Corresponding per-task actor.sample_actions; clip32D coordinate[-1,1], repeat10slots'),
        model_scope='Eight separately trained source-task actors, NOT a shared40-task SAC',
        interpretation='Source teacher competence on disjoint local states; no target transfer or novel-goal claim',
        motor=motor.metadata, feature_contract=motor.feature_encoder.contract)
    recovery = args.output / 'diagnostic-recovery.pkl'
    if args.resume:
        if read(args.output / 'contract.json') != contract:
            raise ValueError('Frozen teacher diagnostic resume changed its contract')
        with recovery.open('rb') as stream:
            progress = pickle.load(stream)
        if progress['contract_sha256'] != content_hash(contract):
            raise ValueError('Frozen teacher diagnostic recovery checksum contract differs')
    else:
        write_json(args.output / 'contract.json', contract)
        progress = dict(contract_sha256=content_hash(contract), cursor=0, receipts=[],
            global_rng=rng_state(), motor_rng=save_motor_rng(motor.policy._rng), learner_rng=None,
            case_references={})
        _save_recovery(recovery, progress)
    learner, sampler, active_key, server = OfficialSAC(args.dsrl_root, 0), np.random.default_rng(0), None, None
    stop = None
    try:
        for ordinal in range(progress['cursor'], len(contract['queries'])):
            row = contract['queries'][ordinal]
            if charged_actions(args.output) + 410 > MAX_ACTIONS or time.monotonic() + 180 >= deadline:
                stop = 'complete_query_action_or_time_reservation'
                break
            key, case = row['task_key'], SourceCase(**row['case'])
            new_task = active_key != key
            if new_task:
                if server is not None:
                    server.close()
                spec = bank[key]
                source_contract = read(spec['contract_path'])
                source_progress = learner.restore(spec['checkpoint_path'], sampler, source_contract)
                if (learner.updates != spec['source_updates'] or source_progress['episode_in_case'] != 4
                        or source_progress['case_index'] != spec['through_case']
                        or len(source_progress['episode_receipts']) != spec['source_episodes']):
                    raise ValueError('Canonical frozen source recovery update count or complete-case boundary differs')
                server = OwnedSimulator(spec['task'], python=args.simulator_python,
                    pythonpath=args.simulator_pythonpath, output=attempt / f'server-{ordinal:02d}', port=args.port)
                active_key = key
                if progress.get('active_task') == key and progress.get('learner_rng') is not None:
                    import jax.numpy as jnp
                    learner.agent._rng = jnp.asarray(progress['learner_rng'])
            output = args.output / f'query-{ordinal:03d}'
            output.mkdir(exist_ok=True)
            raw = output / 'raw-rpc'
            raw.mkdir(exist_ok=True)
            controller = None if row['method'] == 'native' else FrozenTeacher(learner,
                deterministic=row['method'] == 'mean', output=raw)
            frozen = FrozenTeacher(learner, deterministic=True)
            before, updates, replay_length = frozen.frozen_signature(), learner.updates, len(learner.replay)
            replay_sha = hashlib.sha256(pickle.dumps(learner.replay, protocol=5)).hexdigest()
            restore_rng(progress['global_rng'])
            motor.policy._rng = restore_motor_rng(progress['motor_rng'])
            ledger = CostLedger(output / 'physical-journal', max_primitive_steps=MAX_ACTIONS,
                max_settling_steps=MAX_ACTIONS, max_total_actions=MAX_ACTIONS)
            def reset_motor(seed):
                import jax
                motor.policy._rng = jax.random.key(seed)
            def qualify(observation):
                zeros = np.zeros((10, 32), np.float32)
                a, b = motor.infer(observation, server.metadata['instruction'], zeros), motor.infer(observation, server.metadata['instruction'], zeros)
                delta = float(np.max(np.abs(a - b)))
                if not math.isfinite(delta) or delta > 1e-5:
                    raise RuntimeError('Frozen common-motor same-noise qualification failed')
                write_json(output / f'qualification-{time.time_ns()}.json', dict(same_noise_max_delta=delta,
                    feature_parity=motor.feature_encoder.qualify(observation, server.metadata['instruction']), extra_simulator_actions=0))
            captured_motor = CaptureMotor(motor, raw)
            captured_sim = QuerySimulator(server, captured_motor, raw)
            rows, receipt, query_rng = rollout_episode(captured_sim, captured_motor, controller, case=case,
                method='native' if controller is None else 'frozen_source_sac', role='query', ordinal=0,
                episode_id=f'{key}/{case.initial_state}/{row["method"]}', ledger=ledger,
                reset_motor_rng=reset_motor, qualify=qualify if new_task else None, deadline=deadline)
            after = frozen.frozen_signature()
            if (before != after or learner.updates != updates or len(learner.replay) != replay_length
                    or hashlib.sha256(pickle.dumps(learner.replay, protocol=5)).hexdigest() != replay_sha):
                raise RuntimeError('Frozen source actor/critic/optimizer/replay changed during competence queries')
            for item in rows:
                item['sac_observation'] = sac_observation(item['motor_observation'])
                item['sac_next_observation'] = sac_observation(item['motor_next_observation'])
            reference = progress['case_references'].setdefault(case.case_id, captured_sim.initial_reset)
            reset_audit = comparison(reference, captured_sim.initial_reset, camera_max_lsb=1)
            path = output / f'episode-{time.time_ns()}.pkl.gz'
            with gzip.open(path, 'xb', compresslevel=1) as stream:
                pickle.dump(dict(rows=rows, receipt=receipt, method=row['method'], source=bank[key],
                    full_learner_signature_before=before, full_learner_signature_after=after,
                    reset_audit=reset_audit,
                    source_updates=updates, replay_rows=replay_length, replay_sha256=replay_sha,
                    query_rng=query_rng, optimizer_updates=0), stream, protocol=5)
            write_json(output / 'reset-audit.json', reset_audit)
            if not reset_audit['allowed']:
                raise RuntimeError('Source query physical/proprio identity or explicit one-LSB camera bound changed; raw evidence retained')
            progress['receipts'].append(dict(**row, receipt=receipt, trace=str(path.relative_to(args.output)),
                trace_sha256=file_sha256(path), full_learner_signature=before, optimizer_updates=0))
            progress.update(cursor=ordinal + 1, global_rng=rng_state(), motor_rng=save_motor_rng(motor.policy._rng),
                learner_rng=np.asarray(learner.agent._rng).copy(), active_task=key,
                source_checkpoint_sha256=bank[key]['checkpoint_sha256'])
            _save_recovery(recovery, progress)
            write_json(args.output / 'progress.json', dict(completed_queries=progress['cursor'],
                total_action_charge=charged_actions(args.output), latest_recovery_sha256=file_sha256(recovery)))
    finally:
        if server is not None:
            server.close()
    write_json(args.output / 'index.json', dict(format=FORMAT, contract=contract,
        completed=progress['cursor'] == len(contract['queries']), receipts=progress['receipts'], stop_reason=stop,
        total_action_charge=charged_actions(args.output), elapsed_seconds=time.monotonic() - started,
        latest_recovery_sha256=file_sha256(recovery), optimizer_updates=0, replay_insertion=False))


if __name__ == '__main__':
    main()
