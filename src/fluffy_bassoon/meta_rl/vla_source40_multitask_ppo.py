"""One recurrent PPO learner across all40 frozen-OpenPI source tasks.

Owns at most one simulator subprocess. Source tasks cycle independently of
outcomes; actual exposure and physical costs are reported separately.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from .vla_source40_ppo import PI05_RECEIPT, JournaledSimulator, episode_sink, physical_charge, write_json
from .vla_source40_schedule import LIBERO_COMMIT, SOURCE_SUITES, load_schedule


def source_tasks(schedule):
    tasks = schedule['tasks']
    expected = [f'{suite}:{task:02d}' for suite in SOURCE_SUITES for task in range(10)]
    if len(tasks) != 40 or {row['task_key'] for row in tasks} != set(expected):
        raise ValueError('Shared source PPO requires all40 original task identities')
    by_key = {row['task_key']: row for row in tasks}
    return [by_key[key] for key in expected]


def validate_task_metadata(task, metadata):
    if (metadata.get('suite') != task['suite'] or metadata.get('task_id') != task['task_id']
            or metadata.get('bddl_sha256') != task['bddl_sha256']
            or metadata.get('libero_commit') != LIBERO_COMMIT
            or metadata.get('horizon') != 400 or metadata.get('initial_states') != 50
            or metadata.get('fresh_environment_on_reset') is not True
            or metadata.get('settle_steps') != 10):
        raise ValueError('Owned simulator differs from the declared original source task')


class OwnedSimulator:
    """Only the subprocess started here can be terminated or replaced."""
    def __init__(self, task, *, python, pythonpath, output, port, timeout=180.):
        from .vla_libero import SimulatorClient
        self.process, self.client = None, None
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        self.log = (output / 'simulator.log').open('w')
        env = dict(os.environ, PYTHONPATH=str(pythonpath), PYTHONDONTWRITEBYTECODE='1',
                   PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1')
        command = [str(python), '-m', 'fluffy_bassoon.meta_rl.vla_libero', '--mode', 'env-server',
            '--output', str(output), '--port', str(port), '--suite', task['suite'],
            '--task-id', str(task['task_id']), '--horizon', '400', '--fresh-environment-on-reset']
        try:
            self.process = subprocess.Popen(command, env=env, stdout=self.log, stderr=subprocess.STDOUT)
            started = time.monotonic()
            ready = output / 'simulator-ready.json'
            while not ready.exists():
                if self.process.poll() is not None:
                    raise RuntimeError(f'Owned simulator exited{self.process.returncode}; see {output}')
                if time.monotonic() - started >= timeout:
                    raise TimeoutError(f'Owned simulator did not become ready; see {output}')
                time.sleep(.1)
            self.client = SimulatorClient(port)
            self.metadata = dict(self.client.metadata)
            validate_task_metadata(task, self.metadata)
        except BaseException:
            self.close()
            raise

    def reset(self, state, seed):
        return self.client.reset(state, seed)

    def step(self, actions):
        return self.client.step(actions)

    def close(self):
        if self.client is not None:
            try:
                self.client.close()
            except (OSError, EOFError):
                pass
            self.client = None
        if self.process is not None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process = None
        if not self.log.closed:
            self.log.close()


class TaskCycleSimulator:
    """Stable metadata object, task switches only at complete trial boundaries."""
    def __init__(self, tasks, attempts, factory, *, receipts=None):
        if not tasks or attempts < 1 or len({row['task_key'] for row in tasks}) != len(tasks):
            raise ValueError('Require unique source tasks and a positive trial length')
        self.tasks, self.attempts, self.factory = tasks, attempts, factory
        self.receipts = None if receipts is None else Path(receipts)
        if self.receipts is not None:
            self.receipts.mkdir(parents=True, exist_ok=True)
        self.metadata, self.client, self.connected = {}, None, None
        self.qualifying = True
        self.cursor, self.active_pair, self.active_task, self.remaining = 0, None, None, 0
        self.exposure = {row['task_key']: dict(episodes=0, primitive_steps=0, trials_started=0) for row in tasks}
        self._connect(0)

    def _connect(self, index):
        if self.connected == index:
            return
        self.close()
        task = self.tasks[index]
        self.client = self.factory(task)
        try:
            validate_task_metadata(task, self.client.metadata)
        except BaseException:
            self.close()
            raise
        self.connected = index
        self.metadata.clear()
        self.metadata.update(self.client.metadata)

    def begin_collection(self):
        self.qualifying = False

    def reset(self, state, seed):
        pair = (int(state), int(seed))
        if self.qualifying:
            index = 0
        elif self.remaining == 0:
            index = self.cursor % len(self.tasks)
            self.cursor += 1
            self.active_task, self.active_pair, self.remaining = index, pair, self.attempts
            self.exposure[self.tasks[index]['task_key']]['trials_started'] += 1
        else:
            if pair != self.active_pair:
                raise ValueError('A task trial must keep the same state/seed across retry episodes')
            index = self.active_task
        self._connect(index)
        task = self.tasks[index]
        receipt = dict(task_key=task['task_key'], suite=task['suite'], task_id=task['task_id'],
            initial_state=pair[0], scene_seed=pair[1], bddl_sha256=task['bddl_sha256'],
            qualification=self.qualifying, returned=False, settling_steps=10)
        path = None if self.receipts is None else self.receipts / f'reset-{time.time_ns()}.json'
        if path is not None:
            write_json(path, receipt)
        result = self.client.reset(*pair)
        if not isinstance(result.get('initial_state_vector_sha256'), str):
            raise ValueError('Original source reset omitted the actual nominal-state hash')
        if path is not None:
            write_json(path, dict(receipt, returned=True,
                initial_state_vector_sha256=result['initial_state_vector_sha256'],
                initial_observation_sha256=result['initial_observation_sha256']))
        if not self.qualifying:
            self.remaining -= 1
            self.exposure[task['task_key']]['episodes'] += 1
        return dict(result, source_task_key=task['task_key'], source_bddl_sha256=task['bddl_sha256'])

    def step(self, actions):
        result = self.client.step(actions)
        if not self.qualifying:
            self.exposure[self.tasks[self.active_task]['task_key']]['primitive_steps'] += int(result['primitive_steps'])
        return result

    def state_dict(self):
        return dict(format='source40_ppo_task_cursor_v1', cursor=self.cursor,
            active_pair=None if self.active_pair is None else list(self.active_pair),
            active_task=self.active_task, remaining=self.remaining, exposure=copy.deepcopy(self.exposure))

    def load_state_dict(self, state):
        if (state.get('format') != 'source40_ppo_task_cursor_v1'
                or type(state.get('cursor')) is not int or state['cursor'] < 0
                or state.get('remaining') != 0
                or set(state.get('exposure', {})) != set(self.exposure)
                or (state['active_task'] is not None and not 0 <= state['active_task'] < len(self.tasks))):
            raise ValueError('Require a complete-trial task cursor for this source task set')
        if any(set(row) != {'episodes', 'primitive_steps', 'trials_started'}
               or any(type(value) is not int or value < 0 for value in row.values())
               or row['episodes'] != row['trials_started'] * self.attempts
               for row in state['exposure'].values()):
            raise ValueError('Recovered source exposure must contain only complete actual trials')
        self.cursor, self.remaining, self.active_task = state['cursor'], state['remaining'], state['active_task']
        self.active_pair = None if state['active_pair'] is None else tuple(state['active_pair'])
        self.exposure = copy.deepcopy(state['exposure'])
        self.qualifying = False

    def close(self):
        if self.client is not None:
            self.client.close()
        self.client, self.connected = None, None


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('schedule', 'output', 'checkpoint', 'openpi-root', 'simulator-python'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--simulator-pythonpath', required=True)
    ap.add_argument('--source-commit', required=True)
    ap.add_argument('--expected-hostname', required=True)
    ap.add_argument('--max-total-actions', type=int, required=True)
    ap.add_argument('--max-elapsed-seconds', type=float, required=True)
    ap.add_argument('--updates', type=int, required=True)
    ap.add_argument('--attempts', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--port', type=int, default=8766)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--context-length', type=int, default=300)
    ap.add_argument('--memory-mode', choices=('full', 'episode'), default='full')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--resume', action='store_true')
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname:
        raise ValueError('Model computation requires the explicitly allocated Linux host')
    if (args.attempts < 1 or args.updates < 1 or not 0 <= args.seed < 2**32
            or args.max_total_actions < args.attempts * 410 + 10
            or args.max_elapsed_seconds <= 0 or not math.isfinite(args.max_elapsed_seconds)):
        raise ValueError('Invalid complete-trial source action/compute reservation')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed source commit')
    from .vla_libero import FrozenOpenPI
    from .vla_rl2 import RL2Config, make_rl2_policy, train_rl2
    from .vla_source40_corpus import restore_motor_rng, save_motor_rng
    schedule = load_schedule(args.schedule)
    tasks = source_tasks(schedule)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    import fcntl
    lock = (args.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = args.output / 'attempts' / str(time.time_ns())
    attempt.mkdir(parents=True)
    def factory(task):
        return OwnedSimulator(task, python=args.simulator_python, pythonpath=args.simulator_pythonpath,
            output=attempt / f"server-{task['task_key'].replace(':', '-')}-{time.time_ns()}", port=args.port)
    cycle = TaskCycleSimulator(tasks, args.attempts, factory, receipts=args.output / 'task-resets')
    sim = JournaledSimulator(cycle, args.output / 'execution-journal', args.max_total_actions)
    try:
        args.model_config, args.feature_interface = 'pi05_libero', 'vla-prefix'
        motor = FrozenOpenPI(SimpleNamespace(**dict(vars(args), output=attempt)))
        if motor.metadata['receipt_sha256'] != PI05_RECEIPT or motor.horizon != 10:
            raise ValueError('Require the frozen official OpenPI40 motor')
        contract = dict(format='openpi_pi05_source40_shared_ppo_v1', source_commit=args.source_commit,
            source_only=True, source_data_from_targets=False, source_task_keys=[row['task_key'] for row in tasks],
            tasks=[{key: task[key] for key in ('task_key', 'suite', 'task_id', 'bddl_sha256', 'initial_states_sha256')}
                   for task in tasks], schedule_sha256=schedule['schedule_sha256'],
            motor=motor.metadata, feature_contract=motor.feature_encoder.contract,
            noise_contract='32D squashed Gaussian coordinate repeated over native H10; executed coordinate unchanged',
            algorithm='Shared on-policy RL2/GTrXL PPO noise policy; one optimizer across all40 source tasks',
            task_schedule='Outcome-independent round robin; one task per complete K-attempt trial',
            training_initial_state_ids=list(range(40)), maximum_total_actions=args.max_total_actions,
            memory_mode=args.memory_mode, seed=args.seed,
            validation='Separate frozen-policy source-validation task/case allocation, not optimizer input',
            support_query_protocol='Source trials do not set deployment support-shot budgets')
        contract_path = args.output / 'source-contract.json'
        if args.resume:
            if json.loads(contract_path.read_text()) != contract:
                raise ValueError('Shared source PPO continuation contract differs')
        else:
            write_json(contract_path, contract)
        first = sim.reset(0, args.seed)['observation']
        zero = np.zeros((10, 32), np.float32)
        a, b = motor.infer(first, sim.metadata['instruction'], zero), motor.infer(first, sim.metadata['instruction'], zero)
        if not np.allclose(a, b, rtol=0, atol=1e-5):
            raise RuntimeError('Same-noise frozen-motor qualification failed')
        write_json(attempt / 'qualification.json', dict(same_noise_max_delta=float(np.abs(a-b).max()),
            feature_parity=motor.feature_encoder.qualify(first, sim.metadata['instruction'])))
        cycle.begin_collection()
        policy = make_rl2_policy(RL2Config(observation_dim=4104, width=args.width, layers=args.layers,
            heads=args.heads, context_length=args.context_length), device=args.device, seed=args.seed)
        def snapshot():
            return dict(motor_rng=save_motor_rng(motor.policy._rng), task_cycle=cycle.state_dict())
        def restore(saved):
            motor.policy._rng = restore_motor_rng(saved['motor_rng'])
            cycle.load_state_dict(saved['task_cycle'])
        if args.resume:
            for partial in args.output.rglob('*.partial'):
                partial.rename(partial.with_name(partial.name + f'-uncommitted-{time.time_ns()}'))
        _, summary = train_rl2(policy, sim, motor,
            lambda observation: motor.feature_encoder(observation, sim.metadata['instruction']),
            train_states=list(range(40)), heldout_states=list(range(40, 50)),
            attempts=args.attempts, chunk=10, episode_horizon=400,
            max_primitive_steps=args.max_total_actions, max_total_actions=args.max_total_actions,
            physical_cost=lambda: physical_charge(sim.journal)['total_action_charge'],
            updates=args.updates, output_dir=args.output / 'learner', source_contract=contract,
            seed=args.seed, mode=args.memory_mode, max_elapsed_seconds=args.max_elapsed_seconds,
            resume=args.output / 'learner/latest.pt' if args.resume else None,
            episode_sink=episode_sink(args.output / 'raw', contract=contract),
            runtime_state=snapshot, restore_runtime=restore)
        exposure = cycle.state_dict()
        write_json(args.output / 'index.json', dict(source_contract=contract, summary=summary,
            committed_task_exposure=exposure, actual_tasks_with_committed_episodes=sum(
                row['episodes'] > 0 for row in exposure['exposure'].values()),
            physical_cost=physical_charge(sim.journal),
            status='Shared-policy source data; no robot or meta-learning benefit established'))
    finally:
        sim.close()


if __name__ == '__main__':
    main()
