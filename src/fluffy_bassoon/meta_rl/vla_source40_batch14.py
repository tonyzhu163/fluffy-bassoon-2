"""Source-only, native-horizon frozen-pi0.5 PPO with14 complete K4 trials.

This is a new immutable experiment contract. It neither migrates R3 runs nor
authorizes a launch. A caller must fund the supplied physical/time ceilings.
"""
import argparse
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

from .vla_source40_multitask_ppo import OwnedSimulator, TaskCycleSimulator, source_tasks
from .vla_source40_ppo import PI05_RECEIPT, JournaledSimulator, episode_sink, physical_charge, write_json
from .vla_source40_schedule import LIBERO_COMMIT, SOURCE_SUITES, load_schedule


NATIVE_HORIZONS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)
FORMAT = 'openpi_pi05_source40_native_batch14_ppo_v1'


def native_source_tasks(schedule):
    """All40 source identities, interleaved by goal index independent of outcomes."""
    by_key = {row['task_key']: row for row in source_tasks(schedule)}
    return [by_key[f'{suite}:{number:02d}'] for number in range(10) for suite in SOURCE_SUITES]


def validate_native_metadata(task, metadata):
    if (task['suite'] not in NATIVE_HORIZONS or metadata.get('suite') != task['suite']
            or metadata.get('task_id') != task['task_id'] or metadata.get('bddl_sha256') != task['bddl_sha256']
            or metadata.get('libero_commit') != LIBERO_COMMIT
            or metadata.get('horizon') != NATIVE_HORIZONS[task['suite']] or metadata.get('initial_states') != 50
            or metadata.get('fresh_environment_on_reset') is not True or metadata.get('settle_steps') != 10):
        raise ValueError('Owned simulator differs from the exact original source/native-horizon task')


def batch_plan(tasks, cursor, sampler):
    """Cycle real train-state IDs per goal; freeze scene seeds before collection."""
    if len(tasks) != 40 or type(cursor) is not int or cursor < 0:
        raise ValueError('Require the complete40-task source cycle and committed trial cursor')
    plan = []
    for offset in range(14):
        task = tasks[(cursor + offset) % len(tasks)]
        horizon = NATIVE_HORIZONS[task['suite']]
        plan.append(dict(task_key=task['task_key'], suite=task['suite'], task_id=task['task_id'],
            task_cursor=cursor + offset, initial_state=((cursor + offset) // len(tasks)) % 40,
            scene_seed=int(sampler.integers(2**31)), horizon=horizon,
            attempts=4, settle_steps=10, maximum_actions=4 * (horizon + 10)))
    return plan


def setup_only_restart(output):
    """No learner recovery may be replaced after ANY executed training action."""
    output = Path(output)
    cost = physical_charge(output / 'execution-journal')
    if (cost['known_primitive_steps'] or cost['unresolved_primitive_ceiling']
            or any((output / 'raw').glob('*.pkl.gz'))
            or any((output / 'raw-rpc').rglob('*.pkl.gz'))):
        raise ValueError('Missing learner recovery after source execution; preserve evidence and diagnose')
    learner = output / 'learner'
    if learner.exists():
        learner.rename(learner.with_name('learner-uncommitted-setup-' + str(time.time_ns())))
    write_json(output / f'setup-restart-{time.time_ns()}.json', dict(
        reason='No valid learner checkpoint and zero known/uncertain primitive actions',
        preserved_physical_charge=cost, fresh_seeded_learner=True, source_history_discarded=False))


class NativeOwnedSimulator(OwnedSimulator):
    """Reuse owned-process reset/step/close while launching the suite's horizon."""
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
            '--task-id', str(task['task_id']), '--horizon', str(NATIVE_HORIZONS[task['suite']]),
            '--fresh-environment-on-reset']
        try:
            self.process = subprocess.Popen(command, env=env, stdout=self.log, stderr=subprocess.STDOUT)
            started = time.monotonic()
            ready = output / 'simulator-ready.json'
            while not ready.exists():
                if self.process.poll() is not None:
                    raise RuntimeError(f'Owned native simulator exited{self.process.returncode}; see {output}')
                if time.monotonic() - started >= timeout:
                    raise TimeoutError(f'Owned native simulator did not become ready; see {output}')
                time.sleep(.1)
            self.client = SimulatorClient(port)
            self.metadata = dict(self.client.metadata)
            validate_native_metadata(task, self.metadata)
        except BaseException:
            self.close()
            raise


class NativeTaskCycle(TaskCycleSimulator):
    """Complete-trial cursor and fresh-process K4 resets, with native horizons."""
    def _connect(self, index, *, force=False):
        if self.connected == index and not force:
            return
        self.close()
        task = self.tasks[index]
        self.client = self.factory(task)
        try:
            validate_native_metadata(task, self.client.metadata)
        except BaseException:
            self.close()
            raise
        self.connected = index
        self.metadata.clear()
        self.metadata.update(self.client.metadata)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('schedule', 'output', 'checkpoint', 'openpi-root', 'simulator-python'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--simulator-pythonpath', required=True)
    ap.add_argument('--source-commit', required=True)
    ap.add_argument('--expected-hostname', required=True)
    ap.add_argument('--max-total-actions', type=int, required=True)
    ap.add_argument('--stop-unix', type=float, required=True,
        help='Invocation wall deadline from the caller\'s cumulative allocation ledger; not a new budget')
    ap.add_argument('--updates', type=int, default=100)
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
    if (not 0 <= args.seed < 2**32 or not 1 <= args.updates <= 100
            or not 0 < args.max_total_actions <= 1_000_000
            or not math.isfinite(args.stop_unix) or args.stop_unix <= time.time() + 90):
        raise ValueError('Require a funded physical cap<=1M and live invocation deadline;100 updates is a target')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed source commit')
    from .vla_libero import FrozenOpenPI
    from .vla_rl2 import RL2Config, collect_rl2_trial, make_rl2_policy
    from .vla_rl2_batch import train_complete_batches
    from .vla_source40_corpus import restore_motor_rng, save_motor_rng
    from .vla_reset_identity import RESET_IDENTITY
    schedule = load_schedule(args.schedule)
    tasks = native_source_tasks(schedule)
    # Root creation is shared with owned simulator/raw receipts. The trainer's
    # learner/ subdirectory is created independently and atomically recovered.
    args.output.mkdir(parents=True, exist_ok=args.resume)
    import fcntl
    lock = (args.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = args.output / 'attempts' / str(time.time_ns())
    attempt.mkdir(parents=True)
    def factory(task):
        return NativeOwnedSimulator(task, python=args.simulator_python, pythonpath=args.simulator_pythonpath,
            output=attempt / f"server-{task['task_key'].replace(':', '-')}-{time.time_ns()}", port=args.port)
    cycle = NativeTaskCycle(tasks, 4, factory, receipts=args.output / 'task-resets')
    sim = JournaledSimulator(cycle, args.output / 'execution-journal', args.max_total_actions)
    try:
        args.model_config, args.feature_interface = 'pi05_libero', 'vla-prefix'
        motor = FrozenOpenPI(SimpleNamespace(**dict(vars(args), output=attempt)))
        if motor.metadata['receipt_sha256'] != PI05_RECEIPT or motor.horizon != 10:
            raise ValueError('Require the frozen official OpenPI40 motor')
        contract = dict(format=FORMAT, source_commit=args.source_commit,
            source_only=True, source_data_from_targets=False,
            source_task_keys=[row['task_key'] for row in tasks],
            tasks=[{key: task[key] for key in ('task_key', 'suite', 'task_id', 'bddl_sha256', 'initial_states_sha256')}
                for task in tasks], schedule_sha256=schedule['schedule_sha256'],
            motor=motor.metadata, feature_contract=motor.feature_encoder.contract,
            noise_contract='32D squashed Gaussian coordinate repeated over native H10; executed coordinate unchanged',
            algorithm='One shared on-policy RL2/GTrXL PPO noise policy across all40 source goals',
            task_schedule='Outcome-independent interleaved spatial/object/goal/10 round robin by goal index',
            state_schedule='Per-goal sequential cycle through actual train-state IDs0..39; sampler draws scene seeds only',
            training_initial_state_ids=list(range(40)), heldout_initial_state_ids=list(range(40, 50)),
            reset_identity=RESET_IDENTITY, native_horizons=NATIVE_HORIZONS, settling_actions_per_episode=10,
            updates_target=args.updates, complete_trials_per_update=14, attempts_per_trial=4,
            optimization=dict(learning_rate=1e-4, epochs=2, valid_token_weighting=True),
            maximum_total_actions=args.max_total_actions, memory_mode=args.memory_mode, seed=args.seed,
            cap_precedence='Start a new batch only when all14 worst-case K4 trials fit;100 updates is not guaranteed',
            validation='No heldout or target trajectories enter optimizer input',
            support_query_protocol='Source K4 trials do not set deployment support-shot budgets')
        path = args.output / 'source-contract.json'
        if args.resume:
            if not path.exists():
                setup_only_restart(args.output)
                write_json(path, contract)
            elif json.loads(path.read_text()) != contract:
                raise ValueError('Native batch14 continuation contract differs')
        else:
            write_json(path, contract)
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
        sink = episode_sink(args.output / 'raw', contract=contract)
        def snapshot():
            return dict(motor_rng=save_motor_rng(motor.policy._rng), task_cycle=cycle.state_dict())
        def restore(saved):
            motor.policy._rng = restore_motor_rng(saved['motor_rng'])
            cycle.load_state_dict(saved['task_cycle'])
        def collect(current, descriptor, trial, update, behavior_sha):
            selected = tasks[cycle.cursor % len(tasks)]
            if (cycle.remaining != 0 or descriptor['task_cursor'] != cycle.cursor
                    or descriptor['task_key'] != selected['task_key']
                    or descriptor['suite'] != selected['suite'] or descriptor['task_id'] != selected['task_id']
                    or not 0 <= descriptor['initial_state'] < 40
                    or type(descriptor['scene_seed']) is not int or not 0 <= descriptor['scene_seed'] < 2**31
                    or descriptor['horizon'] != NATIVE_HORIZONS[selected['suite']]
                    or descriptor['maximum_actions'] != 4 * (descriptor['horizon'] + 10)
                    or descriptor['attempts'] != 4 or descriptor['settle_steps'] != 10):
                raise ValueError('Planned complete source trial differs from the committed task cursor')
            def save_episode(**kwargs):
                sink(trial=trial, behavior_update=update, behavior_policy_sha256=behavior_sha, **kwargs)
            # Complete-episode PPO traces remain compact. Keep raw inference
            # inputs/outcomes as well, so an interrupted episode retains its
            # actual observations/noises/actions alongside conservative charges.
            from .vla_source40_memory_replication import CaptureMotor, CaptureSimulator
            raw = args.output / 'raw-rpc' / f'trial-{trial:06d}-{time.time_ns()}'
            raw.mkdir(parents=True)
            captured_motor = CaptureMotor(motor, raw)
            captured_sim = CaptureSimulator(sim, captured_motor, raw)
            started = time.monotonic()
            rollout, receipt = collect_rl2_trial(current, captured_sim, captured_motor,
                lambda observation: captured_motor.feature_encoder(observation, sim.metadata['instruction']),
                initial_state=descriptor['initial_state'], scene_seed=descriptor['scene_seed'],
                attempts=4, chunk=10, mode=args.memory_mode, episode_sink=save_episode,
                reset_identity=RESET_IDENTITY)
            if (cycle.remaining != 0 or cycle.cursor != descriptor['task_cursor'] + 1
                    or receipt['primitive_steps'] + receipt['settling_steps'] > descriptor['maximum_actions']):
                raise RuntimeError('Actual complete source trial exceeded its declared task/action boundary')
            return rollout, dict(receipt, descriptor=descriptor, behavior_update=update,
                behavior_policy_sha256=behavior_sha, collection_seconds=time.monotonic()-started)
        training_resume = args.resume and (args.output / 'learner/latest.pt').is_file()
        if args.resume and not training_resume:
            setup_only_restart(args.output)
        if args.resume:
            for partial in args.output.rglob('*.partial'):
                partial.rename(partial.with_name(partial.name + f'-uncommitted-{time.time_ns()}'))
        summary = train_complete_batches(policy, planner=lambda sampler: batch_plan(tasks, cycle.cursor, sampler),
            collector=collect, runtime_state=snapshot, restore_runtime=restore,
            physical_cost=lambda: physical_charge(sim.journal)['total_action_charge'],
            output=args.output / 'learner', source_contract=contract, updates=args.updates,
            max_total_actions=args.max_total_actions, seed=args.seed, mode=args.memory_mode,
            learning_rate=1e-4, epochs=2, stop_unix=args.stop_unix, resume=training_resume)
        exposure = cycle.state_dict()
        write_json(args.output / 'index.json', dict(source_contract=contract, summary=summary,
            committed_task_exposure=exposure, actual_tasks_with_committed_episodes=sum(
                row['episodes'] > 0 for row in exposure['exposure'].values()),
            physical_cost=physical_charge(sim.journal),
            status='Source-only candidate and actual counts; no robot or memory benefit established'))
    finally:
        sim.close()


if __name__ == '__main__':
    main()
