"""Frozen source-PPO support/query deployment on declared target cases.

One common motor serves PPO full/reset controls and optional existing AD/DPT
controls. No target trajectory enters source learning and no optimizer exists.
"""
import argparse
from copy import deepcopy
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
import torch

from .checkpoint import restore_rng, rng_state
from .vla_history import FrozenHistoryController, file_sha256
from .vla_libero import FrozenOpenPI, SimulatorClient, write_json
from .vla_rl2 import FrozenRL2Controller
from .vla_source40_corpus import PI05_RECEIPT, save_motor_rng
from .vla_source40_evaluate import SUPPORT_BUDGETS, TargetCase, evaluate_case, load_candidates
from .vla_source40_frozen_validation import load_models
from .vla_source40_multitask_ppo import source_tasks
from .vla_source40_ppo import physical_charge
from .vla_source40_schedule import content_hash, load_schedule
from .vla_source40_subset_evaluate import TargetServer


PPO_METHODS = ('ppo_full', 'ppo_episode_reset', 'ppo_trained_episode')
DEFAULT_METHODS = ('native', *PPO_METHODS)
METHOD_CHOICES = (*DEFAULT_METHODS, 'tied_gaussian', 'zero', 'ad_full', 'ad_episode_reset',
                  'ad_static', 'dpt_full', 'dpt_episode_reset', 'dpt_static')


class TargetPPOController(FrozenRL2Controller):
    """Use RL2's global Gaussian sampler with complete per-episode recovery."""
    def set_episode_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.generator = None  # The qualified RL2 sampler rejects a local Generator.

    def snapshot(self):
        if self.pending:
            raise ValueError('Target PPO recovery requires completed action feedback')
        def cpu(value):
            if isinstance(value, dict):
                return {key: cpu(item) for key, item in value.items()}
            return value.detach().cpu().clone() if torch.is_tensor(value) else deepcopy(value)
        return dict(format='frozen_target_ppo_context_v1', mode=self.mode, state=cpu(self.state),
            previous_noise=self.previous_noise.copy(), previous_reward=self.previous_reward,
            previous_done=self.previous_done, episode_start=self.episode_start, pending=False,
            global_rng=rng_state(), generator=None)

    def restore(self, saved):
        if (saved.get('format') != 'frozen_target_ppo_context_v1' or saved.get('mode') != self.mode
                or saved.get('pending') is not False or saved.get('generator') is not None):
            raise ValueError('Target PPO context mode or complete-episode boundary differs')
        device = next(self.policy.parameters()).device
        def target(value):
            if isinstance(value, dict):
                return {key: target(item) for key, item in value.items()}
            return value.to(device).clone() if torch.is_tensor(value) else deepcopy(value)
        self.state = target(saved['state'])
        for name in ('previous_noise', 'previous_reward', 'previous_done', 'episode_start', 'pending'):
            setattr(self, name, deepcopy(saved[name]))
        self.generator = None
        restore_rng(saved['global_rng'])


def candidates(models, provenance):
    return dict(
        ppo_full=dict(policy=models['full'], mode='full', kind='ppo', sha256=provenance['full']['policy_sha256']),
        ppo_episode_reset=dict(policy=models['full'], mode='episode', kind='ppo', sha256=provenance['full']['policy_sha256']),
        ppo_trained_episode=dict(policy=models['episode'], mode='episode', kind='ppo', sha256=provenance['episode']['policy_sha256']))


def controller_factory(spec):
    cls = TargetPPOController if spec.get('kind') == 'ppo' else FrozenHistoryController
    return cls(spec['policy'], mode=spec['mode'])


def reservation(path, run, source_commit, action_cap):
    ledger = json.loads(Path(path).read_text())
    record = ledger.get('reservations', {}).get(run)
    if (record is None or record.get('run_id') != run or record.get('source_commit') != source_commit
            or record.get('maximum_total_actions') != action_cap or record.get('purpose') != 'target_evaluation'
            or record.get('campaign_sha256') != ledger.get('campaign_sha256')
            or record.get('source_pretraining_data') is not False or record.get('fresh_ppo_training_data') is not False):
        raise ValueError('Root must first record the exact target-only action escrow; this runner never creates it')
    # Bind the individual reservation, so concurrent unrelated ledger additions
    # do not invalidate an otherwise exact resume.
    return dict(ledger_path=str(path), record=record, record_sha256=content_hash(record))


def load_case(path, expected):
    if file_sha256(path) != expected:
        raise ValueError('Target case plan checksum differs')
    value = json.loads(Path(path).read_text())
    if value.get('format') != 'source40_ppo_target_case_v1':
        raise ValueError('Require one explicit protocol target case')
    case = TargetCase(**value['case'])
    if case.benchmark not in ('libero90', 'plus', 'pro'):
        raise ValueError('Only the existing LIBERO90/Plus/Pro target protocol is supported')
    if file_sha256(path) != expected:
        raise ValueError('Target case plan changed while reading')
    return case, value


def shifted_command(case, plan, *, source_commit, output, port):
    """Use the already qualified shifted workers; do not synthesize new cases."""
    config = plan['simulator']
    command = [str(config['python']), '-m']
    if case.benchmark == 'plus':
        return command + ['fluffy_bassoon.meta_rl.vla_plus_worker', '--output', str(output),
            '--source-commit', source_commit, '--libero-root', str(config['plus_root']),
            '--case-index', str(case.task_id), '--port', str(port), '--horizon', '280', '--settle-steps', '10']
    if case.benchmark == 'pro':
        return command + ['fluffy_bassoon.meta_rl.vla_libero_pro', '--output', str(output),
            '--pro-root', str(config['pro_root']), '--data-root', str(config['data_root']),
            '--case', config['pro_case'], '--port', str(port), '--horizon', '280']
    raise ValueError('Original LIBERO90 uses the existing original TargetServer')


class ShiftedServer:
    def __init__(self, case, plan, *, source_commit, output, port):
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.process, self.client = None, None
        self.log = output.with_suffix('.log').open('w')
        try:
            self.process = subprocess.Popen(shifted_command(case, plan, source_commit=source_commit, output=output, port=port),
                env=dict(os.environ, PYTHONPATH=plan['simulator']['pythonpath']),
                stdout=self.log, stderr=subprocess.STDOUT)
            started = time.monotonic()
            while not (output / 'simulator-ready.json').exists():
                if self.process.poll() is not None or time.monotonic() - started > 180:
                    raise RuntimeError(f'Declared shifted simulator failed to become ready; see {self.log.name}')
                time.sleep(.1)
            self.client = SimulatorClient(port)
            self.metadata = self.client.metadata
            case.validate(self.metadata)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'models', 'case-plan', 'schedule', 'checkpoint', 'openpi-root', 'ledger'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('source-commit', 'run-id', 'expected-hostname', 'models-sha256', 'case-plan-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--max-total-actions', type=int, required=True)
    parser.add_argument('--max-elapsed-seconds', type=float, required=True)
    parser.add_argument('--port', type=int, default=8771)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--support-budgets', type=int, choices=SUPPORT_BUDGETS, nargs='+', default=SUPPORT_BUDGETS)
    parser.add_argument('--methods', choices=METHOD_CHOICES, nargs='+', default=DEFAULT_METHODS)
    parser.add_argument('--student-seed', type=int, default=0)
    parser.add_argument('--students-index', type=Path)
    parser.add_argument('--students-index-sha256')
    parser.add_argument('--assembly', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname:
        raise ValueError('Model/simulator evaluation requires the explicitly allocated Linux host')
    if args.max_total_actions < 1 or not math.isfinite(args.max_elapsed_seconds) or args.max_elapsed_seconds <= 0:
        raise ValueError('Require explicit bounded future target compute/action limits')
    if len({bool(args.students_index), bool(args.students_index_sha256), bool(args.assembly)}) != 1:
        raise ValueError('Optional existing AD/DPT controls require their index hash and assembly together')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed target-evaluation source')
    escrow = reservation(args.ledger, args.run_id, args.source_commit, args.max_total_actions)
    case, plan = load_case(args.case_plan, args.case_plan_sha256)
    schedule = load_schedule(args.schedule)
    if file_sha256(args.checkpoint / 'READY.json') != PI05_RECEIPT:
        raise ValueError('Require the original official OpenPI40 motor')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    import fcntl
    lock = (args.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = args.output / 'attempts' / str(time.time_ns())
    attempt.mkdir(parents=True)
    motor = FrozenOpenPI(SimpleNamespace(**dict(vars(args), output=attempt,
        model_config='pi05_libero', feature_interface='vla-prefix')))
    model_directory = args.output / 'models'
    model_directory.mkdir(exist_ok=True)
    models, source = load_models(args.models, args.models_sha256, motor=motor,
        tasks=[row['task_key'] for row in source_tasks(schedule)], schedule_sha256=schedule['schedule_sha256'],
        output=model_directory, device=args.device)
    selected = candidates(models, source)
    student_costs = None
    if args.students_index:
        students, student_costs = load_candidates(args.students_index, args.students_index_sha256,
            args.assembly, motor, seed=args.student_seed, device=args.device)
        selected.update(students)
    for mode, record in source.items():
        experiment = Path(record['recovery_path']).parents[1]
        record['source_rpc_cost'] = physical_charge(experiment / 'execution-journal')
    config = plan['simulator']
    server = (TargetServer(task_id=case.task_id, python=config['python'], pythonpath=config['pythonpath'],
                output=attempt / 'simulator', port=args.port, libero_root=Path(config['libero_root']))
        if case.benchmark == 'libero90' else ShiftedServer(case, plan, source_commit=args.source_commit,
                output=attempt / 'simulator', port=args.port))
    try:
        case.validate(server.metadata)
        provenance = dict(evaluation_source_commit=args.source_commit, case_plan_sha256=args.case_plan_sha256,
            source_models_sha256=args.models_sha256, source_ppo=source, optional_ad_dpt=student_costs,
            group_maximum_total_actions=args.max_total_actions, escrow=escrow,
            target_only=True, source_pretraining_data=False, deployment_optimizer_updates=0,
            method_source_cost_role=dict(native='No steering-source interactions required',
                ppo_full='Full source model, count this source cost once',
                ppo_episode_reset='Same full weights; source cost shared with ppo_full',
                ppo_trained_episode='Separately trained episode-reset source model'),
            ordinary_dsrl_baseline='Separate existing fresh target-online DSRL pipeline; PPO controls are not DSRL',
            full_protocol_support_grid=set(args.support_budgets) == set(SUPPORT_BUDGETS),
            shifted_scope='Existing Plus/Pro workers are declared development slices; no full shifted-suite result is implied')
        def reset_motor_rng(seed):
            import jax
            motor.policy._rng = jax.random.key(seed)
        def qualify(observation):
            zero = np.zeros((10, 32), np.float32)
            a = motor.infer(observation, server.metadata['instruction'], zero)
            b = motor.infer(observation, server.metadata['instruction'], zero)
            if not np.allclose(a, b, rtol=0, atol=1e-5):
                raise RuntimeError('Common frozen target motor same-noise qualification failed')
            write_json(args.output / 'qualification.json', dict(same_noise_max_delta=float(np.abs(a-b).max()),
                features=motor.feature_encoder.qualify(observation, server.metadata['instruction'])))
        evaluate_case(server, motor, selected, case=case, output=args.output, provenance=provenance,
            max_primitive_steps=args.max_total_actions, max_settling_steps=args.max_total_actions,
            support_budgets=tuple(args.support_budgets), methods=tuple(args.methods),
            controller_factory=controller_factory, reset_motor_rng=reset_motor_rng,
            motor_rng_state=lambda: save_motor_rng(motor.policy._rng), qualify=qualify,
            resume=args.resume, max_elapsed_seconds=args.max_elapsed_seconds)
    finally:
        server.close()


if __name__ == '__main__':
    main()
