"""Fresh on-policy PPO source rollouts for the official frozen OpenPI motor.

One explicit source task/partition is a runtime qualification building block.
It is not an all40 shared-policy meta-training orchestrator.
"""
import argparse
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

from .vla_source40_schedule import SOURCE_SUITES


FORMAT = "frozen_vla_ppo_episode_v1"
PI05_RECEIPT = "ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def physical_charge(journal):
    """Known primitives plus unresolved requested actions and all reset charges."""
    journal = Path(journal)
    outcomes = {path.stem: json.loads(path.read_text()) for path in journal.glob('step-*.json')}
    known = sum(row['primitive_steps'] for row in outcomes.values())
    uncertain = sum(json.loads(path.read_text())['maximum_primitive_steps']
        for path in journal.glob('intent-*.json') if 'step-' + path.stem[7:] not in outcomes)
    resets = [json.loads(path.read_text()) for path in journal.glob('reset-*.json')]
    settling = sum(row['settling_steps'] for row in resets)
    return dict(known_primitive_steps=known, unresolved_primitive_ceiling=uncertain,
        settling_step_charge=settling,
        known_settling_steps=sum(row['settling_steps'] for row in resets if row['returned']),
        total_action_charge=known + uncertain + settling)


class JournaledSimulator:
    """Charge physical RPCs before execution; input data stays read-only."""
    def __init__(self, sim, journal, action_cap):
        self.sim, self.metadata = sim, sim.metadata
        self.journal, self.action_cap = Path(journal), int(action_cap)
        self.journal.mkdir(parents=True, exist_ok=True)

    def _reserve(self, count):
        if physical_charge(self.journal)['total_action_charge'] + count > self.action_cap:
            raise RuntimeError('PPO physical action ceiling would be exceeded')

    def reset(self, initial_state, seed):
        count = int(self.metadata['settle_steps'])
        self._reserve(count)
        path = self.journal / f'reset-{time.time_ns()}.json'
        row = dict(initial_state=int(initial_state), seed=int(seed), settling_steps=count, returned=False)
        write_json(path, row)
        result = self.sim.reset(initial_state, seed)
        write_json(path, dict(row, returned=True,
            initial_observation_sha256=result.get('initial_observation_sha256'),
            initial_state_vector_sha256=result.get('initial_state_vector_sha256')))
        return result

    def step(self, actions):
        count = len(actions)
        self._reserve(count)
        identity = str(time.time_ns())
        write_json(self.journal / f'intent-{identity}.json', dict(maximum_primitive_steps=count))
        result = self.sim.step(actions)
        executed = int(result['primitive_steps'])
        if not 0 < executed <= count:
            raise ValueError('Simulator returned an invalid executed action count')
        write_json(self.journal / f'step-{identity}.json', dict(primitive_steps=executed,
            done=bool(result['done']), success=bool(result['success'])))
        return result

    def close(self):
        return self.sim.close()


def episode_sink(output, *, contract):
    """Return a sink retaining both raw public transitions and PPO likelihoods."""
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    def save(*, trial, attempt, rows, summary, behavior, behavior_update,
             behavior_policy_sha256, initial_recurrent_state):
        required = {'motor_observation', 'motor_next_observation', 'public_features',
                    'executed_noise', 'native_initial_noise', 'executed_actions',
                    'reward', 'raw_rewards', 'done', 'success', 'primitive_steps'}
        if not rows or any(not required.issubset(row) for row in rows):
            raise ValueError('PPO source trace omitted public observations or executed transitions')
        if len(rows) != len(behavior['log_probs']) or int(summary['primitive_steps']) != sum(
                int(row['primitive_steps']) for row in rows):
            raise ValueError('PPO source likelihoods and physical transitions differ')
        for row in rows:
            if (np.asarray(row['public_features']).shape != (4104,)
                    or len(row['raw_rewards']) != int(row['primitive_steps'])
                    or len(row['executed_actions']) != int(row['primitive_steps'])):
                raise ValueError('PPO source duration/features/reward/action lengths differ')
        raw = behavior['raw_actions']
        log_jacobian = (2 * (math.log(2.) - raw - torch.nn.functional.softplus(-2 * raw))).sum(-1)
        metadata = dict(format=FORMAT, source_contract=contract, trial=int(trial), attempt=int(attempt),
            behavior_update=int(behavior_update), behavior_policy_sha256=behavior_policy_sha256,
            initial_recurrent_state='Zero when ppo.initial_recurrent_state is None; otherwise the exact saved CPU cache',
            likelihood='Squashed latent-coordinate likelihood; pre-tanh samples and base-Gaussian likelihood retained',
            credit='Physical episode boundaries are not terminal RL2 trial boundaries')
        values = dict(behavior, initial_recurrent_state=initial_recurrent_state,
            pre_squash_log_probs=behavior['log_probs'] + log_jacobian,
            executed_noises=np.stack([row['executed_noise'] for row in rows]),
            terminated=np.asarray([bool(row['success']) for row in rows], bool),
            truncated=np.asarray([bool(row['done']) and not bool(row['success']) for row in rows], bool),
            primitive_durations=np.asarray([row['primitive_steps'] for row in rows], np.int64))
        path = output / f'trial-{trial:06d}-episode-{attempt:03d}.pkl.gz'
        if path.exists():
            identity = f'-uncommitted-{time.time_ns()}'
            path.rename(path.with_name(path.name + identity))
            sidecar = path.with_suffix('.json')
            if sidecar.exists():
                sidecar.rename(sidecar.with_name(sidecar.name + identity))
        with gzip.open(path, 'xb', compresslevel=1) as stream:
            pickle.dump(dict(metadata=metadata, rows=rows, summary=summary, ppo=values), stream, protocol=5)
        write_json(path.with_suffix('.json'), dict(metadata=metadata, primitive_steps=summary['primitive_steps'],
            decisions=len(rows), path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return save


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'checkpoint', 'openpi-root'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--suite', choices=SOURCE_SUITES, required=True)
    ap.add_argument('--task-id', type=int, required=True)
    ap.add_argument('--partition', choices=('train', 'validation'), required=True)
    ap.add_argument('--source-commit', required=True)
    ap.add_argument('--expected-hostname', required=True)
    ap.add_argument('--max-total-actions', type=int, required=True)
    ap.add_argument('--max-elapsed-seconds', type=float, required=True)
    ap.add_argument('--updates', type=int, required=True)
    ap.add_argument('--attempts', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--context-length', type=int, default=300)
    ap.add_argument('--memory-mode', choices=('full', 'episode'), default='full')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--policy', type=Path)
    ap.add_argument('--policy-sha256')
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname:
        raise ValueError('Model computation requires the explicitly identified allocated Linux host')
    if (not 0 <= args.task_id < 10 or args.attempts < 1 or args.updates < 1
            or not 0 <= args.seed < 2**32 or args.max_total_actions < args.attempts * 410 + 10
            or args.max_elapsed_seconds <= 0 or not math.isfinite(args.max_elapsed_seconds)
            or bool(args.policy) != bool(args.policy_sha256)):
        raise ValueError('Invalid source task or complete-trial physical budget')
    if args.partition == 'validation' and args.policy is None:
        raise ValueError('Validation needs an explicit frozen training policy; validation has no optimizer updates')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed source commit')
    from .vla_libero import FrozenOpenPI, SimulatorClient
    from .vla_rl2 import RL2Config, collect_rl2_trial, load_rl2_policy, make_rl2_policy, train_rl2
    from .vla_source40_corpus import restore_motor_rng, save_motor_rng
    from .checkpoint import rng_state, restore_rng
    from .warmstart import atomic_save
    args.output.mkdir(parents=True, exist_ok=args.resume)
    import fcntl
    lock = (args.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.model_config, args.feature_interface = 'pi05_libero', 'vla-prefix'
    sim = JournaledSimulator(SimulatorClient(args.port), args.output / 'execution-journal', args.max_total_actions)
    if (sim.metadata.get('suite') != args.suite or sim.metadata.get('task_id') != args.task_id
            or sim.metadata.get('horizon') != 400 or sim.metadata.get('initial_states') != 50
            or sim.metadata.get('fresh_environment_on_reset') is not True
            or int(sim.metadata.get('settle_steps', -1)) != 10):
        raise ValueError('Require the declared fresh-reset official source40 task simulator')
    attempt_path = args.output / 'attempts' / str(time.time_ns())
    attempt_path.mkdir(parents=True)
    # Keep previous process attestations/qualification receipts on resume.
    motor = FrozenOpenPI(SimpleNamespace(**dict(vars(args), output=attempt_path)))
    if motor.metadata['receipt_sha256'] != PI05_RECEIPT or motor.horizon != 10:
        raise ValueError('Require the frozen official OpenPI40 pi0.5 motor')
    contract = dict(format='openpi_pi05_source40_ppo_stream_v1', source_commit=args.source_commit,
        source_only=True, source_data_from_targets=False, suite=args.suite, task_id=args.task_id,
        partition=args.partition, motor=motor.metadata, feature_contract=motor.feature_encoder.contract,
        simulator={key: value for key, value in sim.metadata.items() if key != 'pid'},
        noise_contract='32D squashed Gaussian coordinate repeated over native H10; executed coordinate unchanged',
        algorithm='Fresh on-policy RL2/GTrXL PPO over a frozen VLA noise input',
        maximum_total_actions=args.max_total_actions, seed=args.seed, prior_policy_sha256=args.policy_sha256,
        memory_mode=args.memory_mode,
        controller_scope='one explicit task stream; not shared all40 meta-PPO orchestration',
        support_query_protocol='Source collection episodes are distinct from deployment support-shot budgets')
    contract_path = args.output / 'source-contract.json'
    if args.resume:
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError('PPO stream resume contract differs')
    else:
        write_json(contract_path, contract)
    states = list(range(40)) if args.partition == 'train' else list(range(40, 50))
    first = sim.reset(states[0], args.seed)['observation']
    zeros = np.zeros((10, 32), np.float32)
    a, b = motor.infer(first, sim.metadata['instruction'], zeros), motor.infer(first, sim.metadata['instruction'], zeros)
    if not np.allclose(a, b, rtol=0, atol=1e-5):
        raise RuntimeError('Same-noise frozen-motor qualification failed')
    write_json(attempt_path / 'qualification.json', dict(same_noise_max_delta=float(np.abs(a-b).max()),
        feature_parity=motor.feature_encoder.qualify(first, sim.metadata['instruction'])))
    policy = make_rl2_policy(RL2Config(observation_dim=4104, width=args.width, layers=args.layers,
        heads=args.heads, context_length=args.context_length), device=args.device, seed=args.seed)
    if args.policy:
        policy, prior = load_rl2_policy(args.policy, expected_sha256=args.policy_sha256, device=args.device)
        source = prior['contract']['source']
        if source.get('motor', {}).get('receipt_sha256') != PI05_RECEIPT or source.get('feature_contract') != motor.feature_encoder.contract:
            raise ValueError('PPO initial policy motor/features differ')
        if policy.config.observation_dim != 4104 or policy.config.noise_dim != 32:
            raise ValueError('PPO policy dimensions differ from the source interface')
    sink = episode_sink(args.output / 'raw', contract=contract)
    encode = lambda observation: motor.feature_encoder(observation, sim.metadata['instruction'])
    snapshot = lambda: dict(motor_rng=save_motor_rng(motor.policy._rng))
    def restore(saved):
        motor.policy._rng = restore_motor_rng(saved['motor_rng'])
    if args.resume:
        for partial in args.output.rglob('*.partial'):
            partial.rename(partial.with_name(partial.name + f'-uncommitted-{time.time_ns()}'))
    if args.partition == 'train':
        _, summary = train_rl2(policy, sim, motor, encode, train_states=states, heldout_states=list(range(40, 50)),
            attempts=args.attempts, chunk=10, episode_horizon=400, max_primitive_steps=args.max_total_actions,
            max_total_actions=args.max_total_actions, physical_cost=lambda: physical_charge(sim.journal)['total_action_charge'],
            updates=args.updates, output_dir=args.output / 'learner', source_contract=contract,
            seed=args.seed, mode=args.memory_mode, max_elapsed_seconds=args.max_elapsed_seconds,
            resume=args.output / 'learner/latest.pt' if args.resume else None,
            episode_sink=sink, runtime_state=snapshot, restore_runtime=restore)
    else:
        # Frozen validation rollouts are on-policy data, but never optimizer input.
        sampler = np.random.default_rng(args.seed)
        progress = dict(contract=contract, trial=0, sampler=sampler.bit_generator.state,
                        runtime=snapshot(), rng=rng_state())
        recovery = args.output / 'validation-recovery.pt'
        if args.resume:
            import torch
            progress = torch.load(recovery, map_location='cpu', weights_only=False)
            if progress['contract'] != contract:
                raise ValueError('Validation recovery contract changed')
            sampler.bit_generator.state = progress['sampler']
            restore(progress['runtime'])
            restore_rng(progress['rng'])
        else:
            atomic_save(recovery, progress)
        started = time.monotonic()
        for trial in range(progress['trial'], args.updates):
            if (physical_charge(sim.journal)['total_action_charge'] + args.attempts * 410 > args.max_total_actions
                    or time.monotonic() - started >= args.max_elapsed_seconds):
                break
            rollout, receipt = collect_rl2_trial(policy, sim, motor, encode,
                initial_state=int(sampler.choice(states)), scene_seed=int(sampler.integers(0, 2**31-1)),
                attempts=args.attempts, chunk=10,
                mode=args.memory_mode,
                episode_sink=lambda **episode: sink(trial=trial + 1, behavior_update=0,
                    behavior_policy_sha256=args.policy_sha256, **episode))
            atomic_save(args.output / f'validation-trial-{trial+1:06d}.pt', dict(rollout=rollout, receipt=receipt,
                frozen_policy_sha256=args.policy_sha256, optimizer_updates=0))
            progress = dict(contract=contract, trial=trial+1, sampler=sampler.bit_generator.state,
                            runtime=snapshot(), rng=rng_state())
            atomic_save(recovery, progress)
        summary = dict(trials=progress['trial'], optimizer_updates=0)
    write_json(args.output / 'index.json', dict(source_contract=contract, summary=summary,
        physical_cost=physical_charge(sim.journal), status='source runtime data; no robot or meta-learning benefit established'))
    sim.close()


if __name__ == '__main__':
    main()
