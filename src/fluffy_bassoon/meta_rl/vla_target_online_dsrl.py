"""Fresh target-online DSRL learning-onset curves on two predeclared cases.

Target interactions remain evaluation/adaptation data and cannot enter source
controller pretraining. The qualified authors' pixel SAC update is reused.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import socket
import subprocess
import sys
import time

import numpy as np

from .vla_libero import (FrozenOpenPI, OfficialSAC, LIBERO_COMMIT, expand_noise, run_episode,
                         sac_observation, sha256, write_json)
from .vla_source40_corpus import OnlineSourceSimulator, PI05_RECEIPT, restore_motor_rng, save_motor_rng
from .vla_source40_schedule import content_hash
from .vla_source40_subset_evaluate import TargetServer


FORMAT = 'openpi_pi05_target_online_dsrl_onset_v1'
MILESTONES = ((0, 0), (1, 410), (13, 5330), (15, 6150))
CASE_CAP = 8000
TOTAL_CAP = 16000
RECIPE = dict(warmup_condition='replay >500 query chunks after a complete episode',
    batch_size=256, update_to_data=20, fixed_weights_within_episode=True,
    actor_lr=1e-4, critic_lr=3e-4, temp_lr=3e-4, hidden_dims=[128, 128, 128],
    cnn_features=[32, 32, 32, 32], cnn_strides=[2, 1, 1, 1], latent_dim=50,
    discount=.999, tau=.005, num_qs=10, target_entropy='auto',
    masks='0 on success, 1 on timeout/failure', reward='-1 per chunk, 0 on success',
    action_bounds='Gaussian warmup unbounded; learned32D noise clipped[-1,1]',
    query_behavior='Native independent H10 Gaussian before any update; stochastic sample_actions SAC after update',
    upstream_collection='examples/train_utils_sim.py:collect_traj at DSRL7f48937',
    upstream_evaluation='examples/train_utils_sim.py:perform_control_eval at DSRL7f48937',
    interface='Official image64 pixels + proprio8, one camera; no VLA-prefix student features',
    departures='Frozen pi0.5 H10/execute10, actual-duration discounts, fixed predeclared physical cases, bounded learning-onset slice')


def case(task):
    return dict(benchmark='libero90', case_id=f'libero90:{task:02d}/state-0/seed-123',
        suite='libero_90', task_id=task, initial_state=0, scene_seed=123)


def costs(directory):
    totals = {role: dict(known_actions=0, uncertain_actions_ceiling=0, primitive_steps=0,
        settling_steps=0, reset_attempts=0) for role in ('training', 'query', 'qualification')}
    for path in Path(directory).glob('physical-journal/*.json'):
        row = json.loads(path.read_text())
        value = totals[row['role']]
        value['known_actions' if row['returned'] else 'uncertain_actions_ceiling'] += row['actual_steps'] if row['returned'] else row['maximum_steps']
        if row['kind'] == 'reset':
            value['reset_attempts'] += 1
        if row['returned']:
            value['primitive_steps' if row['kind'] == 'step' else 'settling_steps'] += row['actual_steps']
    return totals


def charged(directory):
    return sum(value[name] for value in costs(directory).values() for name in ('known_actions', 'uncertain_actions_ceiling'))


def joint_charge(root):
    return sum(charged(path) for path in Path(root).glob('case-*'))


class JournaledSimulator:
    """Persist full RPC inputs before execution and outcomes before learning."""
    def __init__(self, sim, output, root, *, case_cap=CASE_CAP, total_cap=TOTAL_CAP):
        self.sim, self.metadata, self.output, self.root = sim, sim.metadata, Path(output), Path(root)
        self.case_cap, self.total_cap = case_cap, total_cap
        self.journal = self.output / 'physical-journal'
        self.journal.mkdir(parents=True, exist_ok=True)
        self.role, self.episode_id, self.context = 'qualification', 'qualification', None
        # The root run lock gives this process sole ownership of all its RPCs.
        self.case_charge, self.total_charge = charged(self.output), joint_charge(self.root)

    def _intent(self, kind, maximum, payload):
        if self.case_charge + maximum > self.case_cap or self.total_charge + maximum > self.total_cap:
            raise RuntimeError('Target-online joint/case physical action ceiling would be exceeded')
        identity = str(time.time_ns())
        input_path = self.journal / f'{identity}-input.pkl.gz'
        with gzip.open(input_path, 'xb', compresslevel=1) as stream:
            pickle.dump(payload, stream, protocol=5)
        row = dict(kind=kind, role=self.role, episode_id=self.episode_id, maximum_steps=maximum,
            actual_steps=None, returned=False, input_path=input_path.name, input_sha256=sha256(input_path))
        path = self.journal / f'{identity}.json'
        write_json(path, row)
        self.case_charge += maximum
        self.total_charge += maximum
        return path, row

    def _returned(self, event, actual, outcome):
        path, row = event
        if type(actual) is not int or not 0 <= actual <= row['maximum_steps']:
            raise ValueError('Actual RPC cost differs from the saved physical intent')
        # Charge the returned physical cost even if raw serialization fails.
        self.case_charge += actual - row['maximum_steps']
        self.total_charge += actual - row['maximum_steps']
        row.update(returned=True, actual_steps=actual, outcome_status='writing')
        write_json(path, row)
        trace = path.with_suffix('.pkl.gz')
        with gzip.open(trace, 'xb', compresslevel=1) as stream:
            pickle.dump(outcome, stream, protocol=5)
        write_json(path, dict(row, outcome_status='complete', outcome_path=trace.name, outcome_sha256=sha256(trace)))

    def reset(self, state, seed):
        event = self._intent('reset', 10, dict(initial_state=state, scene_seed=seed))
        result = self.sim.reset(state, seed)
        self._returned(event, 10, result)
        if not result.get('initial_state_vector_sha256') or not result.get('initial_observation_sha256'):
            raise ValueError('Actual target reset omitted physical/public observation identity')
        return result

    def step(self, actions):
        actions = np.asarray(actions, np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7 or not 1 <= len(actions) <= 10 or not np.isfinite(actions).all():
            raise ValueError('Require finite native H10 motor actions before RPC')
        event = self._intent('step', len(actions), dict(context=self.context, proposed_motor_actions=actions))
        result = self.sim.step(actions)
        self._returned(event, int(result['primitive_steps']), result)
        count = int(result['primitive_steps'])
        if (count < 1 or np.asarray(result['executed_actions']).shape != (count, 7)
                or not np.isfinite(result['executed_actions']).all() or len(result['rewards']) != count
                or not np.isfinite(result['rewards']).all() or type(result['done']) not in (bool, np.bool_)
                or type(result['success']) not in (bool, np.bool_)):
            raise ValueError('Simulator omitted actual executed actions/rewards/boundaries')
        return result

    def reserve_episode(self):
        return self.case_charge + 410 <= self.case_cap and self.total_charge + 410 <= self.total_cap


def _signature(learner):
    if hasattr(learner, 'frozen_signature'):
        return learner.frozen_signature()
    from flax import serialization
    return hashlib.sha256(serialization.to_bytes(learner.agent._save_dict)).hexdigest()


def _key(seed):
    import jax
    return jax.random.PRNGKey(seed)


def _save_motor(motor):
    return save_motor_rng(motor.policy._rng)


def _restore_motor(motor, value):
    motor.policy._rng = restore_motor_rng(value)


def _seed(target, role, ordinal):
    return int(content_hash(dict(case=target, role=role, ordinal=ordinal))[:8], 16)


def frozen_query(sim, motor, learner, target):
    """Published native-before-learning / stochastic-SAC-after-learning query."""
    seed = _seed(target, 'query', 0)
    agent_rng, motor_rng = learner.agent._rng, motor.policy._rng
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    signature, replay_length, updates = _signature(learner), len(learner.replay), learner.updates
    learner.agent._rng = _key(seed)
    motor.policy._rng = _key(seed)
    sampler = np.random.default_rng(seed)
    try:
        initial = sim.reset(target['initial_state'], target['scene_seed'])
        observation, rows, steps = initial['observation'], [], 0
        while True:
            proposed = None if updates == 0 else learner.action(observation)
            executed = None if proposed is None else np.clip(proposed, -1., 1.).astype(np.float32)
            noise = sampler.standard_normal((10, 32)).astype(np.float32) if executed is None else expand_noise(executed, 10)
            actions = np.asarray(motor.infer(observation, sim.metadata['instruction'], noise), np.float32)[:min(10, 400 - steps)]
            if not len(actions):
                raise RuntimeError('Query simulator did not finish at its horizon')
            sim.context = dict(observation=observation, proposed_noise=proposed, executed_noise=executed,
                native_initial_noise=noise, behavior='native' if updates == 0 else 'frozen_target_sac', updates=updates)
            outcome = sim.step(actions)
            row = dict(motor_observation=observation, motor_next_observation=outcome['observation'],
                sac_observation=sac_observation(observation), sac_next_observation=sac_observation(outcome['observation']),
                proposed_noise=proposed, executed_noise=executed, native_initial_noise=noise,
                executed_actions=outcome['executed_actions'], raw_rewards=outcome['rewards'],
                reward=0. if outcome['success'] else -1., done=bool(outcome['done']), success=bool(outcome['success']),
                terminated=bool(outcome['success']), truncated=bool(outcome['done']) and not bool(outcome['success']),
                primitive_steps=int(outcome['primitive_steps']), primitive_discount=.999 ** outcome['primitive_steps'])
            rows.append(row)
            steps += row['primitive_steps']
            observation = outcome['observation']
            if outcome['done']:
                break
        if _signature(learner) != signature or len(learner.replay) != replay_length or learner.updates != updates:
            raise RuntimeError('Frozen query changed actor/critic/optimizer, replay or update count')
        return rows, dict(success=bool(outcome['success']), primitive_steps=steps, settling_steps=10,
            chunks=len(rows), raw_return=sum(sum(row['raw_rewards']) for row in rows),
            initial_state=target['initial_state'], scene_seed=target['scene_seed'], episode_random_seed=seed,
            initial_state_vector_sha256=initial['initial_state_vector_sha256'],
            initial_observation_sha256=initial['initial_observation_sha256'], completed=True,
            query_behavior='native' if updates == 0 else 'frozen_target_sac', target_optimizer_updates=updates,
            replay_rows_unchanged=True, weights_unchanged=True, optimizer_updates_during_query=0)
    finally:
        # Queries cannot change the subsequent training randomness or replay.
        learner.agent._rng, motor.policy._rng = agent_rng, motor_rng
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)


def save_trace(output, identity, rows, summary, contract):
    path = Path(output) / 'raw' / f'{identity}-{time.time_ns()}.pkl.gz'
    path.parent.mkdir(exist_ok=True)
    with gzip.open(path, 'xb', compresslevel=1) as stream:
        pickle.dump(dict(rows=rows, summary=summary, contract_sha256=content_hash(contract)), stream, protocol=5)
    return dict(summary, trace=str(path.relative_to(output)), trace_sha256=sha256(path))


def run_case(sim, motor, learner, sampler, *, target, output, root, contract, resume=False, deadline=float('inf')):
    """Continue only from a complete physical episode plus its optimizer burst."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'target-recovery.pkl'
    progress = dict(training_episodes=0, training_chunks=0, training_primitive_steps=0,
        training_settling_steps=0, next_milestone=0, training_receipts=[], query_receipts=[], nominal_state_sha256=None)
    if resume:
        if json.loads((output / 'contract.json').read_text()) != contract:
            raise ValueError('Target learner resume contract changed')
        progress = learner.restore(checkpoint, sampler, contract)
        _restore_motor(motor, progress.pop('motor_rng'))
    else:
        if (output / 'contract.json').exists():
            raise ValueError('Fresh target learner requires a new immutable identity')
        write_json(output / 'contract.json', contract)
    journaled = JournaledSimulator(sim, output, root)

    def save(completed=False, stop_reason=None):
        learner.save(checkpoint, sampler, contract, **progress, motor_rng=_save_motor(motor))
        value = dict(progress, format=FORMAT, completed=completed, case=target,
            target_only=True, source_assembly_admission=False, source_checkpoint_reused=False,
            target_optimizer_updates=learner.updates, replay_rows=len(learner.replay), costs=costs(output),
            total_action_charge=charged(output), maximum_total_actions=CASE_CAP,
            latest_recovery_sha256=sha256(checkpoint), contract_sha256=content_hash(contract),
            stop_reason=stop_reason, warmup_cleared=learner.updates > 0,
            interpretation='Learning-onset diagnostic only; not a converged DSRL reproduction')
        write_json(output / 'index.json', value)
        return value

    if not resume:
        save()
    wrapped = OnlineSourceSimulator(journaled, learner, sampler, sac_observation, warmup=500, batch=256, utd=20)
    for milestone in range(progress['next_milestone'], len(MILESTONES)):
        required_episodes, nominal_action_cap = MILESTONES[milestone]
        while progress['training_episodes'] < required_episodes:
            if time.monotonic() + 90 >= deadline or not journaled.reserve_episode():
                return save(stop_reason='complete_episode_time_or_action_reservation')
            episode = progress['training_episodes']
            journaled.role, journaled.episode_id = 'training', f'training-{episode:03d}'
            before_signature, before_updates = _signature(learner), learner.updates
            motor.policy._rng = _key(_seed(target, 'training_motor', episode))
            def choose(observation):
                noise = wrapped.choose(observation)
                journaled.context = dict(wrapped.pending, native_initial_noise=expand_noise(noise, 10),
                    policy_signature=before_signature)
                return noise
            rows, summary = run_episode(wrapped, motor, choose,
                initial_state=target['initial_state'], scene_seed=target['scene_seed'],
                chunk=10, bounded=False, record_features=False)
            if _signature(learner) != before_signature or learner.updates != before_updates:
                raise RuntimeError('Training actor/critic/optimizer changed within an episode')
            if not rows or not rows[-1]['done'] or len(rows) != len(wrapped.provenance) or summary['primitive_steps'] > 400:
                raise RuntimeError('Require a complete native target training episode')
            metrics = wrapped.finish_episode()
            expected_updates = before_updates + (20 * len(rows) if len(learner.replay) > 500 else 0)
            if learner.updates != expected_updates or any(not np.isfinite(value) for value in metrics.values()):
                raise RuntimeError('Published SAC update count or finite metric qualification failed')
            for row, provenance in zip(rows, wrapped.provenance, strict=True):
                row.update({key.replace('source_', 'target_', 1): value for key, value in provenance.items()},
                    target_only=True, policy_signature=before_signature, target_optimizer_updates_after_episode=learner.updates)
                row['target_task_key'] = row.pop('source_task_key')
            summary.update(completed=True, training_episode=episode, target_optimizer_updates_before=before_updates,
                target_optimizer_updates=learner.updates, replay_rows=len(learner.replay), target_update_metrics=metrics,
                behavior='warmup_gaussian' if before_updates == 0 else 'target_sac',
                policy_signature=before_signature, actor_fixed_within_episode=True)
            if progress['nominal_state_sha256'] is None:
                progress['nominal_state_sha256'] = summary.get('initial_state_vector_sha256')
            if not progress['nominal_state_sha256'] or summary.get('initial_state_vector_sha256') != progress['nominal_state_sha256']:
                raise RuntimeError('Target training changed the declared nominal physical case')
            progress['training_episodes'] += 1
            progress['training_chunks'] += len(rows)
            progress['training_primitive_steps'] += int(summary['primitive_steps'])
            progress['training_settling_steps'] += 10
            summary.update(cumulative_training_primitive_steps=progress['training_primitive_steps'],
                cumulative_training_settling_steps=progress['training_settling_steps'],
                cumulative_training_actions=progress['training_primitive_steps'] + progress['training_settling_steps'])
            progress['training_receipts'].append(save_trace(output, journaled.episode_id, rows, summary, contract))
            save()
            print(json.dumps(dict(task=target['task_id'], episode=episode + 1, chunks=progress['training_chunks'],
                updates=learner.updates, training_actions=summary['cumulative_training_actions'], joint_actions=joint_charge(root))), flush=True)
        if time.monotonic() + 90 >= deadline or not journaled.reserve_episode():
            return save(stop_reason='complete_query_time_or_action_reservation')
        journaled.role, journaled.episode_id = 'query', f'query-{milestone:03d}'
        rows, summary = frozen_query(journaled, motor, learner, target)
        if progress['nominal_state_sha256'] is None:
            progress['nominal_state_sha256'] = summary['initial_state_vector_sha256']
        if summary['initial_state_vector_sha256'] != progress['nominal_state_sha256']:
            raise RuntimeError('Target query changed the declared nominal physical case')
        summary.update(milestone=milestone, training_episode_budget=required_episodes,
            training_action_ceiling=nominal_action_cap, actual_training_episodes=progress['training_episodes'],
            actual_training_primitive_steps=progress['training_primitive_steps'],
            actual_training_settling_steps=progress['training_settling_steps'],
            actual_training_actions=progress['training_primitive_steps'] + progress['training_settling_steps'],
            actual_training_chunks=progress['training_chunks'], query_data_added_to_replay=False,
            policy_signature=_signature(learner))
        progress['query_receipts'].append(save_trace(output, journaled.episode_id, rows, summary, contract))
        progress['next_milestone'] += 1
        archive = output / f'milestone-{milestone:03d}-{time.time_ns()}-recovery.pkl'
        learner.save(archive, sampler, contract, **progress, motor_rng=_save_motor(motor))
        progress['query_receipts'][-1].update(full_recovery_path=archive.name, full_recovery_sha256=sha256(archive))
        save()
    return save(completed=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'checkpoint', 'openpi-root', 'dsrl-root', 'libero-root', 'simulator-python'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('source-commit', 'expected-hostname', 'simulator-pythonpath'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--max-total-actions', type=int, required=True)
    parser.add_argument('--max-elapsed-seconds', type=float, default=2400.)
    parser.add_argument('--learner-seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=8769)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname:
        raise ValueError('Run target-online DSRL only on the explicitly allocated Linux host')
    if args.max_total_actions != TOTAL_CAP or not 0 < args.max_elapsed_seconds <= 2400 or not 0 <= args.learner_seed < 2**31:
        raise ValueError('Require the joint16k physical action contract and <=40min driver ceiling')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed target-online DSRL commit')
    if sha256(args.checkpoint / 'READY.json') != PI05_RECEIPT:
        raise ValueError('Require the qualified official OpenPI pi05_libero motor')
    contract = dict(format=FORMAT, source_commit=args.source_commit, cases=[case(0), case(1)],
        milestones=[dict(training_episodes=episodes, training_action_ceiling=actions) for episodes, actions in MILESTONES],
        maximum_total_actions=TOTAL_CAP, maximum_actions_per_case=CASE_CAP, learner_seed=args.learner_seed,
        maximum_seconds_per_invocation=args.max_elapsed_seconds, motor_receipt_sha256=PI05_RECEIPT,
        recipe=RECIPE, target_only=True, source_assembly_admission=False,
        fresh_actor_replay_optimizer_per_case=True, fixed_query_seed_across_milestones=True,
        planning=dict(training_episodes=30, query_episodes=8, worst_case_training_actions=12300,
            worst_case_query_actions=3280, initial_qualification_actions=20, retry_reserve=400),
        interpretation='Small learning-onset comparison, no converged DSRL or full-suite claim')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    import fcntl
    lock = (args.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume:
        if json.loads((args.output / 'contract.json').read_text()) != contract:
            raise ValueError('Joint target-online resume contract differs')
    else:
        write_json(args.output / 'contract.json', contract)
    attempt = args.output / 'attempts' / str(time.time_ns())
    attempt.mkdir(parents=True)
    args.model_config, args.feature_interface = 'pi05_libero', 'pixel-sac'
    original_output = args.output
    args.output = attempt
    motor = FrozenOpenPI(args)
    args.output = original_output
    deadline = time.monotonic() + args.max_elapsed_seconds
    results = []
    for target in contract['cases']:
        output = args.output / f'case-{target["task_id"]:02d}'
        case_contract = dict(contract_sha256=content_hash(contract), case=target, recipe=RECIPE,
            learner_seed=args.learner_seed, source_commit=args.source_commit, maximum_total_actions=CASE_CAP,
            motor_receipt_sha256=PI05_RECEIPT, libero_commit=LIBERO_COMMIT,
            horizon=400, settle_steps=10, fresh_environment_on_reset=True,
            target_only=True, source_assembly_admission=False)
        index_path = output / 'index.json'
        if index_path.exists() and json.loads(index_path.read_text()).get('completed') is True:
            result = json.loads(index_path.read_text())
            if (result.get('contract_sha256') != content_hash(case_contract) or result.get('case') != target
                    or result.get('training_episodes') != 15 or len(result.get('query_receipts', [])) != 4
                    or sha256(output / 'target-recovery.pkl') != result.get('latest_recovery_sha256')
                    or not result.get('target_only')):
                raise ValueError('Completed target case receipt differs from the joint contract')
            results.append(result)
            continue
        if time.monotonic() + 120 >= deadline:
            break
        server = TargetServer(task_id=target['task_id'], python=args.simulator_python,
            pythonpath=args.simulator_pythonpath, output=attempt / f'server-{target["task_id"]}',
            port=args.port, libero_root=args.libero_root)
        try:
            output.mkdir(parents=True, exist_ok=True)
            if (server.metadata.get('suite') != 'libero_90' or server.metadata.get('task_id') != target['task_id']
                    or server.metadata.get('fresh_environment_on_reset') is not True):
                raise ValueError('Target simulator does not match the predeclared fresh-reset case')
            qualification = JournaledSimulator(server, output, args.output)
            initial = qualification.reset(target['initial_state'], target['scene_seed'])
            zeros = np.zeros((10, 32), np.float32)
            a = motor.infer(initial['observation'], server.metadata['instruction'], zeros)
            b = motor.infer(initial['observation'], server.metadata['instruction'], zeros.copy())
            if motor.horizon != 10 or not np.isfinite(a).all() or not np.allclose(a, b, rtol=0, atol=1e-5):
                raise RuntimeError('Frozen target motor failed native same-noise qualification')
            write_json(attempt / f'qualification-{target["task_id"]}.json', dict(same_noise_max_delta=float(np.abs(a-b).max()),
                nominal_state_sha256=initial['initial_state_vector_sha256'], settling_actions=10,
                simulator={key: value for key, value in server.metadata.items() if key not in ('pid', 'hostname')}))
            np.random.seed(args.learner_seed)
            random.seed(args.learner_seed)
            learner, sampler = OfficialSAC(args.dsrl_root, args.learner_seed), np.random.default_rng(args.learner_seed)
            result = run_case(server, motor, learner, sampler, target=target, output=output, root=args.output,
                contract=case_contract, resume=(output / 'contract.json').exists(), deadline=deadline)
            results.append(result)
            write_json(args.output / 'progress.json', dict(completed_cases=sum(row['completed'] for row in results),
                cases=results, total_action_charge=joint_charge(args.output)))
            if not result['completed']:
                break
        finally:
            server.close()
    write_json(args.output / 'index.json', dict(format=FORMAT, contract=contract,
        completed=len(results) == 2 and all(row['completed'] for row in results),
        cases=results, total_action_charge=joint_charge(args.output), maximum_total_actions=TOTAL_CAP,
        target_only=True, source_assembly_admission=False, source_checkpoint_reused=False))


if __name__ == '__main__':
    main()
