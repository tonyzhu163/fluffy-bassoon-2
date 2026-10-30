"""Complete ragged-trial PPO batches without padding or cross-trial memory.

Reuses the existing collector, policy, GAE and PPO objective. A batch is a
single unpadded time stream with explicit trial terminal/reset boundaries.
No optimization occurs until every declared trial has been collected.
"""
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from .checkpoint import restore_rng, rng_state
from .ppo import update_ppo
from .vla_history import file_sha256
from .vla_rl2 import export_rl2_policy
from .warmstart import atomic_save


FORMAT = 'frozen_vla_rl2_complete_trial_batch_v1'
FIELDS = ('observations', 'raw_actions', 'log_probs', 'values', 'rewards',
          'trial_ends', 'attempt_ends', 'resets')


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _cpu(rollout):
    return {key: value.detach().cpu().clone() for key, value in rollout.items()}


def concatenate_trials(rollouts, *, observation_dim, noise_dim, attempts=4):
    """Real positions only; reset every recurrent-state leaf at each new trial."""
    if not rollouts or attempts < 1:
        raise ValueError('Require a nonempty batch of complete trials')
    arrays, boundaries, offset = [], [], 0
    for trial, rollout in enumerate(rollouts):
        if not set(FIELDS).issubset(rollout):
            raise ValueError('A trial omitted real observations/actions/likelihoods/values/boundaries')
        data = {key: rollout[key].clone() for key in FIELDS}
        length = len(data['observations'])
        if (length < attempts or data['observations'].shape != (length, 1, observation_dim + noise_dim + 3)
                or data['raw_actions'].shape != (length, 1, noise_dim)
                or any(data[key].shape != (length, 1) for key in FIELDS[2:])
                or any(not torch.isfinite(data[key]).all() for key in FIELDS[:5])
                or any(data[key].dtype != torch.bool for key in FIELDS[5:])
                or data['trial_ends'][:-1].any() or not data['trial_ends'][-1].all()
                or int(data['attempt_ends'].sum()) != attempts or not data['attempt_ends'][-1].all()):
            raise ValueError('Require finite unpadded [time,1] complete K-attempt trial data')
        if 'action_mask' in rollout and (rollout['action_mask'].shape != (length, 1)
                or rollout['action_mask'].dtype != torch.bool or not rollout['action_mask'].all()):
            raise ValueError('This batch accepts real complete trials only, never padded or cut positions')
        first_feedback = data['observations'][0, 0, observation_dim:]
        if (not torch.equal(first_feedback[:-1], torch.zeros_like(first_feedback[:-1]))
                or float(first_feedback[-1]) != 1.):
            raise ValueError('Every trial must begin with zero previous noise/reward/done and episode-start1')
        # collect_rl2_trial starts with state=None. During concatenated replay,
        # True invokes core.initial_state for ALL cache/memory/position leaves.
        data['resets'][0] = True
        arrays.append(data)
        boundaries.append(dict(trial=trial, start=offset, end=offset + length, decisions=length))
        offset += length
    merged = {key: torch.cat([trial[key] for trial in arrays], dim=0) for key in FIELDS}
    merged['action_mask'] = torch.ones_like(merged['trial_ends'], dtype=torch.bool)
    return merged, boundaries


@torch.no_grad()
def replay_complete_batch(policy, rollouts):
    """Individual and concatenated log-prob/value replay must match exactly."""
    policy.eval()
    device = next(policy.parameters()).device
    trials = [{key: value.to(device) for key, value in trial.items()} for trial in rollouts]
    merged, boundaries = concatenate_trials(trials, observation_dim=policy.config.observation_dim,
        noise_dim=policy.config.noise_dim)
    individual_log, individual_values = [], []
    for trial in trials:
        log_prob, _, value = policy.evaluate_sequence(trial['observations'], trial['raw_actions'], trial['resets'])
        individual_log.append(log_prob)
        individual_values.append(value)
    log_prob, _, value = policy.evaluate_sequence(merged['observations'], merged['raw_actions'], merged['resets'])
    reference_log, reference_values = torch.cat(individual_log), torch.cat(individual_values)
    metrics = dict(trials=len(trials), valid_tokens=len(log_prob), trial_boundaries=boundaries,
        individual_log_prob_max_error=float((reference_log - merged['log_probs']).abs().max()),
        individual_value_max_error=float((reference_values - merged['values']).abs().max()),
        merged_log_prob_max_error=float((log_prob - reference_log).abs().max()),
        merged_value_max_error=float((value - reference_values).abs().max()),
        all_trial_memory_and_feedback_resets=True)
    if any(metrics[key] != 0. for key in ('individual_log_prob_max_error', 'individual_value_max_error',
            'merged_log_prob_max_error', 'merged_value_max_error')):
        raise RuntimeError('Exact pre-update individual/merged behavior replay failed: ' + json.dumps(metrics))
    return merged, metrics


def state_signature(policy, optimizer):
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
        else:
            digest.update(repr(value).encode())
    visit(dict(policy=policy.state_dict(), optimizer=optimizer.state_dict()))
    return digest.hexdigest()


def receipt_counts(receipts):
    patterns = [''.join('S' if attempt['success'] else 'F' for attempt in row['attempts']) for row in receipts]
    def recovers(pattern):
        return any(pattern[i] == 'F' and 'S' in pattern[i + 1:] for i in range(len(pattern)))
    tasks = {}
    for row in receipts:
        descriptor = row.get('descriptor', {})
        key = descriptor.get('task_key', 'unspecified')
        entry = tasks.setdefault(key, dict(trials=0, episodes=0, initial_state_ids=[], scene_seeds=[],
            initial_state_vector_sha256=[], initial_simulator_state_sha256=[]))
        entry['trials'] += 1
        entry['episodes'] += len(row['attempts'])
        if 'initial_state' in descriptor:
            entry['initial_state_ids'] = sorted(set(entry['initial_state_ids']) | {descriptor['initial_state']})
        if 'scene_seed' in descriptor:
            entry['scene_seeds'] = sorted(set(entry['scene_seeds']) | {descriptor['scene_seed']})
        for field in ('initial_state_vector_sha256', 'initial_simulator_state_sha256'):
            entry[field] = sorted(set(entry[field]) | {
                attempt[field] for attempt in row['attempts'] if attempt.get(field) is not None})
    return dict(trials=len(receipts), episodes=sum(len(row['attempts']) for row in receipts),
        successes=sum(pattern.count('S') for pattern in patterns), failures=sum(pattern.count('F') for pattern in patterns),
        all_success_trials=sum('F' not in pattern for pattern in patterns),
        fail_then_success_trials=sum(recovers(pattern) for pattern in patterns),
        learned_behavior_fail_then_success_trials=sum(recovers(pattern) and row.get('behavior_update', 0) > 0
            for pattern, row in zip(patterns, receipts)), patterns=patterns,
        task_coverage=tasks,
        case_identity='Nominal state IDs and repeated scene seeds are reported separately; seed alone is not a new physical state')


def train_complete_batches(policy, *, planner, collector, runtime_state, restore_runtime,
        physical_cost, output, source_contract, updates=100, batch_trials=14,
        max_total_actions=1_000_000, seed=0, mode='full', learning_rate=1e-4, epochs=2,
        stop_unix=None, resume=False):
    """Checkpoint each complete trial, including a partially collected batch.

    planner(sampler) returns a fixed whole-batch descriptor list. collector
    receives a descriptor, unique global trial ID, and immutable behavior hash.
    Interrupted current trials are not fabricated; their physical journals
    remain charged while the last complete-trial sampler/runtime is restored.
    """
    if sys.platform != 'linux':
        raise ValueError('Training/models/tests require allocated Linux resources')
    if (updates < 1 or batch_trials != 14 or epochs < 1 or mode not in ('full', 'episode')
            or type(max_total_actions) is not int or not 0 < max_total_actions <= 1_000_000
            or source_contract.get('source_only') is not True or source_contract.get('source_data_from_targets') is not False
            or stop_unix is None or not np.isfinite(stop_unix) or learning_rate <= 0):
        raise ValueError('Require14 complete source trials, explicit live time and at most1M physical actions')
    output = Path(output)
    contract = dict(format=FORMAT, source=source_contract, config=asdict(policy.config),
        updates_target=updates, batch_trials=batch_trials, attempts=4, seed=seed, mode=mode,
        maximum_total_actions=max_total_actions, optimizer=dict(name='Adam', learning_rate=learning_rate),
        ppo=dict(epochs=epochs, clip=.2, gamma=1., gae_lambda=1., entropy_coef=.01,
            value_coef=.5, max_grad_norm=1., bounded_entropy=True,
            loss_weighting='Sum over real decision positions / total valid tokens across14 complete trials',
            advantage_normalization='Global valid-token mean/std across all14 trials',
            global_optimizer_steps_per_update=epochs),
        credit='GAE ends at every complete K4 trial; memory resets at every trial; full history within a trial',
        partial_batch='Complete trials retained across wall stops; never optimize fewer than14 or cut episodes to fit')
    policy.requires_grad_(True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    sampler = np.random.default_rng(seed)
    progress = dict(iteration=0, completed_trials=0, phase='collect', plan=None,
        pending_rollouts=[], pending_receipts=[], behavior_sha256=None, behavior_state_signature=None,
        counters=dict(environment_steps=0, settling_steps=0, rollout_forward_samples=0,
            update_forward_samples=0, verification_forward_samples=0, optimizer_steps=0),
        completed_batch_statistics=[], update_receipts=[])
    checkpoint = output / 'latest.pt'
    if resume:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved.get('format') != FORMAT or saved.get('contract') != contract:
            raise ValueError('Batch continuation requires the exact original source/scientific contract')
        policy.load_state_dict(saved['policy'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        sampler.bit_generator.state = saved['sampler']
        restore_rng(saved['rng'])
        restore_runtime(saved['runtime'])
        progress = saved['progress']
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'traces').mkdir()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        _json(output / 'contract.json', contract)

    if (not isinstance(progress.get('pending_rollouts'), list)
            or len(progress['pending_rollouts']) != len(progress.get('pending_receipts', []))
            or len(progress['pending_rollouts']) > batch_trials
            or progress['completed_trials'] != progress['iteration'] * batch_trials + len(progress['pending_rollouts'])
            or ((progress['plan'] is None) != (progress['behavior_sha256'] is None))
            or (progress['plan'] is not None and len(progress['plan']) != batch_trials)):
        raise ValueError('Recovery omitted a complete-trial partial batch or its exact source cursor')
    if progress['plan'] is not None:
        behavior_path = output / 'traces' / f"behavior-{progress['iteration']:06d}.pt"
        if (not behavior_path.is_file() or file_sha256(behavior_path) != progress['behavior_sha256']
                or state_signature(policy, optimizer) != progress['behavior_state_signature']):
            raise ValueError('Pending batch no longer has its exact frozen behavior/optimizer state')
        for index, trial in enumerate(progress['pending_rollouts']):
            concatenate_trials([trial], observation_dim=policy.config.observation_dim, noise_dim=policy.config.noise_dim)
            receipt = progress['pending_receipts'][index]
            if (receipt.get('descriptor') != progress['plan'][index]
                    or receipt.get('behavior_policy_sha256') != progress['behavior_sha256']
                    or receipt.get('behavior_update') != progress['iteration']):
                raise ValueError('Pending trial lost its exact planned case/frozen behavior-policy binding')

    def save():
        digest = atomic_save(checkpoint, dict(format=FORMAT, contract=contract,
            boundary='complete trial; optimizer idle or whole batch ready', policy=policy.state_dict(),
            optimizer=optimizer.state_dict(), sampler=sampler.bit_generator.state,
            rng=rng_state(), runtime=runtime_state(), progress=progress))
        # This derived stream is rebuilt from the atomically committed update
        # receipts. A crash between SGD/checkpoint/log writes cannot create a
        # second, apparently new scientific update when the burst is replayed.
        temporary = output / 'metrics.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in progress['update_receipts']))
        temporary.replace(output / 'metrics.jsonl')
        return digest

    def report(reason):
        all_statistics = progress['completed_batch_statistics'] + [receipt_counts(progress['pending_receipts'])]
        exposure = {}
        for batch in all_statistics:
            for key, row in batch['task_coverage'].items():
                target = exposure.setdefault(key, dict(trials=0, episodes=0, initial_state_ids=[], scene_seeds=[],
                    initial_state_vector_sha256=[], initial_simulator_state_sha256=[]))
                for field in ('trials', 'episodes'):
                    target[field] += row[field]
                for field in ('initial_state_ids', 'scene_seeds', 'initial_state_vector_sha256',
                        'initial_simulator_state_sha256'):
                    target[field] = sorted(set(target[field]) | set(row[field]))
        value = dict(format=FORMAT, completed=progress['iteration'] == updates,
            updates=progress['iteration'], target_updates=updates, counters=progress['counters'],
            complete_source_trials=progress['completed_trials'], pending_complete_trials=len(progress['pending_rollouts']),
            physical_action_charge=physical_cost(), maximum_total_actions=max_total_actions,
            stop_reason=reason, full_recovery_sha256=file_sha256(checkpoint),
            batch_statistics=progress['completed_batch_statistics'],
            pending_statistics=receipt_counts(progress['pending_receipts']),
            committed_source_task_coverage=exposure, actual_tasks_with_complete_trials=len(exposure),
            interpretation='Actual source exposure and optimizer progress; no robot/memory/sample-efficiency benefit established')
        if progress['iteration'] > 0:
            value['candidate_sha256'] = export_rl2_policy(output / 'candidate.pt', policy,
                manifest=dict(contract=contract, updates=progress['iteration'], counters=progress['counters'],
                    pending_complete_trials=len(progress['pending_rollouts']), status='Unvalidated source candidate'))
        _json(output / 'summary.json', value)
        return value

    save()
    while progress['iteration'] < updates:
        if time.time() + 90 >= stop_unix:
            return report('wall_stop_at_complete_trial_boundary')
        if progress['plan'] is None:
            before_plan = copy.deepcopy(sampler.bit_generator.state)
            descriptors = planner(sampler)
            if (len(descriptors) != batch_trials or any(type(row.get('maximum_actions')) is not int
                    or row['maximum_actions'] < 1 for row in descriptors)):
                raise ValueError('A global batch must declare exactly14 complete trials')
            required = sum(row['maximum_actions'] for row in descriptors)
            if physical_cost() + required > max_total_actions:
                sampler.bit_generator.state = before_plan
                save()
                return report('insufficient_actions_for_another_complete_batch')
            behavior_path = output / 'traces' / f"behavior-{progress['iteration']:06d}.pt"
            if behavior_path.exists():
                behavior_path.rename(behavior_path.with_name(behavior_path.name + f'-uncommitted-{time.time_ns()}'))
            behavior_sha = export_rl2_policy(behavior_path, policy, manifest=dict(training_contract=contract,
                behavior_update=progress['iteration'], source_commit=source_contract['source_commit'],
                status='Single frozen behavior policy for14 complete source trials'))
            progress.update(plan=descriptors, behavior_sha256=behavior_sha,
                behavior_state_signature=state_signature(policy, optimizer))
            save()
        remaining = progress['plan'][len(progress['pending_rollouts']):]
        if physical_cost() + sum(row['maximum_actions'] for row in remaining) > max_total_actions:
            return report('retry_cost_prevents_completion_of_preserved_pending_batch')
        while len(progress['pending_rollouts']) < batch_trials:
            if time.time() + 90 >= stop_unix:
                return report('wall_stop_with_preserved_partial_batch')
            descriptor = progress['plan'][len(progress['pending_rollouts'])]
            trial_id = progress['completed_trials'] + 1
            try:
                rollout, receipt = collector(policy, descriptor, trial_id, progress['iteration'], progress['behavior_sha256'])
                concatenate_trials([rollout], observation_dim=policy.config.observation_dim,
                    noise_dim=policy.config.noise_dim)
                if (receipt.get('descriptor') != descriptor or receipt.get('behavior_update') != progress['iteration']
                        or receipt.get('behavior_policy_sha256') != progress['behavior_sha256']):
                    raise RuntimeError('Collected trial lost its exact planned case/frozen behavior-policy binding')
                if state_signature(policy, optimizer) != progress['behavior_state_signature']:
                    raise RuntimeError('Policy or optimizer changed while collecting its global trial batch')
            except BaseException as error:
                _json(output / f'unexpected-collection-{time.time_ns()}.json', dict(stage='complete_source_trial',
                    descriptor=descriptor, trial=trial_id, error=repr(error), completed_updates=progress['iteration'],
                    preserved_complete_trials=len(progress['pending_rollouts']), physical_action_charge=physical_cost(),
                    action='Stopped; retain actual raw/physical evidence and diagnose before any retry or scale decision'))
                raise
            progress['pending_rollouts'].append(_cpu(rollout))
            progress['pending_receipts'].append(receipt)
            progress['completed_trials'] += 1
            progress['counters']['environment_steps'] += int(receipt['primitive_steps'])
            progress['counters']['settling_steps'] += int(receipt['settling_steps'])
            progress['counters']['rollout_forward_samples'] += int(receipt['decisions'])
            progress['phase'] = 'ready_update' if len(progress['pending_rollouts']) == batch_trials else 'collect'
            save()
        # The ready checkpoint is intentionally not overwritten until the
        # entire optimizer burst commits. An interrupted update replays the
        # same collected batch from its original policy/optimizer/RNG state.
        if time.time() + 90 >= stop_unix:
            return report('wall_stop_with_preserved_ready_batch')
        try:
            device = next(policy.parameters()).device
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            merged, replay = replay_complete_batch(policy, progress['pending_rollouts'])
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            replay['seconds'] = time.monotonic() - started
            started = time.monotonic()
            metrics = update_ppo(policy, optimizer, merged, epochs=epochs, minibatch_trials=1,
                clip=.2, gamma=1., gae_lambda=1., entropy_coef=.01, value_coef=.5,
                max_grad_norm=1., bounded_entropy=True, episode_returns=False)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                metrics.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
            metrics['update_seconds'] = time.monotonic() - started
            if (metrics['optimizer_steps'] != epochs or metrics['active_action_samples'] != len(merged['rewards']) * epochs
                    or any(not np.isfinite(value) for value in metrics.values())
                    or any(not torch.isfinite(value).all() for value in policy.state_dict().values())):
                raise RuntimeError('Batch optimizer count, valid-token exposure or finite policy qualification failed')
        except BaseException as error:
            _json(output / f'unexpected-update-{time.time_ns()}.json', dict(stage='ready_batch_update', error=repr(error),
                completed_updates=progress['iteration'], pending_complete_trials=len(progress['pending_rollouts']),
                physical_action_charge=physical_cost(), action='Stopped before proceeding; preserve ready checkpoint and diagnose'))
            raise
        progress['iteration'] += 1
        statistics = receipt_counts(progress['pending_receipts'])
        progress['completed_batch_statistics'].append(dict(update=progress['iteration'], **statistics))
        progress['counters']['update_forward_samples'] += int(metrics['forward_samples'])
        progress['counters']['verification_forward_samples'] += 2 * int(replay['valid_tokens'])
        progress['counters']['optimizer_steps'] += int(metrics['optimizer_steps'])
        row = dict(update=progress['iteration'], replay=replay, metrics=metrics, source_trials=statistics,
            physical_action_charge=physical_cost(), counters=copy.deepcopy(progress['counters']))
        progress['update_receipts'].append(row)
        progress.update(plan=None, pending_rollouts=[], pending_receipts=[], phase='collect',
            behavior_sha256=None, behavior_state_signature=None)
        save()
        print(json.dumps(dict(batch_training=row), allow_nan=False), flush=True)
    return report('requested_complete_batch_updates')
