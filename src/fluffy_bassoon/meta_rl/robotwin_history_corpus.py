"""Native RoboTwin source-history AD/DPT export; no simulator/motor loading.

This schema retains native 2062+1024+4 inputs and physical bound1.5. It is
deliberately not the LIBERO vla_history dataset/trainer schema.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import sys

import numpy as np
import torch

from .gtrxl import GatedTransformerXLCore
from .ppo import RecurrentActorCritic
from .vla_history import file_sha256
from .vla_noise_ppo import PPOConfig, policy_inputs
from .vla_noise_sac import SACConfig, SquashedGaussianActor
from .warmstart import checked_load, validate_weights


SCHEMA = 'robotwin_public_decisions_v1'
FORMAT = 'robotwin_native_history_dataset_v1'
METHODS = ('ppo', 'ppo_reset', 'sac')
TASKS = ('handover_mic', 'pick_diverse_bottles', 'place_a2b_left',
         'place_object_stand', 'place_phone_stand', 'scan_object')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest_array(value):
    digest = hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def parameter_digest(state):
    """Match BehaviorPolicyArchive's tensor digest, not a caller's label string."""
    digest = hashlib.sha256()
    if not isinstance(state, dict) or not state:
        raise ValueError('Acting archive omitted tensor weights')
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor) or not value.isfinite().all():
            raise ValueError('Acting archive has invalid tensor weights')
        value = value.detach().cpu().contiguous()
        digest.update(name.encode() + str(value.dtype).encode() + str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_recovery(path, sha256, method):
    """Only explicit source coordinator/per-method snapshots, never target copies."""
    payload = checked_load(path, sha256, weights_only=False)
    contract = payload.get('contract', {})
    if ('unit' in payload or 'result' in payload or method not in METHODS
            or set(contract.get('source_tasks', [])) != set(TASKS)
            or contract.get('observation_dim') != 2062 or contract.get('native_horizon') != 32
            or tuple(contract.get('noise_shape', [])) != (32, 32)
            or contract.get('action_limit') != 1.5 or contract.get('train_episodes') != 4
            or not contract.get('runtime') or not contract['runtime'].get('norm_stats_sha256')):
        raise ValueError('Require a source-trained native RoboTwin recovery, not a target or LIBERO policy')
    if 'states' in payload:
        if (payload.get('resume_stage') != 'source_or_completed_evaluation_unit'
                or payload.get('boundary') != 'completed source trial/update'):
            raise ValueError('Coordinator is not at a committed source boundary')
        state = payload['states'][method]
    else:
        if payload.get('method') != method or payload.get('boundary') not in (
                'complete trial or declared budget cut', 'completed whole-trial source iteration',
                'exact matched source milestone; pending PPO data retained'):
            raise ValueError('Require a declared per-method source checkpoint boundary')
        if payload.get('resume_stage', 'source') != 'source':
            raise ValueError('Target adaptation cannot provide a teacher or source prefix')
        state = payload
    if tuple(state.get('schedule', {}).get('tasks', [])) != tuple(contract['source_tasks']):
        raise ValueError('Source schedule tasks differ')
    learner = state['learner']
    if method == 'sac':
        if learner.get('target_replay_mode') is not None or learner.get('target_start') is not None:
            raise ValueError('Target-updated SAC cannot supply source labels')
    elif learner.get('kind') != 'gtrxl_noise_ppo' or learner.get('schema_version') != 1:
        raise ValueError('Unsupported recurrent teacher schema')
    count = state.get('completed_source_trials', state['schedule']['trials'])
    if type(count) is not int or count < 1 or count > state['schedule']['trials']:
        raise ValueError('No complete committed source trials')
    return dict(contract=contract, state=state, learner=learner, method=method, trials=count,
                source_iteration=payload.get('source_iteration'))


class CorpusReader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.inputs, self.features, self.actors = {}, {}, {}

    def path(self, value):
        path = (self.root / value).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Corpus reference escapes its source root')
        return path

    def json(self, path, expected=None):
        path = self.path(path)
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError('Corpus JSON checksum differs')
        self.inputs[str(path)] = digest
        return json.loads(path.read_text())

    def arrays(self, ref, names):
        path = self.path(ref['path'])
        if path.stat().st_size != ref['bytes'] or file_sha256(path) != ref['sha256']:
            raise ValueError('Corpus NPZ checksum/size differs')
        self.inputs[str(path)] = ref['sha256']
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(ref['arrays']):
                raise ValueError('Corpus NPZ fields differ')
            result = {name: archive[name].copy() for name in names}
        for name, value in result.items():
            declared = ref['arrays'][name]
            if (value.dtype.hasobject or str(value.dtype) != declared['dtype']
                    or list(value.shape) != declared['shape'] or _digest_array(value) != declared['sha256']
                    or not np.isfinite(value).all()):
                raise ValueError('Corpus array schema/hash/finite-value check failed')
        return result

    def frame(self, ref):
        key = str(self.path(ref['path']))
        if key not in self.features:
            features = self.arrays(ref, ['features'])['features']
            if features.dtype != np.float32 or features.shape != (2062,):
                raise ValueError('Require original native2062 public features')
            self.features[key] = features
        elif self.inputs[key] != ref['sha256']:
            raise ValueError('One frame path has conflicting identities')
        return self.features[key]

    def acting_weights(self, context, method):
        path = self.path(context['policy_checkpoint'])
        file_sha = context.get('policy_checkpoint_sha256')
        key = str(path)
        if key not in self.actors:
            saved = checked_load(path, file_sha)
            actual = parameter_digest(saved.get('state_dict'))
            expected_kind = 'sac_actor' if method == 'sac' else 'ppo_actor_value'
            if saved.get('kind') != expected_kind or saved.get('parameter_sha256') != actual:
                raise ValueError('Acting archive payload/tensor parameter digest differs')
            self.actors[key] = (file_sha, actual)
            self.inputs[key] = file_sha
        if self.actors[key] != (file_sha, context.get('policy_sha256')):
            raise ValueError('Acting tensor parameter digest differs from context')


def _decode(value, arrays):
    if isinstance(value, dict):
        if set(value) == {'numeric_array'}:
            return arrays[value['numeric_array']]
        return {key: _decode(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, arrays) for item in value]
    return value


def _interface(contract):
    runtime = contract['runtime']
    keys = ('norm_stats_sha256', 'rlinf_commit', 'robotwin_commit', 'features',
            'action_transform', 'task_config_name', 'appearance_randomization')
    return dict(runtime={key: runtime.get(key) for key in keys},
                model_path=contract['model_path'], task_episode_limits=contract.get('task_episode_limits'),
                episode_limit=contract['episode_limit'], observation_dim=2062, noise_shape=[32, 32], action_limit=1.5)


def read_source(corpus_root, source, *, allow_legacy_cut=False, partition='train', validation_iteration=None):
    reader = CorpusReader(corpus_root)
    collection = reader.json('contract.json')
    contract, method = source['contract'], source['method']
    if partition not in ('train', 'validation'):
        raise ValueError('Unknown source-local partition')
    phase = 'source' if partition == 'train' else 'source_validation'
    if partition == 'validation' and (type(validation_iteration) is not int or validation_iteration < 1
            or source['source_iteration'] != validation_iteration):
        raise ValueError('Validation requires the exact immutable per-method source-iteration snapshot')
    expected_trials = source['trials'] if partition == 'train' else len(TASKS)
    validation_actor = None
    if partition == 'validation':
        validation_actor = parameter_digest(source['learner']['actor' if method == 'sac' else 'policy'])
    if (collection.get('schema') != SCHEMA or collection.get('source_commit') != contract['source_commit']
            or set(collection.get('source_tasks', [])) != set(TASKS)
            or collection.get('model_path') != contract['model_path']
            # The driver excludes render PCI from checkpoint contracts. It is
            # allocation identity, not a feature/motor/scientific setting. The
            # COMPLETE raw runtime JSON remains hash-pinned by the reader.
            or _canonical({key:value for key,value in reader.json('runtime-contract.json').items()
                           if key != 'actual_render_pci'}) != _canonical({key:value for key,value in contract['runtime'].items()
                           if key != 'actual_render_pci'})):
        raise ValueError('Corpus source/runtime contract differs from source recovery')
    by_trial, excluded = defaultdict(list), defaultdict(int)
    for reset_path in sorted((reader.root / 'episodes').glob('*/reset.json')):
        reset = reader.json(reset_path)
        context = reset.get('context', {})
        if context.get('phase') != phase or context.get('method') != method:
            excluded['non_source_or_other_method_episodes'] += 1
            continue
        pattern = (rf'source:{method}:(\d+)' if partition == 'train' else
                   rf'stage:iteration:{validation_iteration}:source-validation:{method}:([^:]+)')
        match = re.fullmatch(pattern, context.get('trial_id', ''))
        if match is None:
            if partition == 'validation':
                excluded['other_validation_stages'] += 1
                continue
            raise ValueError('Unknown source trial identity')
        if partition == 'validation' and match[1] not in TASKS:
            raise ValueError('Target task cannot enter source validation')
        if partition == 'validation' and context.get('stage_source_iteration') != validation_iteration:
            raise ValueError('Validation stage clock differs from its source snapshot')
        number = int(match[1]) if partition == 'train' else TASKS.index(match[1]) + 1
        if partition == 'train' and number > source['trials']:
            excluded['after_committed_cursor_episodes'] += 1
            continue
        episode_dir = reset_path.parent
        manifest = reader.json(episode_dir / 'manifest.json')
        refs = manifest.get('chunks', [])
        if manifest.get('pending_decision') is not None or not refs:
            raise ValueError('Committed source prefix contains an unresolved/empty episode')
        rows, previous, steps, episode_index = [], reset['initial_observation'], 0, None
        task = reset.get('reset_info', {}).get('task')
        if task not in TASKS or (partition == 'validation' and task != match[1]):
            raise ValueError('Target task cannot enter a source history')
        horizon = contract.get('task_episode_limits', {}).get(task, contract['episode_limit'])
        for index, ref in enumerate(refs):
            chunk = reader.json(episode_dir / ref['path'], ref['sha256'])
            ctx = chunk.get('context', {})
            if (chunk.get('schema') != SCHEMA or chunk.get('status') != 'complete'
                    or chunk.get('episode_id') != reset['episode_id'] or chunk.get('chunk_index') != index + 1
                    or ctx.get('phase') != phase or ctx.get('method') != method
                    or ctx.get('trial_id') != context['trial_id'] or chunk['observation'] != previous):
                raise ValueError('Incomplete, non-source or disconnected native transition')
            decision = reader.arrays(chunk['decision_arrays'], list(chunk['decision_arrays']['arrays']))
            metadata = _decode(chunk['decision_metadata'], decision)
            current_episode = metadata.get('episode_index')
            if type(current_episode) is not int or not 0 <= current_episode < 4:
                raise ValueError('Missing actual K4 episode index')
            if episode_index is not None and current_episode != episode_index:
                raise ValueError('Episode clock changed within one episode')
            episode_index = current_episode
            if metadata.get('episode_limit', horizon) != horizon or metadata.get('chunk_index') != index:
                raise ValueError('Recorded horizon/chunk clock differs from native source contract')
            noise = decision['sampled_noise'].reshape(-1)
            if noise.dtype != np.float32 or noise.shape != (1024,) or (np.abs(noise) > 1.5 + 1e-6).any():
                raise ValueError('Require physical1024D noise bounded at1.5')
            native, learning = chunk['native_result'], chunk['learning_result']
            duration = native['executed_primitive_steps']
            native_arrays = reader.arrays(native['arrays'], ['valid_action_mask'])
            if (type(duration) is not int or not 1 <= duration <= 32
                    or not np.isin(native_arrays['valid_action_mask'], (0, 1)).all()
                    or int(native_arrays['valid_action_mask'].sum()) != duration
                    or learning['episode_primitive_steps'] != steps + duration
                    or steps + duration > horizon):
                raise ValueError('Native action duration/mask/horizon differs')
            done = bool(learning['terminated'] or learning['truncated'])
            if done != (index == len(refs) - 1):
                raise ValueError('Only a complete ended episode may be exported')
            version = ctx.get('policy_version', '')
            update = metadata.get('optimizer_updates') if method == 'sac' else metadata.get('policy_update')
            if type(update) is not int or update < 0:
                raise ValueError('Missing actual pre-action source optimizer clock')
            if version == 'bounded_uniform_warmup':
                if method != 'sac' or not metadata.get('source_warmup') or ctx.get('policy_checkpoint') is not None:
                    raise ValueError('Warmup cannot be attributed to a trained actor')
            else:
                parsed = re.fullmatch(rf'{method}:update:(\d+):([0-9a-f]{{64}})', version)
                if parsed is None or int(parsed[1]) != update or parsed[2] != ctx.get('policy_sha256'):
                    raise ValueError('Behavior version differs from actual source update')
                reader.acting_weights(ctx, method)
                if partition == 'validation' and ctx['policy_sha256'] != validation_actor:
                    raise ValueError('Source-validation actor differs from its frozen source snapshot')
                if partition == 'validation' and ctx.get('stage_source_iteration') != validation_iteration:
                    raise ValueError('Validation chunk stage clock differs')
            if partition == 'validation' and method == 'sac' and metadata.get('online_updates') is not False:
                raise ValueError('Validation feedback cannot update the source actor')
            if 'pre_squash_actions' in metadata and not np.allclose(
                    1.5 * np.tanh(metadata['pre_squash_actions']).reshape(-1), noise, rtol=0, atol=1e-6):
                raise ValueError('Recorded physical noise differs from PPO action sample')
            row = dict(features=reader.frame(chunk['observation']),
                next_features=reader.frame(chunk['next_observation']), noise=noise,
                reward=float(learning['reward']), native_reward=float(native['reward']),
                terminated=bool(learning['terminated']), truncated=bool(learning['truncated']),
                native_terminated=bool(native['terminated']), native_truncated=bool(native['truncated']),
                budget_truncated=bool(learning['budget_truncated']), episode_start=index == 0,
                episode_index=episode_index, primitive_step=steps, episode_limit=horizon,
                duration=duration, update=update, version=version,
                policy_hash=ctx.get('policy_sha256') or 'bounded_uniform_warmup', metadata=metadata)
            if not np.isfinite(row['reward']) or not np.isfinite(row['native_reward']):
                raise ValueError('Nonfinite actual feedback')
            rows.append(row)
            steps += duration
            previous = chunk['next_observation']
        if (manifest.get('completed_chunks') != len(rows) or manifest.get('executed_primitive_steps') != steps):
            raise ValueError('Episode manifest counters differ')
        by_trial[number].append(dict(task=task, episode_index=episode_index, rows=rows, reset=reset))
    trials, quarantined = [], defaultdict(int)
    for number in range(1, expected_trials + 1):
        episodes = sorted(by_trial[number], key=lambda episode: episode['episode_index'])
        complete = (len(episodes) == 4 and [ep['episode_index'] for ep in episodes] == list(range(4))
                    and len({ep['task'] for ep in episodes}) == 1
                    and not any(row['budget_truncated'] for ep in episodes for row in ep['rows']))
        if not complete:
            legacy_prefix = (episodes and len(episodes) <= 4
                and [ep['episode_index'] for ep in episodes] == list(range(len(episodes)))
                and len({ep['task'] for ep in episodes}) == 1)
            if partition == 'train' and allow_legacy_cut and number == source['trials'] and 'source_iterations' not in contract and legacy_prefix:
                excluded['legacy_final_incomplete_trials'] += 1
                for episode in episodes:
                    quarantined[episode['task']] += sum(row['duration'] for row in episode['rows'])
                continue
            raise ValueError('Committed source cursor lacks one unambiguous complete K4 trial')
        rows = [row for ep in episodes for row in ep['rows']]
        rows[0]['trial_start'] = True
        if method != 'sac':
            if len({row['version'] for row in rows}) != 1:
                raise ValueError('A source PPO trial must keep fixed behavior weights')
            reset_mode = source['learner']['config']['reset_each_episode']
            expected = native_tokens(rows, reset_each_episode=reset_mode)
            for index, (row, token) in enumerate(zip(rows, expected, strict=True)):
                metadata = row['metadata']
                if (not np.allclose(metadata['policy_input'], token, rtol=0, atol=1e-6)
                        or metadata['history_index'] != index
                        or bool(metadata['recurrent_reset_mask']) != (reset_mode and row['episode_start'])
                        or bool(metadata['reset_episode_feedback']) != reset_mode):
                    raise ValueError('Recorded PPO inputs/reset mask differ from actual causal history')
        cases = [dict(episode_index=ep['episode_index'],
            requested_seed=ep['reset']['reset_info'].get('requested_seed'),
            actual_seed=ep['reset']['reset_info'].get('actual_seed'),
            initial_observation_sha256=ep['reset']['initial_observation']['sha256']) for ep in episodes]
        if any(type(case['actual_seed']) is not int or type(case['requested_seed']) is not int for case in cases):
            raise ValueError('Missing actual source/reset case identity')
        trials.append(dict(number=number, task=episodes[0]['task'], cases=cases, rows=rows))
    if not trials:
        raise ValueError('No eligible complete source histories')
    selected = {task:sum(row['duration'] for trial in trials if trial['task'] == task for row in trial['rows']) for task in TASKS}
    expected = source['state'].get('task_steps', {})
    actual = {task:selected[task] + quarantined.get(task, 0) for task in TASKS}
    if partition == 'train' and (set(expected) != set(TASKS) or actual != expected
            or sum(actual.values()) != source['state'].get('source_steps')):
        raise ValueError('Selected/quarantined durations differ from frozen source_steps/task_steps')
    if quarantined:
        excluded['legacy_quarantined_actions'] = sum(quarantined.values())
    return reader, trials, dict(excluded)


def native_tokens(rows, *, reset_each_episode=False, carry_across_trials=False):
    """Use real physical noise/reward; targets and update IDs never enter inputs."""
    previous_noise, previous_reward, tokens = np.zeros(1024, np.float32), 0., []
    for index, row in enumerate(rows):
        if (row.get('trial_start', False) and not carry_across_trials) or (reset_each_episode and row['episode_start']):
            previous_noise, previous_reward = np.zeros(1024, np.float32), 0.
        boundary = not reset_each_episode and row['episode_start'] and index > 0
        if row.get('trial_start', False) and not carry_across_trials:
            boundary = False
        tokens.append(policy_inputs(row['features'], previous_noise, previous_reward, boundary,
            row['episode_index'], row['primitive_step'], train_episodes=4, episode_limit=row['episode_limit']))
        previous_noise, previous_reward = row['noise'], row['reward']
    return np.stack(tokens).astype(np.float32)


def load_teacher(recovery, *, device):
    learner, method = recovery['learner'], recovery['method']
    if type(learner.get('updates')) is not int or learner['updates'] < 1:
        raise ValueError('DPT requires an explicitly trained source teacher candidate')
    if method == 'sac':
        contract = learner['contract']
        if (contract['observation_dim'] != 2062 or contract['action_dim'] != 1024 or contract['action_limit'] != 1.5):
            raise ValueError('SAC teacher interface differs')
        config = dict(contract['config'])
        config['hidden_dims'] = tuple(config['hidden_dims'])
        with torch.random.fork_rng(devices=[]):
            policy = SquashedGaussianActor(2062, 1024, SACConfig(**config), 1.5)
        weights, reset = learner['actor'], False
    else:
        config = PPOConfig(**learner['config'])
        if learner['observation_dim'] != 2062 or tuple(learner['noise_shape']) != (32, 32) or config.action_limit != 1.5:
            raise ValueError('PPO teacher interface differs')
        import gymnasium as gym
        with torch.random.fork_rng(devices=[]):
            core = GatedTransformerXLCore(3090, width=config.width, layers=config.layers,
                heads=config.heads, memory_length=config.memory_length, gate_bias=config.gate_bias)
            policy = RecurrentActorCritic(core, gym.spaces.Box(-1., 1., (1024,), np.float32))
        weights, reset = learner['policy'], config.reset_each_episode
    validate_weights(policy, weights)
    policy.load_state_dict(weights, strict=True)
    return policy.to(device).eval().requires_grad_(False), reset


@torch.no_grad()
def teacher_labels(policy, method, rows, *, reset_each_episode):
    reference = next(policy.parameters())
    tokens = native_tokens(rows, reset_each_episode=reset_each_episode)
    state, labels = None, []
    for row, token in zip(rows, tokens, strict=True):
        if method == 'sac':
            inputs = torch.as_tensor(row['features'][None], device=reference.device)
            proposal, _ = policy(inputs, generator=None, deterministic=True)
        else:
            inputs = torch.as_tensor(token[None], device=reference.device)
            reset = torch.tensor([reset_each_episode and row['episode_start']], device=reference.device)
            proposal, _, _, _, _, state = policy.step(inputs, state, reset=reset, deterministic=True)
            proposal = proposal * 1.5
        labels.append(proposal[0].cpu().numpy().copy())
    return np.asarray(labels, np.float32)


def export_corpus(corpus_root, output, *, source_path, source_sha256, source_method,
                  teacher_path, teacher_sha256, teacher_method, objectives=('ad', 'dpt'),
                  device='cpu', allow_legacy_cut=False, partition='train', validation_iteration=None):
    if not objectives or len(set(objectives)) != len(objectives) or not set(objectives) <= {'ad', 'dpt', 'behavior_cloning'}:
        raise ValueError('Choose distinct AD/DPT/BC objectives')
    if partition == 'validation' and 'ad' in objectives:
        raise ValueError('Fixed-policy source validation is BC/DPT, not an AD learning history')
    output, corpus_root = Path(output).resolve(), Path(corpus_root).resolve()
    if output.exists() or output.is_relative_to(corpus_root):
        raise ValueError('Use a fresh output outside the source corpus')
    source = source_recovery(source_path, source_sha256, source_method)
    teacher = source_recovery(teacher_path, teacher_sha256, teacher_method)
    if _canonical(_interface(source['contract'])) != _canonical(_interface(teacher['contract'])):
        raise ValueError('Source and teacher motor/features/horizon interfaces differ')
    reader, trials, excluded = read_source(corpus_root, source, allow_legacy_cut=allow_legacy_cut,
        partition=partition, validation_iteration=validation_iteration)
    policy, teacher_reset = load_teacher(teacher, device=device)
    grouped = defaultdict(list)
    for trial in trials:
        rows = trial['rows']
        rows[0]['trial_start'] = True
        # Teacher state is rebuilt independently for each actual source K4.
        labels = teacher_labels(policy, teacher_method, rows, reset_each_episode=teacher_reset)
        for row, label in zip(rows, labels, strict=True):
            row['target'] = label
        grouped[trial['task']].extend(rows)
    arrays, task_keys = {}, sorted(grouped)
    fields = dict(observations='features', next_observations='next_features', executed_noise='noise', rewards='reward', native_rewards='native_reward',
        terminated='terminated', truncated='truncated', native_terminated='native_terminated',
        native_truncated='native_truncated', budget_truncated='budget_truncated', episode_starts='episode_start',
        episode_indices='episode_index', primitive_steps='primitive_step', episode_limits='episode_limit',
        durations='duration', behavior_updates='update', behavior_versions='version')
    joined = [row for task in task_keys for row in grouped[task]]
    if 'ad' in objectives:
        for task, rows in grouped.items():
            if (len({row['update'] for row in rows}) < 2 or len({row['policy_hash'] for row in rows}) < 2
                    or any(b['update'] < a['update'] for a, b in zip(rows, rows[1:]))):
                raise ValueError(f'AD requires actual chronological behavior policy changes for {task}')
    for name, key in fields.items():
        arrays[name] = np.asarray([row[key] for row in joined])
    arrays['observations'] = arrays['observations'].astype(np.float32)
    arrays['next_observations'] = arrays['next_observations'].astype(np.float32)
    arrays['executed_noise'] = arrays['executed_noise'].astype(np.float32)
    arrays['rewards'] = arrays['rewards'].astype(np.float32)
    arrays['native_rewards'] = arrays['native_rewards'].astype(np.float32)
    arrays['source_trial_starts'] = np.asarray([row.get('trial_start', False) for row in joined], bool)
    arrays['history_inputs'] = np.concatenate([native_tokens(grouped[task], carry_across_trials=True) for task in task_keys])
    arrays['lengths'] = np.asarray([len(grouped[task]) for task in task_keys], np.int64)
    inputs = dict(reader.inputs)
    inputs[str(Path(source_path).resolve())], inputs[str(Path(teacher_path).resolve())] = source_sha256, teacher_sha256
    if any(file_sha256(path) != digest for path, digest in inputs.items()):
        raise ValueError('Source/teacher changed during export; use immutable snapshots')
    output.mkdir(parents=True, exist_ok=False)
    datasets = {}
    for objective in objectives:
        target = np.asarray([row['target'] for row in joined], np.float32) if objective == 'dpt' else arrays['executed_noise'].copy()
        if not np.isfinite(target).all() or (np.abs(target) > 1.5 + 1e-6).any():
            raise ValueError('Teacher labels must be finite physical1024D/bound1.5 actions')
        path = output / f'{objective}.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays, target_noise=target)
        datasets[objective] = dict(path=str(path), sha256=file_sha256(path), objective=objective)
    case_ids = [hashlib.sha256(_canonical(dict(task=trial['task'],
        actual_seeds=[case['actual_seed'] for case in trial['cases']])).encode()).hexdigest() for trial in trials]
    metadata = dict(format=FORMAT, completed=True, datasets=datasets, task_keys=task_keys,
        partition=partition, validation_iteration=validation_iteration, case_ids=case_ids,
        source_method=source_method, teacher_method=teacher_method, input_sha256=inputs,
        source_contract=source['contract'], source_recovery_sha256=source_sha256, teacher_recovery_sha256=teacher_sha256,
        teacher_status='source-trained candidate; deterministic mean competence not assumed',
        interface=_interface(source['contract']), policy_input_dim=3090,
        label_rule='deterministic physical1.5*tanh(mean); recurrent teacher rebuilt per actual source trial',
        student_history='same-task source trials across updates; actual feedback crosses source-trial boundaries; original clocks retained',
        source_trials=[dict(number=trial['number'], task=trial['task'], cases=trial['cases'], decisions=len(trial['rows'])) for trial in trials],
        excluded=excluded, counts=dict(decisions=len(joined), episodes=sum(row['episode_start'] for row in joined),
            complete_K4_trials=len(trials), committed_source_actions=source['state']['source_steps'],
            selected_actions=sum(row['duration'] for row in joined)),
        additional_simulator_actions=0, additional_vla_queries=0,
        consumer='Native-token exporter only; LIBERO vla_history_train/deployment is incompatible')
    (output / 'index.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus-root', 'source-recovery', 'teacher-recovery', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('source-sha256', 'teacher-sha256', 'source-commit', 'expected-hostname'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--source-method', choices=METHODS, required=True)
    parser.add_argument('--teacher-method', choices=METHODS, required=True)
    parser.add_argument('--objectives', choices=('ad', 'dpt', 'behavior_cloning'), nargs='+', default=['ad', 'dpt'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--allow-legacy-cut', action='store_true')
    parser.add_argument('--partition', choices=('train', 'validation'), default='train')
    parser.add_argument('--validation-iteration', type=int)
    args = parser.parse_args(argv)
    if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname or args.threads < 1:
        raise ValueError('Export only on the explicitly identified allocated Linux host')
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed exporter commit')
    torch.set_num_threads(args.threads)
    result = export_corpus(args.corpus_root, args.output, source_path=args.source_recovery,
        source_sha256=args.source_sha256, source_method=args.source_method,
        teacher_path=args.teacher_recovery, teacher_sha256=args.teacher_sha256, teacher_method=args.teacher_method,
        objectives=tuple(args.objectives), device=args.device, allow_legacy_cut=args.allow_legacy_cut,
        partition=args.partition, validation_iteration=args.validation_iteration)
    print(json.dumps(dict(datasets=result['datasets'], counts=result['counts'])))


if __name__ == '__main__':
    main()
