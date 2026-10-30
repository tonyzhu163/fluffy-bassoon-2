"""Export committed source40 PPO histories with paired AD/DPT-style targets.

Labeling replays real feedback through a frozen teacher; it neither executes
teacher proposals nor reuses the behavior policy's recurrent caches.
"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import pickle
import socket
import subprocess
import sys

import numpy as np
import torch

from .vla_history import file_sha256, save_dataset, trial_tokens, validate_trial
from .vla_rl2 import load_rl2_policy
from .vla_source40_ppo import FORMAT as RAW_FORMAT, PI05_RECEIPT, write_json
from .vla_source40_schedule import SOURCE_SUITES
from .warmstart import checked_load


FORMAT = 'openpi_source40_ppo_corpus_v1'
OBJECTIVES = ('ad', 'dpt', 'behavior_cloning')


def source_keys(source):
    """Accept only the original source40 contracts, never a target rollout."""
    if (source.get('source_only') is not True or source.get('source_data_from_targets') is not False
            or source.get('motor', {}).get('receipt_sha256') != PI05_RECEIPT
            or not source.get('feature_contract') or not source.get('noise_contract')):
        raise ValueError('Require explicit source-only OpenPI40 motor/features/noise provenance')
    allowed = {f'{suite}:{task:02d}' for suite in SOURCE_SUITES for task in range(10)}
    if source.get('format') == 'openpi_pi05_source40_shared_ppo_v1':
        keys = source.get('source_task_keys', [])
        if len(keys) != 40 or set(keys) != allowed:
            raise ValueError('Shared PPO contract must declare all40 original source tasks')
    elif source.get('format') == 'openpi_pi05_source40_ppo_stream_v1':
        task = source.get('task_id')
        if type(task) is not int:
            raise ValueError('Require an original source task')
        keys = [f"{source.get('suite')}:{task:02d}"]
        if not set(keys) <= allowed:
            raise ValueError('Require an original source task')
    else:
        raise ValueError('Unknown source40 PPO collection contract')
    return set(keys)


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def load_committed_trials(run, recovery_sha256):
    """Read only complete trials committed by the explicitly pinned recovery."""
    run = Path(run).resolve()
    declared_source = json.loads((run / 'source-contract.json').read_text())
    validation = declared_source.get('partition') == 'validation'
    recovery = run / ('validation-recovery.pt' if validation else 'learner/latest.pt')
    saved = checked_load(recovery, recovery_sha256, weights_only=False)
    source = saved.get('contract', {}) if validation else saved.get('contract', {}).get('source', {})
    keys = source_keys(source)
    if validation:
        if type(saved.get('trial')) is not int or saved['trial'] < 1 or not source.get('prior_policy_sha256'):
            raise ValueError('Require a committed frozen source-validation recovery')
        iterations = saved['trial']
        contract = dict(source=source, mode=source.get('memory_mode'))
        if contract['mode'] not in ('full', 'episode'):
            raise ValueError('Unknown frozen validation reset mode')
    else:
        contract = saved.get('contract', {})
        if (saved.get('version') != 1 or saved.get('boundary') != 'between_trials'
            or type(saved.get('iteration')) is not int or saved['iteration'] < 1
            or contract.get('format') != 'frozen_vla_rl2_training_v1'
            or contract.get('mode') not in ('full', 'episode')
            or contract.get('config', {}).get('observation_dim') != 4104
            or contract.get('config', {}).get('noise_dim') != 32
            or type(contract.get('attempts')) is not int or contract['attempts'] < 1):
            raise ValueError('Require a committed source40 full-trial PPO recovery')
        iterations = saved['iteration']
    if declared_source != source:
        raise ValueError('Source contract differs from the committed recovery')
    partition = source.get('partition', 'train')
    if partition not in ('train', 'validation'):
        raise ValueError('Unknown source case partition')
    states = range(40) if partition == 'train' else range(40, 50)
    inputs = {str(recovery): recovery_sha256}
    trials = []
    validation_counters = dict(environment_steps=0, settling_steps=0, optimizer_steps=0)
    attempts = contract.get('attempts')
    for number in range(1, iterations + 1):
        path = run / (f'validation-trial-{number:06d}.pt' if validation else f'learner/traces/trial-{number:06d}.pt')
        digest = file_sha256(path)
        trace = checked_load(path, digest, weights_only=False)
        inputs[str(path)] = digest
        update = 0 if validation else number - 1
        if validation:
            behavior_sha = source['prior_policy_sha256']
            if trace.get('frozen_policy_sha256') != behavior_sha or trace.get('optimizer_updates') != 0:
                raise ValueError('Frozen source validation changed behavior policy or optimizer state')
        else:
            behavior_path = run / f'learner/traces/behavior-{update:06d}.pt'
            behavior_sha = trace.get('behavior_policy_sha256')
            behavior = checked_load(behavior_path, behavior_sha)
            inputs[str(behavior_path)] = behavior_sha
            if (trace.get('behavior_update') != update
                or trace.get('source_commit') != source.get('source_commit')
                or trace.get('initial_recurrent_state') != 'zero'
                or behavior.get('manifest', {}).get('training_contract') != contract
                or behavior['manifest'].get('behavior_update') != update):
                raise ValueError('Committed trial/behavior provenance differs')
        rollout, receipt = trace['rollout'], trace['receipt']
        validation_counters['environment_steps'] += int(receipt['primitive_steps'])
        validation_counters['settling_steps'] += int(receipt['settling_steps'])
        if attempts is None:
            attempts = len(receipt.get('attempts', []))
        if attempts < 1 or len(receipt.get('attempts', [])) != attempts:
            raise ValueError('Committed trial is missing a complete episode')
        rows, offset, task_key = [], 0, None
        for attempt in range(1, attempts + 1):
            raw_path = run / f'raw/trial-{number:06d}-episode-{attempt:03d}.pkl.gz'
            sidecar = raw_path.with_suffix('.json')
            sidecar_sha = file_sha256(sidecar)
            metadata = json.loads(sidecar.read_text())
            if file_sha256(raw_path) != metadata.get('sha256'):
                raise ValueError('Raw PPO episode checksum differs')
            inputs[str(sidecar)], inputs[str(raw_path)] = sidecar_sha, metadata['sha256']
            # These are own, hash-checked collector artifacts, not downloaded pickles.
            with gzip.open(raw_path, 'rb') as stream:
                episode = pickle.load(stream)
            meta, summary, actual = episode['metadata'], episode['summary'], episode['rows']
            key = f"{summary.get('suite')}:{summary.get('task_id', -1):02d}"
            if (meta != metadata.get('metadata') or meta.get('format') != RAW_FORMAT
                    or meta.get('source_contract') != source or meta.get('trial') != number
                    or meta.get('attempt') != attempt or meta.get('behavior_update') != update
                    or meta.get('behavior_policy_sha256') != behavior_sha
                    or dict(summary, attempt=attempt) != receipt['attempts'][attempt - 1]
                    or summary.get('initial_state') not in states
                    or key not in keys or (task_key is not None and task_key != key)
                    or not actual or any(row.get('source_task_key') != key for row in actual)
                    or not actual[-1]['done'] or any(row['done'] for row in actual[:-1])):
                raise ValueError('Raw PPO episode changed task, case, boundary or provenance')
            task_key = key
            # Receipt adds the attempt index to the original raw summary.
            for field in ('observations', 'raw_actions', 'rewards', 'attempt_ends', 'trial_ends', 'resets'):
                if not np.array_equal(_array(episode['ppo'][field]),
                                      _array(rollout[field][offset:offset + len(actual)])):
                    raise ValueError('Raw PPO episode differs from its committed trial')
            offset += len(actual)
            rows.extend(actual)
        if offset != len(rollout['observations']):
            raise ValueError('Committed PPO trial has missing or extra decisions')
        base = dict(observations=np.stack([row['public_features'] for row in rows]).astype(np.float32),
            executed_noise=np.stack([row['executed_noise'] for row in rows]).astype(np.float32),
            rewards=np.asarray([row['reward'] for row in rows], np.float32),
            dones=np.asarray([row['done'] for row in rows], bool),
            episode_starts=np.asarray([True] + [row['done'] for row in rows[:-1]], bool),
            teacher_steps=np.full(len(rows), update, np.int64))
        base['target_noise'] = base['executed_noise'].copy()
        validate_trial(base, objective='behavior_cloning')
        if not np.allclose(np.tanh(_array(rollout['raw_actions'])[:, 0]),
                           base['executed_noise'], atol=1e-6, rtol=0):
            raise ValueError('Executed PPO noise differs from its recorded pre-tanh action')
        if not np.allclose(_array(rollout['observations'])[:, 0],
                           trial_tokens(base, mode=contract['mode']), atol=1e-6, rtol=0):
            raise ValueError('PPO tokens differ from causal actual feedback')
        trials.append(dict(number=number, task_key=task_key, arrays=base,
            initial_state=receipt['initial_state'], scene_seed=receipt['scene_seed']))
    return dict(run=run, recovery=saved, counters=validation_counters if validation else saved['counters'],
                contract=contract, source=source,
                partition=partition, trials=trials, inputs=inputs)


@torch.no_grad()
def label_trial(teacher, trial, *, mode):
    """Rebuild teacher memory from real history, starting empty per source trial."""
    if mode not in ('full', 'episode'):
        raise ValueError('Unknown teacher reset convention')
    reference = next(teacher.parameters())
    state, labels = None, []
    tokens = trial_tokens(trial, mode=mode)
    for token, start in zip(tokens, trial['episode_starts'], strict=True):
        inputs = torch.as_tensor(token[None], device=reference.device, dtype=reference.dtype)
        reset = torch.tensor([bool(start) and mode == 'episode'], device=reference.device)
        proposal, _, _, _, _, state = teacher.step(inputs, state, reset=reset, deterministic=True)
        labels.append(proposal[0].cpu().numpy().copy())
    result = np.asarray(labels, np.float32)
    if not np.isfinite(result).all() or (np.abs(result) > 1).any():
        raise ValueError('Teacher emitted invalid bounded noise labels')
    return result


def export_corpus(run, output, *, recovery_sha256, teacher_path, teacher_sha256,
                  objectives=('ad', 'dpt'), device='cpu'):
    if not objectives or len(set(objectives)) != len(objectives) or not set(objectives) <= set(OBJECTIVES):
        raise ValueError('Choose distinct supported export objectives')
    output, run = Path(output).resolve(), Path(run).resolve()
    if output.exists() or output.is_relative_to(run):
        raise ValueError('Require a fresh output outside the immutable source run')
    snapshot = load_committed_trials(run, recovery_sha256)
    # Loading a model initializes temporary parameters; preserve caller RNG even
    # though deterministic labeling itself does not sample actions.
    with torch.random.fork_rng(devices=[]):
        teacher, manifest = load_rl2_policy(teacher_path, expected_sha256=teacher_sha256, device=device)
    teacher_contract = manifest.get('contract', {})
    teacher_source = teacher_contract.get('source', {})
    teacher_keys = source_keys(teacher_source)
    source = snapshot['source']
    if (teacher_source.get('partition', 'train') != 'train'
            or teacher_contract.get('mode') not in ('full', 'episode')
            or type(manifest.get('updates')) is not int or manifest['updates'] < 1
            or any(source[key] != teacher_source[key] for key in ('feature_contract', 'noise_contract'))
            or teacher.config.observation_dim != 4104 or teacher.config.noise_dim != 32
            or not {trial['task_key'] for trial in snapshot['trials']} <= teacher_keys):
        raise ValueError('Teacher must be source-trained with matching tasks, motor, features and noise')
    grouped, source_trials = defaultdict(list), defaultdict(list)
    for trial in snapshot['trials']:
        values = dict(trial['arrays'])
        values['target_noise'] = label_trial(teacher, values, mode=teacher_contract['mode'])
        grouped[trial['task_key']].append(values)
        source_trials[trial['task_key']].append({key: trial[key] for key in ('number', 'initial_state', 'scene_seed')})
    paired = {objective: [] for objective in objectives}
    task_keys = sorted(grouped)
    for key in task_keys:
        history = {field: np.concatenate([trial[field] for trial in grouped[key]]) for field in grouped[key][0]}
        for objective in objectives:
            values = dict(history)
            if objective != 'dpt':
                values['target_noise'] = values['executed_noise'].copy()
            # A PPO trial has fixed weights. AD needs multiple chronological
            # source trials of the SAME task with actual intervening updates.
            if objective == 'ad' and values['teacher_steps'][0] == values['teacher_steps'][-1]:
                raise ValueError(f'AD requires a learning trajectory for {key}; use dpt/behavior_cloning for fixed-policy data')
            validate_trial(values, objective=objective)
            paired[objective].append(values)
    pinned_inputs = dict(snapshot['inputs'], **{str(Path(teacher_path).resolve()): teacher_sha256})
    for path, digest in pinned_inputs.items():
        if file_sha256(path) != digest:
            raise ValueError('Source/teacher artifacts changed during export; use an immutable snapshot')
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(source_contract=source, local_scene_split=snapshot['partition'],
        motor_sha256=PI05_RECEIPT, feature_contract=source['feature_contract'], noise_contract=source['noise_contract'],
        trial_ids=[f'ppo:{recovery_sha256}:{key}' for key in task_keys], task_keys=task_keys,
        source_trials=dict(source_trials), input_sha256=pinned_inputs, teacher_manifest=manifest,
        teacher_policy_sha256=teacher_sha256, teacher_status='candidate; independent competence not assumed',
        label_rule='deterministic tanh(mean); teacher memory replayed from actual feedback, reset per source trial',
        history_rule='Chronological source trials grouped by task; original episode boundaries retained',
        collection_costs_count_once=dict(committed_recovery_counters=snapshot['counters'],
            scope='Committed source trials only; full collector qualifications/retries remain separately charged'),
        additional_simulator_actions=0, additional_vla_queries=0,
        data_construction='Paired chronological-query DPT-style variant, not independent-query original DPT')
    datasets = {}
    for objective, trials in paired.items():
        path = output / f'{objective}.npz'
        result = save_dataset(path, trials, objective=objective, metadata=metadata)
        datasets[objective] = dict(path=str(path), sha256=result['sha256'], transitions=result['transitions'])
    index = dict(format=FORMAT, completed=True, metadata=metadata, datasets=datasets)
    write_json(output / 'index.json', index)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'teacher', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('recovery-sha256', 'teacher-sha256', 'source-commit', 'expected-hostname'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--objectives', nargs='+', choices=OBJECTIVES, default=['ad', 'dpt'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname:
        raise ValueError('Run labeling on the explicitly identified allocated Linux host')
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed exporter commit')
    if args.threads < 1:
        raise ValueError('Thread count must be positive')
    torch.set_num_threads(args.threads)
    result = export_corpus(args.source_run, args.output, recovery_sha256=args.recovery_sha256,
        teacher_path=args.teacher, teacher_sha256=args.teacher_sha256, objectives=tuple(args.objectives), device=args.device)
    print(json.dumps(result['datasets'], allow_nan=False))


if __name__ == '__main__':
    main()
