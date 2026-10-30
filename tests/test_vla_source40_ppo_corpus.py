"""PPO relabeling must preserve executed history and rebuild teacher memory."""
import gzip
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.vla_history import file_sha256, load_dataset, trial_tokens
from fluffy_bassoon.meta_rl.vla_rl2 import (RL2Config, collect_rl2_trial,
    load_rl2_policy, make_rl2_policy, train_rl2)
from fluffy_bassoon.meta_rl.vla_source40_ppo import PI05_RECEIPT, episode_sink
from fluffy_bassoon.meta_rl.vla_source40_ppo_corpus import (export_corpus,
    label_trial, load_committed_trials)
from fluffy_bassoon.meta_rl.vla_source40_schedule import SOURCE_SUITES
from fluffy_bassoon.meta_rl.warmstart import atomic_save


def make_source(tmp_path, *, updates=2, mode='full', shared=False):
    run = tmp_path / 'source'
    run.mkdir()
    source = dict(format='openpi_pi05_source40_ppo_stream_v1', source_only=True,
        source_data_from_targets=False, source_commit='b' * 40,
        suite='libero_spatial', task_id=0, partition='train', memory_mode=mode,
        motor=dict(receipt_sha256=PI05_RECEIPT), feature_contract=dict(kind='test-prefix-v2'),
        noise_contract='32D squashed coordinate repeated H10')
    if shared:
        source['format'] = 'openpi_pi05_source40_shared_ppo_v1'
        source['source_task_keys'] = [f'{suite}:{task:02d}' for suite in SOURCE_SUITES for task in range(10)]
    (run / 'source-contract.json').write_text(json.dumps(source))
    calls = 0
    def episode(sim, motor, choose, *, initial_state, scene_seed, chunk, bounded, feedback):
        nonlocal calls
        task = (calls // 2) % 2 if shared else 0
        calls += 1
        return recorded_episode(choose, feedback, initial_state, scene_seed, task)
    policy = make_rl2_policy(RL2Config(observation_dim=4104, width=8, layers=1, heads=2, context_length=8))
    _, _ = train_rl2(policy, SimpleNamespace(metadata=dict(settle_steps=10)), None,
        lambda obs: obs['features'], train_states=[0, 1], heldout_states=[40], attempts=2,
        chunk=10, episode_horizon=2, max_primitive_steps=updates * 4, updates=updates,
        output_dir=run / 'learner', source_contract=source, epochs=1, mode=mode,
        episode_runner=episode, episode_sink=episode_sink(run / 'raw', contract=source))
    return run


def recorded_episode(choose, feedback, initial_state, scene_seed, task=0):
    rows = []
    for index in range(2):
        features = np.zeros(4104, np.float32)
        features[:3] = [initial_state / 10, task / 10, index / 2]
        obs = dict(features=features)
        action = choose(obs)
        done, reward = index == 1, -1.
        feedback(action, reward, done)
        rows.append(dict(motor_observation=obs, motor_next_observation=obs,
            public_features=features, source_task_key=f'libero_spatial:{task:02d}',
            executed_noise=action, native_initial_noise=np.repeat(action[None], 10, axis=0),
            executed_actions=np.zeros((1, 7), np.float32), reward=reward,
            raw_rewards=[0.], done=done, success=False, primitive_steps=1))
    return rows, dict(initial_state=initial_state, scene_seed=scene_seed, success=False,
        suite='libero_spatial', task_id=task, chunks=2, primitive_steps=2,
        initial_observation_sha256=f'{initial_state}:{scene_seed}')


def export_args(run):
    return dict(recovery_sha256=file_sha256(run / 'learner/latest.pt'),
        teacher_path=run / 'learner/candidate.pt', teacher_sha256=file_sha256(run / 'learner/candidate.pt'))


def test_paired_export_uses_frozen_teacher_and_identical_actual_context(tmp_path):
    run = make_source(tmp_path)
    before = {str(path): file_sha256(path) for path in run.rglob('*') if path.is_file()}
    rng = torch.get_rng_state().clone()
    result = export_corpus(run, tmp_path / 'paired', **export_args(run))
    assert torch.equal(rng, torch.get_rng_state())
    ad, _ = load_dataset(result['datasets']['ad']['path'])
    dpt, _ = load_dataset(result['datasets']['dpt']['path'])
    assert len(ad) == len(dpt) == 1
    for field in ad[0]:
        if field != 'target_noise':
            np.testing.assert_array_equal(ad[0][field], dpt[0][field])
    np.testing.assert_array_equal(ad[0]['target_noise'], ad[0]['executed_noise'])
    np.testing.assert_array_equal(trial_tokens(ad[0]), trial_tokens(dpt[0]))
    assert ad[0]['teacher_steps'].tolist() == [0] * 4 + [1] * 4
    args = export_args(run)
    teacher, _ = load_rl2_policy(args['teacher_path'], expected_sha256=args['teacher_sha256'])
    snapshot = load_committed_trials(run, args['recovery_sha256'])
    expected = []
    for trial in snapshot['trials']:
        state = None
        # Independent replay uses saved behavior TOKENS, never its saved cache.
        saved = torch.load(run / f"learner/traces/trial-{trial['number']:06d}.pt", weights_only=False)
        for token in saved['rollout']['observations']:
            proposal, _, _, _, _, state = teacher.step(token, state, deterministic=True)
            expected.append(proposal[0].detach().numpy())
    np.testing.assert_array_equal(dpt[0]['target_noise'], np.stack(expected))
    assert not np.array_equal(dpt[0]['target_noise'], ad[0]['target_noise'])
    assert all(file_sha256(path) == digest for path, digest in before.items())
    assert result['metadata']['additional_simulator_actions'] == 0
    assert result['metadata']['additional_vla_queries'] == 0


def test_single_fixed_policy_trial_is_dpt_or_bc_not_ad(tmp_path):
    run = make_source(tmp_path, updates=1)
    with pytest.raises(ValueError, match='learning trajectory'):
        export_corpus(run, tmp_path / 'invalid', **export_args(run))
    assert not (tmp_path / 'invalid').exists()
    result = export_corpus(run, tmp_path / 'valid', objectives=('dpt', 'behavior_cloning'), **export_args(run))
    assert set(result['datasets']) == {'dpt', 'behavior_cloning'}


def test_shared_stream_groups_same_task_across_updates_without_mixing_goals(tmp_path):
    run = make_source(tmp_path, updates=4, shared=True)
    result = export_corpus(run, tmp_path / 'paired', **export_args(run))
    ad, manifest = load_dataset(result['datasets']['ad']['path'])
    assert manifest['metadata']['task_keys'] == ['libero_spatial:00', 'libero_spatial:01']
    assert ad[0]['teacher_steps'].tolist() == [0] * 4 + [2] * 4
    assert ad[1]['teacher_steps'].tolist() == [1] * 4 + [3] * 4
    assert [row['number'] for row in result['metadata']['source_trials']['libero_spatial:00']] == [1, 3]


def test_teacher_rebuilds_its_own_reset_mode_and_uses_real_past_feedback(tmp_path):
    run = make_source(tmp_path, mode='episode')
    args = export_args(run)
    snapshot = load_committed_trials(run, args['recovery_sha256'])
    teacher, _ = load_rl2_policy(args['teacher_path'], expected_sha256=args['teacher_sha256'])
    trial = snapshot['trials'][0]['arrays']
    episode_labels = label_trial(teacher, trial, mode='episode')
    full_labels = label_trial(teacher, trial, mode='full')
    # Episode starts have identical public input. Later tokens legitimately
    # differ because the recorded behavior sampled different actual actions.
    np.testing.assert_array_equal(episode_labels[0], episode_labels[2])
    assert not np.array_equal(full_labels[2:], episode_labels[2:])
    changed = dict(trial, rewards=trial['rewards'].copy(), executed_noise=trial['executed_noise'].copy())
    changed['rewards'][1] = 20.
    changed['executed_noise'][1] = .9
    changed_labels = label_trial(teacher, changed, mode='full')
    np.testing.assert_array_equal(full_labels[:2], changed_labels[:2])
    assert not np.array_equal(full_labels[2:], changed_labels[2:])
    # Targets cannot enter teacher history, including the next query's input.
    changed['target_noise'] = np.full_like(trial['target_noise'], 999.)
    np.testing.assert_array_equal(changed_labels, label_trial(teacher, changed, mode='full'))


@pytest.mark.parametrize('update_checksum', [False, True])
def test_corrupt_raw_history_is_rejected_even_with_rewritten_sidecar(tmp_path, update_checksum):
    run = make_source(tmp_path)
    path = run / 'raw/trial-000001-episode-001.pkl.gz'
    with gzip.open(path, 'rb') as stream:
        saved = pickle.load(stream)
    saved['rows'][0]['reward'] = 7.
    with gzip.open(path, 'wb') as stream:
        pickle.dump(saved, stream)
    if update_checksum:
        sidecar = path.with_suffix('.json')
        metadata = json.loads(sidecar.read_text())
        metadata['sha256'] = file_sha256(path)
        sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='checksum|causal actual feedback'):
        export_corpus(run, tmp_path / 'invalid', **export_args(run))


def test_export_ignores_uncommitted_tail_and_requires_pinned_recovery(tmp_path):
    run = make_source(tmp_path)
    (run / 'raw/trial-000003-episode-001.pkl.gz').write_bytes(b'incomplete uncommitted trial')
    args = export_args(run)
    with pytest.raises(ValueError, match='checksum'):
        export_corpus(run, tmp_path / 'invalid', **dict(args, recovery_sha256='0' * 64))
    result = export_corpus(run, tmp_path / 'valid', **args)
    assert result['datasets']['ad']['transitions'] == 8
    assert not any('000003' in path for path in result['metadata']['input_sha256'])


def test_validation_exports_dpt_without_optimizer_updates_or_false_ad(tmp_path):
    run = make_source(tmp_path)
    args = export_args(run)
    teacher, _ = load_rl2_policy(args['teacher_path'], expected_sha256=args['teacher_sha256'])
    source = json.loads((run / 'source-contract.json').read_text())
    source.update(partition='validation', prior_policy_sha256=args['teacher_sha256'])
    validation = tmp_path / 'validation'
    validation.mkdir()
    (validation / 'source-contract.json').write_text(json.dumps(source))
    sink = episode_sink(validation / 'raw', contract=source)
    rollout, receipt = collect_rl2_trial(teacher, SimpleNamespace(metadata=dict(settle_steps=10)), None,
        lambda obs: obs['features'], initial_state=40, scene_seed=99, attempts=2, chunk=10,
        episode_runner=lambda sim, motor, choose, *, feedback, initial_state, scene_seed, **kw:
            recorded_episode(choose, feedback, initial_state, scene_seed),
        episode_sink=lambda **ep: sink(trial=1, behavior_update=0,
            behavior_policy_sha256=args['teacher_sha256'], **ep))
    atomic_save(validation / 'validation-trial-000001.pt', dict(rollout=rollout, receipt=receipt,
        frozen_policy_sha256=args['teacher_sha256'], optimizer_updates=0))
    atomic_save(validation / 'validation-recovery.pt', dict(contract=source, trial=1))
    args['recovery_sha256'] = file_sha256(validation / 'validation-recovery.pt')
    result = export_corpus(validation, tmp_path / 'val-data', objectives=('dpt',), **args)
    assert result['metadata']['local_scene_split'] == 'validation'
    assert result['metadata']['collection_costs_count_once']['committed_recovery_counters'] == dict(
        environment_steps=4, settling_steps=20, optimizer_steps=0)
    with pytest.raises(ValueError, match='learning trajectory'):
        export_corpus(validation, tmp_path / 'false-ad', **args)


@pytest.mark.parametrize('field,value', [('source_data_from_targets', True), ('feature_contract', {'kind': 'other'})])
def test_target_or_incompatible_teacher_is_rejected(tmp_path, field, value):
    run = make_source(tmp_path)
    args = export_args(run)
    saved = torch.load(args['teacher_path'], weights_only=False)
    saved['manifest']['contract']['source'][field] = value
    atomic_save(tmp_path / 'incompatible.pt', saved)
    with pytest.raises(ValueError, match='source-only|matching tasks'):
        export_corpus(run, tmp_path / 'invalid', **dict(args, teacher_path=tmp_path / 'incompatible.pt',
            teacher_sha256=file_sha256(tmp_path / 'incompatible.pt')))
