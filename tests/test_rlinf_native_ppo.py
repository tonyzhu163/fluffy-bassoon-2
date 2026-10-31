"""Allocated Linux CPU tests: contracts, real config composition, no models/envs."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from fluffy_bassoon.meta_rl.rlinf_native_ppo import (
    ActionJournal, prior_gpu_charge, restore_rng, sha, snapshot_rng, validate_recipe,
)


@pytest.fixture
def recipe(monkeypatch, tmp_path):
    from hydra import compose, initialize_config_dir
    code = Path(__file__).resolve().parents[1]
    for key, value in dict(NATIVE_PPO_OUTPUT=str(tmp_path), NATIVE_PPO_ACCOUNTING_ROOT=str(tmp_path/'actions'),
            NATIVE_PPO_PLACEMENT='0-7', NATIVE_PPO_WORLD='8', NATIVE_PPO_DEADLINE='9999999999',
            ROBOTWIN_ASSETS_PATH='/qualified/assets', NATIVE_PPO_MODEL='/qualified/fa8',
            NATIVE_PPO_INITIALIZER_REVISION='fa8df6ed103db0f5549c122f3a17c00ba6426c98').items():
        monkeypatch.setenv(key, value)
    assert Path(os.environ['EMBODIED_PATH']).is_dir(), 'Use the actual pinned RLinf config checkout'
    with initialize_config_dir(version_base='1.1', config_dir=str(code/'configs/native_ppo')):
        return compose(config_name='adjust_bottle_published_batch_two')


def test_published_batch_and_h400_compose(recipe):
    result = validate_recipe(recipe)
    assert result['accumulation_steps'] == 8
    assert result['nominal_all_in_actions'] == 921600
    assert recipe.actor.optim.lr_scheduler == 'constant'
    assert not recipe.actor.fsdp_config.save_full_model_weights


def test_four_rank_preserves_sixtyfour_microbatch_weights(recipe):
    recipe.native_pilot.world_size = 4
    recipe.cluster.component_placement = {'actor, env, rollout': '0-3'}
    result = validate_recipe(recipe)
    assert result['accumulation_steps'] == 16
    assert 4 * result['accumulation_steps'] == 8 * 8 == 64


@pytest.mark.parametrize('field,value', [('actor.global_batch_size', 1024),
    ('env.train.max_episode_steps', 200), ('actor.model.openpi.train_expert_only', False),
    ('actor.optim.lr_scheduler', 'cosine'), ('runner.only_eval', True)])
def test_rejects_scientific_recipe_drift(recipe, field, value):
    from omegaconf import OmegaConf
    OmegaConf.update(recipe, field, value)
    with pytest.raises(ValueError, match='contract differs'):
        validate_recipe(recipe)


def test_unresolved_calls_remain_charged_across_fresh_attempt(tmp_path):
    first = ActionJournal(tmp_path, rank=0, world=8, attempt='r1')
    slot = first.begin(100000, phase='training')
    first.finish(slot, 90000)
    first.begin(20000, phase='training')  # unresolved: all20000 remain spent
    second = ActionJournal(tmp_path, rank=0, world=8, attempt='r2')
    assert second.charged() == 110000
    with pytest.raises(RuntimeError, match='escrow'):
        second.begin(15001, phase='after')
    slot = second.begin(15000, phase='after')
    second.finish(slot, 12000)
    assert second.charged() == 122000
    with pytest.raises(ValueError, match='overwrite'):
        ActionJournal(tmp_path, rank=0, world=8, attempt='r2')
    with pytest.raises(ValueError, match='rank partition'):
        ActionJournal(tmp_path, rank=0, world=4, attempt='r3').charged()


def test_prior_gpu_clock_requires_terminal_hash_bound_unique_receipts(tmp_path):
    path = tmp_path/'clock.json'
    path.write_text(json.dumps(dict(status='finished', maximum_aggregate_gpu_seconds=21600,
        charged_gpu_seconds=800)))
    row = dict(clock_path=str(path), sha256=sha(path))
    assert prior_gpu_charge(dict(prior_attempts=[row])) == 800
    with pytest.raises(ValueError, match='Duplicate'):
        prior_gpu_charge(dict(prior_attempts=[row, row]))
    path.write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        prior_gpu_charge(dict(prior_attempts=[row]))


def test_worker_rng_roundtrip_cpu_no_models():
    import random
    import numpy as np
    import torch
    state = snapshot_rng()
    first = (random.random(), np.random.rand(), torch.rand(4))
    restore_rng(state)
    second = (random.random(), np.random.rand(), torch.rand(4))
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


def test_official_runtime_import_and_baseline_precedes_training(monkeypatch, tmp_path):
    # Import actual pinned native worker/runner dependencies on allocated CPU;
    # replace execution only, never construct actors, envs or a Ray cluster.
    from fluffy_bassoon.meta_rl import rlinf_native_ppo_runtime as runtime
    events = []
    class Handle:
        def __init__(self, value=None):
            self.value = [] if value is None else value
        def wait(self):
            return self.value
    class Group:
        def __init__(self, name):
            self.name = name
        def pilot_rng(self, *, restore=False):
            events.append((self.name, 'rng', restore)); return Handle()
        def pilot_phase(self, name, **kwargs):
            events.append(('phase', name)); return Handle()
        def pilot_evaluation_iteration(self, iteration):
            return Handle()
        def pilot_pair_receipt(self):
            return Handle([dict(same_ids=True)])
    runner = object.__new__(runtime.PilotRunner)
    runner.cfg = SimpleNamespace(native_pilot=SimpleNamespace(mode='train', output=str(tmp_path),
        initializer_revision='fa8'), runner={})
    runner.global_step = 0
    runner.actor, runner.rollout, runner.env = [Group(name) for name in ('actor', 'rollout', 'env')]
    runner.update_rollout_weights = lambda: events.append(('sync',))
    runner.evaluate = lambda: events.append(('evaluate', runner.global_step)) or {'success': .5}
    runner._capture_boundary = lambda: events.append(('boundary', runner.global_step))
    monkeypatch.setattr(runtime, 'validate_recipe', lambda _: {})
    def official_run(self):
        for step in (1, 2):
            events.append(('official_optimizer', step))
            self.global_step = step
    monkeypatch.setattr(runtime.EmbodiedRunner, 'run', official_run)
    runner.run()
    assert events.index(('evaluate', 0)) < events.index(('official_optimizer', 1))
    assert events.index(('official_optimizer', 2)) < events.index(('evaluate', 2))
    assert events.count(('evaluate', 0)) == events.count(('evaluate', 2)) == 1
    assert json.loads((tmp_path/'after-evaluation.json').read_text())['global_step'] == 2


def test_boundary_capture_follows_official_environment_wait(monkeypatch):
    from fluffy_bassoon.meta_rl import rlinf_native_ppo_runtime as runtime
    events = []
    runner = object.__new__(runtime.PilotRunner)
    runner._capture_boundary = lambda: events.append('capture')
    monkeypatch.setattr(runtime.EmbodiedRunner, '_log_step_metrics', lambda self: events.append('official_wait'))
    runner._log_step_metrics()
    assert events == ['official_wait', 'capture']


def test_sampler_and_eval_rng_survive_whole_boundary_restore(tmp_path):
    import torch
    from fluffy_bassoon.meta_rl.rlinf_native_ppo_runtime import SidecarMixin
    class FakeEnvironmentWorker(SidecarMixin):
        pilot_role = 'env'
    worker = FakeEnvironmentWorker()
    worker._rank = 0; worker._trajectory_step = 8; worker.version = 2
    worker._prefetched_train_bootstrap = None
    worker._pilot_first_resets = {'before': [{'public_observation_sha256': 'saved'}]}
    worker._pilot_rng = snapshot_rng()
    env = SimpleNamespace(reset_state_ids=torch.tensor([40, 41]), _current_seed_index=2,
        _generator=torch.Generator().manual_seed(17), success_seeds=torch.arange(8), seed=0)
    worker.env_list = [env]; worker.eval_env_list = []
    worker.last_obs_list = []; worker.last_intervened_info_list = []
    receipt = worker.pilot_capture(str(tmp_path))
    env.reset_state_ids = torch.tensor([99, 99]); env._current_seed_index = 7
    worker._pilot_first_resets = {}; worker._trajectory_step = 99
    restored = worker.pilot_restore(str(tmp_path))
    assert restored['snapshot_sha256'] == receipt['sha256']
    assert worker._trajectory_step == 8 and env._current_seed_index == 2
    assert env.reset_state_ids.tolist() == [40, 41]
    assert worker._pilot_first_resets['before'][0]['public_observation_sha256'] == 'saved'
    env.success_seeds = torch.arange(9)
    with pytest.raises(ValueError, match='seed inventory'):
        worker.pilot_restore(str(tmp_path))


def test_original_rollout_is_saved_before_official_training_shuffle(monkeypatch, tmp_path):
    import torch
    from fluffy_bassoon.meta_rl import rlinf_native_ppo_runtime as runtime
    actor = object.__new__(runtime.PilotActor)
    actor.cfg = SimpleNamespace(native_pilot=SimpleNamespace(output=str(tmp_path)),
        env=SimpleNamespace(train=SimpleNamespace(rollout_epoch=4)))
    actor._rank = 0; actor.optimizer_steps = 0
    actor.model = SimpleNamespace(set_global_step=lambda step: None)
    assert not hasattr(actor, 'global_step')
    actor.set_global_step(0)  # Actual author setter updates version, never global_step.
    final_values = torch.arange(9 * 8).reshape(9, 8, 1)
    actor.rollout_batch = dict(prev_values=final_values, dones=torch.zeros(9, 8, 50),
        forward_inputs=dict(chains=torch.ones(8, 8, 6, 50, 32), denoise_inds=torch.zeros(8, 8, 5)))
    def official_adv(self):
        self.rollout_batch['advantages'] = torch.ones(8, 8, 1)
        return {'unchanged_metric': 1}
    monkeypatch.setattr(runtime.EmbodiedFSDPActor, 'compute_advantages_and_returns', official_adv)
    assert actor.compute_advantages_and_returns() == {'unchanged_metric': 1}
    saved = torch.load(next((tmp_path/'training-rollouts').glob('*.pt')), weights_only=False)
    assert torch.equal(saved['rollout_batch']['prev_values'], final_values)
    assert saved['rollout_batch']['forward_inputs']['chains'].shape[2] == 6
    assert saved['producing_policy_iteration'] == 0 and saved['collection_iteration'] == 1
    assert saved['evaluation_data'] is False
    actor.rollout_batch['prev_values'] = final_values[:-1].flatten()
    assert torch.equal(saved['rollout_batch']['prev_values'], final_values)
    first_path = next((tmp_path/'training-rollouts').glob('*.pt'))
    first_sha = sha(first_path)
    actor.rollout_batch['prev_values'] = final_values
    actor.set_global_step(1)
    actor.compute_advantages_and_returns()
    paths = sorted((tmp_path/'training-rollouts').glob('*.pt'))
    assert len(paths) == 2 and sha(first_path) == first_sha
    second = torch.load(paths[1], weights_only=False)
    assert second['producing_policy_iteration'] == 1 and second['collection_iteration'] == 2
    assert not hasattr(actor, 'global_step')
    with pytest.raises(RuntimeError, match='never overwrite'):
        actor.compute_advantages_and_returns()


def test_native_history_multirank_episode_contract(monkeypatch, tmp_path):
    """One CPU fixture:2ranks×4rounds, mixed S/F, padding, GAE and DPT refs."""
    import torch
    from rlinf.algorithms.registry import calculate_adv_and_returns
    from rlinf.utils.nested_dict_process import process_nested_dict_for_adv, process_nested_dict_for_train
    from fluffy_bassoon.meta_rl import rlinf_native_ppo_runtime as runtime
    from fluffy_bassoon.meta_rl.rlinf_native_ppo_data import reconstruct_rows, record_gaussian_parameters

    def observation(number):
        return dict(main_images=torch.full((1, 2, 2, 3), number, dtype=torch.uint8),
            wrist_images=torch.full((1, 2, 2, 2, 3), number+1, dtype=torch.uint8),
            states=torch.full((1, 14), float(number)), task_descriptions=['adjust bottle'])
    traces = []
    for rank in range(2):
        for round_index in range(4):
            success = (rank+round_index) % 2 == 0
            elapsed = 0
            for chunk in range(2):
                mask = torch.tensor([[True, False] if success and chunk == 0 else
                    [False, False] if success else [True, True]])
                duration = int(mask.sum())
                current = observation(rank*40+round_index*4+elapsed)
                next_obs = observation(rank*40+round_index*4+elapsed+duration)
                rewards = torch.tensor([[0., 1.]]) if success and chunk == 0 else torch.zeros(1, 2)
                terms = torch.tensor([[False, success]])
                truncs = torch.tensor([[False, not success and chunk == 1]])
                actions = torch.arange(28).float().reshape(1, 2, 14)+rank+round_index
                traces.append(dict(evaluation_data=False, task_key='adjust_bottle', rank=rank,
                    collection_iteration=1, producing_policy_iteration=0, round_index=round_index,
                    chunk_index=chunk, local_env_indices=[0], global_env_indices=[rank],
                    episode_serials=torch.tensor([round_index+1]),
                    reset_state_ids=torch.tensor([100+rank*10+round_index]),
                    primitive_time_before=torch.tensor([elapsed]), current_observation=current,
                    proposed_motor_actions=actions, measured_valid_mask=True,
                    outputs=([next_obs], rewards, terms, truncs,
                        [dict(valid_action_mask=mask, final_observation=next_obs)])))
                elapsed += duration
    # Evaluation never enters this reader, even if interleaved in a file list.
    evaluation = dict(traces[0], evaluation_data=True)
    teacher = dict(checkpoint_path='/fixed/source/global_step_25/actor', checkpoint_sha256='a'*64,
        actor_config_sha256='b'*64, norm_stats_sha256='c'*64,
        sampling_seed_rule='predeclared query seed; no target checkpoint selection')
    rows = reconstruct_rows([evaluation, *reversed(traces)], teacher)
    assert len(rows) == 16 and sum(row['duration'] > 0 for row in rows) == 12
    assert sum(row['duration'] == 0 for row in rows) == 4  # preserve post-success padding
    assert all(row['submitted_actions'].shape == (2, 14) for row in rows)
    assert all(row['executed_actions'].shape[0] == row['duration'] for row in rows)
    assert all(row['current_observation']['wrist_images'].shape[0] == 2 for row in rows)
    assert all(torch.equal(row['next_observation']['states'], row['final_observation']['states']) for row in rows)
    assert all(row['dpt_teacher_action'] is None and row['dpt_label_teacher'] == teacher for row in rows)
    assert all(row['current_observation']['task_descriptions'] == 'adjust bottle' for row in rows)
    for row in rows:
        for earlier in row['history_row_indices']:
            assert rows[earlier]['rank'] == row['rank'] and rows[earlier]['duration'] > 0
    assert any(row['history_row_indices'] and row['episode_serial'] > 1 for row in rows)

    for rank in range(2):
        ordered = sorted((trace for trace in traces if trace['rank'] == rank),
            key=lambda trace: (trace['round_index'], trace['chunk_index']))
        raw = dict(rewards=torch.cat([trace['outputs'][1][None] for trace in ordered]),
            prev_logprobs=torch.zeros(8, 1, 2, 14),
            prev_values=torch.zeros(12, 1, 1), dones=torch.zeros(12, 1, 2, dtype=torch.bool),
            terminations=torch.zeros(12, 1, 2, dtype=torch.bool),
            truncations=torch.zeros(12, 1, 2, dtype=torch.bool),
            forward_inputs=dict(chains=torch.ones(8, 1, 6, 2, 32),
                denoise_inds=torch.zeros(8, 1, 5, dtype=torch.long)))
        for round_index in range(4):
            raw['prev_values'][round_index*3+2] = .75  # true final bootstrap row
            for chunk in range(2):
                source = ordered[round_index*2+chunk]
                slot = round_index*3+chunk+1
                raw['terminations'][slot] = source['outputs'][2]
                raw['truncations'][slot] = source['outputs'][3]
                raw['dones'][slot] = source['outputs'][2] | source['outputs'][3]
        batch = process_nested_dict_for_adv(raw, 4)  # Actual author's unflattening.
        batch['loss_mask'] = torch.tensor([[ordered[round_index*2+chunk]['measured_valid_mask'] and
            bool(ordered[round_index*2+chunk]['outputs'][4][-1]['valid_action_mask'].any())
            for round_index in range(4)] for chunk in range(2)])[:,:,None]
        gae = calculate_adv_and_returns(task_type='embodied', adv_type='gae', reward_type='chunk_level',
            rewards=batch['rewards'], dones=batch['dones'], values=batch['prev_values'],
            loss_mask=batch['loss_mask'], gamma=.99, gae_lambda=.95, normalize_advantages=True,
            group_size=1)
        batch.update(gae)
        actor = object.__new__(runtime.PilotActor)
        actor.cfg = SimpleNamespace(native_pilot=SimpleNamespace(output=str(tmp_path)),
            env=SimpleNamespace(train=SimpleNamespace(rollout_epoch=4)))
        actor._rank = rank; actor.optimizer_steps = 0
        actor.model = SimpleNamespace(set_global_step=lambda step: None)
        actor.set_global_step(0); actor.rollout_batch = batch
        monkeypatch.setattr(runtime.EmbodiedFSDPActor, 'compute_advantages_and_returns', lambda self: {})
        actor.compute_advantages_and_returns()
        saved = torch.load(tmp_path/'training-rollouts'/
            f'iteration-0001-policy-0000-rank-{rank:02d}.pt', weights_only=False)
        assert saved['rollout_batch']['prev_values'].shape == (3, 4, 1)
        assert torch.all(saved['rollout_batch']['prev_values'][-1] == .75)
        assert torch.equal(saved['rollout_batch']['loss_mask'], batch['loss_mask'])
        assert torch.equal(saved['rollout_batch']['returns'], gae['returns'])
        flattened = process_nested_dict_for_train(batch, torch.arange(8).flip(0))
        assert flattened['prev_values'].shape[0] == 8
        assert saved['rollout_batch']['prev_values'].shape[0] == 3

    # Actual mean/std observer returns the author's exact tensor objects and
    # consumes no additional random draw or second sampler evaluation.
    calls, parameters = [], []
    mean, std = torch.ones(1, 2, 32), torch.full((1, 2, 32), .3)
    def author(*args, **kwargs):
        calls.append(1); return mean, std
    observer = record_gaussian_parameters(author, lambda: parameters, runtime.cpu_tree)
    rng = torch.get_rng_state().clone()
    actual_mean, actual_std = observer(mean, mean, torch.tensor([0]),
        noise_method='flow_sde', noise_level=.3, num_steps=5)
    assert actual_mean is mean and actual_std is std and calls == [1]
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(parameters[0]['mean'], mean) and torch.equal(parameters[0]['std'], std)
