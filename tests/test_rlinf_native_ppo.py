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
        def pilot_phase(self, name):
            events.append(('phase', name)); return Handle()
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
    actor._rank = 0; actor.global_step = 0; actor.optimizer_steps = 0
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
