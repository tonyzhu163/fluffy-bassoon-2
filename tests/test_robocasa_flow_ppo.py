import numpy as np
import pytest

from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo import first_success_reward


def test_transient_success_pays_once_without_later_chunk_cancellation():
    successes = np.array([[False, False], [True, False], [False, True],
                          [True, True], [False, False]])
    seen = np.zeros(2, dtype=bool)
    rewards = []
    for success in successes:
        rewards.append(first_success_reward(success, seen))
        seen |= success
    np.testing.assert_array_equal(rewards, [[0, 0], [1, 0], [0, 1], [0, 0], [0, 0]])
    np.testing.assert_array_equal(np.sum(rewards, axis=0), [1, 1])


def test_first_success_rejects_a_broadcast_latch():
    with pytest.raises(ValueError, match='latch shape'):
        first_success_reward([True, False], False)


from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo import (
    GlobalActionJournal, flow_contract, flow_episode_ids, flow_logprob_parity, flow_seeds,
    flow_value_parity, keep_frozen_image_encoder_in_eval, train_case_seeds,
    verify_grpo_group_reset)


def test_frozen_image_mode_is_preserved_without_changing_expert_training():
    import torch
    model = torch.nn.Module()
    model.img = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(.1))
    model.expert = torch.nn.Linear(4, 4)
    model.img.requires_grad_(False)
    model.train()
    keep_frozen_image_encoder_in_eval(model)
    assert model.training and model.expert.training and not model.img.training
    assert all(not child.training for child in model.img.modules())
    assert all(parameter.requires_grad for parameter in model.expert.parameters())
    model.img[0].weight.requires_grad_(True)
    with pytest.raises(ValueError, match='frozen image'):
        keep_frozen_image_encoder_in_eval(model)


def test_pre_sgd_parity_uses_executed_chunk_sum_and_rejects_padded_dims():
    import torch
    old = torch.zeros(2, 50, 12)
    assert flow_logprob_parity(old.clone(), old)['passed']
    # Individually tiny errors can corrupt the joint PPO ratio.
    assert not flow_logprob_parity(old + 1e-5, old)['passed']
    assert not flow_logprob_parity(old + float('nan'), old)['passed']
    with pytest.raises(ValueError, match='executed action'):
        flow_logprob_parity(torch.zeros(2, 50, 32), torch.zeros(2, 50, 32))


def test_pre_sgd_value_parity_rejects_denoise_step_mean():
    import torch
    steps = torch.tensor([[0.1, 0.5, 0.9], [0.2, 0.2, 0.8]])  # per-denoise-step values
    chosen = steps[torch.arange(2), torch.tensor([2, 0])]
    assert flow_value_parity(chosen.clone(), chosen[:, None])['passed']
    # Unpatched RLinf stores the mean over steps but recomputes the chosen step.
    assert not flow_value_parity(chosen, steps.mean(dim=1, keepdim=True))['passed']
    assert not flow_value_parity(chosen + float('nan'), chosen[:, None])['passed']


def test_flow_factory_stabilizes_native_regions_before_constructor(monkeypatch):
    from types import SimpleNamespace
    from fluffy_bassoon.meta_rl import rlinf_robocasa_flow_ppo_runtime as runtime
    calls = []
    monkeypatch.setattr(runtime, 'stable_counter_regions', lambda: calls.append('regions'))
    monkeypatch.setattr(runtime, 'counted_native_factory', lambda factory, **kwargs: factory)
    monkeypatch.setattr(runtime, 'native_fingerprint', lambda env: {'physical': 'recorded'})
    def construct():
        calls.append('constructor')
        return SimpleNamespace(reset=lambda: 'live observation')
    env = runtime.fresh_factory(construct, 123, 999)()
    assert calls == ['regions', 'constructor']
    assert env.reset() == 'live observation'
    assert env._flow_physical == {'physical': 'recorded'}


def flow_config(task='TurnOffStove', envs=128, rounds=1, maximum=12210000, **seeds):
    train_seed, eval_seed = flow_seeds(task)
    horizon = dict(max_episode_steps=600, max_steps_per_rollout_epoch=600, task_filter=[task])
    return dict(
        env=dict(train=dict(horizon, total_num_envs=envs, rollout_epoch=rounds,
                            seed=seeds.get('train', train_seed), split='target'),
                 eval=dict(horizon, total_num_envs=64, rollout_epoch=1, seed=seeds.get('eval', eval_seed))),
        actor=dict(global_batch_size=256, model=dict(num_action_chunks=50)),
        algorithm=dict(update_epoch=4), runner=dict(max_epochs=130, save_interval=10),
        native_pilot=dict(maximum_actions=maximum))


def test_ten_million_contract_arithmetic():
    contract = flow_contract(flow_config())
    assert contract['optimizer_steps_per_update'] == 24
    assert contract['nominal_policy_controls'] == 130 * 128 * 600 + 2 * 64 * 600 == 10060800
    assert contract['nominal_policy_controls'] + contract['reset_reserve_controls'] <= 12210000
    assert contract['physics_tick_limit'] == (600 + 128) * 128


def test_distinct_training_and_evaluation_episode_lengths():
    cfg = flow_config(maximum=None)
    cfg['env']['train'].update(max_episode_steps=400, max_steps_per_rollout_epoch=400)
    cfg['env']['eval'].update(max_episode_steps=750, max_steps_per_rollout_epoch=750)
    cfg['actor']['model']['num_action_chunks'] = 5
    cfg['actor']['global_batch_size'] = 2048
    contract = flow_contract(cfg)
    assert contract['horizon'] == 400 and contract['eval_horizon'] == 750
    assert contract['chunks_per_episode'] == 80 and contract['eval_chunks_per_episode'] == 150
    assert contract['optimizer_steps_per_update'] == 20
    assert contract['TRAIN_policy_controls'] == 130 * 128 * 400
    assert contract['nominal_policy_controls'] == 130 * 128 * 400 + 2 * 64 * 750
    assert contract['physics_tick_limit'] == (750 + 128) * 128


@pytest.mark.parametrize('phase,episode,rollout', [
    ('train', 400, 405), ('eval', 750, 745), ('eval', 751, 751), ('train', 0, 0),
])
def test_phase_horizons_reject_partial_chunks_and_inconsistent_rollouts(phase, episode, rollout):
    cfg = flow_config(maximum=None)
    cfg['actor']['model']['num_action_chunks'] = 5
    cfg['env'][phase].update(max_episode_steps=episode, max_steps_per_rollout_epoch=rollout)
    with pytest.raises(ValueError, match='Each phase horizon'):
        flow_contract(cfg)


@pytest.mark.parametrize('episode_length', [400, 750])
def test_environment_finishes_at_its_own_phase_horizon(monkeypatch, tmp_path, episode_length):
    from types import SimpleNamespace
    from fluffy_bassoon.meta_rl import rlinf_robocasa_flow_ppo_runtime as runtime
    written = []
    monkeypatch.setattr(runtime, 'write', lambda path, value: written.append(value))
    class Original:
        def step(self, actions, auto_reset):
            return 'native step'
    owner = SimpleNamespace(_flow_contract={'horizon': 400}, _flow_phase='evaluation',
                            _flow_iteration=0, _rank=0)
    wrapped = object.__new__(runtime.native_class(Original, owner))
    wrapped.cfg = SimpleNamespace(max_episode_steps=episode_length)
    wrapped._flow_steps = 0
    wrapped.success_once = [False]
    wrapped._flow_first_success = [None]
    wrapped.current_info_list = [{'success': False}]
    wrapped._flow_episode_ids, wrapped.env_seeds = ['case'], [123]
    wrapped._flow_root, wrapped._flow_key, wrapped._flow_initial = tmp_path, 'case', {}
    wrapped.env = SimpleNamespace(get_env_attr=lambda key: [dict(
        controls=wrapped._flow_steps + 3, reset_controls=3)])
    for _ in range(episode_length - 1):
        assert wrapped.step(actions=object()) == 'native step'
        assert not written
    wrapped.step(actions=object())
    assert len(written) == 1 and written[0]['completed']
    assert written[0]['rows'][0]['native_counts']['controls'] == episode_length + 3


@pytest.mark.parametrize('chunk,batch,epochs,iterations,steps', [
    (50, 256, 1, 40, 240),
    (50, 256, 4, 40, 960),
    (10, 512, 1, 40, 600),
    (50, 256, 1, 160, 960),
])
def test_executed_cadence_controls_likelihood_and_optimizer_exposure(
        chunk, batch, epochs, iterations, steps):
    cfg = flow_config(maximum=None)
    cfg['runner']['max_epochs'] = iterations
    cfg['actor']['global_batch_size'] = batch
    cfg['actor']['model'].update(num_action_chunks=chunk, action_dim=12, num_steps=5,
        openpi=dict(action_horizon=50, action_chunk=chunk, num_steps=5, ignore_last=True))
    cfg['algorithm']['update_epoch'] = epochs
    contract = flow_contract(cfg)
    assert contract['predicted_action_horizon'] == 50
    assert contract['chunks_per_episode'] == 600 // chunk
    assert contract['likelihood_coordinates'] == chunk * 12
    assert contract['total_optimizer_steps'] == steps
    assert contract['TRAIN_policy_controls'] == iterations * 128 * 600


def test_contract_rejects_unexecuted_likelihood_tail_or_mismatched_sampler():
    cfg = flow_config()
    cfg['actor']['model'].update(num_action_chunks=10,
        openpi=dict(action_horizon=50, action_chunk=50))
    with pytest.raises(ValueError, match='likelihood slices'):
        flow_contract(cfg)
    cfg['actor']['model']['openpi']['action_chunk'] = 10
    cfg['actor']['model']['num_steps'] = 5
    cfg['actor']['model']['openpi']['num_steps'] = 10
    with pytest.raises(ValueError, match='denoising steps'):
        flow_contract(cfg)


def test_short_chunk_parity_checks_only_executed_coordinates():
    import torch
    values = torch.zeros(2, 10, 12)
    assert flow_logprob_parity(values, values.clone())['shape'] == [2, 10, 12]
    with pytest.raises(ValueError, match='executed action'):
        flow_logprob_parity(torch.zeros(2, 50, 12), values)


def test_train_seeds_are_unique_disjoint_and_layout_independent():
    banks = []
    for envs, rounds in ((128, 1), (64, 2), (32, 4)):
        contract = flow_contract(flow_config(envs=envs, rounds=rounds))
        local = envs // 8
        seeds = [seed for iteration in range(130) for round_index in range(rounds) for rank in range(8)
                 for seed in train_case_seeds(contract, iteration=iteration, round_index=round_index,
                                              rank=rank, local_envs=local)]
        assert len(seeds) == len(set(seeds)) == 130 * 128
        assert not set(seeds) & set(range(contract['eval_seed'], contract['eval_seed'] + 64))
        banks.append(set(seeds))
    assert banks[0] == banks[1] == banks[2]
    with pytest.raises(ValueError, match='schedule'):
        train_case_seeds(contract, iteration=0, round_index=4, rank=0, local_envs=4)


def test_contract_rejects_mislabelled_or_underbudgeted_runs():
    with pytest.raises(ValueError, match='seeds must be'):
        flow_contract(flow_config(train=2100000))  # OpenDrawer's range on TurnOffStove
    with pytest.raises(ValueError, match='below nominal'):
        flow_contract(flow_config(maximum=1000000))
    config = flow_config()
    config['env']['eval']['task_filter'] = ['OpenDrawer']
    with pytest.raises(ValueError, match='exactly one task'):
        flow_contract(config)
    assert flow_contract(flow_config(maximum=None))['maximum_actions'] is None


def test_global_journal_records_without_a_ceiling(tmp_path):
    journal = GlobalActionJournal(tmp_path, rank=0, world=8, attempt='a', maximum_actions=None)
    journal.finish(journal.begin(50, phase='training'), 50)
    assert journal.charged() == 50


def test_actor_batch_drop_removes_only_named_duplicates():
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import retained_batch
    batch = dict(prev_logprobs=1, forward_inputs={'observation/image': 2, 'chains': 3, 'action': 4})
    kept = retained_batch(batch, ['forward_inputs.observation/image', 'forward_inputs.chains'])
    assert kept == dict(prev_logprobs=1, forward_inputs={'action': 4})
    assert retained_batch(batch, []) == batch
    with pytest.raises(ValueError, match='absent'):
        retained_batch(batch, ['forward_inputs.observation/images'])


def test_vlm_critic_ppo_and_grpo_contracts_and_grouped_seeds():
    assert flow_seeds('TurnOnSinkFaucet') == (2800000, 680100)
    ppo_cfg = flow_config(task='TurnOnSinkFaucet')
    ppo_cfg['actor']['model'].update(add_value_head=True, openpi=dict(value_after_vlm=True))
    ppo = flow_contract(ppo_cfg)
    assert ppo['adv_type'] == 'gae' and ppo['loss_type'] == 'actor_critic'
    assert ppo['group_size'] == 1 and ppo['add_value_head'] and ppo['value_after_vlm']
    assert ppo['scenes_per_iteration'] == 128

    grpo_cfg = flow_config(task='OpenDrawer', envs=128, rounds=1)
    grpo_cfg['algorithm'].update(adv_type='grpo', loss_type='actor', group_size=8)
    grpo_cfg['env']['train']['group_size'] = 8
    grpo_cfg['env']['eval']['group_size'] = 1
    grpo_cfg['actor']['model'].update(add_value_head=False, openpi=dict(value_after_vlm=False))
    grpo = flow_contract(grpo_cfg)
    assert grpo['adv_type'] == 'grpo' and grpo['loss_type'] == 'actor'
    assert grpo['group_size'] == 8 and not grpo['add_value_head']
    assert grpo['scenes_per_iteration'] == 16 and grpo['episodes_per_iteration'] == 128

    all_episode_ids, all_scene_seeds = [], []
    for iteration in range(130):
        for rank in range(8):
            seeds = train_case_seeds(grpo, iteration=iteration, round_index=0, rank=rank, local_envs=16)
            assert seeds[:8] == [seeds[0]] * 8 and seeds[8:] == [seeds[8]] * 8
            assert seeds[0] != seeds[8]
            ids = flow_episode_ids(grpo['task'], seeds, group_size=8)
            assert len(ids) == len(set(ids)) == 16
            all_episode_ids.extend(ids)
            all_scene_seeds.extend([seeds[0], seeds[8]])
    assert len(all_episode_ids) == len(set(all_episode_ids)) == 130 * 128
    assert len(all_scene_seeds) == len(set(all_scene_seeds)) == 130 * 16
    assert not set(all_scene_seeds) & set(range(grpo['eval_seed'], grpo['eval_seed'] + 64))

    bad_grpo = flow_config()
    bad_grpo['algorithm'].update(adv_type='grpo', loss_type='actor', group_size=8)
    bad_grpo['actor']['model']['add_value_head'] = True
    with pytest.raises(ValueError, match='GRPO requires'):
        flow_contract(bad_grpo)


def test_verify_grpo_group_reset_enforces_within_group_scene_identity():
    snapshot = dict(
        case=dict(seeds=[100, 100, 101, 101]),
        physical=[dict(xml='a'), dict(xml='a'), dict(xml='b'), dict(xml='b')],
        task_descriptions=['open drawer', 'open drawer', 'open drawer 2', 'open drawer 2'],
        states=np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]], dtype=np.float32),
    )
    verify_grpo_group_reset(snapshot, 2)
    broken = dict(snapshot, states=np.array([[1.0, 2.0], [1.0, 9.0], [3.0, 4.0], [3.0, 4.0]], dtype=np.float32))
    with pytest.raises(ValueError, match='initial public states'):
        verify_grpo_group_reset(broken, 2)


def test_periodic_val_check_interval_contract_accounting():
    cfg = flow_config(task='OpenDrawer', maximum=15000000)
    cfg['runner']['val_check_interval'] = 10
    contract = flow_contract(cfg)
    assert contract['eval_interval'] == 10
    assert contract['eval_panels'] == 1 + 130 // 10 == 14
    assert contract['nominal_policy_controls'] == 130 * 128 * 600 + 14 * 64 * 600
    assert contract['nominal_policy_controls'] + contract['reset_reserve_controls'] <= 15000000


def test_reconfigure_or_create_subproc_env_batches_pipe_commands_in_parallel(monkeypatch):
    from types import SimpleNamespace
    import rlinf.envs.sim.robocasa.venv as robocasa_venv
    import rlinf.envs.venv as base_venv
    from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo_env import reconfigure_or_create_subproc_env

    events = []
    created = []

    class FakeRemote:
        def __init__(self, idx):
            self.idx = idx

        def send(self, msg):
            events.append(('send', self.idx, msg[0], msg[1].data()))

        def recv(self):
            events.append(('recv', self.idx))
            return None

    class FakeSubprocEnv:
        def __init__(self, fns):
            created.append(len(fns))
            self.is_closed = False
            self.workers = [SimpleNamespace(parent_remote=FakeRemote(i)) for i in range(len(fns))]

        def close(self):
            self.is_closed = True

    monkeypatch.setattr(robocasa_venv, 'RobocasaSubprocEnv', FakeSubprocEnv)
    monkeypatch.setattr(base_venv, 'CloudpickleWrapper', lambda fn: SimpleNamespace(data=fn))

    env = reconfigure_or_create_subproc_env(None, [lambda: 'e0', lambda: 'e1'])
    assert created == [2] and not events
    reused = reconfigure_or_create_subproc_env(env, [lambda: 'n0', lambda: 'n1'])
    assert reused is env and created == [2]
    # All sends happen before any blocking recv so worker reconfigures run concurrently.
    assert events == [
        ('send', 0, 'reconfigure', 'n0'),
        ('send', 1, 'reconfigure', 'n1'),
        ('recv', 0),
        ('recv', 1),
    ]


def test_async_chunk_io_writes_and_flushes_deterministically(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import gzip
    import torch
    from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo_runtime import FlowEnvironment, _write_chunk_archive

    worker = object.__new__(FlowEnvironment)
    worker._flow_io_pool = ThreadPoolExecutor(max_workers=2)
    worker._flow_io_futures = []
    path = tmp_path / 'chunk01.pt.gz'
    payload = dict(phase='training', iteration=1, current_observation=dict(states=torch.ones(2, 4)))
    worker._submit_flow_io(_write_chunk_archive, path, payload)
    worker._flush_flow_io()
    assert worker._flow_io_futures == []
    with gzip.open(path, 'rb') as stream:
        loaded = torch.load(stream, weights_only=False)
    assert loaded['current_observation_sha256']
    assert torch.equal(loaded['current_observation']['states'], torch.ones(2, 4))
    worker._flow_io_pool.shutdown(wait=True)
