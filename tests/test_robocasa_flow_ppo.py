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
