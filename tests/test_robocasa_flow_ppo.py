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
    GlobalActionJournal, flow_contract, flow_seeds, train_case_seeds)


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
