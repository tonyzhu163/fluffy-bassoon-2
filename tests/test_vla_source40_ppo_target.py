"""Frozen PPO target support/query semantics; run on allocated Linux only."""
from copy import deepcopy
import hashlib
import json
import pickle

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.vla_rl2 import RL2ActorCritic, RL2Config
from fluffy_bassoon.meta_rl.vla_source40_evaluate import SUPPORT_BUDGETS, _episode_rng, evaluate_case
from fluffy_bassoon.meta_rl.vla_source40_ppo_target import (PPO_METHODS, TargetPPOController,
    candidates, controller_factory, reservation, shifted_command)
from fluffy_bassoon.meta_rl.vla_libero_pro import initial_state_receipt
from test_vla_source40_evaluate import Simulator, Motor, case


def policy():
    torch.manual_seed(7)
    return RL2ActorCritic(RL2Config(observation_dim=4104, noise_dim=32, width=8, layers=1, heads=2, context_length=8))


def specs():
    return candidates(dict(full=policy(), episode=policy()),
        dict(full=dict(policy_sha256='full-source-hash'), episode=dict(policy_sha256='separate-reset-source-hash')))


def test_global_rl2_rng_seed_avoids_the_unsupported_local_generator():
    controller = TargetPPOController(policy())
    _episode_rng(controller, 123)
    assert controller.generator is None
    first = controller.act(np.zeros(4104, np.float32))
    controller.observe(executed_noise=first, reward=-1., done=True)
    saved = controller.snapshot()
    _episode_rng(controller, 456)
    expected = controller.act(np.zeros(4104, np.float32))
    controller.observe(executed_noise=expected, reward=0., done=True)
    torch.randn(100)
    other = TargetPPOController(controller.policy.base)
    other.restore(saved)
    _episode_rng(other, 456)
    np.testing.assert_array_equal(other.act(np.zeros(4104, np.float32)), expected)
    assert other.generator is None
    assert saved['previous_reward'] == -1. and saved['previous_done']
    assert saved['state']['length'].item() == 1


def test_same_full_weights_reset_lesion_and_separately_trained_reset_are_distinct():
    selected = specs()
    assert selected['ppo_episode_reset']['policy'] is selected['ppo_full']['policy']
    assert selected['ppo_episode_reset']['sha256'] == selected['ppo_full']['sha256']
    assert selected['ppo_trained_episode']['policy'] is not selected['ppo_full']['policy']
    assert selected['ppo_trained_episode']['sha256'] != selected['ppo_full']['sha256']
    for method in PPO_METHODS:
        controller = controller_factory(selected[method])
        _episode_rng(controller, 10)
        noise = controller.act(np.zeros(4104, np.float32))
        controller.observe(executed_noise=noise, reward=-1., done=True)
        _episode_rng(controller, 11)
        controller.act(np.zeros(4104, np.float32))
        assert controller.state['length'].item() == (2 if method == 'ppo_full' else 1)


def test_all_shot_budgets_are_actual_supports_then_a_separate_query(tmp_path):
    sim, motor, selected = Simulator(), Motor(), specs()
    result = evaluate_case(sim, motor, selected, case=case(), output=tmp_path,
        provenance=dict(group_maximum_total_actions=10000), max_primitive_steps=10000, max_settling_steps=1000,
        support_budgets=SUPPORT_BUDGETS, methods=('ppo_full',), controller_factory=controller_factory)
    assert sim.episodes == 20 and len(result['episodes']) == 20
    assert [row['support_budget'] for row in result['trials']] == list(SUPPORT_BUDGETS)
    query_seeds = [row['episode_random_seed'] for row in result['episodes'] if row['role'] == 'query']
    assert len(set(query_seeds)) == 1
    assert result['weights_unchanged'] and result['budget_matched_primary_eligible']
    assert all(len(row['support_episode_ids']) == row['support_budget'] for row in result['trials'])
    assert result['physical_costs']['support']['reset_attempts'] == 15
    assert result['physical_costs']['query']['reset_attempts'] == 5


def test_interrupted_query_restores_actual_support_context_and_charges_unknown_rpc(tmp_path):
    selected = specs()
    sim, motor = Simulator(fail_on_step=3), Motor()
    kwargs = dict(case=case(), output=tmp_path, provenance={}, max_primitive_steps=10000,
        max_settling_steps=1000, support_budgets=(1,), methods=('ppo_full',), controller_factory=controller_factory)
    with pytest.raises(RuntimeError, match='injected RPC'):
        evaluate_case(sim, motor, selected, **kwargs)
    with (tmp_path / 'evaluation-recovery.pkl').open('rb') as stream:
        saved = pickle.load(stream)
    assert saved['next_episode'] == 1 and saved['controller']['format'] == 'frozen_target_ppo_context_v1'
    assert saved['controller']['state']['length'].item() == 2
    assert saved['controller']['pending'] is False
    sim.fail_on_step = None
    result = evaluate_case(sim, motor, selected, resume=True, **kwargs)
    assert len(result['episodes']) == 2 and not result['budget_matched_primary_eligible']
    assert result['physical_costs']['query']['uncertain_primitive_steps_ceiling'] == 10
    assert result['weights_unchanged']


def test_escrow_is_read_only_and_unrelated_new_reservations_do_not_change_identity(tmp_path):
    path = tmp_path / 'ledger.json'
    record = dict(run_id='future-target', source_commit='a' * 40, maximum_total_actions=5000,
        campaign_sha256='campaign', purpose='target_evaluation', source_pretraining_data=False,
        fresh_ppo_training_data=False)
    value = dict(campaign_sha256='campaign', reservations={'future-target': record})
    path.write_text(json.dumps(value))
    first = reservation(path, 'future-target', 'a' * 40, 5000)
    value['reservations']['unrelated'] = dict(something='new')
    path.write_text(json.dumps(value))
    assert reservation(path, 'future-target', 'a' * 40, 5000) == first
    with pytest.raises(ValueError, match='escrow'):
        reservation(path, 'future-target', 'a' * 40, 6000)


def test_shifted_workers_and_actual_pro_nominal_state_receipt():
    from fluffy_bassoon.meta_rl.vla_source40_evaluate import TargetCase
    plus = TargetCase('plus', 'declared-plus', 'libero_object', 667, 0, 123)
    pro = TargetCase('pro', 'declared-pro', 'libero_object_swap', 1, 0, 123)
    config = dict(simulator=dict(python='qualified-python', pythonpath='qualified-path', plus_root='/plus',
        pro_root='/pro', data_root='/pro-data', pro_case='swap'))
    assert 'fluffy_bassoon.meta_rl.vla_plus_worker' in shifted_command(plus, config, source_commit='a' * 40, output='/fresh', port=1234)
    assert 'fluffy_bassoon.meta_rl.vla_libero_pro' in shifted_command(pro, config, source_commit='a' * 40, output='/fresh', port=1234)
    observation = dict(image=np.zeros((2, 2, 3), np.uint8), wrist=np.ones((2, 2, 3), np.uint8), state=np.zeros(8, np.float32))
    nominal = np.arange(110, dtype=np.float64)
    receipt = initial_state_receipt(0, nominal, observation)
    assert receipt['initial_state_vector_sha256'] == hashlib.sha256(nominal.tobytes()).hexdigest()
    changed = deepcopy(observation)
    changed['state'][0] = 1.
    other = initial_state_receipt(0, nominal, changed)
    assert receipt['initial_state_vector_sha256'] == other['initial_state_vector_sha256']
    assert receipt['initial_observation_sha256'] != other['initial_observation_sha256']
