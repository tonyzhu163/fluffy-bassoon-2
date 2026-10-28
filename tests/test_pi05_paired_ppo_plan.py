"""Identical private plans, explicit budgets and behavioral admission for paired PPO."""

from collections import Counter
from copy import deepcopy
import random

import pytest

from fluffy_bassoon.meta_rl.pi05_balanced_plan import canonical_sha256, plan_sha256
from fluffy_bassoon.meta_rl.pi05_paired_ppo_plan import (
    GATE_PROTOCOL, PairedPPOPlanProvider, make_paired_ppo_contract, online_budget,
    require_live_competence_gate,
)


INFOS = (25, None, 21, None, 1, None, 4, 4)


def live_gate():
    return dict(protocol=GATE_PROTOCOL, status="passed", complete=True,
        evidence_kind="live_same_interface", sampler="StochasticPi05BridgePolicy.sample_chunk",
        deterministic=True, denoise_steps=4, action_chunk=5, active_dims=7, residual_scale=1.,
        core="persistent_ctm", width=64, ticks=4,
        source_commit="a" * 40, evidence_sha256="b" * 64,
        motor_sha256="1" * 64, norm_stats_sha256="2" * 64, rlinf_sources={"native": "3" * 64},
        evaluated_theta_degrees=[-45., 0., 45.], decision="Synthetic reviewed gate fixture only",
        comparisons={str(theta): dict(bridge=[[1, 1]], zero_bridge=[[0, 0]], external_oracle=[[1, 1]])
                     for theta in (-45., 0., 45.)})


def warmstart_parent():
    return dict(family="native_pi05_history_v1", source_commit="c" * 40, iteration=100,
        compatibility=dict(core="persistent_ctm",
            motor_sha256="1" * 64, norm_stats_sha256="2" * 64, rlinf_sources={"native": "3" * 64},
            bridge=dict(core="persistent_ctm", width=64, ticks=4, residual_scale=1.),
            gaussian_policy=dict(noise_scale=.05, active_dims=7, residual_scale=1.)),
        data=dict(balanced_plan_sha256="d" * 64),
        pretraining_counters=dict(teacher_simulator_steps=3840, optimizer_steps=100))


def test_both_routes_get_identical_balanced_plans_without_mutable_rng_dependence():
    before = random.getstate()
    direct, dpt = PairedPPOPlanProvider(INFOS, seed=8), PairedPPOPlanProvider(INFOS, seed=8)
    assert random.getstate() == before
    all_hashes, all_noise_seeds = [], []
    for iteration in range(20):
        plan = direct(iteration)
        assert plan == dpt(iteration)
        assert plan["iteration"] == iteration and plan["update"] == iteration + 1
        assert plan["plan_sha256"] == plan_sha256(plan)
        assert Counter(zip(plan["theta_degrees"], plan["object_indices"], strict=True)) == Counter(
            (theta, obj) for theta in (-45., 0., 45.) for obj in (0, 1))
        assert len(plan["scenes"]) == 4 and all(len(batch) == 6 for batch in plan["scenes"])
        assert all(row["object"] == plan["object_indices"][lane] and row["overlay"] < 12
                   for batch in plan["scenes"] for lane, row in enumerate(batch))
        assert len({row["episode_id"] for batch in plan["scenes"] for row in batch}) == 24
        all_hashes.append(plan["plan_sha256"])
        all_noise_seeds.append(plan["motor_residual_rng_seed"])
    assert len(set(all_hashes)) == len(set(all_noise_seeds)) == 20
    altered = direct(0)
    altered["scenes"][0][0]["episode_id"] = -1
    assert direct(0) == dpt(0)
    provenance = direct.provenance()
    digest = provenance.pop("provider_sha256")
    assert canonical_sha256(provenance) == digest


def test_resume_uses_completed_iteration_and_exact_cumulative_online_budget():
    provider = PairedPPOPlanProvider(INFOS, seed=91)
    provider(0)
    provider(7)
    assert provider(11) == PairedPPOPlanProvider(INFOS, seed=91)(11)
    assert online_budget(20) == dict(completed_updates=20, global_batch_trials=6, trials=120,
        attempts=480, simulator_steps=38400, policy_chunks=7680, native_prefix_queries=7680,
        native_suffix_queries=30720, optimizer_steps_max=80)
    assert provider(19)["cumulative_online_budget"] == online_budget(20)
    for invalid in (-1, 20, True, 1.):
        with pytest.raises(ValueError):
            provider(invalid)
    for invalid in (0, 21, True):
        with pytest.raises(ValueError):
            PairedPPOPlanProvider(INFOS, updates=invalid)


@pytest.mark.parametrize("change", [
    dict(status="unresolved"), dict(status="failed"), dict(complete=False),
    dict(evidence_kind="offline_flow_loss"), dict(sampler="execute_labels"),
    dict(residual_scale=.1), dict(width=128), dict(ticks=1), dict(denoise_steps=10),
    dict(evaluated_theta_degrees=[-45., 45.]), dict(evidence_sha256=None), dict(decision=""),
])
def test_offline_loss_incomplete_or_different_interface_cannot_admit_ppo(change):
    receipt = dict(live_gate(), **change)
    with pytest.raises(ValueError):
        require_live_competence_gate(receipt)


def test_live_gate_requires_paired_rectangular_binary_outcomes():
    receipt = live_gate()
    assert require_live_competence_gate(receipt) == receipt
    receipt["comparisons"]["0.0"]["bridge"] = [[.5, 1.]]
    with pytest.raises(ValueError, match="binary"):
        require_live_competence_gate(receipt)
    receipt = live_gate()
    receipt["comparisons"]["0.0"]["bridge"] = [[1]]
    with pytest.raises(ValueError, match="matched rectangular"):
        require_live_competence_gate(receipt)


def test_contract_pins_shared_optimizer_noise_and_separate_dpt_costs():
    provider = PairedPPOPlanProvider(INFOS, seed=4)
    parent = warmstart_parent()
    contract = make_paired_ppo_contract(provider, source_commit="e" * 40,
        warmstart_sha256="f" * 64, warmstart_provenance=parent, competence_gate=live_gate())
    digest = contract.pop("contract_sha256")
    assert canonical_sha256(contract) == digest
    assert contract["plans"]["planned_online_budget_per_route"]["simulator_steps"] == 38400
    assert contract["optimization"]["epochs"] == 4 and contract["optimization"]["target_kl"] == .05
    assert contract["initialization"]["dpt"]["parent"] == parent
    assert contract["resource_proposal"]["gpus"] == 2
    assert contract["resource_proposal"]["driver_elapsed_seconds_max_per_route"] == 3600
    bad_parent = deepcopy(parent)
    bad_parent["compatibility"]["core"] = "transformer_xl"
    with pytest.raises(ValueError, match="compatible balanced"):
        make_paired_ppo_contract(provider, source_commit="e" * 40, warmstart_sha256="f" * 64,
            warmstart_provenance=bad_parent, competence_gate=live_gate())
    different_motor = dict(live_gate(), motor_sha256="4" * 64)
    with pytest.raises(ValueError, match="motor/runtime identities"):
        make_paired_ppo_contract(provider, source_commit="e" * 40, warmstart_sha256="f" * 64,
            warmstart_provenance=parent, competence_gate=different_motor)
