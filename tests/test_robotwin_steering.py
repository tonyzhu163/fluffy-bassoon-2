"""Run on allocated Linux CPU only; synthetic tests are not live qualification."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.rl.robotwin_steering import (
    ARMS, TASKS, CoefficientSteering, FixedActionResidual, FixedKVResidual, clipped_coefficients,
    fixed_basis, validate_interface,
)
from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
from fluffy_bassoon.meta_rl.robotwin_steering_screen import (
    build_plan, next_episode, run_complete_episode, validate_admission, verify_source_conventions,
    RLINF_COMMIT, ROBOTWIN_COMMIT,
    PHASE_CAPS, SOURCE_METHODS,
)


@pytest.fixture
def interface():
    return dict(schema="motus_coefficient_interface_v1", action_horizon=32, flow_steps=5,
        decoded_action_dim=14, public_dim=2062, coefficient_dim=4, coefficient_bound=1.,
        gripper_delta=0., joint_units="radians", gripper_indices=[6, 13], arm_joint_bound=.05,
        noise_delta_bound=1.5, kv_delta_bound=.1, basis_seed=20261031, kv_heads=1, head_dim=4,
        kv_layers=2, motor_sha256="a"*64, normalization_sha256="b"*64, joint_convention_sha256="c"*64,
        joint_lower=[-3.]*6+[0.]+[-3.]*6+[0.], joint_upper=[3.]*6+[1.]+[3.]*6+[1.])


def test_four_dimensional_bounds_and_reproducible_fixed_basis():
    proposed, applied = clipped_coefficients([2., -2., .5, 0.])
    np.testing.assert_array_equal(applied, [1., -1., .5, 0.])
    assert proposed[0] == 2
    np.testing.assert_array_equal(fixed_basis(7, (4, 32, 32)), fixed_basis(7, (4, 32, 32)))
    with pytest.raises(ValueError):
        clipped_coefficients([float("nan")]*4)


@pytest.mark.parametrize("fault", ["units", "range", "bound", "gripper"])
def test_interface_rejects_unverified_or_wrong_physical_conventions(interface, fault):
    if fault == "units": interface["joint_units"] = "normalized"
    if fault == "range": interface["joint_lower"][0] = 4.
    if fault == "bound": interface["arm_joint_bound"] = 2.
    if fault == "gripper": interface["gripper_delta"] = .02
    with pytest.raises(ValueError): validate_interface(interface)


def test_fixed_kv_zero_parity_padding_nonmutation_and_bound(interface):
    hook = FixedKVResidual(validate_interface(interface))
    cache = tuple((torch.randn(1, 5, 1, 4), torch.randn(1, 5, 1, 4)) for _ in range(2))
    saved = deepcopy(cache)
    mask = torch.tensor([[True, True, True, False, False]])
    features = torch.zeros(1, 2062)
    with torch.no_grad():
        zero = hook(cache, mask, features)
        hook.coefficients.copy_(torch.tensor([1., -1., .9, -.7]))
        edited = hook(cache, mask, features)
    assert all(torch.equal(a, b) for pair, other in zip(cache, zero) for a, b in zip(pair, other))
    assert all(torch.equal(a, b) for pair, other in zip(cache, saved) for a, b in zip(pair, other))
    for before, after in zip(cache[-1], edited[-1]):
        assert torch.equal(before[~mask], after[~mask])
        assert (after-before).abs().max() <= interface["kv_delta_bound"]+1e-6


def test_physical_constant_basis_has_in_support_odd_inverse_and_unchanged_grippers(interface):
    hook = FixedActionResidual(validate_interface(interface))
    base, features = torch.zeros(1, 32, 14), torch.zeros(1, 2062)
    with torch.no_grad():
        assert torch.equal(hook(base, features).actions, base)
        hook.coefficients[0] = .4
        forward = hook(base, features)
        hook.coefficients[0] = -.4
        inverse = hook(base, features)
    torch.testing.assert_close(forward.requested_delta, -inverse.requested_delta)
    assert forward.requested_delta.abs().max() <= .02
    assert torch.equal(forward.actions[..., [6, 13]], base[..., [6, 13]])
    assert torch.count_nonzero(hook.basis[0]) == 4
    assert hook.config()["constant_basis_joint_indices"] == [0, 1, 7, 8]
    # This proves algebraic support, NOT an actual simulator or clipped-actuator inverse.


@pytest.mark.parametrize("horizon,expected_cases,expected_bound", [(400, 3, 26800), (500, 2, 25000)])
def test_native_horizon_derivation_includes_all_confirmations_and_controls(horizon, expected_cases, expected_bound):
    plan = build_plan(dict.fromkeys(TASKS, horizon))
    assert plan["maximum_failure_cases"] == expected_cases
    assert plan["worst_case_actions"] == expected_bound <= 30000
    assert plan["maximum_actions_per_failure_case"] == (3*(3+2)+2)*horizon
    assert plan["reset_policy_actions"] == 0
    with pytest.raises(ValueError): build_plan(dict.fromkeys(TASKS, 2000))


def simulate_plan(all_search_fail=False):
    plan, records = build_plan(dict.fromkeys(TASKS, 400)), {}
    while (spec := next_episode(plan, records)) is not None:
        success = not (spec["stage"] == "development" and spec["task"] == TASKS[0] and spec["case"] < 3)
        if spec["stage"] == "search" and all_search_fail: success = False
        records[spec["id"]] = dict(spec=spec, success=success, primitive_steps=400,
            reset_identity=dict(actual_seed=spec["reset_seed"], observation_sha256="obs", features_sha256="features"))
    return plan, records


def test_predeclared_grid_has_one_task_67_complete_episodes_and_no_query_cases():
    plan, records = simulate_plan()
    assert len(records) == 16+3*(3*3+3*2+2) == 67
    assert sum(r["primitive_steps"] for r in records.values()) == plan["worst_case_actions"] == 26800
    assert {r["spec"]["task"] for r in records.values() if r["spec"]["stage"] != "development"} == {TASKS[0]}
    for case in range(3):
        confirms = [r for r in records.values() if r["spec"]["stage"] == "confirmation" and r["spec"]["case"] == case]
        for arm in ("native", *ARMS):
            assert len([r for r in confirms if r["spec"]["arm"] == arm]) == 2
        for repeat in range(2):
            assert len({r["spec"]["motor_seed"] for r in confirms if r["spec"]["id"].endswith(str(repeat))}) == 1
    assert not plan["locked_queries_opened"]


def test_all_failure_restricted_search_does_not_invent_teacher_or_repeat_more():
    _, records = simulate_plan(all_search_fail=True)
    assert len(records) == 16+27
    assert not any(r["spec"]["stage"] == "confirmation" for r in records.values())


class FakeControl:
    native_horizon = 3
    def __init__(self, failure=False):
        self.env = SimpleNamespace(corpus_writer=None)
        self.elapsed_steps, self.active, self.resets, self.failure = 0, False, 0, failure
        self.coefficients = np.zeros(4, np.float32)
        self.metadata = []
    def episode_limit_for_task(self, task): return 5
    def set_recording_context(self, value): pass
    def set_coefficients(self, arm, value): self.coefficients = np.clip(np.asarray(value, np.float32), -1, 1)
    def reset(self, task, seed, *, motor_seed):
        self.resets += 1; self.elapsed_steps = 0; self.active = True
        self.last_reset_info = dict(actual_seed=seed, observation_sha256="obs", features_sha256="features", reset_policy_actions=0)
        return np.zeros(2062, np.float32)
    def record_decision_metadata(self, value): self.metadata.append(value)
    def step(self, coefficients):
        if self.failure: raise RuntimeError("injected physical uncertainty")
        self.coefficients = np.clip(coefficients, -1, 1)
        actual = min(3, 5-self.elapsed_steps); self.elapsed_steps += actual; self.active = self.elapsed_steps < 5
        return np.ones(2062, np.float32), 0., False, not self.active, dict(primitive_steps=actual, success=False)
    def snapshot(self): return dict(elapsed_steps=self.elapsed_steps)


def episode_spec():
    return dict(id="fixture", task=TASKS[0], case=0, reset_seed=7, motor_seed=8,
                arm="noise", coefficients=[0.]*4, horizon=5)


def ledger(tmp_path, cap=20):
    return CampaignBudget(tmp_path/"ledger", campaign_id="fresh-test", total_actions=cap,
        source_caps={}, phase_caps={"a1":cap}, contract={}, run_id="test", source_commit="a"*40)


def test_episode_runner_finishes_native_horizon_and_controller_observes_actual_clipped_actions(tmp_path):
    control, observations = FakeControl(), []
    controller = SimpleNamespace(act=lambda obs: np.full(4, 2., np.float32),
        decision_metadata=lambda: dict(policy_version="sac:update:0:fixture", source_warmup=True),
        observe=lambda *args: observations.append(args))
    with ledger(tmp_path) as budget:
        row = run_complete_episode(control, episode_spec(), budget, run_id="test", controller=controller)
        assert row["complete"] and row["primitive_steps"] == 5 and row["chunks"] == 2
        assert budget.snapshot()["actual_actions"] == 5 and not budget.snapshot()["unresolved_decisions"]
    assert [args[6] for args in observations] == [3, 2]
    assert all(np.array_equal(args[1], np.ones(4)) for args in observations)
    assert all(meta["source_warmup"] for meta in control.metadata)


def test_full_episode_admission_rejects_short_allowance_before_reset(tmp_path):
    control = FakeControl()
    with ledger(tmp_path, cap=4) as budget, pytest.raises(RuntimeError, match="full native"):
        run_complete_episode(control, episode_spec(), budget, run_id="test")
    assert control.resets == 0


def test_source_episode_preserves_exact_lineage_method_in_canonical_ledger(tmp_path):
    method = "sac_calibration_minus_train0"
    control, spec = FakeControl(), dict(episode_spec(), method=method)
    with CampaignBudget(tmp_path/"ledger", campaign_id="fixture", total_actions=20,
            source_caps={method:10}, phase_caps={"source":20}, contract={}, run_id="test", source_commit="a"*40) as budget:
        run_complete_episode(control, spec, budget, run_id="test", phase="source")
        snapshot = budget.snapshot()
        assert snapshot["by_method"]["source/"+method] == 5
        assert "source/noise" not in snapshot["by_method"]


def test_failed_physical_call_retains_charge_and_blocks_automatic_reexecution(tmp_path):
    control = FakeControl(failure=True)
    with ledger(tmp_path) as budget:
        with pytest.raises(RuntimeError, match="uncertainty"):
            run_complete_episode(control, episode_spec(), budget, run_id="test")
        assert budget.snapshot()["pending_actions"] == 3
    with ledger(tmp_path) as budget:
        assert budget.snapshot()["uncertain_actions"] == 3
        with pytest.raises(RuntimeError, match="repeated/uncertain"):
            run_complete_episode(control, episode_spec(), budget, run_id="test")


def test_source_units_are_bound_to_actual_files_and_verified_ranges(tmp_path, interface):
    path = tmp_path/"joint.py"; path.write_text("# fixture only\n")
    urdf = tmp_path/"robot.urdf"
    indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    urdf.write_text('<robot>'+''.join(f'<joint name="arm{i}" type="revolute"><limit lower="-3" upper="3"/></joint>'
                                    for i in indices)+'</robot>')
    receipt = dict(arm_joint_units="radians", gripper_indices=[6, 13], gripper_mode="absolute",
        maximum_arm_delta=.05, joint_lower=interface["joint_lower"], joint_upper=interface["joint_upper"],
        rlinf_commit=RLINF_COMMIT, robotwin_commit=ROBOTWIN_COMMIT,
        reset_policy_action_bound=0,
        source_files=[dict(root="robotwin", path="joint.py", sha256=hashlib.sha256(path.read_bytes()).hexdigest())],
        joint_limit_records=[dict(root="robotwin", path="robot.urdf", joint=f"arm{i}", action_index=i,
            sha256=hashlib.sha256(urdf.read_bytes()).hexdigest()) for i in indices])
    verify_source_conventions(receipt, interface, dict(robotwin=tmp_path))
    bad = deepcopy(receipt); bad["joint_lower"][0] = -2.
    with pytest.raises(ValueError): verify_source_conventions(bad, interface, dict(robotwin=tmp_path))
    path.write_text("changed\n")
    with pytest.raises(ValueError, match="deployed source"):
        verify_source_conventions(receipt, interface, dict(robotwin=tmp_path))


def test_canonical_decision_coefficients_are_actual_clipped_actions_and_bound_to_mapping(interface):
    class Env:
        noise_shape, observation_dim, corpus_writer, active = (32, 32), 2062, None, False
        def __init__(self):
            self.backend = SimpleNamespace(model=SimpleNamespace(num_steps=5, device="cpu"), contract={},
                _kv_cache=((torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4)),)*2)
        def reset(self, task, seed): return np.zeros(2062, np.float32)
        def record_decision_metadata(self, value): self.metadata = value
        def step(self, noise): return np.zeros(2062, np.float32), 0., False, False, dict(last_valid_agent_request=np.ones(14, np.float32))
    env = Env()
    control = CoefficientSteering(env, interface, interface_sha256="d"*64)
    control.set_coefficients("noise", np.zeros(4))
    control.reset(TASKS[0], 4, motor_seed=5)
    assert np.array_equal(control.student_observation(np.zeros(2062))[-14:], np.zeros(14))
    control.record_decision_metadata(dict(steering_lineage_id="fixture", steering_condition_id="audit-private"))
    control.step(np.array([2., -.5, -2., 0.], np.float32))
    np.testing.assert_array_equal(env.metadata["steering_coefficients"], [1., -.5, -1., 0.])
    np.testing.assert_array_equal(env.metadata["proposed_steering_coefficients"], [2., -.5, -2., 0.])
    assert env.metadata["steering_interface_sha256"] == "d"*64
    assert env.metadata["steering_mapping_sha256"] == control.mapping_sha256()
    assert env.metadata["steering_condition_id"] == "audit-private"
    assert control.physical_basis().shape == (4, 14)
    assert np.array_equal(env.metadata["student_previous_agent_command"], np.zeros(14))
    assert np.array_equal(control.student_observation(np.zeros(2062))[-14:], np.ones(14))
    control.reset(TASKS[0], 4, motor_seed=5)
    assert np.array_equal(control.student_observation(np.zeros(2062))[-14:], np.zeros(14))
    canonical = control.mapping_sha256()
    control.set_coefficients("noise_coefficients", np.zeros(4))
    assert control.arm == "noise" and control.mapping_sha256() == canonical
    control.set_coefficients("native_KV_coefficients", np.zeros(4))
    assert control.arm == "kv"
    control.set_coefficients("decoded_ALOHA_joint_coefficients", np.zeros(4))
    assert control.arm == "action"


def test_allocation_ticket_cannot_replenish_shared_caps_or_bind_old_ledger(tmp_path):
    scope = dict(schema="frozen_pi05_endtoend_scope_v1", phase_id="fresh-pilot",
        phase_limits=dict(cumulative_gpu_seconds=28800, all_in_simulator_actions=300000,
                          outputs_gib=64, fresh_cpu_qualification_worker_seconds=1800),
        first_stage_limits_within_phase=dict(gpu_seconds=7200, all_in_simulator_actions=30000))
    output = tmp_path/"child"
    ticket = dict(scope_sha256="a"*64, source_commit="b"*40, hostname="job-fixture", gpu_count=1,
        output=str(output), phase_output_root=str(tmp_path), reserved_gpu_seconds=100,
        prior_phase_gpu_seconds=200, other_reserved_phase_gpu_seconds=0, prior_a1_gpu_seconds=200,
        action_ledger=dict(total_actions=300000, phase_caps=PHASE_CAPS, source_caps=dict.fromkeys(SOURCE_METHODS, 19200),
            campaign_id="fresh-pilot", contract=dict(scope_sha256="a"*64, phase_id="fresh-pilot",
                                                     motor_sha256="c"*64, candidate_tasks=list(TASKS))))
    def check(value): validate_admission(scope, value, scope_sha="a"*64, source="b"*40,
        output=output, hostname="job-fixture", interface=dict(motor_sha256="c"*64))
    check(ticket)
    for key in ("prior_phase_gpu_seconds", "prior_a1_gpu_seconds"):
        bad = deepcopy(ticket); bad[key] = 28800
        with pytest.raises(ValueError): check(bad)
    bad = deepcopy(ticket); bad["action_ledger"]["campaign_id"] = "old-closed-budget"
    with pytest.raises(ValueError, match="fresh shared phase"): check(bad)
