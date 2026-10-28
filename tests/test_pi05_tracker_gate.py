"""Capacity admission also requires complete zero-reference repeat stability."""

from copy import deepcopy

import pytest

from fluffy_bassoon.meta_rl.pi05_tracker_gate import compare_zero_trajectories, review_tracker_gate


def outcome(bridge, zero, oracle):
    return dict(bridge=[bridge], zero_bridge=[zero], external_oracle=[oracle])


def trajectory():
    return dict(completed=True, native_noise_seed=810000, outcomes=[[1, 0]],
        plan=dict(theta_degrees=[-45., -45.], scenes=[[dict(episode_id=1), dict(episode_id=2)]],
                  reset_seeds=[17]),
        initial_state=dict(recurrent_memory="fresh None", previous_command=[[0.] * 7] * 2,
                           pending_reward=[0., 0.], attempt_index=0, reset_seeds=[17]),
        chunks=[dict(attempt=0, chunk=chunk, observation_sha256="a" * 64,
            prepared_observation_sha256="b" * 64, native_kv_sha256="c" * 64,
            feedback_sha256="d" * 64, initial_noise_sha256="e" * 64,
            decoded_actions=[[[0.] * 7] * 5] * 2, prepared_commands=[[[0.] * 7] * 5] * 2,
            issued_commands=[[[0.] * 7] * 5] * 2, success_per_step=[[0, 0]] * 5)
            for chunk in range(16)])


def stable_repeats():
    return {str(theta): compare_zero_trajectories(trajectory(), trajectory()) for theta in (-45., 0., 45.)}


def test_original_fitted_teacher_does_not_qualify_a_gain():
    raw = {"-45.0": outcome([1, 0], [0, 0], [1, 0]),
           "0.0": outcome([1, 1], [1, 1], [1, 1]),
           "45.0": outcome([0, 0], [1, 0], [1, 0])}
    receipt = review_tracker_gate(raw, stable_repeats())
    assert not receipt["passed"] and receipt["status"] == "unresolved"
    assert receipt["totals"] == dict(bridge=3, zero_bridge=3, external_oracle=4)


def test_small_capacity_screen_requires_each_perturbation_and_overall_gain():
    raw = {"-45.0": outcome([1, 0], [0, 0], [1, 0]),
           "0.0": outcome([1, 1], [1, 1], [1, 1]),
           "45.0": outcome([1, 0], [1, 0], [1, 0])}
    assert review_tracker_gate(raw, stable_repeats())["passed"]
    raw["45.0"] = outcome([0, 0], [0, 0], [1, 0])
    assert not review_tracker_gate(raw, stable_repeats())["passed"]


def test_incomplete_heldout_conditions_cannot_pass():
    assert review_tracker_gate({})["status"] == "incomplete"
    raw = {str(theta): outcome([1, 1], [0, 0], [1, 1]) for theta in (-45., 0., 45.)}
    del raw["45.0"]["external_oracle"]
    assert not review_tracker_gate(raw)["passed"]


def test_same_first_prefix_and_outcome_cannot_hide_later_trajectory_divergence():
    original, repeated = trajectory(), trajectory()
    assert compare_zero_trajectories(original, repeated)["stable"]
    repeated["chunks"][9]["observation_sha256"] = "f" * 64
    receipt = compare_zero_trajectories(original, repeated)
    assert receipt["complete"] and receipt["outcomes_identical"]
    assert not receipt["stable"] and not receipt["exact_trajectory_identical"]
    assert receipt["first_difference"] == dict(chunk_index=9, fields=["observation_sha256"])


@pytest.mark.parametrize("field", ["decoded_actions", "prepared_commands", "issued_commands"])
def test_all_actual_command_arrays_participate_in_stability_check(field):
    original, repeated = trajectory(), deepcopy(trajectory())
    repeated["chunks"][3][field][0][0][0] = .001
    receipt = compare_zero_trajectories(original, repeated)
    assert not receipt["stable"] and field in receipt["first_difference"]["fields"]


def test_repeat_requires_complete_provenance_outcomes_and_all_chunks():
    original = trajectory()
    for change in (dict(completed=False), dict(chunks=original["chunks"][:-1]),
                   dict(native_noise_seed=810001), dict(outcomes=[[0, 0]]), dict(initial_state={})):
        assert not compare_zero_trajectories(original, dict(deepcopy(original), **change))["stable"]
    incomplete = deepcopy(original)
    del incomplete["chunks"][4]["feedback_sha256"]
    assert not compare_zero_trajectories(original, incomplete)["complete"]


def test_nominal_capacity_gain_cannot_pass_without_stable_zero_repeats():
    raw = {str(theta): outcome([1, 1], [0, 0], [1, 1]) for theta in (-45., 0., 45.)}
    assert review_tracker_gate(raw)["status"] == "repeat_stability_unresolved"
    repeats = stable_repeats()
    repeats["45.0"]["stable"] = False
    receipt = review_tracker_gate(raw, repeats)
    assert not receipt["passed"] and receipt["status"] == "zero_reference_unstable"
    del repeats["45.0"]
    assert not review_tracker_gate(raw, repeats)["passed"]
    stopped = review_tracker_gate(raw, stable_repeats(), bounded_stop=True)
    assert not stopped["passed"] and stopped["status"] == "bounded_stop"
