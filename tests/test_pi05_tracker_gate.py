"""Live capacity admission rejects a nominal average hiding a failed perturbation."""

from fluffy_bassoon.meta_rl.pi05_tracker_gate import review_tracker_gate


def outcome(bridge, zero, oracle):
    return dict(bridge=[bridge], zero_bridge=[zero], external_oracle=[oracle])


def test_original_fitted_teacher_does_not_qualify_a_gain():
    raw = {"-45.0": outcome([1, 0], [0, 0], [1, 0]),
           "0.0": outcome([1, 1], [1, 1], [1, 1]),
           "45.0": outcome([0, 0], [1, 0], [1, 0])}
    receipt = review_tracker_gate(raw)
    assert not receipt["passed"] and receipt["status"] == "unresolved"
    assert receipt["totals"] == dict(bridge=3, zero_bridge=3, external_oracle=4)


def test_small_capacity_screen_requires_each_perturbation_and_overall_gain():
    raw = {"-45.0": outcome([1, 0], [0, 0], [1, 0]),
           "0.0": outcome([1, 1], [1, 1], [1, 1]),
           "45.0": outcome([1, 0], [1, 0], [1, 0])}
    assert review_tracker_gate(raw)["passed"]
    raw["45.0"] = outcome([0, 0], [0, 0], [1, 0])
    assert not review_tracker_gate(raw)["passed"]


def test_incomplete_heldout_conditions_cannot_pass():
    assert review_tracker_gate({})["status"] == "incomplete"
    raw = {str(theta): outcome([1, 1], [0, 0], [1, 1]) for theta in (-45., 0., 45.)}
    del raw["45.0"]["external_oracle"]
    assert not review_tracker_gate(raw)["passed"]
