"""Finite-arm selection needs separate actual repeats and honest query costs."""

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_noise_headroom import analyze, make_arms, source_task


def records():
    rows = []
    for state in (0, 1):
        for repeat in (0, 1):
            rows.append(dict(kind="native", arm=None, initial_state=state, repeat=repeat,
                success=False, primitive_steps=10, settling_steps=10,
                initial_observation_sha256=str(state), trajectory_sha256="native"))
            for arm in range(8):
                # The apparent winner in selection fails its actual repeat.
                success = (repeat == 0 and arm == 1) or (repeat == 1 and arm == 0)
                rows.append(dict(kind="fixed", arm=arm, initial_state=state, repeat=repeat,
                    success=success, primitive_steps=10, settling_steps=10,
                    initial_observation_sha256=str(state), trajectory_sha256=f"{state}-{arm}-{success}"))
    return rows


def test_arms_are_frozen_bounded_seeded_coordinates_with_a_zero_control():
    first = make_arms(0)
    assert first.shape == (8, 32) and first.dtype == np.float32
    np.testing.assert_array_equal(first[0], 0.)
    np.testing.assert_array_equal(first, make_arms(0))
    assert np.max(np.abs(first)) <= 1 and not np.array_equal(first, make_arms(1))


def test_headroom_never_substitutes_selection_max_for_actual_repeat_success():
    result = analyze(records())
    selected = result["repeat0_selected_repeat1_evaluation"]
    assert selected["success_rate"] == 0. and selected["gain_over_zero"] == -1.
    assert all(row["selected_arm"] == 1 for row in selected["trials"])
    assert result["repeatability"]["success_matches"] == 12
    assert result["actual_cost"]["episodes"] == 36
    assert result["actual_cost"]["primitive_steps"] == 360


def test_bandit_curve_counts_exploration_and_uses_only_measured_native_attempts():
    result = analyze(records())
    curves = result["simple_search_bandit"]
    assert curves[-1]["attempt_budget"] == 9 and curves[-1]["mean_primitive_steps"] == 90
    assert curves[2]["explored_arms"] == 2 and curves[2]["success_last"] == 0.
    assert curves[1]["native_same_attempt_budget"] is not None
    assert curves[2]["native_same_attempt_budget"] is None
    with pytest.raises(ValueError, match="36 actual"):
        analyze(records()[:-1])


def test_frozen_source_gate_rejects_test_and_excluded_tasks():
    task = dict(suite="libero_90", task_id=57, split="train", teacher_collection_allowed=True,
                source_state_indices=[0, 1])
    assert source_task([task], "libero_90", 57) is task
    for label in ("test", "test_auxiliary_excluded_90"):
        with pytest.raises(ValueError, match="no test/excluded"):
            source_task([dict(task, split=label)], "libero_90", 57)
