"""Decoded chunk-history causality and physical reset semantics."""
import numpy as np
import pytest

from fluffy_bassoon.meta_rl.chunk_history import ChunkHistoryEncoder, ChunkHistoryLayout, trial_tokens


def fixture():
    layout = ChunkHistoryLayout(3, 2, 2)
    data = dict(features=np.arange(12, dtype=np.float32).reshape(4, 3),
                bases=np.ones((4, 2, 2), dtype=np.float32),
                issued=np.arange(16, dtype=np.float32).reshape(4, 2, 2),
                delivered_masks=np.array([[1, 1], [1, 0], [1, 1], [1, 1]], dtype=np.bool_),
                rewards=np.array([0, 1, 0, 0], dtype=np.float32),
                terminated=np.array([0, 1, 0, 0], dtype=np.bool_),
                truncated=np.array([0, 0, 0, 1], dtype=np.bool_),
                physics_seconds=np.array([1, .1, 1, 1], dtype=np.float32),
                episode_starts=np.array([1, 0, 1, 0], dtype=np.bool_))
    return layout, data


def test_current_and_future_outcomes_do_not_reach_query():
    layout, data = fixture()
    original, _ = trial_tokens(layout, **data)
    data["issued"][2:] += 100
    data["rewards"][2:] += 10
    changed, _ = trial_tokens(layout, **data)
    np.testing.assert_array_equal(changed[:3], original[:3])
    assert not np.array_equal(changed[3], original[3])


@pytest.mark.parametrize("mode", ["full", "episode", "static"])
def test_boundary_feedback_and_attention_reset_agree(mode):
    layout, data = fixture()
    tokens, resets = trial_tokens(layout, mode=mode, **data)
    feedback = tokens[:, layout.current_dim:-1]
    np.testing.assert_array_equal(resets, [True, mode == "static", mode != "full", mode == "static"])
    assert not feedback[0].any()
    if mode == "full":
        np.testing.assert_array_equal(feedback[2, :4], data["issued"][1].reshape(-1))
        np.testing.assert_array_equal(feedback[2, -4:], np.array([1, 1, 0, .1], dtype=np.float32))
    elif mode == "episode":
        assert not feedback[2].any()
        assert feedback[1].any() and feedback[3].any()
    else:
        assert not feedback.any()


def test_online_snapshot_preserves_tail_and_trial_reset():
    layout, data = fixture()
    encoder = ChunkHistoryEncoder(layout)
    _, reset = encoder.query(data["features"][0], data["bases"][0], episode_start=True)
    assert reset
    command = data["issued"][1].copy()
    saved = command.copy()
    encoder.observe(command, data["delivered_masks"][1], 1., terminated=True,
                    truncated=False, physics_seconds=.1)
    command[:] = -999
    token, reset = encoder.query(data["features"][2], data["bases"][2], episode_start=True)
    assert not reset
    np.testing.assert_array_equal(token[layout.current_dim:layout.current_dim + 4], saved.reshape(-1))
    np.testing.assert_array_equal(layout.residual(saved, data["bases"][1]), saved - 1)
    with pytest.raises(ValueError, match="Observe"):
        encoder.reset_trial()
    encoder.observe(saved, data["delivered_masks"][1], 0., terminated=False,
                    truncated=True, physics_seconds=.1)
    encoder.reset_trial()
    token, reset = encoder.query(data["features"][0], data["bases"][0], episode_start=True)
    assert reset and not token[layout.current_dim:-1].any()


def test_rejects_unmeasured_noop_and_inconsistent_physical_boundaries():
    layout, data = fixture()
    encoder = ChunkHistoryEncoder(layout)
    encoder.query(data["features"][0], data["bases"][0], episode_start=True)
    with pytest.raises(ValueError, match="positive physical duration"):
        encoder.observe(data["issued"][0], data["delivered_masks"][0], 0.,
                        terminated=False, truncated=False, physics_seconds=0.)
    encoder.observe(data["issued"][0], data["delivered_masks"][0], 0.,
                    terminated=False, truncated=False, physics_seconds=.1)
    with pytest.raises(ValueError, match="disagrees"):
        encoder.query(data["features"][1], data["bases"][1], episode_start=True)
