import copy
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent


def load_signals():
    spec = spec_from_file_location("_signals_under_test", HERE / "signals.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Signals


Signals = load_signals()
contract = json.loads((HERE.parent / "frozen-vla-tracking-feedback-20261123-r1/source-model.json").read_text())
prior = np.array(contract["prior_matrix"])


def raw(n=7, angle=0, rgb=255, xyz=(0, 0, 0), jaw=0.03):
    return {"robot0_joint_pos": np.full(n, angle),
            "agentview_image": np.full((32, 32, 3), rgb, np.uint8),
            "robot0_eef_pos": np.array(xyz),
            "robot0_gripper_qpos": np.array([jaw / 2, -jaw / 2])}


def drawer(x, size=128, **kwargs):
    obs = raw(**kwargs)
    image = np.full((size, size, 3), 180, np.uint8)
    # Draw the source cabinet scanlines in unflipped native camera storage.
    rows = (np.arange(size) + 0.5) * 128 / size
    image[::-1][(rows >= 73) & (rows < 81), :int(x * size / 128)] = 30
    obs["agentview_image"] = image
    return obs


def support(n=7):
    # Three orthogonal calibration commands with a target response distinct
    # from the source. Include the preceding command in every response.
    x = np.column_stack([np.eye(3), np.vstack([np.zeros(3), np.eye(3)[:2]])])
    return {"inputs": x, "response": x @ prior - 0.03 * np.eye(3), "joints": np.zeros((3, n))}


def chunk():
    return np.tile([0.98, 0.05, -0.22, 1/3, -0.52, 0.05, 1], (5, 1))


@pytest.mark.parametrize("n", [6, 7])
def test_posture_conditioned_fit_matches_ridge_and_damped_inverse(n):
    s = support(n)
    s["joints"][1, 0] = np.pi  # orthogonal joint kernel suppresses this row only
    query = raw(n)
    sig = Signals(s, query)
    x = s["inputs"][[0, 2]]
    y = s["response"][[0, 2]]
    ref_fit = prior.copy()
    ref_fit[:3] += np.linalg.solve(x[:, :3].T @ x[:, :3] + np.eye(3),
                                   x[:, :3].T @ (y - x @ prior))
    body = ref_fit[:3] + ref_fit[3:]
    source = prior[:3] + prior[3:]
    inverse = np.linalg.solve(body @ body.T + contract["source_residual_power"] * np.eye(3), body)
    matrix = np.eye(3) + (inverse @ (source - body).T).T
    a = chunk()
    orig = a.copy()
    corrected = sig.correct(query, a)
    np.testing.assert_allclose(corrected[:, :3], np.clip(a[:, :3] @ matrix, -1, 1), atol=1e-7)
    assert corrected[0, 0] == 1 and np.all(np.abs(corrected) <= 1)
    np.testing.assert_array_equal(corrected[:, 3:], a[:, 3:])
    np.testing.assert_array_equal(a, orig)
    assert not np.shares_memory(corrected, a)
    # Same physical joint posture across the 2π wrap.
    np.testing.assert_allclose(sig.correct(raw(n, angle=2*np.pi), a), corrected, atol=1e-7)
    np.testing.assert_allclose(sig.correct(raw(n, angle=np.pi), a), a, atol=1e-7)
    motion, progress = sig.features(query, a)
    np.testing.assert_allclose(motion[9:12], np.r_[a[0, :3], np.zeros(3)] @ ref_fit)
    assert motion.shape == (19,) and progress.shape == (13,)
    assert np.isfinite(motion).all() and np.isfinite(progress).all()


def test_history_uses_executed_delta_and_actual_preceding_command():
    before = drawer(32)
    after = drawer(54, angle=np.pi, xyz=(0.05, 0.02, -0.01), jaw=0.01)
    sig = Signals(support(), before)
    fork = copy.deepcopy(sig)
    issued = -chunk()
    pending = chunk() * 0.8
    # Deliberately different command, followed by a jump to the prior-only posture.
    at_boundary = sig.features(before, issued)[0]
    sig.features(after, pending)
    sig.correct(after, pending)
    np.testing.assert_array_equal(sig.features(before, issued)[0], at_boundary)
    sig.observe(before, issued, after)
    motion, progress = sig.features(after, pending)
    achieved = after["robot0_eef_pos"] - before["robot0_eef_pos"]
    np.testing.assert_allclose(motion[:3], pending[0, :3])
    np.testing.assert_allclose(motion[3:6], issued[0, :3])
    np.testing.assert_allclose(motion[9:12], np.r_[pending[0, :3], issued[0, :3]] @ prior)
    np.testing.assert_allclose(motion[12:15], achieved)
    np.testing.assert_allclose(motion[15:18], at_boundary[9:12] - achieved)
    assert motion[-1] == 1
    assert progress[0] > 0.5 > progress[6]
    np.testing.assert_allclose(progress[[5, 11, 12]], [0.01, 0.03, 0.8])
    assert fork.features(after, pending)[0][-1] == 0
    # MuJoCo buffers may be overwritten after observe returns.
    before["agentview_image"].fill(0)
    issued.fill(0)
    np.testing.assert_array_equal(sig.features(after, pending)[0], motion)
    np.testing.assert_array_equal(sig.features(after, pending)[1], progress)
    sig.observe(after, pending[:2], after)
    np.testing.assert_array_equal(sig.features(after, pending)[0], motion)


@pytest.mark.parametrize("size", [128, 256])
def test_current_front_crosses_native_boundary_in_both_directions(size):
    reset = drawer(32, size)
    sig = Signals(support(), reset)
    previous = reset
    # The partly reclosed front is still 18 pixels from reset.
    for x, opened in ((48, False), (53, True), (50, False), (54, True)):
        obs = drawer(x, size)
        old_progress = sig.features(previous, chunk())[1][:6]
        sig.observe(previous, chunk(), obs)
        progress = sig.features(obs, chunk())[1]
        np.testing.assert_array_equal(progress[6:12], old_progress)
        assert progress[0] == float(opened)
        assert sig.cue(obs) == ("put the bowl in the top drawer" if opened else "open the top drawer")
        previous = obs


def test_subpixel_front_votes_and_missing_edges():
    reset, partial = drawer(32), drawer(51)
    # The first bright pixel at x=51 has gray 110. Interpolation at gray
    # 90 moves its edge by 0.75 pixels versus 0.4 at the reset's gray 180.
    partial["agentview_image"][::-1][73:81, 51] = 110
    thresholds = json.loads((HERE / "drawer-calibration.json").read_bytes())["crossing_px_by_demo"]
    expected = sum(v < 19.35 for v in thresholds.values()) / len(thresholds)
    assert 0 < expected < 1
    sig = Signals(support(), reset)
    assert sig.features(partial, chunk())[1][0] == pytest.approx(expected)
    assert sig.cue(raw(rgb=0)) == "open the top drawer"
    assert Signals(support(), raw(rgb=255)).cue(partial) == "open the top drawer"


def test_zero_excitation_first_window_and_public_only_inputs():
    s = support()
    s["inputs"][:] = 0
    reset = raw()
    sig = Signals(s, reset)
    a = chunk()
    np.testing.assert_allclose(sig.correct(reset, a), a, atol=1e-7)
    # Native observations may carry auxiliary labels in a wrapper. They never
    # affect these features or enter a command/observation update.
    extra = dict(reset, arm=object(), success=object(), object_state=object(), reward=object())
    other = Signals(dict(s, case=object()), extra)
    for observed, expected in zip(other.features(extra, np.vstack([a, -a])), sig.features(reset, a), strict=True):
        np.testing.assert_array_equal(observed, expected)
    other.observe(extra, a, extra)
    sig.observe(reset, a, reset)
    np.testing.assert_array_equal(other.features(extra, a)[0], sig.features(reset, a)[0])


def test_absolute_load_is_independent_of_cwd_and_other_workers(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    modules = {name: types.ModuleType(name) for name in ("common", "motion")}
    path = sys.path.copy()
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    sig = load_signals()(support(), raw())
    assert sig.cue(raw()) == "open the top drawer"
    assert sys.path == path and all(sys.modules[name] is module for name, module in modules.items())


def test_rejects_nonfinite_support_and_invalid_actions_before_history_updates():
    s = support()
    s["response"][0, 0] = np.nan
    with pytest.raises(ValueError):
        Signals(s, raw())
    sig = Signals(support(), raw())
    for invalid in (np.empty((0, 7)), np.zeros((5, 6)), np.full((5, 7), np.inf), np.full((5, 7), 1.2)):
        with pytest.raises(ValueError):
            sig.correct(raw(), invalid)
    with pytest.raises(ValueError):
        sig.observe(raw(), np.zeros((6, 7)), raw())
    assert sig.features(raw(), chunk())[0][-1] == 0
