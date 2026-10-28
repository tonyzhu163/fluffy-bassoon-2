"""Oracles must establish useful experience headroom without leaking hidden state."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.diagnostics import bandit_baseline, gain_baseline
from fluffy_bassoon.meta_rl.environments import BernoulliBandit, HiddenGainReach


def fixed_gain_env(gain, horizon):
    env = HiddenGainReach(horizon=horizon)
    env._sample_hidden = lambda: setattr(env, "_gain", gain)
    env._start_attempt = lambda: np.array([0., 0., .75, 0.], dtype=np.float32)
    return env


def test_short_reach_requires_reusing_identified_dynamics_on_ambiguous_starts():
    carried, reset = [], []
    for gain in (-1., 1.):
        oracle = gain_baseline(fixed_gain_env(gain, 4), 1, "oracle")
        np.testing.assert_array_equal(oracle, np.ones(4))
        carried.append(gain_baseline(fixed_gain_env(gain, 4), 1, "identify_carry"))
        reset.append(gain_baseline(fixed_gain_env(gain, 4), 1, "identify_reset"))
    np.testing.assert_array_equal(np.mean(carried, 0), [.5, 1., 1., 1.])
    np.testing.assert_array_equal(np.mean(reset, 0), [.5, .5, .5, .5])


def test_long_reach_does_not_demonstrate_cross_attempt_headroom():
    for gain in (-1., 1.):
        rewards = gain_baseline(fixed_gain_env(gain, 8), 1, "identify_reset")
        np.testing.assert_array_equal(rewards, np.ones(4))


def test_bernoulli_oracle_is_a_common_noise_upper_bound():
    env = BernoulliBandit(attempts=32, arms=6)
    oracle_rewards, oracle_regret = bandit_baseline(env, 12, "oracle", np.random.default_rng(3))
    np.testing.assert_array_equal(oracle_regret, np.zeros(32))
    for mode in ("random", "win_stay", "thompson"):
        rewards, regret = bandit_baseline(env, 12, mode, np.random.default_rng(3))
        assert ((regret >= 0) & (regret <= 1)).all()
        assert (oracle_rewards >= rewards).all()


def test_diagnostics_reject_unsupported_modes_and_gain_families():
    with pytest.raises(ValueError, match="Unknown"):
        bandit_baseline(BernoulliBandit(), 1, "bad", np.random.default_rng(1))
    with pytest.raises(ValueError, match="signed unit"):
        gain_baseline(HiddenGainReach(gains=(.5, 1.)), 1, "identify_reset")


@pytest.mark.skipif(not os.environ.get("SLURM_STEP_ID"), reason="Diagnostics require an allocated Linux step")
def test_diagnostic_cli_keeps_raw_trials_and_source_identity(tmp_path):
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    output = tmp_path / "diagnostics"
    result = subprocess.run([sys.executable, "-m", "fluffy_bassoon.meta_rl.diagnostics",
                             "--source-commit", source, "--output-dir", str(output),
                             "--trials", "4", "--bandit-attempts", "8", "--gain-horizons", "4"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["source_commit"] == source
    assert summary["bandit"]["8"]["oracle"]["cumulative_expected_regret"] == 0
    assert summary["gain"]["4"]["oracle"]["mean_success"] == 1
    with np.load(output / "bernoulli-8-oracle.npz") as raw:
        assert raw["rewards"].shape == (4, 8) and not raw["regret"].any()
