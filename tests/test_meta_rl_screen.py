"""The matched screen seals its cell matrix, budgets, recovery and ablations."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from fluffy_bassoon.meta_rl.screen import (SCREEN_TASKS, VARIANTS, command_options, load_manifest,
                                          matching_width, parser, resolve_task_defaults)


def test_training_options_keep_booleans_as_switches_and_paths_as_arguments():
    assert command_options(dict(source_commit="abc", output_dir=Path("a path"),
                                bounded_synchronisation=True, decay_per_observation=False)) == [
        "--source-commit", "abc", "--output-dir", "a path", "--bounded-synchronisation"]


def test_mutated_manifest_is_rejected_before_any_work(tmp_path):
    from fluffy_bassoon.evaluation_stages import sha256
    path = tmp_path / "manifest.json"
    path.write_text('{"version": 1}')
    path.with_suffix(".sha256").write_text(sha256(path))
    path.write_text('{"version": 2}')
    with pytest.raises(ValueError, match="checksum"):
        load_manifest(path)


@pytest.mark.parametrize("label,core,ticks", VARIANTS)
def test_parameter_matching_fits_every_screen_core_and_next_width_exceeds_cap(label, core, ticks):
    from fluffy_bassoon.meta_rl.cores import build_policy
    from fluffy_bassoon.meta_rl.environments import BernoulliBandit
    env = BernoulliBandit(attempts=64, arms=6)
    settings = dict(ticks=ticks, stm_length=32, ltm_tokens=32, synapse_depth=8,
                    bounded_synchronisation=core == "persistent_ctm",
                    decay_per_observation=core == "persistent_ctm",
                    transformer_memory_length=128, transformer_layers=2, looped_memory_tokens=8)
    selected = matching_width(core, env, settings, 100_000)
    assert 85_000 <= selected["parameters"] <= 100_000
    next_config = dict(selected["core_config"], width=selected["width"] + 4)
    assert sum(p.numel() for p in build_policy(core, next_config, env).parameters()) > 100_000
    assert selected["retained_state_bytes_per_lane"] > 0
    env.close()


def _plan_args(*options):
    return parser().parse_args(["plan", "--source-commit", "abc", "--output-dir", "out", *options])


@pytest.mark.parametrize("task", SCREEN_TASKS)
def test_task_defaults_fill_attempts_horizon_offset_and_drop_arms_for_grids(task):
    args = _plan_args("--task", task)
    arms = resolve_task_defaults(args)
    assert args.attempts == SCREEN_TASKS[task]["attempts"]
    assert args.reward_offset == SCREEN_TASKS[task]["reward_offset"]
    grid = task in ("darkroom", "keydoor")
    assert args.horizon == {"darkroom": 20, "keydoor": 50}.get(task, 1)
    assert arms == (None if grid else 6)


def test_explicit_screen_settings_override_task_defaults_and_bandits_reject_horizons():
    args = _plan_args("--task", "darkroom", "--attempts", "8", "--horizon", "10", "--reward-offset", ".25")
    resolve_task_defaults(args)
    assert (args.attempts, args.horizon, args.reward_offset) == ([8], 10, .25)
    with pytest.raises(ValueError, match="one decision"):
        resolve_task_defaults(_plan_args("--task", "bernoulli", "--horizon", "2"))


@pytest.mark.parametrize("task", ("darkroom", "keydoor"))
@pytest.mark.parametrize("label,core,ticks", VARIANTS)
def test_parameter_matching_fits_every_screen_core_on_grid_trials(task, label, core, ticks):
    from fluffy_bassoon.meta_rl.train import make_env
    attempts, horizon = max(SCREEN_TASKS[task]["attempts"]), {"darkroom": 20, "keydoor": 50}[task]
    env = make_env(task, attempts, horizon)
    settings = dict(ticks=ticks, stm_length=32, ltm_tokens=32, synapse_depth=8,
                    bounded_synchronisation=core == "persistent_ctm",
                    decay_per_observation=core == "persistent_ctm",
                    transformer_memory_length=attempts * horizon, transformer_layers=2,
                    looped_memory_tokens=8)
    selected = matching_width(core, env, settings, 100_000)
    assert 85_000 <= selected["parameters"] <= 100_000
    env.close()


@pytest.mark.skipif(not os.environ.get("SLURM_STEP_ID"), reason="Screen CLI requires an allocated Linux step")
def test_screen_cli_runs_a_sealed_cell_and_records_frozen_ablations(tmp_path):
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    root = tmp_path / "screen"
    commands = (["plan", "--source-commit", source, "--output-dir", str(root),
                 "--attempts", "2", "--seeds", "1000", "--updates", "1", "--num-envs", "2",
                 "--eval-trials", "2", "--ppo-epochs", "1", "--minibatch-trials", "2", "--device", "cpu"],
                ["run", "--manifest", str(root / "manifest.json"), "--index", "5", "--attempt-id", "r1"])
    for options in commands:
        result = subprocess.run([sys.executable, "-m", "fluffy_bassoon.meta_rl.screen", *options],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["members"]) == 7
    attempt = next((root / "members").glob("005-*/r1"))
    receipt = json.loads((attempt / "receipt.json").read_text())
    assert receipt["completed"] and receipt["cost"]["measured_updates"] == 1
    diagnostic = json.loads((attempt / "diagnostics/summary.json").read_text())
    assert diagnostic["model_tensors_unchanged"]
    assert len(diagnostic["conditions"]) == 12
    assert diagnostic["conditions"]["greedy_reward_blind"]["reward_blind"]
    assert diagnostic["conditions"]["greedy_reset"]["reset_each_attempt"]


@pytest.mark.skipif(not os.environ.get("SLURM_STEP_ID"), reason="Screen CLI requires an allocated Linux step")
def test_screen_cli_runs_a_sealed_dark_room_cell_without_bandit_arms(tmp_path):
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    root = tmp_path / "screen"
    commands = (["plan", "--source-commit", source, "--output-dir", str(root), "--task", "darkroom",
                 "--attempts", "2", "--horizon", "4", "--seeds", "1000", "--updates", "1",
                 "--num-envs", "2", "--eval-trials", "2", "--ppo-epochs", "1", "--minibatch-trials", "2",
                 "--device", "cpu"],
                ["run", "--manifest", str(root / "manifest.json"), "--index", "1", "--attempt-id", "r1"])
    for options in commands:
        result = subprocess.run([sys.executable, "-m", "fluffy_bassoon.meta_rl.screen", *options],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((root / "manifest.json").read_text())
    assert {(m["task"], m["horizon"], m["arms"]) for m in manifest["members"]} == {("darkroom", 4, None)}
    assert manifest["training"]["reward_offset"] == 0.
    attempt = next((root / "members").glob("001-*/r1"))
    receipt = json.loads((attempt / "receipt.json").read_text())
    assert receipt["completed"] and "--arms" not in receipt["command"]
    config = json.loads((attempt / "training/config.json").read_text())
    assert (config["contract"]["task"], config["contract"]["horizon"]) == ("darkroom", 4)
    diagnostic = json.loads((attempt / "diagnostics/summary.json").read_text())
    assert diagnostic["conditions"]["greedy_reward_blind"]["reward_blind"]
