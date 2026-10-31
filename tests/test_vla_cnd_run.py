"""CND driver preflight contracts; run on allocated Linux only."""

import copy
import json
from types import SimpleNamespace

import pytest

from fluffy_bassoon.meta_rl.vla_cnd_run import compatible_prior, parser, validate_scene_banks


def test_cnd_driver_defaults_use_qualified_sac_features_and_prespecified_budgets():
    args = parser().parse_args(["--output", "/tmp/cnd", "--source-commit", "a" * 40,
        "--openpi-root", "/tmp/openpi", "--dsrl-root", "/tmp/dsrl", "--checkpoint", "/tmp/motor",
        "--prior-recovery", "/tmp/prior", "--prior-contract", "/tmp/contract"])
    assert args.feature_interface == "sac" and args.history_updates == 500 and args.attempts == 4
    assert args.anneal_decisions == [35, 55] and args.min_teacher_success == .7
    assert not args.qualify_only
    validate_scene_banks(args.train_states, args.validation_states, args.calibration_states,
                         args.eval_states, initial_states=50)


def test_pi05_qualification_can_stop_before_automatic_cnd_collection():
    args = parser().parse_args(["--output", "/tmp/pi05-gate", "--source-commit", "a" * 40,
        "--openpi-root", "/tmp/openpi", "--dsrl-root", "/tmp/dsrl", "--checkpoint", "/tmp/pi05",
        "--prior-recovery", "/tmp/pi05-teacher", "--prior-contract", "/tmp/contract",
        "--model-config", "pi05_libero", "--chunk", "10", "--qualify-only"])
    assert args.qualify_only and args.model_config == "pi05_libero" and args.chunk == 10


def test_calibration_and_deployment_must_not_share_scenes():
    with pytest.raises(ValueError, match="disjoint"):
        validate_scene_banks([0], [1], [2], [2], initial_states=50)


def test_prior_migration_allows_only_readable_added_metadata():
    prior = dict(motor={"weights": "same"}, noise="same", dsrl_commit="same", chunk=20,
                 horizon=400, simulator={"task_id": 57, "bddl_sha256": "same"})
    current = copy.deepcopy(prior)
    current["simulator"].update(bddl_file="task.bddl", problem_folder="libero_90")
    compatible_prior(prior, current)
    current["simulator"]["bddl_sha256"] = "changed"
    with pytest.raises(ValueError, match="scientific interface"):
        compatible_prior(prior, current)


@pytest.mark.parametrize("successes,qualified", [(5, False), (6, True)])
def test_teacher_gate_uses_actual_completed_episodes_and_preserves_receipts(tmp_path, monkeypatch,
                                                                         successes, qualified):
    from fluffy_bassoon.meta_rl import vla_cnd_run as driver
    checkpoint = tmp_path / "source.pkl"
    checkpoint.write_bytes(b"immutable-checkpoint")
    args = SimpleNamespace(output=tmp_path, calibration_states=list(range(24, 32)),
                           min_teacher_success=.7, seed=0, chunk=20)
    sim = SimpleNamespace(metadata={"settle_steps": 10})
    seen = []
    def episode(sim, motor, choose, **kwargs):
        seen.append(kwargs["initial_state"])
        return [], dict(success=len(seen) <= successes, primitive_steps=20)
    monkeypatch.setattr(driver, "run_episode", episode)
    monkeypatch.setattr(driver, "learner_fingerprint", lambda _: "unchanged")
    result = driver.qualify_teacher(args, sim, None, None, checkpoint)
    assert result["qualified"] is qualified and result["completed_episodes"] == 8
    assert result["successes"] == successes and result["primitive_steps"] == 160
    assert result["settling_steps"] == 80 and result["teacher_unchanged"]
    assert seen == list(range(24, 32))
    receipts = [json.loads(line) for line in (tmp_path / "teacher-calibration-episodes.jsonl").read_text().splitlines()]
    assert len(receipts) == 8 and sum(row["success"] for row in receipts) == successes
