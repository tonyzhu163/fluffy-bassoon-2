"""Continuation cost ceiling: complete episode rounding must remain bounded."""

import pytest

from fluffy_bassoon.meta_rl.vla_teacher_continue import continuation_target, parser


def test_continuation_rounding_reserves_whole_final_episode():
    target = continuation_target(4064, 20000, 400)
    assert target == 23665
    # The longest last episode can begin at target - 1, still below the ceiling.
    assert target - 1 + 400 - 4064 == 20000
    with pytest.raises(ValueError, match="complete final episode"):
        continuation_target(4064, 399, 400)


def test_continuation_does_not_expose_new_optimizer_or_scene_bank_defaults():
    args = parser().parse_args(["--output", "/tmp/continue", "--source-commit", "a" * 40,
        "--openpi-root", "/tmp/openpi", "--dsrl-root", "/tmp/dsrl", "--checkpoint", "/tmp/motor",
        "--prior-recovery", "/tmp/prior", "--prior-contract", "/tmp/contract"])
    assert args.additional_max_steps == 20000 and args.calibration_states == list(range(24, 32))
    assert not hasattr(args, "batch_size") and not hasattr(args, "utd")
