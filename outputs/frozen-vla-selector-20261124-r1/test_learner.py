"""Synthetic fork fixtures exercise selector semantics only."""

import itertools
import json

import pytest

from learner import choose, fit


KINDS = ("factorized", "generic")


def fork(motion, progress, wins, arm="Panda", seed=0):
    return {
        "motion": [motion], "progress": [progress],
        "case": {"arm": arm, "suite": "unit", "task_id": 0, "init_state_id": 0, "seed": seed},
        "outcomes": {key: key in wins for key in ("continue", "correct", "cue")},
    }


def isolated_forks():
    return [
        fork(m, p, {winner}, arm=arm, seed=i)
        for arm in ("Panda", "UR5e")
        for i, (m, p, winner) in enumerate(((1, -1, "correct"), (-1, 1, "cue"), (-1, -1, "continue")))
    ]


@pytest.mark.parametrize("kind", KINDS)
def test_isolated_decisions_after_json_round_trip(kind):
    model = json.loads(json.dumps(fit(isolated_forks(), kind), allow_nan=False))
    assert model["kind"] == kind
    assert choose(model, [0.9], [-0.9]) == "correct"
    assert choose(model, [-0.9], [0.9]) == "cue"
    assert choose(model, [-0.9], [-0.9]) == "continue"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("baseline,correct,cue", list(itertools.product([False, True], repeat=3)))
def test_all_single_fork_outcomes_and_ties(kind, baseline, correct, cue):
    row = fork(0, 0, set())
    row["outcomes"] = {"continue": baseline, "correct": correct, "cue": cue}
    model = fit([row], kind)
    # With constant features the fitted intercept equals the observed paired gain.
    for action in ("correct", "cue"):
        assert model["heads"][action]["intercept"] == int(row["outcomes"][action]) - int(baseline)
    expected = "continue"
    if not baseline and correct != cue:
        expected = "correct" if correct else "cue"
    assert choose(model, [0], [0]) == expected


def test_matched_capacity_and_feature_access():
    # Only progress predicts correction. Motion is constant, so the factorized
    # correction head has zero mean gain while the generic head can use progress.
    rows = [
        fork(0, -1, {"continue"}),
        fork(0, 1, {"correct"}, arm="UR5e"),
    ]
    models = [fit(rows, kind) for kind in KINDS]
    for model in models:
        fitted = sum(len(head["slopes"]) + 1 for head in model["heads"].values())
        assert fitted == model["trainable_parameters"] == 34
    specialized, pooled = models
    assert choose(specialized, [0], [1]) == "continue"
    assert choose(pooled, [0], [1]) == "correct"
    assert choose(pooled, [0], [-1]) == "continue"


@pytest.mark.parametrize("kind", KINDS)
def test_split_excludes_held_out_features_and_labels(kind):
    rows = [fork(0, 2, {"correct"}), fork(8, 4, {"cue"}, arm="UR5e")]
    mixed = [*rows, fork(1e8, -1e8, {"continue"}, arm="Kinova3")]
    model = fit(mixed, kind)
    assert model == fit(rows, kind)
    assert model["center"] == [4, 3]
    assert model["inverse_scale"] == [0.25, 1]
    with pytest.raises(ValueError, match="at least one paired source row"):
        fit(mixed[2:], kind)


@pytest.mark.parametrize("key,value", [
    ("motion", []), ("motion", [[1]]), ("motion", [1, 2]), ("motion", ["1"]),
    ("progress", [float("nan")]), ("progress", [float("inf")]),
    ("outcomes", {"continue": False, "correct": 1, "cue": False}),
    ("outcomes", {"continue": True, "correct": False}),
    ("outcomes", {"continue": True, "correct": False, "cue": False, "invalid": True}),
    ("case", {"arm": "Panda"}),
])
def test_invalid_source_rows(key, value):
    rows = isolated_forks()
    rows[-1][key] = value
    with pytest.raises(ValueError):
        fit(rows, "generic")


def test_invalid_queries_and_kind():
    rows = isolated_forks()
    model = fit(rows, "factorized")
    with pytest.raises(ValueError, match="dimensions"):
        choose(model, [1, 2], [3])
    with pytest.raises(ValueError, match="non-finite"):
        choose(model, [0], [float("inf")])
    with pytest.raises(ValueError, match="kind"):
        fit(rows, "unknown")
