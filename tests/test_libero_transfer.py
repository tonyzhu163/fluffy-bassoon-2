"""Leakage regressions for the independent 40 -> 90 experiment."""
from copy import deepcopy

import pytest

from fluffy_bassoon import libero_transfer as lt


def record(key, suite, goal, instruction):
    return dict(task_key=key, suite=suite, goal_binding_sha256=goal, normalized_instruction=instruction)


def test_transitive_overlap_cannot_enter_strict_test():
    rows = [record("train", "libero_goal", "g1", "i1"),
            record("same_goal", "libero_90", "g1", "i2"),
            record("linked_instruction", "libero_90", "g2", "i2"),
            record("unseen", "libero_90", "g3", "i3")]
    result = lt.partition(rows)
    assert [r["split"] for r in result] == ["train40", "test90_overlap_control", "test90_overlap_control", "test90_strict"]
    assert all(not r["meta_gradient_allowed"] and not r["selection_allowed"] for r in result[1:])
    assert "split" not in rows[0]


def fixture_manifest():
    rows = [record(f"{s}:{i}", s, f"{s}-g-{i}", f"{s}-i-{i}")
            for s, n in [(s, 10) for s in lt.TRAIN_SUITES] + [(lt.TEST_SUITE, 90)] for i in range(n)]
    rows[40]["goal_binding_sha256"] = rows[0]["goal_binding_sha256"]
    rows = lt.partition(rows)
    value = dict(tasks=rows, counts={"train40": 40, "test90_strict": 89, "test90_overlap_control": 1})
    value["metadata_sha256"] = lt.content_hash(value)
    return value


def test_rehashed_leakage_is_rejected():
    value = fixture_manifest()
    lt.validate(value)
    altered = deepcopy(value)
    altered["tasks"][40]["split"] = "test90_strict"
    altered["metadata_sha256"] = lt.content_hash({k: v for k, v in altered.items() if k != "metadata_sha256"})
    with pytest.raises(ValueError, match="Invalid split"):
        lt.validate(altered)


def test_tampering_and_duplicate_task_ids_are_rejected():
    value = fixture_manifest()
    value["tasks"][0]["normalized_instruction"] = "edited"
    with pytest.raises(ValueError, match="metadata_sha256"):
        lt.validate(value)
    value = fixture_manifest()
    value["tasks"][1]["task_key"] = value["tasks"][0]["task_key"]
    value["metadata_sha256"] = lt.content_hash({k: v for k, v in value.items() if k != "metadata_sha256"})
    with pytest.raises(ValueError, match="distinct"):
        lt.validate(value)
