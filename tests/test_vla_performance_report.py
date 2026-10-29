"""Outcome aggregation must not turn repeated attempts into independent scenes."""

import json

import pytest

from fluffy_bassoon.meta_rl.vla_performance_report import load_episodes, paired_comparison, report, summarize


def rows(method, outcomes):
    return [dict(method=method, split="eval", task_id=57, scene_seed=scene, trial_seed=0,
                 attempt=index + 1, success=bool(success), primitive_steps=400,
                 completed=True, cumulative_train_steps=1000 if method != "native" else 0)
            for scene, trial in enumerate(outcomes) for index, success in enumerate(trial)]


def test_attempt_success_and_retry_success_are_separate():
    result = summarize(rows("history", [[0, 1, 0, 1], [1, 0, 1, 0]]))
    assert result["independent_scenes"] == 2
    assert result["mean_trial_success"] == .5
    assert result["success_at_least_once"] == 1.
    assert result["final_attempt_success"] == .5
    assert result["by_attempt"][-1]["mean_cumulative_primitive_steps"] == 1600


def test_pairing_requires_same_scenes_and_attempt_budget():
    first = rows("history", [[0, 1], [1, 1]])
    second = rows("native", [[0, 0], [1, 0]])
    result = paired_comparison(first, second, repetitions=100)
    assert result["independent_scenes"] == 2
    assert result["mean_trial_success_difference"] == .5
    with pytest.raises(ValueError, match="identical task/scene"):
        paired_comparison(first, second[:2])
    assert paired_comparison(first[:2], second[:2])["bootstrap_95_percent_interval"] is None


def test_duplicate_records_cannot_inflate_sample_size(tmp_path):
    data = rows("native", [[0, 1]])
    path = tmp_path / "episodes.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in data + data[:1]))
    with pytest.raises(ValueError, match="Duplicate"):
        load_episodes(path)


def test_unequal_budgets_not_pooled_and_training_not_retry_trials():
    data = rows("history", [[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="unequal"):
        summarize(data[:-1])
    for row in data:
        row["split"] = "train"
    result = report(data)
    assert next(iter(result["cells"].values()))["completed_episodes"] == 4


def test_namespaced_history_pairs_dsrl_despite_different_source_costs():
    history = rows("ad_history", [[0, 1], [1, 1]])
    source = rows("dsrl", [[0, 0], [1, 0]])
    static = rows("ad_static", [[0, 0], [1, 1]])
    reset = rows("ad_reset", [[0, 0], [1, 0]])
    for row in history + static + reset:
        row["cumulative_train_steps"] = 8854
    for row in source:
        row["cumulative_train_steps"] = 4064
    paired = report(history + source + static + reset)["paired"]
    dsrl = paired["ad_history-dsrl/eval/train_steps=8854"]
    assert dsrl["mean_trial_success_difference"] == .5
    assert dsrl["candidate_cumulative_train_steps"] == 8854
    assert dsrl["reference_cumulative_train_steps"] == 4064
    assert paired["ad_history-ad_static/eval/train_steps=8854"]["mean_trial_success_difference"] == .25
    assert paired["ad_history-ad_reset/eval/train_steps=8854"]["independent_scenes"] == 2


def test_rl2_pairs_online_dsrl_and_separately_trained_reset():
    data = sum([rows(method, [[0, 1], [1, 1]]) for method in
                ("rl2", "rl2_reset", "rl2_trained_reset", "dsrl_online")], [])
    paired = report(data)["paired"]
    for reference in ("rl2_reset", "rl2_trained_reset", "dsrl_online"):
        assert paired[f"rl2-{reference}/eval/train_steps=1000"]["mean_trial_success_difference"] == 0.


def test_ambiguous_reference_checkpoints_are_not_selected_by_training_cost():
    history = rows("dpt_history", [[0, 1], [1, 1]])
    source = rows("dsrl", [[0, 0], [1, 0]])
    other_checkpoint = rows("dsrl", [[1, 1], [1, 1]])
    for row in other_checkpoint:
        row["cumulative_train_steps"] = 4000
    result = report(history + source + other_checkpoint)["paired"]["dpt_history-dsrl/eval/train_steps=1000"]
    assert "Ambiguous" in result["unpaired"]
    assert result["reference_cumulative_train_steps"] == [1000, 4000]
    assert "mean_trial_success_difference" not in result
