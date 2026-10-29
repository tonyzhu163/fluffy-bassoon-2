"""Whole-case data amount is distinct from teacher cost and AD admissibility."""
import json
from pathlib import Path

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_history import FIELDS, load_dataset, save_dataset
from fluffy_bassoon.meta_rl.vla_source40_data_scales import make_scales, prefix_selection
from fluffy_bassoon.meta_rl.vla_source40_schedule import content_hash
from fluffy_bassoon.meta_rl.vla_source40_training import load_assembly
from test_vla_source40_training import source


def warmup_then_learning(path):
    """First complete case is fixed warmup; second contains actual clock changes."""
    index = json.loads(path.read_text())
    for objective in ("ad", "dpt"):
        selected = index["publications"][-1]["datasets"][objective]
        dataset = path.parent / selected["path"]
        trials, manifest = load_dataset(dataset, expected_sha256=selected["sha256"])
        trials[0]["teacher_steps"] = np.array([0, 0, 1, 2], dtype=np.int64)
        dataset.unlink()
        dataset.with_suffix(".json").unlink()
        rewritten = save_dataset(dataset, trials, objective=objective, metadata=manifest["metadata"])
        selected["sha256"] = rewritten["sha256"]
    path.write_text(json.dumps(index))
    return path


def histories(tmp_path):
    train = warmup_then_learning(source(tmp_path / "train", "train"))
    validation = source(tmp_path / "validation", "validation")
    return train, validation


def test_prefix_uses_all_episodes_of_first_cases_not_transition_quota():
    index = dict(episode_receipts=[dict(case_id=case, episode_in_case=episode,
        chunks=chunks, initial_state=state, initial_state_vector_sha256=str(state) * 64)
        for case, state, values in (("a", 0, (2, 3)), ("b", 1, (4, 1)))
        for episode, chunks in enumerate(values)])
    result = prefix_selection(index, 1)
    assert result["case_ids"] == ["a"]
    assert result["episodes"] == 2 and result["transitions"] == 5
    assert result["nominal_state_ids"] == [0]
    with pytest.raises(ValueError, match="positive"):
        prefix_selection(index, 3)
    with pytest.raises(ValueError, match="positive"):
        prefix_selection(index, True)


def test_raw_warmup_prefix_is_bc_dpt_not_fabricated_ad(tmp_path):
    train, validation = histories(tmp_path)
    result = make_scales(train, validation, tmp_path / "scales", case_counts=(1, 2),
        qualification_subset=True)
    warmup = result["scales"][0]
    assert warmup["ad_status"] == "raw_only_fixed_teacher" and warmup["assembly"] is None
    assert warmup["cases"] == 1 and warmup["episodes"] == 2 and warmup["transitions"] == 2
    bc, manifest = load_dataset(warmup["datasets"]["train_behavior_cloning"]["path"])
    assert manifest["objective"] == "behavior_cloning"
    np.testing.assert_array_equal(bc[0]["teacher_steps"], [0, 0])
    dpt, manifest = load_dataset(warmup["datasets"]["train_dpt"]["path"])
    assert manifest["objective"] == "dpt" and len(dpt[0]["observations"]) == 2
    assert not (tmp_path / "scales/cases-001/train-ad.npz").exists()
    assert not (tmp_path / "scales/cases-001/assembly.json").exists()


def test_learning_prefix_is_existing_small_pilot_assembly_and_keeps_full_cost(tmp_path):
    train, validation = histories(tmp_path)
    result = make_scales(train, validation, tmp_path / "scales", case_counts=(1, 2),
        qualification_subset=True)
    learned = result["scales"][1]
    assert learned["ad_status"] == "learning_history"
    assembly = load_assembly(learned["assembly"]["path"], allow_small_pilot=True)
    assert assembly["small_pilot"] and assembly["source_task_keys"] == ["libero_spatial:00"]
    with pytest.raises(ValueError, match="explicit"):
        load_assembly(learned["assembly"]["path"])
    original_index = json.loads(train.read_text())
    teacher_sha = original_index["publications"][-1]["source_checkpoint_sha256"]
    for scale in result["scales"]:
        assert scale["original_collection_costs_count_once"]["train"]["actual_primitive_steps"] == 32
        assert scale["original_collection_costs_count_once"]["train"]["source_updates"] == 40
        trials, manifest = load_dataset(scale["datasets"]["train_dpt"]["path"])
        original = manifest["metadata"]["selected_source_histories"][0]
        assert original["selected_publication"]["source_checkpoint_sha256"] == teacher_sha
        assert original["episodes"] == 4  # Original complete teacher history retained.
        assert original["used_prefix"]["episodes"] == scale["episodes"]
    assert result["additional_simulator_actions"] == 0
    assert result["content_sha256"] == content_hash({k: v for k, v in result.items() if k != "content_sha256"})


def test_validation_arrays_are_identical_for_every_data_amount(tmp_path):
    train, validation = histories(tmp_path)
    result = make_scales(train, validation, tmp_path / "scales", case_counts=(1, 2),
        qualification_subset=True)
    original_index = json.loads(validation.read_text())
    for objective in ("ad", "dpt"):
        item = original_index["publications"][-1]["datasets"][objective]
        original, _ = load_dataset(validation.parent / item["path"], expected_sha256=item["sha256"])
        for scale in result["scales"]:
            actual, manifest = load_dataset(scale["datasets"][f"validation_{objective}"]["path"])
            for field in FIELDS:
                np.testing.assert_array_equal(actual[0][field], original[0][field])
            assert manifest["metadata"]["local_scene_split"] == "validation"
            assert "used_prefix" not in manifest["metadata"]["selected_source_histories"][0]


def test_default_admission_and_impossible_data_scale_reject_before_output(tmp_path):
    train, validation = histories(tmp_path)
    output = tmp_path / "scales"
    with pytest.raises(ValueError, match="qualification-subset"):
        make_scales(train, validation, output, case_counts=(1, 2))
    with pytest.raises(ValueError, match="positive"):
        make_scales(train, validation, output, case_counts=(3,), qualification_subset=True)
    with pytest.raises(ValueError, match="increasing"):
        make_scales(train, validation, output, case_counts=(2, 1), qualification_subset=True)
    assert not output.exists()
