"""CPU checks of the frozen LIBERO split loader."""
import json

import pytest

from fluffy_bassoon import libero_split


def test_split_loads_with_frozen_counts_and_training_tasks():
    split = libero_split.load()
    assert {name: len(libero_split.tasks(split, name)) for name in libero_split.SPLITS} == libero_split.COUNTS
    training = libero_split.motor_training_tasks(split)
    assert len(training) == 78 and all(task["suite"] == "libero_90" for task in training)
    held_out = {task["task_key"] for name in ("validation", "test", "test_auxiliary_excluded_90")
                for task in libero_split.tasks(split, name)}
    assert not held_out & {task["task_key"] for task in training}
    with pytest.raises(ValueError):
        libero_split.tasks(split, "dev")


def test_split_rejects_edited_content(tmp_path):
    split = json.loads(libero_split.DEFAULT.read_text())
    split["tasks"][0]["split"] = "validation"
    path = tmp_path/"split.json"
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="metadata_sha256"):
        libero_split.load(path)
