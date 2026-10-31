"""Source-case role isolation and the fixed complete-trial budget."""
import copy

import pytest

from fluffy_bassoon.meta_rl.vla_plus_worker import CASES, SOURCE_TRAINING_CASES, case_contract
from fluffy_bassoon.meta_rl.vla_rlinf_plus_corpus import source_plan, validate_source_case


def test_plus_source_trials_cannot_become_clean_validation_or_development_cases(tmp_path):
    plan = source_plan(0)
    assert len(plan) == len({row["trial_id"] for row in plan}) == 6
    assert {row["case_index"] for row in plan} == {2387, 1438}
    assert {row["initial_state"] for row in plan} == {0, 1, 2}
    assert {row["split"] for row in plan} == {"train"}
    assert len(plan) * 4 * 280 == 6720
    assert not set(SOURCE_TRAINING_CASES) & set(CASES)
    for case in CASES:
        with pytest.raises(ValueError, match="explicit development/source-training role"):
            case_contract(tmp_path, case, role="source_training")


def valid_metadata():
    task = dict(task_key="libero_90:47", split="train", teacher_collection_allowed=True,
                source_state_indices=list(range(40)), goal_binding_sha256="goal")
    metadata = dict(benchmark_kind="plus", source_training=True, development=False, role="source_training",
        task_id=2387, horizon=280, settle_steps=10, initial_states=50, fresh_environment_on_reset=True,
        origin_base=dict(train_task_key="libero_90:47"), plus_case=dict(role="source_training",
            runtime_index=2387, attachment_train_lineage=dict(goal_binding_sha256="goal")))
    return task, metadata


@pytest.mark.parametrize("mutation", ["development", "case", "lineage", "task_split"])
def test_plus_source_gate_rejects_wrong_roles_cases_and_lineage(mutation):
    task, metadata = valid_metadata()
    validate_source_case(metadata, task, 2387)
    metadata = copy.deepcopy(metadata)
    if mutation == "development":
        metadata["development"] = True
    elif mutation == "case":
        metadata["plus_case"]["runtime_index"] = 667
    elif mutation == "lineage":
        metadata["origin_base"]["train_task_key"] = "libero_90:57"
    else:
        task["split"] = "validation"
    with pytest.raises(ValueError, match="Plus TRAIN"):
        validate_source_case(metadata, task, 2387)
