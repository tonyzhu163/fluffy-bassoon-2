"""Source40 assembly isolation and fixed-budget matrix contracts (allocated Linux)."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_history import FIELDS, file_sha256, load_dataset, save_dataset
from fluffy_bassoon.meta_rl.vla_source40_corpus import PI05_RECEIPT
from fluffy_bassoon.meta_rl.vla_source40_schedule import case_plan, content_hash
from fluffy_bassoon.meta_rl.vla_source40_training import (ASSEMBLY_FORMAT, FEATURE_KIND, INDEX_FORMAT,
    NOISE_CONTRACT, assemble, load_assembly, load_source, parser, training_commands)


def write(path, value):
    path.write_text(json.dumps(value))


def source(root, partition, *, task="libero_spatial:00", state_hash=None):
    root.mkdir()
    suite, number = task.split(":")
    feature = dict(kind=FEATURE_KIND, checkpoint_receipt_sha256=PI05_RECEIPT,
        model_config="pi05_libero", observation_dim=4104)
    cases = case_plan(task, partition, 2, 2, 0)
    contract = dict(format="openpi_pi05_libero40_persistent_source_v1", source_only=True,
        source_data_from_targets=False, task=dict(task_key=task, suite=suite, task_id=int(number)),
        partition=partition, history_id=f"{task}/{partition}/seed-0", cases=cases,
        motor=dict(receipt_sha256=PI05_RECEIPT, weights_frozen=True),
        feature_contract=feature, noise_contract=NOISE_CONTRACT)
    write(root / "contract.json", contract)
    receipts = []
    for case in cases:
        for episode in range(2):
            receipts.append(dict(case_id=case["case_id"], initial_state=case["initial_state"],
                scene_seed=case["scene_seed"], episode_in_case=episode, suite=suite, task_id=int(number),
                initial_state_vector_sha256=state_hash or content_hash(dict(task=task, state=case["initial_state"])),
                chunks=1))
    publications = []
    for through_case, count in ((0, 2), (1, 4)):
        checkpoint = root / f"case-{through_case:05d}-source.pkl"
        checkpoint.write_bytes(f"source actor {partition} {through_case}".encode())
        actor_sha = file_sha256(checkpoint)
        noise = np.arange(count * 32, dtype=np.float32).reshape(count, 32) / 100
        trial = dict(observations=np.full((count, 4104), .01, np.float32),
            executed_noise=noise, target_noise=noise.copy(), rewards=np.zeros(count, np.float32),
            dones=np.ones(count, bool), episode_starts=np.ones(count, bool),
            teacher_steps=np.arange(count, dtype=np.int64))
        datasets = {}
        for objective in ("ad", "dpt"):
            value = deepcopy(trial)
            if objective == "dpt":
                value["target_noise"] = value["target_noise"] + .5
            path = root / f"prefix-{through_case}" / f"{objective}.npz"
            manifest = save_dataset(path, [value], objective=objective, metadata=dict(source_contract=contract,
                task_key=task, global_task_split="source40", local_scene_split=partition,
                feature_contract=feature, motor_sha256=PI05_RECEIPT, noise_contract=NOISE_CONTRACT,
                trial_ids=[contract["history_id"]], source_case_ids=[case["case_id"] for case in cases[:through_case + 1]],
                source_generation_episodes=count, source_generation_chunks=count,
                source_actor_checkpoint_sha256=actor_sha))
            datasets[objective] = dict(path=str(path.relative_to(root)), sha256=manifest["sha256"])
        publications.append(dict(status="paired_ad_dpt", through_case=through_case, case_count=through_case + 1,
            episodes=count, transitions=count, datasets=datasets, source_checkpoint=checkpoint.name,
            source_checkpoint_sha256=actor_sha))
    index = dict(format=INDEX_FORMAT, completed=True, collection_qualified=True, source_only=True,
        task_key=task, partition=partition, source_updates=40, chunks=4, collection_mode="complete_cases",
        case_coverage=dict(all_requested_cases_complete=True), source_contract_sha256=content_hash(contract),
        feature_contract=feature, motor_receipt_sha256=PI05_RECEIPT, noise_contract=NOISE_CONTRACT,
        episode_receipts=receipts, publications=publications, actual_primitive_steps=32, actual_settling_steps=40)
    path = root / "index.json"
    write(path, index)
    return path


def test_final_prefix_is_selected_once_not_concatenated(tmp_path):
    path = source(tmp_path / "source", "train")
    result = load_source(path, "train")
    assert result["selected"]["through_case"] == 1
    assert len(result["datasets"]["ad"]["observations"]) == 4  # Not 2+4 overlapping rows.
    assert result["datasets"]["dpt"]["target_noise"][0, 0] == .5
    for field in set(FIELDS) - {"target_noise"}:
        np.testing.assert_array_equal(result["datasets"]["ad"][field], result["datasets"]["dpt"][field])


def test_explicit_pilot_assembles_disjoint_cases_and_preserves_histories(tmp_path):
    train = source(tmp_path / "train", "train")
    val = source(tmp_path / "val", "validation")
    output = tmp_path / "assembly"
    value = assemble([train], [val], output, allow_small_pilot=True)
    assert value["format"] == ASSEMBLY_FORMAT and value["small_pilot"]
    assert value["collection_costs_count_once"]["train"]["actual_primitive_steps"] == 32
    trials, manifest = load_dataset(output / "train-ad.npz")
    assert len(trials) == 1 and manifest["transitions"] == 4
    assert manifest["metadata"]["trial_ids"] == ["libero_spatial:00/train/seed-0"]
    assert load_assembly(output / "assembly.json", allow_small_pilot=True) == value
    with pytest.raises(ValueError, match="explicit"):
        load_assembly(output / "assembly.json")


def test_full_assembly_requires_actual_all40_coverage(tmp_path):
    train = source(tmp_path / "train", "train")
    val = source(tmp_path / "val", "validation")
    with pytest.raises(ValueError, match="all40"):
        assemble([train], [val], tmp_path / "assembly")
    assert not (tmp_path / "assembly").exists()


def test_duplicate_task_or_prefix_indices_cannot_multiply_experience(tmp_path):
    train = source(tmp_path / "train", "train")
    val = source(tmp_path / "val", "validation")
    with pytest.raises(ValueError, match="one source index"):
        assemble([train, train], [val], tmp_path / "assembly", allow_small_pilot=True)


def test_nominal_state_hash_collision_rejected_even_with_different_ids(tmp_path):
    train = source(tmp_path / "train", "train", state_hash="a" * 64)
    val = source(tmp_path / "val", "validation", state_hash="a" * 64)
    with pytest.raises(ValueError, match="state_hashes"):
        assemble([train], [val], tmp_path / "assembly", allow_small_pilot=True)


@pytest.mark.parametrize("change,match", [
    (lambda value: value.update(task_key="libero_90:00"), "source40"),
    (lambda value: value.update(source_only=False), "source40"),
    (lambda value: value.update(motor_receipt_sha256="0" * 64), "OpenPI40"),
    (lambda value: value.update(partition="validation"), "partition"),
    (lambda value: value["episode_receipts"][0].update(initial_state=40), "partitions"),
    (lambda value: value["publications"].pop(), "entire completed history"),
    (lambda value: value["publications"][-1]["datasets"]["ad"].update(path="../elsewhere.npz"), "relative"),
])
def test_source_contract_failures_are_rejected(tmp_path, change, match):
    path = source(tmp_path / "source", "train")
    index = json.loads(path.read_text())
    change(index)
    write(path, index)
    with pytest.raises(ValueError, match=match):
        load_source(path, "train")


def test_noncovering_smoke_needs_explicit_pilot_flag(tmp_path):
    path = source(tmp_path / "source", "train")
    index = json.loads(path.read_text())
    index.update(collection_mode="chunk_target", case_coverage=dict(all_requested_cases_complete=False))
    write(path, index)
    with pytest.raises(ValueError, match="complete-case coverage"):
        load_source(path, "train")
    assert load_source(path, "train", allow_small_pilot=True)["episodes"] == 4


def test_actual_ad_dpt_context_mismatch_rejected(tmp_path):
    path = source(tmp_path / "source", "train")
    index = json.loads(path.read_text())
    artifact = index["publications"][-1]["datasets"]["dpt"]
    data_path = path.parent / artifact["path"]
    trials, manifest = load_dataset(data_path)
    data_path.unlink()
    data_path.with_suffix(".json").unlink()
    trials[0]["observations"][0, 0] += 1
    replacement = save_dataset(data_path, trials, objective="dpt", metadata=manifest["metadata"])
    artifact["sha256"] = replacement["sha256"]
    write(path, index)
    with pytest.raises(ValueError, match="identical actual histories"):
        load_source(path, "train")


def test_matrix_explicitly_matches_budgets_and_seeds(tmp_path):
    train = source(tmp_path / "train", "train")
    val = source(tmp_path / "val", "validation")
    value = assemble([train], [val], tmp_path / "assembly", allow_small_pilot=True)
    args = parser().parse_args(["train", "--assembly", str(tmp_path / "assembly/assembly.json"),
        "--output-dir", str(tmp_path / "students"), "--updates", "25", "--seeds", "0", "7",
        "--source-commit", "f" * 40, "--expected-hostname", "allocated-host", "--max-elapsed-seconds", "300"])
    commands = training_commands(args, value)
    assert len(commands) == 8
    assert {(row["objective"], row["mode"], row["seed"]) for row in commands} == {
        (objective, mode, seed) for objective in ("ad", "dpt") for mode in ("full", "static") for seed in (0, 7)}
    for row in commands:
        command = row["command"]
        assert command[command.index("--updates") + 1] == "25"
        assert command[command.index("--seed") + 1] == str(row["seed"])
        assert command[command.index("--max-elapsed-seconds") + 1] == "300.0"
