"""Two-goal admission checks, using real numeric source/assembly formats."""
from copy import deepcopy
import json

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_history import file_sha256, load_dataset, save_dataset
from fluffy_bassoon.meta_rl.vla_source40_corpus import PI05_RECEIPT
from fluffy_bassoon.meta_rl.vla_source40_schedule import case_plan, content_hash
from fluffy_bassoon.meta_rl.vla_source40_training import FEATURE_KIND, INDEX_FORMAT, NOISE_CONTRACT, load_assembly, load_source
import fluffy_bassoon.meta_rl.vla_source40_two_goal_qualification as module


def source(root, task, partition):
    root.mkdir()
    suite, number = task.split(":")
    feature = dict(kind=FEATURE_KIND, checkpoint_receipt_sha256=PI05_RECEIPT,
        model_config="pi05_libero", observation_dim=4104)
    cases = case_plan(task, partition, 2, 4, 0)
    contract = dict(format="openpi_pi05_libero40_persistent_source_v1", source_only=True,
        source_data_from_targets=False, task=dict(task_key=task, suite=suite, task_id=int(number)),
        partition=partition, history_id=f"{task}/{partition}/seed-0", cases=cases,
        motor=dict(receipt_sha256=PI05_RECEIPT, weights_frozen=True),
        feature_contract=feature, noise_contract=NOISE_CONTRACT,
        source_recipe=dict(warmup_chunks=500, batch_size=256, utd=20),
        learner_seed=int(content_hash(dict(task=task, partition=partition))[:8], 16),
        validation="Independent SAC/replay on disjoint validation cases; no validation interactions enter train replay or student gradients")
    (root / "contract.json").write_text(json.dumps(contract))
    actor, recovery = root / "case-00001-source.pkl", root / "source-recovery.pkl"
    actor.write_bytes(f"actor {task} {partition}".encode())
    recovery.write_bytes(f"complete numeric-fixture learner {task} {partition}".encode())
    receipts = []
    for ordinal in range(8):
        case = cases[ordinal // 4]
        receipts.append(dict(case_id=case["case_id"], initial_state=case["initial_state"],
            scene_seed=case["scene_seed"], episode_in_case=ordinal % 4, suite=suite,
            task_id=int(number), chunks=80,
            initial_state_vector_sha256=content_hash(dict(task=task, state=case["initial_state"])),
            source_updates=0 if ordinal < 6 else 1600 * (ordinal - 5),
            source_stages={"warmup_gaussian" if ordinal < 7 else "sac_online": 80}))
    ends = np.zeros(640, bool)
    ends[79::80] = True
    starts = np.zeros(640, bool)
    starts[::80] = True
    trial = dict(observations=np.zeros((640, 4104), np.float32),
        executed_noise=np.zeros((640, 32), np.float32), target_noise=np.zeros((640, 32), np.float32),
        rewards=np.zeros(640, np.float32), dones=ends, episode_starts=starts,
        teacher_steps=np.r_[np.zeros(560, np.int64), np.full(80, 1600, np.int64)])
    datasets = {}
    for objective in ("ad", "dpt"):
        value = deepcopy(trial)
        if objective == "dpt":
            value["target_noise"] += .5
        path = root / f"{objective}.npz"
        manifest = save_dataset(path, [value], objective=objective, metadata=dict(source_contract=contract,
            task_key=task, global_task_split="source40", local_scene_split=partition,
            feature_contract=feature, motor_sha256=PI05_RECEIPT, noise_contract=NOISE_CONTRACT,
            trial_ids=[contract["history_id"]], source_case_ids=[case["case_id"] for case in cases],
            source_generation_episodes=8, source_generation_chunks=640,
            source_actor_checkpoint_sha256=file_sha256(actor)))
        datasets[objective] = dict(path=path.name, sha256=manifest["sha256"])
    index = dict(format=INDEX_FORMAT, completed=True, collection_qualified=True, source_only=True,
        task_key=task, partition=partition, source_updates=3200, chunks=640,
        collection_mode="complete_cases", case_coverage=dict(all_requested_cases_complete=True),
        source_contract_sha256=content_hash(contract), feature_contract=feature,
        motor_receipt_sha256=PI05_RECEIPT, noise_contract=NOISE_CONTRACT,
        episode_receipts=receipts, actual_primitive_steps=640, actual_settling_steps=90,
        latest_recovery_sha256=file_sha256(recovery), publications=[dict(status="paired_ad_dpt",
            through_case=1, case_count=2, episodes=8, transitions=640, datasets=datasets,
            source_checkpoint=actor.name, source_checkpoint_sha256=file_sha256(actor))])
    path = root / "index.json"
    path.write_text(json.dumps(index))
    return path


@pytest.fixture(scope="module")
def histories(tmp_path_factory):
    root = tmp_path_factory.mktemp("two-goal-source")
    return {partition: [source(root / f"{partition}-{i}", task, partition)
        for i, task in enumerate(module.TASKS)] for partition in ("train", "validation")}


def qualified(histories, output, **kwargs):
    return module.qualify_two_goal(histories["train"], histories["validation"], output,
        train_sha256=[file_sha256(path) for path in histories["train"]],
        validation_sha256=[file_sha256(path) for path in histories["validation"]],
        qualification_subset=True, **kwargs)


def test_two_goal_assembly_uses_existing_format_and_keeps_both_histories_costs(histories, tmp_path):
    output = tmp_path / "assembly"
    receipt = qualified(histories, output)
    assembly = load_assembly(output / "assembly.json", allow_small_pilot=True)
    assert set(assembly["source_task_keys"]) == set(module.TASKS)
    assert assembly["validation_task_keys"] == assembly["source_task_keys"]
    assert assembly["small_pilot"] is True and assembly["source_only"] is True
    for partition in ("train", "validation"):
        assert receipt["collection_costs_count_once"][partition]["charged_total_actions"] == 1460
        assert receipt["collection_costs_count_once"][partition]["source_updates"] == 6400
        for objective in ("ad", "dpt"):
            trials, manifest = load_dataset(output / f"{partition}-{objective}.npz")
            assert len(trials) == 2 and manifest["transitions"] == 1280
            assert manifest["metadata"]["source_task_keys"] == assembly["source_task_keys"]
            assert len(manifest["metadata"]["selected_source_histories"]) == 2
    assert receipt["additional_simulator_actions"] == 0
    assert len({row["learner_seed"] for rows in receipt["evidence"].values() for row in rows}) == 4
    assert all(row["learned_behavior_chunks"] == 80 for rows in receipt["evidence"].values() for row in rows)
    assert receipt["assembly_file_sha256"] == file_sha256(output / "assembly.json")
    with pytest.raises(ValueError, match="explicit"):
        load_assembly(output / "assembly.json")


def test_missing_second_validation_or_file_pin_rejects_before_output(histories, tmp_path):
    output = tmp_path / "assembly"
    with pytest.raises(ValueError, match="two hash-pinned"):
        module.qualify_two_goal(histories["train"], histories["validation"][:1], output,
            train_sha256=[file_sha256(path) for path in histories["train"]],
            validation_sha256=[file_sha256(histories["validation"][0])], qualification_subset=True)
    with pytest.raises(ValueError, match="SHA256"):
        module.qualify_two_goal(histories["train"], histories["validation"], output,
            train_sha256=["0" * 64] * 2,
            validation_sha256=[file_sha256(path) for path in histories["validation"]], qualification_subset=True)
    assert not output.exists()


def test_single_goal_repeated_as_second_goal_is_not_a_two_goal_experiment(histories, tmp_path):
    paths = dict(histories, train=[histories["train"][0]] * 2)
    with pytest.raises(ValueError, match="exactly Spatial00"):
        qualified(paths, tmp_path / "assembly")


def test_postwarmup_checks_actual_behavior_and_complete_recovery(histories):
    path = histories["train"][0]
    source = load_source(path, "train", allow_small_pilot=True)
    index = json.loads(path.read_text())
    contract = json.loads((path.parent / "contract.json").read_text())
    missing_behavior = deepcopy(index)
    for row in missing_behavior["episode_receipts"]:
        row["source_stages"] = dict(warmup_gaussian=row["chunks"])
    with pytest.raises(ValueError, match="fixed behavior-stage receipts"):
        module.postwarmup_evidence(source, missing_behavior, contract)
    source["datasets"]["ad"]["teacher_steps"][:] = 0
    with pytest.raises(ValueError, match="chronological postwarmup"):
        module.postwarmup_evidence(source, index, contract)
    source = load_source(path, "train", allow_small_pilot=True)
    index["latest_recovery_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="recovery checksum"):
        module.postwarmup_evidence(source, index, contract)


def test_reused_train_validation_seed_and_feature_change_reject_before_assembly(histories, tmp_path, monkeypatch):
    original = module.postwarmup_evidence
    monkeypatch.setattr(module, "postwarmup_evidence", lambda *args: dict(original(*args), learner_seed=7))
    with pytest.raises(ValueError, match="four independent"):
        qualified(histories, tmp_path / "assembly")
    monkeypatch.setattr(module, "postwarmup_evidence", original)
    loader = module.load_source
    def mismatch(*args, **kwargs):
        result = loader(*args, **kwargs)
        if result["task"] == "libero_10:00":
            result["feature_contract"] = dict(result["feature_contract"], extra="different")
        return result
    monkeypatch.setattr(module, "load_source", mismatch)
    with pytest.raises(ValueError, match="one exact frozen"):
        qualified(histories, tmp_path / "assembly")
    assert not (tmp_path / "assembly").exists()


def test_explicit_subset_flag_is_required_before_reading_any_source(tmp_path):
    with pytest.raises(ValueError, match="qualification-subset"):
        module.qualify_two_goal([], [], tmp_path / "assembly", train_sha256=[], validation_sha256=[])
