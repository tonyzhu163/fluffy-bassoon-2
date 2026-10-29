"""Recompute compact robot counts from mirrored JSON; no model imports."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
verification = json.loads((ROOT / "remote-verification.json").read_text())
result = dict(model_revision="6222623f635769bfc73c9472e29fab9b7fd8e027", runs={})
for run, audit in verification["runs"].items():
    root = ROOT / "raw" / run
    for item in audit["checked_files"]:
        path = ROOT / "raw" / item["path"]
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], str(path)
    contract = json.loads((root / "experiment/contract.json").read_text())
    summary = json.loads((root / "experiment/summary.json").read_text())
    rows = [json.loads(line) for line in (root / "experiment/episodes.jsonl").read_text().splitlines()]
    assert audit["exit_code"] == 0 and summary["completed"] is True
    assert contract["motor"]["revision"] == result["model_revision"]
    assert contract["motor"]["strict_loading"] == dict(strict=True, loaded_tensors=667, missing_keys=[], unexpected_keys=[])
    assert all(row["completed"] for row in rows) if "trials" not in summary else True
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("method", row.get("local_scene_split"))].append(row)
    methods = {key: dict(successes=sum(int(row["success"]) for row in bank), episodes=len(bank),
        primitive_steps=sum(row["primitive_steps"] for row in bank),
        source_optimizer_updates=sum(row["source_updates_after"] - row["source_updates_before"] for row in bank),
        success_by_attempt={str(attempt): dict(successes=sum(int(row["success"]) for row in bank if row["attempt"] == attempt),
            episodes=sum(row["attempt"] == attempt for row in bank)) for attempt in range(1, 5)})
        for key, bank in groups.items()}
    primitives = sum(row["primitive_steps"] for row in rows)
    settling = summary.get("actual_settling_steps", summary.get("settling_steps"))
    assert primitives == summary.get("actual_primitive_steps", summary.get("primitive_steps"))
    run_result = dict(source_commit=contract["source_commit"], episodes=len(rows), primitive_steps=primitives,
        settling_steps=settling, total_simulator_actions=primitives + settling,
        worker_elapsed_seconds=summary["elapsed_seconds"], methods=methods,
        summary_sha256=hashlib.sha256((root / "experiment/summary.json").read_bytes()).hexdigest(),
        episodes_sha256=hashlib.sha256((root / "experiment/episodes.jsonl").read_bytes()).hexdigest(),
        verified_expected_hashes=audit["expected_hash_checks"], focused_test_summary=audit["focused_test_summary"])
    if "trials" in summary:
        index = json.loads((root / "experiment/index.json").read_text())
        assert index == summary and len(index["trials"]) == 6
        assert sum(t["primitive_steps"] for t in index["trials"]) == primitives
        assert len({trial["trial_id"] for trial in index["trials"]}) == 6
        run_result.update(task_key=index["task_key"], trials=6, paired_contexts_bitwise_equal=audit["paired_contexts_bitwise_equal"],
            index_sha256=hashlib.sha256((root / "experiment/index.json").read_bytes()).hexdigest(),
            pilot_datasets=index.get("pilot_datasets"),
            source_steps_by_attempt={str(attempt): sum(row["primitive_steps"] for row in rows if row["attempt"] == attempt)
                                    for attempt in range(1, 5)},
            reset_variation=[dict(trial_id=trial["trial_id"],
                nominal_state_hashes=len({episode["initial_state_vector_sha256"] for episode in trial["episodes"]}),
                observed_initial_hashes=len({episode["initial_observation_sha256"] for episode in trial["episodes"]}))
                for trial in index["trials"]], final_candidate_expert_qualification=False)
        train = {trial["trial_id"] for trial in index["trials"] if trial["scene"]["split"] == "train"}
        validation = {trial["trial_id"] for trial in index["trials"] if trial["scene"]["split"] == "validation"}
        if index.get("benchmark_kind") == "plus":
            assert len(train) == 6 and not validation
            assert index["role"] == "source_training" and index["global_task_split"] == "train"
            assert {trial["case_index"] for trial in index["trials"]} == {2387, 1438}
            assert {row["case_index"] for row in rows}.isdisjoint({290, 667})
            run_result["source_cases"] = {str(case): dict(
                successes=sum(int(row["success"]) for row in rows if row["case_index"] == case), episodes=12,
                trials=[dict(trial_id=trial["trial_id"], success=[episode["success"] for episode in trial["episodes"]],
                    primitive_steps=[episode["primitive_steps"] for episode in trial["episodes"]])
                    for trial in index["trials"] if trial["case_index"] == case]) for case in (2387, 1438)}
            run_result["student_validation_scope"] = "no Plus validation exports; common clean validation shared across data arms"
        else:
            assert len(train) == 4 and len(validation) == 2 and not train & validation
            run_result["student_validation_scope"] = "local scenes of the same source training goal; not a held-out semantic-task bank"
    else:
        assert len(rows) == 40 and summary["total_simulator_steps"] == primitives + settling
        assert {row["split"] for row in rows} == {"development"}
        assert len({row["case_index"] for row in rows}) == 2
        for case in range(2):
            for method in methods:
                bank = [row for row in rows if row["case_index"] == case and row["method"] == method]
                assert len(bank) == 4 and {row["attempt"] for row in bank} == {1, 2, 3, 4}
                expected = summary["success_by_case_method"][f"{case}/{method}"]
                assert expected == dict(successes=sum(int(row["success"]) for row in bank), episodes=4)
        run_result.update(benchmark=summary["benchmark"], development_cases=2, nominal_scene_indices=[0],
            method_count=5, meta_controller_evaluated=False, official_final_score=False,
            case_metadata=contract["cases"], source_config={k: contract[k] for k in ("batch_size", "utd", "warmup_chunks")})
    result["runs"][run] = run_result
result["all_completed_cost"] = {key: sum(run[key] for run in result["runs"].values())
    for key in ("episodes", "primitive_steps", "settling_steps", "total_simulator_actions", "worker_elapsed_seconds")}
result["interpretation"] = "One goal-family smoke test; success ceilings give weak failure/recovery evidence. No frozen meta-controller performance or qualified final-actor expertise is established."
(ROOT / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"all_completed_cost": result["all_completed_cost"],
    "runs": {key: dict(episodes=value["episodes"], methods=value["methods"]) for key, value in result["runs"].items()}}, indent=2))
