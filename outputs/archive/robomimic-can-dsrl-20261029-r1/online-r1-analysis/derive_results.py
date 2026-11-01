"""Descriptive two-scene online-control audit; stdlib artifact counts only."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
SCENES = [2_000_000, 2_000_001]


def read(name):
    return json.loads((RAW / name).read_text())


def hashed(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


checks = []


def check(name, valid, evidence=None):
    checks.append(dict(name=name, passed=bool(valid), evidence=evidence))


manifest = read("remote-manifest.json")
source = read("source-manifest.json")
for name, evidence in manifest["raw_files"].items():
    path = RAW / name
    check("mirror:" + name, path.stat().st_size == evidence["bytes"] and hashed(path) == evidence["sha256"])

plan = read("online/plan.json")
admission = read("online/online-admission.json")
online = read("online/online/online-evaluation.json")
summary = read("online/online/summary.json")
contract = read("online/online/contract.json")
probe = read("online/online/gpu-probe.json")
pilot_admission = read("pilot/pilot-admission.json")
pilot_plan = read("pilot/plan.json")
reference_contract = read("pilot/dsrl-na/contract.json")
reference_summary = read("pilot/dsrl-na/summary.json")
reference_manifest = read("pilot/dsrl-na/recovery-manifest.json")
check("exit_zero", (RAW / "online/exit-code.txt").read_text().strip() == "0" and plan["cli_returncode"] == 0)
check("online_complete", admission["admitted"] and admission["completed"] and online["completed"] and summary["completed"])
check("requested_budget", admission["requested_scenes"] == 2 and admission["requested_episodes"] == 8
      and admission["requested_primitive_steps"] == online["primitive_steps"] == 2400)
check("summary_evaluation_hashes", hashed(RAW / "online/online/summary.json") == plan["online_summary_sha256"]
      and hashed(RAW / "online/online/online-evaluation.json") == plan["online_evaluation_sha256"])
check("summary_contains_same_evaluation", all(summary[key] == value for key, value in online.items()))
check("pilot_artifact_identity", all(hashed(RAW / "pilot" / name) == value for name, value in plan["pilot_artifact_sha256"].items()))
check("recorded_source", contract["source_commit"] == plan["source_commit"] == admission["source_commit"] == source["online"]["git_head"])
check("reference_source", reference_contract["source_commit"] == plan["training_source_commit"] == source["pilot"]["git_head"])
for path, evidence in source["pilot"]["files"].items():
    check("unchanged_source:" + path, source["online"]["files"][path] == evidence)
check("gpu_allocation", probe["status"] == "passed" and probe["purpose"] == "training"
      and probe["hostname"] == contract["job"] == plan["hostname"] and len(probe["devices"]) == 1
      and probe["compute_processes_before"] == [] and bool(probe["devices"][0]["allocation_check"]))
for key in ("checkpoint_sha256", "normalization_sha256", "frozen_motor_sha256", "runtime",
            "robomimic_commit", "robomimic_env_sha256"):
    check("runtime_input_identity:" + key, contract[key] == reference_contract[key] == plan["runtime_identity"][key])
check("training_reference_contract", online["training_source_contract"] == reference_contract)
check("training_reference_counters", online["training_num_timesteps"] == 1196 and online["training_learner_updates"] == 5980
      and reference_summary["primitive_steps"] == 28800 and reference_summary["learner_updates"] == 5980)
check("recorded_immutable_reference", online["reference_unchanged"] and online["motor_unchanged"]
      and plan["immutable_reference_unchanged"] and plan["source_checkpoint_unchanged"])
for name, expected in plan["reference_sha256"].items():
    check("reference_postrun_hash:" + name, manifest["reference_postrun"][name]["sha256"] == expected
          and online["reference_sha256"][name] == expected)
    if name != "recovery-manifest.json":
        check("original_manifest:" + name, reference_manifest["files"][name] == expected)
for name, key in (("state_5000.pt", "checkpoint_sha256"), ("normalization.npz", "normalization_sha256")):
    check("frozen_input_postrun:" + name, manifest["inputs_postrun"][name]["sha256"] == contract[key])

matrices = {"online DSRL-NA": {metric: [scene[metric] for scene in online["scenes"]]
                              for metric in ("successes", "release_logged_successes")}}
rows_by_name = {"native": read("qualification/native/evaluation.json")["native"][:2]}
for method in ("dsrl-na", "dsrl-na-retry", "rl2", "episode-reset"):
    for condition, rows in read("pilot/" + method + "/evaluation.json").items():
        rows_by_name[method + "/" + condition] = rows[:2]
for name, rows in rows_by_name.items():
    check(name + ":paired_seeds", [row["scene_seeds"] for row in rows] == [[seed] for seed in SCENES])
    check(name + ":matched_budget", sum(row["primitive_steps"] for row in rows) == 2400 and all(row["attempts"] == 4 for row in rows))
    matrices[name] = {metric: [row[metric][0] for row in rows] for metric in ("successes", "release_logged_successes")}
    for row in rows:
        expected = plan["paired_scene_sha256"][str(row["scene_seeds"][0])]
        check(name + ":scene_hash:" + str(row["scene_seeds"][0]), row["initial_observation_sha256"] == [[expected]] * 4)

scenes = online["scenes"]
check("online_seeds", [row["scene_seed"] for row in scenes] == SCENES)
check("fresh_start_each_scene", all(row["starting_learner_sha256"] == online["starting_learner_sha256"] for row in scenes)
      and all(row["starting_replay_transitions"] == 7200 for row in scenes))
for row in scenes:
    seed = row["scene_seed"]
    check("online_rng:" + str(seed), row["rng_seed"] == seed + 1)
    check("online_trial_complete:" + str(seed), row["completed"] and row["stop_reason"] is None
          and row["next_attempt_index"] == 4 and row["primitive_steps"] == 1200 and row["chunks"] == 300
          and all(record["episode_complete"] and record["adaptation_complete"] and record["primitive_steps"] == 300
                  for record in row["attempt_records"]))
    expected = plan["paired_scene_sha256"][str(seed)]
    check("online_reset_identity:" + str(seed), row["initial_observation_sha256"] == [expected] * 4
          and all(record["initial_observation_sha256"] == expected for record in row["attempt_records"]))
    check("online_updates:" + str(seed), row["adaptation_train_calls"] == 300
          and row["adaptation_learner_updates"] == 6000 and row["adaptation_optimizer_steps_total"] == 21000
          and row["adaptation_optimizer_updates"] == dict(actor=6000, critic=6000, entropy=6000, noise_critic=3000))
    check("online_changed_fresh_copy:" + str(seed), all(value != row["starting_learner_sha256"][key]
                                                        for key, value in row["final_learner_sha256"].items()))
check("online_total_updates", sum(row["adaptation_train_calls"] for row in scenes) == 600
      and online["adaptation_learner_updates"] == 12000 and online["adaptation_optimizer_steps_total"] == 42000)
check("motor_query_count", summary["motor_batch_calls"] == 600 * 31 and summary["motor_samples"] == 600 * (1 + 30 * 256))
check("fourfold_update_ratio", plan["comparison_metadata"]["released_training_lanes"] == 4
      and plan["comparison_metadata"]["online_adaptation_lanes"] == 1
      and plan["comparison_metadata"]["online_updates_per_new_transition_relative_to_released_training"] == 4)

results = {}
differences = {}
for name, by_metric in matrices.items():
    results[name] = {}
    for metric, matrix in by_metric.items():
        check(name + ":binary_shape:" + metric, len(matrix) == 2 and all(len(values) == 4 and all(type(v) is bool for v in values) for values in matrix))
        results[name][metric] = dict(scene_seeds=SCENES, per_scene_attempt_matrix=matrix,
                                     per_scene_bits=["".join(str(int(v)) for v in values) for values in matrix],
                                     per_attempt_success_counts=[sum(row[i] for row in matrix) for i in range(4)],
                                     per_scene_success_counts=list(map(sum, matrix)), successes=sum(map(sum, matrix)),
                                     episodes_recorded=8, descriptive_episode_success_rate=sum(map(sum, matrix)) / 8,
                                     scenes_with_any_success=sum(any(row) for row in matrix))
        if name != "online DSRL-NA":
            delta = [[int(a) - int(b) for a, b in zip(online_row, frozen_row)]
                     for online_row, frozen_row in zip(matrices["online DSRL-NA"][metric], matrix)]
            differences.setdefault(name, {})[metric] = dict(online_minus_comparator_by_scene_attempt=delta,
                                                           per_scene_success_count_difference=list(map(sum, delta)),
                                                           total_success_count_difference=sum(map(sum, delta)),
                                                           per_attempt_success_count_difference=[sum(row[i] for row in delta) for i in range(4)])

result = dict(generated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
              source_commit=contract["source_commit"], reference_source_commit=reference_contract["source_commit"],
              reference_training_seed=1, paired_scene_seeds=SCENES,
              inference_unit="two paired initial scenes; one training seed; eight episodes are not independent trials",
              checks=checks, failed_checks=[row for row in checks if not row["passed"]],
              configurations=results, online_minus_comparators=differences,
              compute=dict(new_primitive_steps=2400, chunks=600, train_calls=600,
                           actor_updates=12000, critic_updates=12000, entropy_updates=12000, noise_critic_updates=6000,
                           total_optimizer_invocations=42000, motor_batch_calls=summary["motor_batch_calls"],
                           motor_samples=summary["motor_samples"],
                           frozen_comparator_deployment_optimizer_invocations=0,
                           frozen_comparator_deployment_motor_samples=600,
                           online_relative_updates_per_new_transition_vs_training=4,
                           online_driver_elapsed_seconds=summary["elapsed_seconds"],
                           allocation_elapsed_seconds=admission["elapsed_seconds"],
                           allocated_gpu_minutes=admission["allocated_gpu_minutes"],
                           gpu_name=probe["devices"][0]["cuda_name"], peak_cuda_memory_gib=summary["peak_cuda_memory_gib"]),
              fresh_scene_starting_learner_sha256=online["starting_learner_sha256"],
              starting_replay_transitions_per_scene=[row["starting_replay_transitions"] for row in scenes],
              immutable_reference_postrun=manifest["reference_postrun"],
              analysis_script_sha256=hashed(Path(__file__)),
              raw_manifest_sha256=hashed(RAW / "remote-manifest.json"),
              source_manifest_sha256=hashed(RAW / "source-manifest.json"),
              limitations=["Stdlib metadata/count/hash audit only; no model, tensor, replay, optimizer or simulator loaded.",
                           "Reference checkpoints were hashed on Baige without loading them and were not copied to macOS.",
                           "Fresh learner/optimizer/replay isolation is supported by runtime-recorded starting hashes and counts; replay transition contents were not independently inspected.",
                           "A common starting RNG seed does not couple later random draws: online optimizer work consumes RNG, and adaptation already occurs within attempt one.",
                           "Two scenes and one training seed do not establish a method ranking, statistical significance or a broad negative result for online adaptation.",
                           "Deployment interactions match, while online uses additional optimization and fourfold the released training update-to-data ratio."])
(ROOT / "RESULTS.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(dict(checks=len(checks), failed_checks=result["failed_checks"],
                      paired_totals={name: metrics["successes"]["successes"] for name, metrics in results.items()}), indent=2))
