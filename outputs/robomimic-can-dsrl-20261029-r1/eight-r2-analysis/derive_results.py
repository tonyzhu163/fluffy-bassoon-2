"""Reproduce descriptive eight-r2 metadata/count/hash audit; stdlib only."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
SEEDS, SCENES = [1, 2], list(range(2_000_000, 2_000_008))
METHODS = ["dsrl-na", "dsrl-na-retry", "rl2", "episode-reset"]
METRICS = ["successes", "release_logged_successes"]


def read(name):
    return json.loads((RAW / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


checks = []


def check(name, passed):
    checks.append(dict(name=name, passed=bool(passed)))


def finite_numbers(value):
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_numbers(item) for item in value)
    return math.isfinite(value) if isinstance(value, (int, float)) else True


def summarize(matrix):
    return dict(per_scene_attempt_matrix=matrix, scene_seeds=SCENES,
                per_scene_bits=["".join(str(int(v)) for v in row) for row in matrix],
                per_attempt_success_counts=[sum(row[index] for row in matrix) for index in range(4)],
                per_scene_success_counts=list(map(sum, matrix)), successes=sum(map(sum, matrix)),
                recorded_episodes=32, descriptive_episode_success_rate=sum(map(sum, matrix)) / 32,
                scenes_with_any_success=sum(any(row) for row in matrix), scenes=8)


def paired(left, right):
    difference = [[int(a) - int(b) for a, b in zip(xs, ys)] for xs, ys in zip(left, right)]
    counts = list(map(sum, difference))
    return dict(scene_seeds=SCENES, per_scene_per_attempt_difference=difference,
                per_scene_success_count_difference=counts,
                per_attempt_success_count_difference=[sum(row[index] for row in difference) for index in range(4)],
                total_success_count_difference=sum(counts), mean_scene_rate_difference=sum(counts) / 32,
                positive_scenes=sum(v > 0 for v in counts), tied_scenes=sum(v == 0 for v in counts),
                negative_scenes=sum(v < 0 for v in counts))


manifest = read("remote-manifest.json")
for name, entry in manifest["raw_files"].items():
    path = RAW / name
    check("mirror:" + name, path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"])
admission = read("campaign/campaign-admission.json")
plan = read("campaign/plan.json")
parent = read("campaign/parent-gpu-probe.json")
phases = read("campaign/phase-records.json")
stage_events = [json.loads(line) for line in (RAW / "campaign/stages.jsonl").read_text().splitlines()]
stages = {(row["seed"], row["method"]): row for row in phases}
check("exit_zero", (RAW / "campaign/exit-code.txt").read_text().strip() == "0")
check("eight_complete_matched_arms", admission["admitted"] and admission["matched_training_interactions"]
      and admission["completed_arms"] == admission["required_arms"] == 8 and not admission["incomplete_arms"])
check("selected_design", plan["seeds"] == SEEDS and plan["selected_seed_count"] == 2
      and plan["target_steps"] == admission["target_steps"] == 28800
      and plan["selected_layout"] == "one independent arm per GPU")
check("phase_records_equal_admission", phases == admission["stage_records"])
check("campaign_end_record", stage_events[-1]["stage"] == "campaign_ended"
      and stage_events[-1]["matched_training_interactions"] and stage_events[-1]["completed_arms"] == 8)
check("source_commit", plan["source_commit"] == admission["source_commit"] == parent["source_commit"]
      == manifest["source"]["git_head"])
check("parent_hash", digest(RAW / "campaign/parent-gpu-probe.json") == plan["parent_probe_sha256"])
check("parent_eight_gpu_receipt", parent["status"] == "passed" and parent["purpose"] == "training"
      and len(parent["devices"]) == 8 and len({row["uuid"] for row in parent["devices"]}) == 8)
check("parent_source_hashes", parent["source_files_sha256"] == plan["source_files_sha256"])
check("assigned_unique_workers", sorted(row["gpu"] for row in phases) == list(range(8)))
check("gpu_hour_accounting_within_one_millisecond",
      abs(admission["allocation_gpu_hours"] * 3600 / 8 - admission["elapsed_seconds"]) < .001)
check("elapsed_within_cap", admission["elapsed_seconds"] < admission["wall_budget_seconds"] == 3480)
check("fixture_log_hash", digest(RAW / "campaign/runtime-probe-tests.log")
      == plan["runtime_probe_migration"]["cpu_fixtures"]["log_sha256"])
check("fixture_pass_record", plan["runtime_probe_migration"]["cpu_fixtures"]["passed"]
      and "55 passed, 27 subtests passed" in (RAW / "campaign/runtime-probe-tests.log").read_text())
for path, entry in manifest["source"]["files"].items():
    if path in plan["source_files_sha256"]:
        check("postrun_source_hash:" + path, entry["sha256"] == plan["source_files_sha256"][path])
for path, expected in plan["runtime_probe_migration"]["cpu_fixtures"]["test_source_sha256"].items():
    check("fixture_source_hash:" + path, manifest["source"]["files"][path]["sha256"] == expected)
check("probe_migration_source_hash", manifest["source"]["files"]["scripts/[train]gpu_probe.py"]["sha256"]
      == plan["runtime_probe_migration"]["new_sha256"])

per_seed, matrices, budgets = {}, {}, {}
identity_keys = ["runtime", "robomimic_commit", "robomimic_env_sha256", "checkpoint_sha256",
                 "normalization_sha256", "frozen_motor_sha256"]
for seed in SEEDS:
    per_seed[str(seed)], matrices[seed] = {}, {}
    for method in METHODS:
        arm = f"seed{seed}/{method}"
        base = RAW / "campaign" / arm
        contract, summary = read("campaign/" + arm + "/contract.json"), read("campaign/" + arm + "/summary.json")
        evaluation, probe = read("campaign/" + arm + "/evaluation.json"), read("campaign/" + arm + "/gpu-probe.json")
        updates = [json.loads(line) for line in (base / "updates.jsonl").read_text().splitlines()]
        stage = stages[(seed, method)]
        check(arm + ":source_and_seed", contract["source_commit"] == admission["source_commit"]
              and contract["seed"] == seed and contract["resume"] is None)
        check(arm + ":complete", summary["completed"] and stage["completed"] and stage["returncode"] == 0
              and not summary["budget_reached"] and summary["primitive_steps"] == contract["steps"] == 28800)
        check(arm + ":summary_identity", summary == stage["summary"])
        check(arm + ":geometry", contract["envs"] == 4 and contract["attempts"] == 4 and contract["eval_trials"] == 8)
        check(arm + ":core_contract", all(contract[key] == value for key, value in plan["core"].items()))
        for key in identity_keys:
            check(arm + ":common_identity:" + key, contract[key] == plan["runtime_identity"][key])
        for name, key in (("contract.json", "contract_sha256"), ("evaluation.json", "evaluation_sha256"),
                          ("gpu-probe.json", "gpu_probe_sha256")):
            check(arm + ":launcher_hash:" + name, digest(base / name) == stage[key])
        selected = probe["selected_device"]
        check(arm + ":worker_gpu_identity", probe["status"] == "passed"
              and probe["source_commit"] == admission["source_commit"] and probe["parent_gpu_count"] == 8
              and probe["parent_probe_sha256"] == plan["parent_probe_sha256"]
              and int(probe["cuda_visible_devices"]) == stage["gpu"]
              and selected["uuid"] == parent["devices"][stage["gpu"]]["uuid"]
              and probe["pid_allocation"]["own_gpu_uuids"] == [selected["uuid"]]
              and set(probe["scheduler_nvidia_visible_devices"]) == set(parent["gpu_identity_mapping"]["assigned_nvml_uuids"]))
        check(arm + ":finite_training_metrics", finite_numbers(updates))
        if method.startswith("dsrl"):
            check(arm + ":optimizer_budget", summary["optimizer_steps"] == dict(actor=5980, critic=5980, noise_critic=2990, entropy=5980)
                  and summary["learner_updates"] == 5980 and summary["warmup_primitive_steps"] == 24016)
            check(arm + ":motor_budget", summary["training_motor_samples"] == 7200 + 299 * 30 * 256
                  and summary["evaluation_motor_samples"] == 2400)
            recovery = read("campaign/" + arm + "/recovery-manifest.json")
            check(arm + ":recovery_manifest", digest(base / "recovery-manifest.json") == stage["recovery"]["manifest_sha256"])
            for name, expected in recovery["files"].items():
                check(arm + ":postrun_recovery:" + name,
                      manifest["postrun_recoveries"][arm][name]["sha256"] == expected == stage["recovery"]["files"][name])
            parameters, optimizer_steps, replay_error = summary["trainable_parameters"], sum(summary["optimizer_steps"].values()), None
        else:
            check(arm + ":six_complete_ppo_updates", len(updates) == 6 and [row["update"] for row in updates] == list(range(1, 7))
                  and [row["primitive_steps"] for row in updates] == list(range(4800, 28801, 4800)))
            check(arm + ":ppo_budget", summary["updates"] == 6 and summary["optimizer_steps"] == 24
                  and sum(row["optimizer_steps"] for row in updates) == 24
                  and summary["training_motor_samples"] == summary["evaluation_motor_samples"] == 7200)
            replay_error = max(row["replay_error"] for row in updates)
            check(arm + ":replay_and_gradients", replay_error == 0 and all(row["gradient_norm"] > 0 for row in updates))
            check(arm + ":postrun_recovery", manifest["postrun_recoveries"][arm]["recovery.pt"]["sha256"] == stage["recovery"]["recovery_sha256"])
            parameters, optimizer_steps = summary["parameters"], summary["optimizer_steps"]
        budgets[arm] = dict(training_primitive_steps=summary["primitive_steps"],
                            warmup_primitive_steps=summary.get("warmup_primitive_steps", 0),
                            evaluation_primitive_steps=summary["evaluation_primitive_steps"], parameters=parameters,
                            optimizer_invocations=optimizer_steps, optimizer_step_breakdown=summary["optimizer_steps"],
                            training_motor_samples=summary["training_motor_samples"], evaluation_motor_samples=summary["evaluation_motor_samples"],
                            whole_method_elapsed_seconds=summary["elapsed_seconds"], peak_cuda_memory_gib=summary["peak_cuda_memory_gib"],
                            progress_records=len(updates), max_recorded_replay_error=replay_error)
        for condition, rows in evaluation.items():
            key = method + "/" + condition
            check(arm + "/" + condition + ":scene_geometry", [row["scene_seeds"] for row in rows] == [[s] for s in SCENES]
                  and all(row["primitive_steps"] == 1200 and row["attempts"] == 4 for row in rows))
            if method in ("rl2", "episode-reset"):
                expected = method == "episode-reset" or condition == "memory_reset"
                check(arm + "/" + condition + ":reset_contract", all(row["reset_each_attempt"] is expected for row in rows))
            for row in rows:
                scene = str(row["scene_seeds"][0])
                check(arm + "/" + condition + ":paired_reset:" + scene,
                      row["initial_observation_sha256"] == [plan["paired_scenes"][scene]] * 4)
            per_seed[str(seed)][key], matrices[seed][key] = {}, {}
            for metric in METRICS:
                matrix = [row[metric][0] for row in rows]
                check(arm + "/" + condition + ":binary_matrix:" + metric,
                      all(len(row) == 4 and all(type(v) is bool for v in row) for row in matrix))
                per_seed[str(seed)][key][metric] = summarize(matrix)
                matrices[seed][key][metric] = matrix

for name, key in (("state_5000.pt", "checkpoint_sha256"), ("normalization.npz", "normalization_sha256")):
    check("postrun_frozen_input:" + name, manifest["inputs_postrun"][name]["sha256"] == plan["runtime_identity"][key])

comparators = ["rl2/memory_reset", "episode-reset/normal", "rl2/reward_blind", "dsrl-na/frozen", "dsrl-na-retry/frozen"]
effects, pooled = {}, {}
for seed in SEEDS:
    effects[str(seed)] = {name: {metric: paired(matrices[seed]["rl2/normal"][metric], matrices[seed][name][metric])
                                 for metric in METRICS} for name in comparators}
    check(f"seed{seed}:trained_reset_identity_control", matrices[seed]["episode-reset/normal"] == matrices[seed]["episode-reset/memory_reset"])
for config in per_seed["1"]:
    pooled[config] = {}
    for metric in METRICS:
        values = [per_seed[str(seed)][config][metric] for seed in SEEDS]
        pooled[config][metric] = dict(seed_success_totals=[row["successes"] for row in values],
                                      successes=sum(row["successes"] for row in values), recorded_episodes=64,
                                      descriptive_episode_success_rate=sum(row["successes"] for row in values) / 64,
                                      per_attempt_success_counts=[sum(row["per_attempt_success_counts"][i] for row in values) for i in range(4)],
                                      scenes_with_any_success_by_training_seed=[row["scenes_with_any_success"] for row in values])
pooled_effects = {name: {metric: dict(seed_total_success_count_differences=[effects[str(seed)][name][metric]["total_success_count_difference"] for seed in SEEDS],
                                     descriptive_total_success_count_difference=sum(effects[str(seed)][name][metric]["total_success_count_difference"] for seed in SEEDS),
                                     mean_scene_rate_difference=sum(effects[str(seed)][name][metric]["total_success_count_difference"] for seed in SEEDS) / 64)
                         for metric in METRICS} for name in comparators}

native_rows = read("native-qual-r4/evaluation.json")["native"]
native_contract = read("native-qual-r4/contract.json")
historical_native = {metric: summarize([row[metric][0] for row in native_rows]) for metric in METRICS}
check("historical_native_scene_identity", all(row["initial_observation_sha256"] == [plan["paired_scenes"][str(row["scene_seeds"][0])]] * 4 for row in native_rows))

result = dict(generated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
              source_commit=admission["source_commit"], run_id=plan["run_id"], training_seeds=SEEDS, evaluation_scene_seeds=SCENES,
              inference_unit="two training seeds crossed with the same eight paired initial scenes; four retries per cell; not 64 independent trials",
              checks=checks, failed_checks=[row for row in checks if not row["passed"]],
              per_seed=per_seed, pooled_descriptive=pooled, paired_scene_effects=effects, pooled_descriptive_effects=pooled_effects,
              historical_native_reference=dict(source_commit=native_contract["source_commit"], seed=native_contract["seed"],
                                                metrics=historical_native, included_in_final_pooled_counts=False,
                                                limitation="old 4090 qualification reference; no newly measured seed2 native baseline"),
              method_budgets=budgets,
              campaign_compute=dict(training_primitive_steps=sum(row["training_primitive_steps"] for row in budgets.values()),
                                     evaluation_primitive_steps=sum(row["evaluation_primitive_steps"] for row in budgets.values()),
                                     optimizer_invocations=sum(row["optimizer_invocations"] for row in budgets.values()),
                                     training_motor_samples=sum(row["training_motor_samples"] for row in budgets.values()),
                                     driver_elapsed_seconds=admission["elapsed_seconds"],
                                     driver_stages_end_elapsed_seconds=stage_events[-1]["elapsed_seconds"],
                                     driver_based_gpu_hours=admission["allocation_gpu_hours"],
                                     combined_driver_gpu_hours_with_prior_budget_reserve=admission["combined_gpu_hours_with_prior_reserve"],
                                     scheduler_billing_or_terminal_timestamp_established=False),
              resource_identity=admission["resource_identity"],
              raw_remote_manifest_sha256=digest(RAW / "remote-manifest.json"), analysis_script_sha256=digest(Path(__file__)),
              limitations=["Only stdlib artifact metadata/counts/hashes used; no model/tensor/replay/optimizer/simulator/test execution.",
                           "Checkpoint payload hashes were independently streamed on Baige and no checkpoint was downloaded or deserialized.",
                           "Two seeds and eight shared scenes are too few for broad method rankings; no independence assumption or significance test on 64 episode records.",
                           "The earlier 4090 pilot repeats seed1 with another source/hardware execution. It is not pooled as an independent third training seed and no bitwise reproducibility claim is made.",
                           "The historical native reference is separate and has no matching newly measured seed2 native evaluation.",
                           "All training interaction budgets match at 28800, but optimizer, parameter and motor-query budgets differ; recurrent policies have only six PPO updates.",
                           "Frozen motor identity is contract/qualified-driver workflow evidence; source checkpoint hashes agree but no independent model-tensor inspection was performed.",
                           "Raw hardware identifier is NVIDIA RPBZZZ6 in the user-approved Pro6K queue; no marketing model decoding is asserted.",
                           "GPU-hours are computed from eight GPUs times the driver elapsed clock, not an independently established scheduler allocation/billing interval.",
                           "This is unchanged low-dimensional Can, not VLA evidence, convergence evidence, cross-task meta-learning, or an online DSRL evaluation."])
(ROOT / "RESULTS.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(dict(checks=len(checks), failed_checks=result["failed_checks"], pooled_successes={name:value["successes"]["successes"] for name,value in pooled.items()},
                      primary_effects=pooled_effects), indent=2))
