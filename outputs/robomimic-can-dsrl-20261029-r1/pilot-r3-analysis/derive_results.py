"""Derive descriptive scene-paired results from mirrored JSON; stdlib only."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
METHODS = ["dsrl-na", "dsrl-na-retry", "rl2", "episode-reset"]
SCENES = list(range(2_000_000, 2_000_008))


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


checks = []


def check(name, passed, evidence=None):
    checks.append(dict(name=name, passed=bool(passed), evidence=evidence))


manifest = read(RAW / "remote-manifest.json")
for name, entry in manifest.items():
    path = RAW / name
    check("mirror:" + name, path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"])

admission = read(RAW / "pilot/pilot-admission.json")
plan = read(RAW / "pilot/plan.json")
qualification = read(RAW / "qualification/qualification/summary.json")
qual_contract = read(RAW / "qualification/qualification/contract.json")
native_contract = read(RAW / "qualification/native/contract.json")
parent_probe = read(RAW / "pilot/parent-gpu-probe.json")
check("completed_matched_admission", admission["matched_training_interactions"]
      and not admission["incomplete_arms"] and admission["target_steps"] == 28800)
check("parent_probe_hash", digest(RAW / "pilot/parent-gpu-probe.json") == plan["parent_probe_sha256"])
check("parent_two_gpu_probe", parent_probe["status"] == "passed" and len(parent_probe["devices"]) == 2)
check("qualification_passed", qualification["passed"] and qualification["motor_unchanged"]
      and qualification["environment_parity"]["passed"] and qualification["dsrl_recovery_roundtrip"]
      and qualification["ppo_recovery_roundtrip"]["passed"])
check("qualification_likelihood_replay", qualification["likelihood_replay_max_error"] == 0
      and qualification["ppo_full_trial_likelihood_replay_max_error"] == 0)

rows_by_config = {"native": read(RAW / "qualification/native/evaluation.json")["native"]}
contracts, budgets = {}, {}
stage_by_method = {stage["method"]: stage for stage in admission["stage_records"]}
for method in METHODS:
    directory = RAW / "pilot" / method
    contract = contracts[method] = read(directory / "contract.json")
    summary = read(directory / "summary.json")
    stage = stage_by_method[method]
    updates = [json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
    for condition, rows in read(directory / "evaluation.json").items():
        rows_by_config[method + "/" + condition] = rows
    check(method + ":source_seed_target", contract["source_commit"] == admission["source_commit"]
          and contract["seed"] == 1 and contract["envs"] == 4 and contract["attempts"] == 4
          and contract["steps"] == 28800 and contract["eval_trials"] == 8)
    check(method + ":complete", stage["returncode"] == 0 and summary["completed"]
          and not summary["budget_reached"] and summary["primitive_steps"] == 28800)
    for filename, field in (("contract.json", "contract_sha256"), ("evaluation.json", "evaluation_sha256"),
                            ("gpu-probe.json", "gpu_probe_sha256")):
        check(method + ":recorded_hash:" + filename, digest(directory / filename) == stage[field])
    probe = read(directory / "gpu-probe.json")
    check(method + ":gpu_source", probe["status"] == "passed" and probe["source_commit"] == admission["source_commit"]
          and probe["parent_probe_sha256"] == plan["parent_probe_sha256"] and probe["parent_gpu_count"] == 2)
    numeric = []
    for row in updates:
        numeric.extend(value for value in row.values() if isinstance(value, (int, float)))
        numeric.extend(row.get("metrics", {}).values())
    check(method + ":finite_logged_training_metrics", all(math.isfinite(value) for value in numeric))
    if method.startswith("dsrl"):
        expected = dict(actor=5980, critic=5980, noise_critic=2990, entropy=5980)
        check(method + ":optimizer_budget", summary["optimizer_steps"] == expected and summary["learner_updates"] == 5980)
        check(method + ":warmup_budget", summary["warmup_primitive_steps"] == 24016)
        check(method + ":motor_query_budget", summary["training_motor_samples"] == 7200 + 299 * 30 * 256)
        check(method + ":recovery_manifest_recorded", digest(directory / "recovery-manifest.json") == stage["recovery"]["manifest_sha256"])
        optimizer_steps = sum(summary["optimizer_steps"].values())
        parameters = summary["trainable_parameters"]
        replay_error = None
    else:
        check(method + ":six_complete_updates", len(updates) == 6 and [row["update"] for row in updates] == list(range(1, 7))
              and [row["primitive_steps"] for row in updates] == list(range(4800, 28801, 4800)))
        check(method + ":ppo_optimizer_budget", summary["updates"] == 6 and summary["optimizer_steps"] == 24
              and sum(row["optimizer_steps"] for row in updates) == 24)
        replay_error = max(row["replay_error"] for row in updates)
        check(method + ":likelihood_replay", replay_error == 0)
        check(method + ":nonzero_finite_gradient", all(row["gradient_norm"] > 0 for row in updates))
        check(method + ":motor_query_budget", summary["training_motor_samples"] == 7200)
        optimizer_steps, parameters = summary["optimizer_steps"], summary["parameters"]
    budgets[method] = dict(training_primitive_steps=summary["primitive_steps"],
                           warmup_primitive_steps=summary.get("warmup_primitive_steps", 0),
                           evaluation_primitive_steps=summary["evaluation_primitive_steps"],
                           trainable_parameters=parameters, optimizer_steps=optimizer_steps,
                           optimizer_step_breakdown=summary["optimizer_steps"],
                           training_motor_samples=summary["training_motor_samples"],
                           evaluation_motor_samples=summary["evaluation_motor_samples"],
                           whole_method_elapsed_seconds=summary["elapsed_seconds"],
                           peak_cuda_memory_gib=summary["peak_cuda_memory_gib"],
                           training_progress_records=len(updates), max_logged_replay_error=replay_error)

identity_keys = ["checkpoint_sha256", "normalization_sha256", "frozen_motor_sha256", "robomimic_commit",
                 "robomimic_env_sha256", "runtime"]
for key in identity_keys:
    values = [contract[key] for contract in contracts.values()] + [qual_contract[key], native_contract[key]]
    check("common_identity:" + key, all(value == values[0] for value in values))
check("scientific_source_hashes_match_parent", plan["scientific_source_sha256"] == parent_probe["source_files_sha256"])

results, matrices, scene_hashes = {}, {}, {}
for config, rows in rows_by_config.items():
    check(config + ":scene_shape", len(rows) == 8 and [row["scene_seeds"] for row in rows] == [[seed] for seed in SCENES])
    check(config + ":episode_budget", sum(row["primitive_steps"] for row in rows) == 9600
          and all(row["attempts"] == 4 for row in rows))
    if "/" in config and config.split("/")[0] in ("rl2", "episode-reset"):
        method, condition = config.split("/")
        expected_reset = method == "episode-reset" or condition == "memory_reset"
        check(config + ":reset_contract", all(row["reset_each_attempt"] is expected_reset for row in rows))
    for row in rows:
        hashes = row["initial_observation_sha256"]
        check(config + ":retry_identity:" + str(row["scene_seeds"][0]), len(hashes) == 4 and all(item == hashes[0] for item in hashes))
        seed = str(row["scene_seeds"][0])
        scene_hashes.setdefault(seed, hashes[0])
        check(config + ":paired_scene:" + seed, scene_hashes[seed] == hashes[0])
    results[config], matrices[config] = {}, {}
    for metric in ("successes", "release_logged_successes"):
        matrix = [row[metric][0] for row in rows]
        check(config + ":binary_shape:" + metric, all(len(values) == 4 and all(type(v) is bool for v in values) for values in matrix))
        matrices[config][metric] = matrix
        results[config][metric] = dict(per_attempt_success_counts=[sum(values[index] for values in matrix) for index in range(4)],
                                       successes=sum(map(sum, matrix)), episodes_recorded=32,
                                       descriptive_episode_success_rate=sum(map(sum, matrix)) / 32,
                                       scene_success_counts=list(map(sum, matrix)),
                                       scenes_with_any_success=sum(any(values) for values in matrix),
                                       scenes=8, per_scene_bits=["".join(str(int(v)) for v in values) for values in matrix])

pairs = [("rl2/normal", name) for name in ("rl2/memory_reset", "episode-reset/normal", "rl2/reward_blind",
                                          "dsrl-na/frozen", "dsrl-na-retry/frozen", "native")]
pairs += [("episode-reset/normal", "episode-reset/memory_reset"), ("dsrl-na/frozen", "dsrl-na-retry/frozen")]
paired = {}
for left, right in pairs:
    key = left + " minus " + right
    paired[key] = {}
    for metric in ("successes", "release_logged_successes"):
        a, b = matrices[left][metric], matrices[right][metric]
        differences = [[int(x) - int(y) for x, y in zip(xs, ys)] for xs, ys in zip(a, b)]
        counts = list(map(sum, differences))
        paired[key][metric] = dict(scene_seeds=SCENES, per_scene_per_attempt_difference=differences,
                                   per_scene_success_count_difference=counts,
                                   per_attempt_success_count_difference=[sum(row[i] for row in differences) for i in range(4)],
                                   total_success_count_difference=sum(counts),
                                   mean_scene_rate_difference=sum(counts) / (8 * 4),
                                   positive_scenes=sum(value > 0 for value in counts),
                                   tied_scenes=sum(value == 0 for value in counts),
                                   negative_scenes=sum(value < 0 for value in counts))

payload = dict(generated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
               source_commit=admission["source_commit"], qualification_source_commit=qual_contract["source_commit"],
               training_seeds=[1], evaluation_scene_seeds=SCENES, inference_unit="paired initial scene; n=8, one training seed",
               raw_remote_manifest_sha256=digest(RAW / "remote-manifest.json"), analysis_script_sha256=digest(Path(__file__)),
               checks=checks, failed_checks=[entry for entry in checks if not entry["passed"]],
               configurations=results, scene_paired_differences=paired, budgets=budgets,
               pilot_total_training_primitive_steps=sum(item["training_primitive_steps"] for item in budgets.values()),
               pilot_total_evaluation_primitive_steps=sum(item["evaluation_primitive_steps"] for item in budgets.values()),
               pilot_elapsed_seconds=admission["elapsed_seconds"], pilot_allocated_gpu_minutes=admission["allocated_gpu_minutes"],
               pilot_gpu_names=[row["cuda_name"] for row in parent_probe["devices"]],
               online_evaluation_status=admission["online_status"],
               limitations=["No tests, model, tensor or simulator computation in this audit; stdlib JSON/hash/counts only.",
                            "Checkpoint tensors and optimizer/replay payloads were not read or copied; recovery evidence is launcher-recorded metadata.",
                            "One training seed and eight paired scenes; 32 episodes are repeated measurements within those scenes, not 32 independent trials.",
                            "Initial motor identity is recorded across contracts; qualification records motor_unchanged and successful driver exit is workflow evidence for its final frozen-weight check.",
                            "Total environment training budget is matched, while parameter, optimizer and decoder-query budgets are not.",
                            "DSRL-NA progress logging stops before the final update; completed summary counters, not the last progress row, establish its final budget.",
                            "This pilot contains frozen DSRL evaluation, without online weight-adaptation evaluation."])
(ROOT / "RESULTS.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
print(json.dumps(dict(failed_checks=payload["failed_checks"], raw_files=len(manifest), checks=len(checks),
                      primary=paired["rl2/normal minus episode-reset/normal"]["successes"]), indent=2))
