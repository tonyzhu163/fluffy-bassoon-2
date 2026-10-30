"""Assemble source40 learning histories and train matched frozen-VLA controllers.

Assembly selects one final committed AD/DPT prefix per task and partition.
Training is imitation of actual source histories, not a robot-performance test.
"""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np

from .vla_history import FIELDS, file_sha256, load_dataset, save_dataset
from .vla_source40_corpus import PI05_RECEIPT
from .vla_source40_schedule import SOURCE_SUITES, content_hash


INDEX_FORMAT = "openpi_pi05_libero40_source_history_index_v1"
ASSEMBLY_FORMAT = "openpi_pi05_libero40_training_assembly_v1"
FEATURE_KIND = "openpi_final_prefix_visual_mean_language_mean_public_state_v2"
NOISE_CONTRACT = "32D tied across native H10; warmup unbounded N(0,I), then SAC clipped[-1,1]; no rescaling"
ALL_TASKS = {f"{suite}:{i:02d}" for suite in SOURCE_SUITES for i in range(10)}
BUDGET_VALIDATION_TASKS = {f"{suite}:{i:02d}" for suite in SOURCE_SUITES for i in (0, 1)}
CAMPAIGN_FORMAT = "libero40_source_1m_campaign_v1"
OBJECTIVES = ("ad", "dpt")
MODES = ("full", "static")


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_campaign(path=None, expected_sha256=None):
    if path is None and expected_sha256 is None:
        return None
    if path is None or expected_sha256 is None or file_sha256(path) != expected_sha256:
        raise ValueError("Campaign file requires its matching explicit SHA256")
    value = _json(path)
    train, validation = value.get("source_training_task_keys", []), value.get("source_validation_task_keys", [])
    if (value.get("format") != CAMPAIGN_FORMAT or value.get("maximum_total_actions") != 1_000_000
            or value.get("campaign_sha256") != content_hash({k: v for k, v in value.items() if k != "campaign_sha256"})
            or len(train) != len(ALL_TASKS) or set(train) != ALL_TASKS
            or len(validation) != len(BUDGET_VALIDATION_TASKS) or set(validation) != BUDGET_VALIDATION_TASKS):
        raise ValueError("Require the hash-bound all40-training/fixed8-validation 1M-action campaign")
    return dict(path=str(Path(path).resolve()), file_sha256=expected_sha256,
        campaign_sha256=value["campaign_sha256"], maximum_total_actions=value["maximum_total_actions"],
        source_training_task_keys=train, source_validation_task_keys=validation)


def _member(root, relative):
    path = Path(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("Source artifact must be relative to its history directory")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Source artifact escapes its history directory")
    return resolved


def _validate_physical_budget(index, contract, *, complete_prefix=False):
    """Verify the physical ledger, independent of advertised candidate-bank size."""
    cap = index.get("maximum_total_actions") if complete_prefix else contract.get("maximum_total_actions")
    costs = index.get("physical_cost_accounting", {})
    fields = {name: costs.get(name) for name in
              ("known_primitive_steps", "uncertain_primitive_steps_ceiling", "primitive_step_budget_charge")}
    fields.update({name: index.get(name) for name in
        ("actual_primitive_steps", "actual_settling_steps", "settling_step_budget_charge",
         "uncertain_settling_steps_ceiling", "total_action_budget_charge")})
    if (type(cap) is not int or cap < 1 or index.get("maximum_total_actions") != cap
            or any(type(value) is not int or value < 0 for value in fields.values())):
        raise ValueError("Physical-budget history requires matching explicit action cap and ledger counts")
    if (fields["known_primitive_steps"] != fields["actual_primitive_steps"]
            or fields["primitive_step_budget_charge"] != fields["known_primitive_steps"] + fields["uncertain_primitive_steps_ceiling"]
            or fields["settling_step_budget_charge"] != fields["actual_settling_steps"] + fields["uncertain_settling_steps_ceiling"]
            or fields["total_action_budget_charge"] != fields["primitive_step_budget_charge"] + fields["settling_step_budget_charge"]
            or fields["total_action_budget_charge"] > cap
            or index.get("unused_action_reservation") != cap - fields["total_action_budget_charge"]):
        raise ValueError("Physical-budget ledger omits or exceeds charged primitive/settling actions")
    coverage = index.get("case_coverage", {})
    if (index.get("all_started_cases_complete") is not True
            or coverage.get("all_started_cases_complete") is not True
            or index.get("stopping_reason") not in (("immutable_complete_prefix",) if complete_prefix else
                ("physical_action_budget", "candidate_bank_exhausted"))):
        raise ValueError("Physical-budget history must stop at complete actual case boundaries")


def load_source(path, partition, *, allow_small_pilot=False, prefix_snapshot_admission=None):
    """Validate immutable receipts and retain exactly one final paired prefix."""
    path = Path(path).resolve()
    index, root = _json(path), path.parent
    contract = _json(root / "contract.json")
    complete_prefix = prefix_snapshot_admission is not None
    if complete_prefix:
        from .vla_source40_topup_assembly import verify_snapshot
        verify_snapshot(path, partition, prefix_snapshot_admission)
    task = index.get("task_key")
    if (index.get("format") != INDEX_FORMAT or (index.get("completed") is not True and not complete_prefix)
            or index.get("source_only") is not True or task not in ALL_TASKS
            or index.get("partition") != partition or index.get("source_updates", 0) <= 0
            or not (index.get("collection_qualified") or index.get("smoke_qualified"))):
        raise ValueError("Require a completed qualified source40 history in the declared partition")
    mode = index.get("collection_mode")
    if (not allow_small_pilot and (mode not in ("complete_cases", "physical_budget")
            or (mode == "complete_cases" and index.get("case_coverage", {}).get("all_requested_cases_complete") is not True))):
        raise ValueError("Full collection requires actual complete-case coverage; smoke needs --allow-small-pilot")
    feature = index.get("feature_contract", {})
    if (index.get("motor_receipt_sha256") != PI05_RECEIPT
            or feature.get("checkpoint_receipt_sha256") != PI05_RECEIPT
            or feature.get("kind") != FEATURE_KIND or feature.get("model_config") != "pi05_libero"
            or feature.get("observation_dim") != 4104 or index.get("noise_contract") != NOISE_CONTRACT):
        raise ValueError("Require the official OpenPI40 motor, prefix-v2 features and tied-H10 noise contract")
    if (contract.get("format") != "openpi_pi05_libero40_persistent_source_v1"
            or content_hash(contract) != index.get("source_contract_sha256")
            or contract.get("source_only") is not True or contract.get("source_data_from_targets") is not False
            or contract.get("task", {}).get("task_key") != task or contract.get("partition") != partition
            or contract["task"].get("suite") != task.split(":")[0]
            or contract["task"].get("task_id") != int(task.split(":")[1])
            or contract.get("feature_contract") != feature or contract.get("noise_contract") != NOISE_CONTRACT
            or contract.get("motor", {}).get("receipt_sha256") != PI05_RECEIPT
            or contract.get("motor", {}).get("weights_frozen") is not True):
        raise ValueError("History index and hash-bound source contract differ")
    if mode == "physical_budget":
        if contract.get("collection_mode") != mode:
            raise ValueError("Physical-budget collection mode differs from its hash-bound contract")
        _validate_physical_budget(index, contract, complete_prefix=complete_prefix)
    paired = [row for row in index.get("publications", []) if row.get("status") == "paired_ad_dpt"]
    if not paired or len({row["through_case"] for row in paired}) != len(paired):
        raise ValueError("Require unique committed paired-prefix versions")
    selected = max(paired, key=lambda row: row["through_case"])
    receipts = index.get("episode_receipts", [])
    if (selected.get("transitions") != index.get("chunks")
            or selected.get("episodes") != len(receipts)
            or selected.get("case_count") != selected["through_case"] + 1):
        raise ValueError("Latest committed prefix must cover the entire completed history")
    checkpoint = _member(root, selected["source_checkpoint"])
    if file_sha256(checkpoint) != selected["source_checkpoint_sha256"]:
        raise ValueError("Source actor checkpoint checksum differs")
    cases = contract["cases"][:selected["through_case"] + 1]
    if (len(cases) != selected["case_count"] or len({case["case_id"] for case in cases}) != len(cases)
            or (not allow_small_pilot and mode == "complete_cases" and len(cases) != len(contract["cases"]))):
        raise ValueError("Invalid complete source case identities")
    if mode == "physical_budget":
        coverage = index["case_coverage"]
        if (coverage.get("started_cases") != len(cases) or coverage.get("completed_cases") != len(cases)
                or coverage.get("available_cases") != len(contract["cases"])
                or (index["stopping_reason"] == "candidate_bank_exhausted" and len(cases) != len(contract["cases"]))):
            raise ValueError("Physical-budget actual case coverage differs from selected complete prefix")
        if not complete_prefix and index["stopping_reason"] == "physical_action_budget" and len(cases) < len(contract["cases"]):
            next_case = contract["cases"][len(cases)]
            required = next_case["episodes"] * (contract["horizon"] + 10)
            if index["unused_action_reservation"] >= required:
                raise ValueError("Physical-budget history stopped before exhausting its next complete-case reservation")
    expected = {case["case_id"]: case for case in cases}
    if [(row["case_id"], row["episode_in_case"]) for row in receipts] != [
            (case["case_id"], episode) for case in cases for episode in range(case["episodes"])]:
        raise ValueError("Actual episode receipts must follow complete chronological cases")
    seen, states, state_hashes = {}, set(), set()
    for row in receipts:
        case = expected.get(row["case_id"])
        bank = range(40) if partition == "train" else range(40, 50)
        if (case is None or case.get("split") != partition or row["initial_state"] not in bank
                or row["initial_state"] != case["initial_state"] or row["scene_seed"] != case["scene_seed"]
                or row.get("suite") != contract["task"]["suite"]
                or row.get("task_id") != contract["task"]["task_id"]
                or not isinstance(row.get("initial_state_vector_sha256"), str)
                or len(row["initial_state_vector_sha256"]) != 64):
            raise ValueError("Actual episode receipt leaks across source case/state partitions")
        episode = row["episode_in_case"]
        lane = seen.setdefault(case["case_id"], set())
        if episode in lane or not 0 <= episode < case["episodes"]:
            raise ValueError("Duplicate or invalid actual episode identity")
        lane.add(episode)
        states.add((task, row["initial_state"]))
        state_hashes.add((task, row["initial_state_vector_sha256"]))
    if (any(seen.get(case["case_id"]) != set(range(case["episodes"])) for case in cases)
            or sum(row["chunks"] for row in receipts) != index["chunks"]):
        raise ValueError("Selected history contains an incomplete case or mismatched actual chunks")
    datasets, manifests = {}, {}
    for objective in OBJECTIVES:
        artifact = selected["datasets"][objective]
        trials, manifest = load_dataset(_member(root, artifact["path"]), expected_sha256=artifact["sha256"])
        metadata = manifest["metadata"]
        if (len(trials) != 1 or manifest["objective"] != objective or manifest["transitions"] != index["chunks"]
                or (manifest["observation_dim"], manifest["noise_dim"]) != (4104, 32)
                or metadata.get("source_contract") != contract or metadata.get("global_task_split") != "source40"
                or metadata.get("local_scene_split") != partition or metadata.get("task_key") != task
                or metadata.get("trial_ids") != [contract["history_id"]]
                or metadata.get("source_case_ids") != [case["case_id"] for case in cases]
                or metadata.get("source_actor_checkpoint_sha256") != selected["source_checkpoint_sha256"]
                or metadata.get("motor_sha256") != PI05_RECEIPT or metadata.get("feature_contract") != feature
                or metadata.get("noise_contract") != NOISE_CONTRACT
                or metadata.get("source_generation_episodes") != len(receipts)
                or metadata.get("source_generation_chunks") != index["chunks"]):
            raise ValueError("Paired dataset provenance does not match the final source history")
        if int(trials[0]["dones"].sum()) != len(receipts):
            raise ValueError("Dataset boundaries differ from actual complete episode receipts")
        datasets[objective], manifests[objective] = trials[0], manifest
    for field in set(FIELDS) - {"target_noise"}:
        if not np.array_equal(datasets["ad"][field], datasets["dpt"][field]):
            raise ValueError("AD/DPT must share identical actual histories; only target labels may differ")
    return dict(task=task, history_id=contract["history_id"], collection_mode=mode,
        index_path=str(path), index_sha256=file_sha256(path),
        feature_contract=feature, selected=selected, datasets=datasets, manifests=manifests,
        states=states, state_hashes=state_hashes, case_ids=set(expected),
        costs={**{key: index.get(key) for key in ("actual_primitive_steps", "actual_settling_steps", "source_updates")},
            "charged_total_actions": index.get("total_action_budget_charge", index["actual_primitive_steps"] + index["actual_settling_steps"]),
            "uncertain_primitive_steps_ceiling": index.get("physical_cost_accounting", {}).get("uncertain_primitive_steps_ceiling", 0),
            "uncertain_settling_steps_ceiling": index.get("uncertain_settling_steps_ceiling", 0)},
        episodes=len(receipts))


def assemble(train_indices, validation_indices, output, *, allow_small_pilot=False,
             campaign_path=None, campaign_sha256=None, topup_admission_path=None, topup_admission_sha256=None):
    """Join different tasks, never different prefix versions of one history."""
    if topup_admission_path is not None or topup_admission_sha256 is not None:
        if campaign_path is not None or campaign_sha256 is not None or allow_small_pilot:
            raise ValueError('Top-up complete-prefix admission is separate from legacy/pilot campaign admission')
        from .vla_source40_topup_assembly import load_admission
        campaign = load_admission(topup_admission_path, topup_admission_sha256)
        prefix_admission = campaign
    else:
        campaign = load_campaign(campaign_path, campaign_sha256)
        prefix_admission = None
    expected_tasks = {"train": ALL_TASKS,
        "validation": BUDGET_VALIDATION_TASKS if campaign is not None else ALL_TASKS}
    source_options = dict(allow_small_pilot=allow_small_pilot)
    if prefix_admission is not None:
        source_options['prefix_snapshot_admission'] = prefix_admission
    groups = {partition: [load_source(path, partition, **source_options) for path in paths]
              for partition, paths in (("train", train_indices), ("validation", validation_indices))}
    for partition, sources in groups.items():
        tasks = [row["task"] for row in sources]
        if not tasks or len(set(tasks)) != len(tasks):
            raise ValueError("Provide exactly one source index per task and partition; prefixes cannot be concatenated")
        if not allow_small_pilot and set(tasks) != expected_tasks[partition]:
            raise ValueError("Default assembly requires all40 training tasks and the campaign's complete validation task set")
        groups[partition] = sorted(sources, key=lambda row: row["task"])
    train_tasks, validation_tasks = ({row["task"] for row in groups[partition]} for partition in ("train", "validation"))
    if (not validation_tasks.issubset(train_tasks)
            or (campaign is None and train_tasks != validation_tasks)):
        raise ValueError("Different validation task coverage requires the declared all40/fixed8 campaign")
    sources = groups["train"] + groups["validation"]
    if not allow_small_pilot and campaign is None and any(row["collection_mode"] == "physical_budget" for row in sources):
        raise ValueError("Full physical-budget assembly requires the explicit 1M-action campaign")
    if campaign is not None and sum(row["costs"]["charged_total_actions"] for row in sources) > campaign["maximum_total_actions"]:
        raise ValueError("Source histories exceed the campaign's total physical-action ceiling")
    if len({content_hash(row["feature_contract"]) for row in sources}) != 1:
        raise ValueError("Frozen feature contracts differ across source histories")
    for key in ("history_id", "case_ids", "states", "state_hashes"):
        def identities(partition):
            return {identity for row in groups[partition] for identity in
                    ([row[key]] if key == "history_id" else row[key])}
        if identities("train") & identities("validation"):
            raise ValueError(f"Train/validation source overlap: {key}")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(format=ASSEMBLY_FORMAT, source_only=True, motor_sha256=PI05_RECEIPT,
        feature_contract=sources[0]["feature_contract"], noise_contract=NOISE_CONTRACT,
        small_pilot=allow_small_pilot, source_task_keys=[row["task"] for row in groups["train"]],
        validation_task_keys=[row["task"] for row in groups["validation"]], campaign=campaign,
        datasets={}, selected_sources={}, collection_costs_count_once={},
        interpretation="Source imitation candidates only; no claim of robot gain or in-context adaptation")
    for partition, rows in groups.items():
        selected_sources = [dict(index_path=row["index_path"], index_sha256=row["index_sha256"],
            task=row["task"], history_id=row["history_id"], selected_publication=row["selected"],
            episodes=row["episodes"], case_ids=sorted(row["case_ids"]),
            task_initial_state_ids=[list(value) for value in sorted(row["states"])],
            task_nominal_state_hashes=[list(value) for value in sorted(row["state_hashes"])]) for row in rows]
        result["selected_sources"][partition] = selected_sources
        result["collection_costs_count_once"][partition] = {key: sum(row["costs"][key] for row in rows)
            for key in ("actual_primitive_steps", "actual_settling_steps", "source_updates", "charged_total_actions",
                        "uncertain_primitive_steps_ceiling", "uncertain_settling_steps_ceiling")}
        result["datasets"][partition] = {}
        for objective in OBJECTIVES:
            path = output / f"{partition}-{objective}.npz"
            manifest = save_dataset(path, [row["datasets"][objective] for row in rows], objective=objective,
                metadata=dict(motor_sha256=PI05_RECEIPT, feature_contract=result["feature_contract"],
                    noise_contract=NOISE_CONTRACT, global_task_split="source40", local_scene_split=partition,
                    trial_ids=[row["history_id"] for row in rows], source_case_ids=[case for row in rows for case in sorted(row["case_ids"])],
                    source_task_keys=[row["task"] for row in rows], campaign=campaign, selected_source_histories=selected_sources,
                    small_pilot=allow_small_pilot, publication_rule="One latest complete paired prefix per task; overlapping versions excluded",
                    teacher_status="Source SAC candidates; no expert/optimality claim"))
            result["datasets"][partition][objective] = dict(path=path.name, sha256=manifest["sha256"],
                trials=manifest["trials"], transitions=manifest["transitions"])
    result["assembly_sha256"] = content_hash(result)
    _write(output / "assembly.json", result)
    return result


def load_assembly(path, *, allow_small_pilot=False):
    path = Path(path).resolve()
    value = _json(path)
    if (value.get("format") != ASSEMBLY_FORMAT or value.get("source_only") is not True
            or value.get("motor_sha256") != PI05_RECEIPT
            or value.get("assembly_sha256") != content_hash({k: v for k, v in value.items() if k != "assembly_sha256"})):
        raise ValueError("Assembly format or hash differs")
    if value.get("small_pilot") and not allow_small_pilot:
        raise ValueError("Small technical corpus requires explicit --allow-small-pilot")
    if not value.get("small_pilot") and set(value["source_task_keys"]) != ALL_TASKS:
        raise ValueError("Full assembly must cover all40 source tasks")
    campaign = value.get("campaign")
    expected_validation = BUDGET_VALIDATION_TASKS if campaign is not None else ALL_TASKS
    validation_tasks = value.get("validation_task_keys", value["source_task_keys"])
    if not value.get("small_pilot") and set(validation_tasks) != expected_validation:
        raise ValueError("Assembly validation task set differs from its declared campaign")
    if campaign is not None:
        # The campaign must remain independently available and hash-bound when
        # the assembled data are admitted for controller training.
        if campaign.get('kind') == 'source40_topup_complete_prefix_admission_v1':
            from .vla_source40_topup_assembly import load_admission
            actual_campaign = load_admission(campaign['path'], campaign['file_sha256'])
        else:
            actual_campaign = load_campaign(campaign["path"], campaign["file_sha256"])
        if actual_campaign != campaign:
            raise ValueError("Assembly campaign provenance differs")
    for partition in ("train", "validation"):
        for objective in OBJECTIVES:
            item = value["datasets"][partition][objective]
            _, manifest = load_dataset(_member(path.parent, item["path"]), expected_sha256=item["sha256"])
            if (manifest["objective"] != objective or manifest["metadata"].get("local_scene_split") != partition
                    or manifest["metadata"].get("motor_sha256") != value["motor_sha256"]
                    or manifest["metadata"].get("feature_contract") != value["feature_contract"]
                    or manifest["metadata"].get("noise_contract") != value["noise_contract"]
                    or manifest["metadata"].get("source_task_keys") != (value["source_task_keys"] if partition == "train" else validation_tasks)
                    or manifest["metadata"].get("campaign") != campaign
                    or manifest["metadata"].get("selected_source_histories") != value["selected_sources"][partition]
                    or manifest["metadata"].get("trial_ids") != [row["history_id"] for row in value["selected_sources"][partition]]):
                raise ValueError("Assembled dataset contract differs")
    return value


def training_commands(args, assembly):
    """Every objective/memory control receives the same explicit optimizer budget."""
    commands = []
    for seed in args.seeds:
        for objective in OBJECTIVES:
            for mode in MODES:
                output = args.output_dir / f"{objective}-{mode}-seed-{seed}"
                train, validation = (assembly["datasets"][partition][objective] for partition in ("train", "validation"))
                command = [sys.executable, "-m", "fluffy_bassoon.meta_rl.vla_history_train",
                    "--train-data", str(args.assembly.parent / train["path"]), "--train-sha256", train["sha256"],
                    "--validation-data", str(args.assembly.parent / validation["path"]), "--validation-sha256", validation["sha256"],
                    "--output-dir", str(output), "--mode", mode, "--seed", str(seed),
                    "--source-commit", args.source_commit, "--expected-hostname", args.expected_hostname]
                for name in ("updates", "batch_size", "learning_rate", "width", "layers", "heads", "context_length",
                             "max_window", "eval_every", "validation_batches", "max_elapsed_seconds", "threads", "device"):
                    command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
                commands.append(dict(objective=objective, mode=mode, seed=seed, output=output, command=command))
    return commands


def train_matrix(args):
    if sys.platform != "linux" or socket.gethostname() != args.expected_hostname:
        raise ValueError("Train only on the explicitly identified allocated Linux host")
    import torch
    from .vla_history import HistoryConfig, HistoryNoisePolicy, export_policy, load_policy
    from .vla_history_train import RESUME_FORMAT
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()):
        raise ValueError("Require the exact clean reviewed source commit")
    assembly = load_assembly(args.assembly, allow_small_pilot=args.allow_small_pilot)
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**32 for seed in args.seeds):
        raise ValueError("Require distinct valid seeds")
    commands = training_commands(args, assembly)
    contract = dict(format="openpi_pi05_libero40_matched_training_v1", assembly_sha256=assembly["assembly_sha256"],
        source_commit=args.source_commit, seeds=args.seeds,
        commands=[dict(objective=row["objective"], mode=row["mode"], seed=row["seed"], command=row["command"]) for row in commands],
        final_selection="Requested fixed final update; candidate.pt is separately the best validation NLL export",
        cost="Collection counted once; each of four controllers per seed has its own updates and training time")
    if args.resume:
        if _json(args.output_dir / "contract.json") != contract:
            raise ValueError("Matrix resume contract differs")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        _write(args.output_dir / "contract.json", contract)
    completed = []
    for row in commands:
        output, command = row["output"], list(row["command"])
        recovery = output / "latest.pt"
        if args.resume and recovery.exists():
            saved = torch.load(recovery, map_location="cpu", weights_only=False)
            if saved.get("format") != RESUME_FORMAT or saved.get("update", 0) > args.updates:
                raise ValueError("Invalid matrix controller recovery")
            run = saved["update"] < args.updates
            if run:
                command += ["--resume", str(recovery)]
        else:
            run = True
        if run:
            subprocess.run(command, check=True)
        saved = torch.load(recovery, map_location="cpu", weights_only=False)
        if saved.get("format") != RESUME_FORMAT or saved.get("update") != args.updates:
            raise RuntimeError("Controller stopped before fixed update budget; preserved recovery, no final candidate promoted")
        model = HistoryNoisePolicy(HistoryConfig(**saved["contract"]["config"]))
        model.load_state_dict(saved["policy"], strict=True)
        final_manifest = dict(contract=saved["contract"], update=saved["update"],
            selection="fixed requested final update", assembly_sha256=assembly["assembly_sha256"],
            full_recovery_sha256=file_sha256(recovery), status="Unvalidated robot-control candidate")
        if (output / "final.pt").exists():
            previous, metadata = load_policy(output / "final.pt", expected_sha256=file_sha256(output / "final.pt"))
            if metadata != final_manifest or any(not torch.equal(value, previous.state_dict()[key])
                    for key, value in model.state_dict().items()):
                raise ValueError("Existing immutable final candidate differs from its recovery")
        else:
            export_policy(output / "final.pt", model, manifest=final_manifest)
        completed.append(dict(objective=row["objective"], mode=row["mode"], seed=row["seed"], updates=saved["update"],
            elapsed_seconds=saved["elapsed_seconds"], final_path=str(output / "final.pt"),
            final_sha256=file_sha256(output / "final.pt"), full_recovery_path=str(recovery), full_recovery_sha256=file_sha256(recovery)))
        _write(args.output_dir / "index.json", dict(contract=contract, completed=len(completed) == len(commands), controllers=completed))
    return completed


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="operation", required=True)
    assemble_parser = sub.add_parser("assemble")
    assemble_parser.add_argument("--train-index", type=Path, nargs="+", required=True)
    assemble_parser.add_argument("--validation-index", type=Path, nargs="+", required=True)
    assemble_parser.add_argument("--output-dir", type=Path, required=True)
    assemble_parser.add_argument("--allow-small-pilot", action="store_true")
    assemble_parser.add_argument("--campaign", type=Path)
    assemble_parser.add_argument("--campaign-sha256")
    train = sub.add_parser("train")
    train.add_argument("--assembly", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--seeds", type=int, nargs="+", required=True)
    train.add_argument("--updates", type=int, required=True)
    train.add_argument("--source-commit", required=True)
    train.add_argument("--expected-hostname", required=True)
    train.add_argument("--max-elapsed-seconds", type=float, required=True, help="Per-controller cumulative ceiling")
    for name, default in (("batch-size", 8), ("width", 128), ("layers", 2), ("heads", 4),
                          ("context-length", 300), ("max-window", 128), ("eval-every", 100),
                          ("validation-batches", 4), ("threads", 4)):
        train.add_argument("--" + name, type=int, default=default)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--allow-small-pilot", action="store_true")
    train.add_argument("--resume", action="store_true")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if args.operation == "assemble":
        assemble(args.train_index, args.validation_index, args.output_dir, allow_small_pilot=args.allow_small_pilot,
                 campaign_path=args.campaign, campaign_sha256=args.campaign_sha256)
    else:
        args.assembly, args.output_dir = args.assembly.resolve(), args.output_dir.resolve()
        train_matrix(args)


if __name__ == "__main__":
    main()
