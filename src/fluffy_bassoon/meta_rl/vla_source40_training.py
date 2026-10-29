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
OBJECTIVES = ("ad", "dpt")
MODES = ("full", "static")


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _member(root, relative):
    path = Path(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("Source artifact must be relative to its history directory")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Source artifact escapes its history directory")
    return resolved


def load_source(path, partition, *, allow_small_pilot=False):
    """Validate immutable receipts and retain exactly one final paired prefix."""
    path = Path(path).resolve()
    index, root = _json(path), path.parent
    contract = _json(root / "contract.json")
    task = index.get("task_key")
    if (index.get("format") != INDEX_FORMAT or index.get("completed") is not True
            or index.get("source_only") is not True or task not in ALL_TASKS
            or index.get("partition") != partition or index.get("source_updates", 0) <= 0
            or not (index.get("collection_qualified") or index.get("smoke_qualified"))):
        raise ValueError("Require a completed qualified source40 history in the declared partition")
    if (not allow_small_pilot and (index.get("collection_mode") != "complete_cases"
            or index.get("case_coverage", {}).get("all_requested_cases_complete") is not True)):
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
            or (not allow_small_pilot and len(cases) != len(contract["cases"]))):
        raise ValueError("Invalid complete source case identities")
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
    return dict(task=task, history_id=contract["history_id"], index_path=str(path), index_sha256=file_sha256(path),
        feature_contract=feature, selected=selected, datasets=datasets, manifests=manifests,
        states=states, state_hashes=state_hashes, case_ids=set(expected),
        costs={key: index.get(key) for key in ("actual_primitive_steps", "actual_settling_steps", "source_updates")},
        episodes=len(receipts))


def assemble(train_indices, validation_indices, output, *, allow_small_pilot=False):
    """Join different tasks, never different prefix versions of one history."""
    groups = {partition: [load_source(path, partition, allow_small_pilot=allow_small_pilot) for path in paths]
              for partition, paths in (("train", train_indices), ("validation", validation_indices))}
    for partition, sources in groups.items():
        tasks = [row["task"] for row in sources]
        if not tasks or len(set(tasks)) != len(tasks):
            raise ValueError("Provide exactly one source index per task and partition; prefixes cannot be concatenated")
        if not allow_small_pilot and set(tasks) != ALL_TASKS:
            raise ValueError("Default assembly requires all40 source tasks in both partitions")
        groups[partition] = sorted(sources, key=lambda row: row["task"])
    if {row["task"] for row in groups["train"]} != {row["task"] for row in groups["validation"]}:
        raise ValueError("Train and validation must cover the same source task set")
    sources = groups["train"] + groups["validation"]
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
        datasets={}, selected_sources={}, collection_costs_count_once={},
        interpretation="Source imitation candidates only; no claim of robot gain or in-context adaptation")
    for partition, rows in groups.items():
        selected_sources = [dict(index_path=row["index_path"], index_sha256=row["index_sha256"],
            task=row["task"], history_id=row["history_id"], selected_publication=row["selected"],
            episodes=row["episodes"], case_ids=sorted(row["case_ids"]),
            task_initial_state_ids=sorted(row["states"]), task_nominal_state_hashes=sorted(row["state_hashes"])) for row in rows]
        result["selected_sources"][partition] = selected_sources
        result["collection_costs_count_once"][partition] = {key: sum(row["costs"][key] for row in rows)
            for key in ("actual_primitive_steps", "actual_settling_steps", "source_updates")}
        result["datasets"][partition] = {}
        for objective in OBJECTIVES:
            path = output / f"{partition}-{objective}.npz"
            manifest = save_dataset(path, [row["datasets"][objective] for row in rows], objective=objective,
                metadata=dict(motor_sha256=PI05_RECEIPT, feature_contract=result["feature_contract"],
                    noise_contract=NOISE_CONTRACT, global_task_split="source40", local_scene_split=partition,
                    trial_ids=[row["history_id"] for row in rows], source_case_ids=[case for row in rows for case in sorted(row["case_ids"])],
                    source_task_keys=result["source_task_keys"], selected_source_histories=selected_sources,
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
    for partition in ("train", "validation"):
        for objective in OBJECTIVES:
            item = value["datasets"][partition][objective]
            _, manifest = load_dataset(_member(path.parent, item["path"]), expected_sha256=item["sha256"])
            if (manifest["objective"] != objective or manifest["metadata"].get("local_scene_split") != partition
                    or manifest["metadata"].get("motor_sha256") != value["motor_sha256"]
                    or manifest["metadata"].get("feature_contract") != value["feature_contract"]
                    or manifest["metadata"].get("noise_contract") != value["noise_contract"]
                    or manifest["metadata"].get("source_task_keys") != value["source_task_keys"]
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
        assemble(args.train_index, args.validation_index, args.output_dir, allow_small_pilot=args.allow_small_pilot)
    else:
        args.assembly, args.output_dir = args.assembly.resolve(), args.output_dir.resolve()
        train_matrix(args)


if __name__ == "__main__":
    main()
