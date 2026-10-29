"""Train matched full/static AD/DPT students on clean or augmented source data.

Both arms reuse the identical clean validation files. Source costs are counted
once per collection; adding Plus data is explicitly additional source data.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import socket
import subprocess
import sys
import time
import traceback

import numpy as np

from .vla_rlinf_libero import FEATURE_KIND, MODEL_SHA256


CONTEXT = ("observations", "executed_noise", "rewards", "dones", "episode_starts", "teacher_steps")
PLUS_SOURCE_CASES = {2387, 1438}


def disjoint_identities(training, validation):
    for key in ("trial_id", "case_id"):
        train, val = [row[key] for row in training], [row[key] for row in validation]
        if len(set(train)) != len(train) or len(set(val)) != len(val) or set(train) & set(val):
            raise ValueError(f"Duplicate or overlapping train/validation {key}")


def paired_contexts(first, second):
    if len(first) != len(second):
        raise ValueError("Paired AD/DPT trial count differs")
    for a, b in zip(first, second, strict=True):
        if any(a[key].dtype != b[key].dtype or a[key].shape != b[key].shape
               or a[key].tobytes() != b[key].tobytes() for key in CONTEXT):
            raise ValueError("Paired AD/DPT executed contexts differ")


def _index(path, kind):
    from .vla_history import file_sha256
    path = Path(path).resolve()
    index = json.loads(path.read_text())
    expected = "rlinf130_retry_corpus_index_v1" if kind == "clean" else "rlinf130_plus_source_corpus_index_v1"
    if (index.get("format") != expected or index.get("completed") is not True
            or index.get("model_sha256") != MODEL_SHA256
            or index.get("feature_contract", {}).get("kind") != FEATURE_KIND
            or index.get("global_task_split") != "train" or not index.get("trials")):
        raise ValueError(f"Require a complete, explicitly compatible {kind} RLinf source corpus")
    if kind == "plus" and (index.get("benchmark_kind") != "plus" or index.get("role") != "source_training"):
        raise ValueError("Plus development/test data cannot enter source training")
    return dict(path=str(path), sha256=file_sha256(path), kind=kind, index=index)


def _dataset(source, receipt):
    from .vla_history import file_sha256, load_dataset
    root = Path(source["path"]).parent
    path = (root / receipt["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Dataset path escaped its source collection")
    trials, manifest = load_dataset(path, expected_sha256=receipt["sha256"])
    meta, index = manifest["metadata"], source["index"]
    if (meta.get("model_sha256") != MODEL_SHA256 or meta.get("global_task_split") != "train"
            or meta.get("task_key") != index["task_key"] or meta.get("feature_contract") != index["feature_contract"]
            or meta.get("motor_sha256") != index["motor_receipt_sha256"]
            or meta.get("noise_contract") != index["noise_contract"]
            or manifest["observation_dim"] != 4104 or manifest["noise_dim"] != 32):
        raise ValueError("Dataset changed motor, feature/noise coordinates or source task")
    return trials, manifest, dict(path=str(path), sha256=receipt["sha256"],
        manifest_sha256=file_sha256(path.with_suffix(".json")))


def prepare(clean_index, plus_index, arm, output):
    from .vla_history import file_sha256, save_dataset
    output = Path(output)
    clean = _index(clean_index, "clean")
    sources = [clean]
    if arm == "augmented":
        if plus_index is None:
            raise ValueError("Augmented training requires the declared Plus source index")
        sources.append(_index(plus_index, "plus"))
    elif arm != "clean" or plus_index is not None:
        raise ValueError("Clean training accepts only the clean source index")
    common = {key: clean["index"][key] for key in
              ("task_key", "model_sha256", "motor_receipt_sha256", "feature_contract", "noise_contract", "frozen_split_sha256")}
    for source in sources:
        if any(source["index"].get(key) != value for key, value in common.items()):
            raise ValueError("Source collections changed shared motor/features/noise/goal-family contracts")
    training, validation = [], []
    collected = {objective: {split: [] for split in ("train", "validation")} for objective in ("ad", "dpt")}
    raw_training = []
    costs = []
    for source in sources:
        index = source["index"]
        source_train, source_validation = 0, 0
        for trial in index["trials"]:
            split = trial["scene"]["split"]
            if split not in ("train", "validation") or source["kind"] == "plus" and split != "train":
                raise ValueError("Only clean local validation is allowed; Plus is training-only")
            state = int(trial["scene"]["initial_state"])
            if source["kind"] == "plus":
                case = trial["plus_case"]
                if (case.get("runtime_index") not in PLUS_SOURCE_CASES
                        or case.get("role") != "source_training" or case.get("split") != "source_training"):
                    raise ValueError("Plus case is outside the predeclared source-training subset")
                case_id = f"{case['case_id']}:state-{state}"
            else:
                case_id = f"original:{index['task_key']}:state-{state}"
            identity = dict(trial_id=trial["trial_id"], case_id=case_id)
            (training if split == "train" else validation).append(identity)
            pair, receipts = {}, {}
            for objective in ("ad", "dpt"):
                data, manifest, receipt = _dataset(source, trial["datasets"][objective])
                if (manifest["objective"] != objective or len(data) != 1
                        or manifest["metadata"]["trial_ids"] != [trial["trial_id"]]
                        or manifest["metadata"].get("local_scene_split") != split):
                    raise ValueError("Trial objective, identity or local scene split differs")
                pair[objective], receipts[objective] = data, receipt
                collected[objective][split].extend(data)
            paired_contexts(pair["ad"], pair["dpt"])
            if split == "train":
                source_train += int(trial["primitive_steps"])
                raw_training.append(dict(source_index=source["path"], source_index_sha256=source["sha256"],
                    source_root=str(Path(source["path"]).parent), source_kind=source["kind"],
                    trial=trial, datasets=receipts, identity=identity))
            else:
                source_validation += int(trial["primitive_steps"])
        if source_train + source_validation != index["actual_primitive_steps"]:
            raise ValueError("Collection cost does not equal its actual complete trials")
        costs.append(dict(kind=source["kind"], index=source["path"], index_sha256=source["sha256"],
            collection_primitive_steps=index["actual_primitive_steps"], training_primitive_steps=source_train,
            validation_primitive_steps=source_validation, settling_steps=index["actual_settling_steps"]))
    disjoint_identities(training, validation)
    if not training or not validation:
        raise ValueError("Require nonempty disjoint training and clean validation trials")
    output.mkdir(parents=True, exist_ok=False)
    paths = {objective: {} for objective in ("ad", "dpt")}
    cost = dict(sources=costs, collection_primitive_steps=sum(x["collection_primitive_steps"] for x in costs),
        training_primitive_steps=sum(x["training_primitive_steps"] for x in costs),
        validation_primitive_steps=sum(x["validation_primitive_steps"] for x in costs),
        settling_steps=sum(x["settling_steps"] for x in costs),
        rule="count each source collection once; AD/DPT and model copies share these interactions")
    for objective in paths:
        path = output / f"{objective}-train.npz"
        receipt = save_dataset(path, collected[objective]["train"], objective=objective, metadata=dict(
            trial_ids=[row["trial_id"] for row in training], case_ids=[row["case_id"] for row in training],
            motor_sha256=common["motor_receipt_sha256"], model_sha256=MODEL_SHA256,
            feature_contract=common["feature_contract"], noise_contract=common["noise_contract"],
            arm=arm, task_key=common["task_key"], global_task_split="train", local_scene_split="train",
            total_source_primitive_steps=cost["collection_primitive_steps"], collection_cost=cost,
            source_indices=[dict(path=s["path"], sha256=s["sha256"]) for s in sources]))
        paths[objective]["train"] = dict(path=str(path), sha256=receipt["sha256"])
        # Copy the exact original clean validation bytes for BOTH data arms.
        val_receipt = clean["index"]["pilot_datasets"][objective]["validation"]
        val, val_manifest, val_source = _dataset(clean, val_receipt)
        if val_manifest["metadata"]["trial_ids"] != [r["trial_id"] for r in validation]:
            raise ValueError("Clean validation bundle differs from its declared source trials")
        paired_contexts(val, collected[objective]["validation"])
        if any(a["target_noise"].tobytes() != b["target_noise"].tobytes()
               for a, b in zip(val, collected[objective]["validation"], strict=True)):
            raise ValueError("Clean validation query labels changed during aggregation")
        destination = output / f"{objective}-validation.npz"
        shutil.copyfile(val_source["path"], destination)
        shutil.copyfile(Path(val_source["path"]).with_suffix(".json"), destination.with_suffix(".json"))
        if file_sha256(destination) != val_source["sha256"]:
            raise ValueError("Clean validation changed while copying")
        paths[objective]["validation"] = dict(path=str(destination), sha256=val_source["sha256"])
    for source in sources:
        if file_sha256(source["path"]) != source["sha256"]:
            raise ValueError("A source index changed during assembly")
    return dict(arm=arm, common=common, paths=paths, collection_cost=cost, training_trials=raw_training,
                validation_identities=validation, training_identities=training)


def train_supervised(args, assembled, registry):
    import torch
    from .vla_history import HistoryConfig, HistoryNoisePolicy, export_policy, file_sha256
    from .vla_history_train import main as train_history
    from .vla_libero import write_json
    for objective in ("ad", "dpt"):
        for mode in ("full", "static"):
            name, paths = f"{objective}_{mode}", assembled["paths"][objective]
            directory = args.output / "models" / f"{objective}-{mode}"
            argv = ["--train-data", paths["train"]["path"], "--train-sha256", paths["train"]["sha256"],
                "--validation-data", paths["validation"]["path"], "--validation-sha256", paths["validation"]["sha256"],
                "--output-dir", str(directory), "--mode", mode, "--width", "128", "--layers", "2",
                "--heads", "4", "--context-length", "300", "--updates", "500", "--batch-size", "4",
                "--max-window", "64", "--learning-rate", "0.0003", "--eval-every", "50",
                "--validation-batches", "2", "--max-elapsed-seconds", "400", "--seed", "0",
                "--device", "cuda:0", "--source-commit", args.source_commit,
                "--expected-hostname", socket.gethostname()]
            train_history(argv)
            latest = torch.load(directory / "latest.pt", map_location="cpu", weights_only=False)
            record = dict(objective=objective, mode=mode, seed=0, requested_updates=500,
                actual_updates=latest["update"], train_sha256=paths["train"]["sha256"],
                validation_sha256=paths["validation"]["sha256"],
                recovery=dict(path=str((directory / "latest.pt").relative_to(args.output)),
                              sha256=file_sha256(directory / "latest.pt")))
            registry["models"][name] = record
            write_json(args.output / "registry.json", registry)
            if latest["update"] != 500:
                raise RuntimeError("Student reached its time bound before 500 updates; recovery retained, arm incomplete")
            policy = HistoryNoisePolicy(HistoryConfig(**latest["contract"]["config"]))
            policy.load_state_dict(latest["policy"], strict=True)
            final = directory / "final-candidate.pt"
            export_policy(final, policy.eval().requires_grad_(False), manifest=dict(contract=latest["contract"],
                update=500, selection="final_update", data_arm=args.arm,
                collection_cost=assembled["collection_cost"]))
            best_path = directory / "candidate.pt"
            best = torch.load(best_path, map_location="cpu", weights_only=False)["manifest"]
            record.update(primary=dict(path=str(final.relative_to(args.output)), sha256=file_sha256(final),
                update=500, selection="final_update"), best_validation=dict(
                path=str(best_path.relative_to(args.output)), sha256=file_sha256(best_path), update=best["update"]),
                normalization="training rows only", elapsed_seconds=latest["elapsed_seconds"])
            write_json(args.output / "registry.json", registry)
            del policy, latest
            torch.cuda.empty_cache()


def training_replay(assembled):
    """Verify raw training episodes against the exact meta-learning contexts."""
    from .vla_history import file_sha256, load_dataset
    from .vla_retry_teacher import append_episode_to_replay
    from types import SimpleNamespace
    source = SimpleNamespace(replay=[])
    provenance, steps, episodes = [], 0, 0
    for entry in assembled["training_trials"]:
        root, trial = Path(entry["source_root"]).resolve(), entry["trial"]
        if trial["scene"]["split"] != "train":
            raise ValueError("Validation interactions cannot enter source SAC replay")
        data, manifest = load_dataset(entry["datasets"]["ad"]["path"],
            expected_sha256=entry["datasets"]["ad"]["sha256"])
        all_rows, traces = [], []
        for receipt in trial["episodes"]:
            relative = receipt.get("trace") or str(Path(trial["source_checkpoint"]).parent /
                f"episode-{receipt['attempt']:03d}.pkl.gz")
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or file_sha256(path) != receipt["trace_sha256"]:
                raise ValueError("Raw source episode path/hash changed")
            with gzip.open(path, "rb") as stream:
                saved = pickle.load(stream)  # Trusted local rollouts, never downloaded pickle metadata.
            rows, summary = saved["rows"], saved["summary"]
            if (not rows or not rows[-1]["done"] or any(row["done"] for row in rows[:-1])
                    or summary["initial_state"] != trial["scene"]["initial_state"]
                    or summary["scene_seed"] != trial["scene"]["scene_seed"]
                    or sum(row["primitive_steps"] for row in rows) != receipt["primitive_steps"]):
                raise ValueError("Raw episode differs from its declared complete source trial")
            all_rows.extend(rows)
            append_episode_to_replay(source, rows)
            steps += int(receipt["primitive_steps"])
            episodes += 1
            traces.append(dict(path=str(path), sha256=receipt["trace_sha256"],
                               primitive_steps=receipt["primitive_steps"]))
        expected = data[0]
        actual = dict(observations=np.stack([r["public_features"] for r in all_rows]).astype(np.float32),
            executed_noise=np.stack([r["executed_noise"] for r in all_rows]).astype(np.float32),
            rewards=np.asarray([r["reward"] for r in all_rows], np.float32),
            dones=np.asarray([r["done"] for r in all_rows], bool))
        if any(actual[key].shape != expected[key].shape or actual[key].tobytes() != expected[key].tobytes() for key in actual):
            raise ValueError("Source SAC replay is not the same real training interaction stream as the students")
        provenance.append(dict(source_index=entry["source_index"], source_index_sha256=entry["source_index_sha256"],
            trial_id=trial["trial_id"], traces=traces, dataset=entry["datasets"]["ad"],
            original_source_contract=manifest["metadata"]["source_contract"]))
    if steps != assembled["collection_cost"]["training_primitive_steps"] or not source.replay:
        raise ValueError("Source training interaction accounting differs from the shared dataset")
    return source.replay, provenance, steps, episodes


def train_source_sac(args, assembled, registry):
    import torch
    import jax
    from .vla_history import file_sha256
    from .vla_libero import DSRL_COMMIT, OfficialSAC, append_json, write_json
    if args.dsrl_root is None:
        raise ValueError("Source SAC pretraining requires the pinned original DSRL source")
    if len(jax.devices("gpu")) != 1 or jax.default_backend() != "gpu":
        raise ValueError("Source SAC must compute on the single allocated JAX GPU")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    directory = args.output / "source-sac"
    directory.mkdir()
    learner = OfficialSAC(args.dsrl_root, 0)
    replay, provenance, training_steps, episodes = training_replay(assembled)
    learner.replay = replay
    sampler = np.random.default_rng(0)
    contract = dict(format="rlinf130_offline_source_sac_v1", source_commit=args.source_commit,
        arm=args.arm, dsrl_commit=DSRL_COMMIT, seed=0, batch_size=32, requested_updates=5000,
        model_sha256=MODEL_SHA256, motor_receipt_sha256=assembled["common"]["motor_receipt_sha256"],
        feature_contract=assembled["common"]["feature_contract"], noise_contract=assembled["common"]["noise_contract"],
        task_key=assembled["common"]["task_key"], replay_rows=len(replay),
        training_primitive_steps=training_steps, collection_cost=assembled["collection_cost"],
        source_training_provenance=provenance, training_datasets=assembled["paths"], additional_environment_interactions=0,
        source_policy="fresh official SAC offline updates on the same train-only real replay as full/static students",
        limitation="one goal family; raw RGB/state actor does not solve future multi-task instruction conditioning")
    write_json(directory / "contract.json", contract)
    registry["source_sac"] = dict(checkpoint=None, contract=dict(
        path=str((directory / "contract.json").relative_to(args.output)), sha256=file_sha256(directory / "contract.json")),
        updates=0, requested_updates=5000, replay_rows=len(replay), training_primitive_steps=training_steps,
        additional_environment_interactions=0, seed=0, batch_size=32, offline=True)
    write_json(args.output / "registry.json", registry)
    started = time.monotonic()
    for update in range(1, 5001):
        metrics = learner.update(32, sampler)
        terminal = update == 5000 or time.monotonic() - started >= 600
        if update == 1 or update % 250 == 0 or terminal:
            record = dict(update=update, elapsed_seconds=time.monotonic() - started, **metrics)
            append_json(directory / "metrics.jsonl", record)
            print(json.dumps(dict(source_sac_training=record)), flush=True)
            path = directory / "source-recovery.pkl"
            learner.save(path, sampler, contract, episodes=episodes, primitive_steps=training_steps,
                offline_updates=update, elapsed_seconds=record["elapsed_seconds"],
                runtime_rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
            registry["source_sac"].update(updates=update, checkpoint=dict(
                path=str(path.relative_to(args.output)), sha256=file_sha256(path)),
                elapsed_seconds=record["elapsed_seconds"])
            write_json(args.output / "registry.json", registry)
        if terminal:
            break
    if learner.updates != 5000:
        raise RuntimeError("Source SAC reached its time bound before 5000 updates; complete recovery retained")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--arm", choices=("clean", "augmented"), required=True)
    p.add_argument("--clean-index", type=Path, required=True)
    p.add_argument("--plus-index", type=Path)
    p.add_argument("--phase", choices=("supervised", "source-sac", "all"), default="supervised")
    p.add_argument("--dsrl-root", type=Path)
    args = p.parse_args(argv)
    if sys.platform != "linux":
        raise ValueError("Train models only on allocated Linux")
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get("FLUFFY_GPU_EXECUTION", "baige"), os.environ)
    from .vla_libero import attest_process_gpu, write_json
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()):
        raise ValueError("Require exact clean reviewed training source")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        assembled = prepare(args.clean_index, args.plus_index, args.arm, args.output / "data")
        write_json(args.output / "assembly.json", assembled)
        import torch
        if torch.cuda.device_count() != 1 or torch.version.cuda != "12.8":
            raise ValueError("Require the allocated qualified CUDA-12.8 student runtime")
        torch.ones(4, device="cuda").sum().item()
        attest_process_gpu(args.output / "student-binding.json", graphics=False)
        registry = dict(format="rlinf130_trained_arm_v1", arm=args.arm, completed=False,
            phase=args.phase,
            source_commit=args.source_commit, model_sha256=MODEL_SHA256,
            feature_contract=assembled["common"]["feature_contract"],
            noise_contract=assembled["common"]["noise_contract"], models={},
            collection_cost=assembled["collection_cost"],
            runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, hostname=socket.gethostname()),
            scope="single original goal family; local technical pilot, not broad task generalization")
        write_json(args.output / "registry.json", registry)
        if args.phase in ("supervised", "all"):
            train_supervised(args, assembled, registry)
        if args.phase in ("source-sac", "all"):
            train_source_sac(args, assembled, registry)
        registry.update(completed=True, elapsed_seconds=time.monotonic() - started)
        write_json(args.output / "registry.json", registry)
        return 0
    except Exception as error:
        write_json(args.output / "failure.json", dict(error=repr(error), traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic() - started))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
