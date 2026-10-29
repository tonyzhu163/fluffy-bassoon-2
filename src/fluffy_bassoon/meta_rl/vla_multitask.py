"""Validate semantic task splits and assemble same-context AD/DPT collections.

Collection stays in independently pinned task runs. This module merges only
declared training/validation tasks; held-out tasks have no supervised exports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .vla_features import FEATURE_KIND


CONTEXT_FIELDS = ("observations", "executed_noise", "rewards", "dones", "episode_starts", "teacher_steps")


def validate_manifest(manifest):
    if manifest.get("format") != "frozen_vla_multitask_v1":
        raise ValueError("Require an explicit versioned multi-task manifest")
    tasks = manifest.get("tasks", [])
    required = {"task_key", "suite", "task_id", "semantic_task", "split"}
    if not tasks or any(not isinstance(t, dict) or not required.issubset(t) for t in tasks):
        raise ValueError("Require explicit task identities and semantic split labels")
    for key in ("task_key", "semantic_task"):
        values = [t[key] for t in tasks]
        if any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
            raise ValueError("Duplicate or missing semantic task identity across splits")
    if len({(t["suite"], t["task_id"]) for t in tasks}) != len(tasks):
        raise ValueError("The same benchmark task occurs more than once")
    if {t["split"] for t in tasks} != {"train", "validation", "test"}:
        raise ValueError("Declare nonempty train, validation and held-out test task banks")
    for task in tasks:
        if (not isinstance(task["suite"], str) or not task["suite"]
                or type(task["task_id"]) is not int or task["task_id"] < 0):
            raise ValueError("Invalid benchmark task coordinates")
        if task["split"] == "test" and task.get("datasets"):
            raise ValueError("Held-out semantic test tasks cannot supply supervised training data")
    return tasks


def assert_context_equal(ad, dpt):
    if len(ad) != len(dpt) or any(not np.array_equal(a[key], b[key])
        for a, b in zip(ad, dpt, strict=True) for key in CONTEXT_FIELDS):
        raise ValueError("AD and DPT must share exact executed contexts; only targets may change")


class SimulatorTaskBank:
    """Route whole trials among already allocated, pinned simulator workers.

    No task index enters policy observations. An internal episode identity maps
    to one semantic task and one benchmark initial-state index. Resetting that
    identity for the next attempt keeps the task, instruction and scene fixed.
    The caller launches workers and owns their resource/allocation lifecycle.
    """
    def __init__(self, manifest, clients):
        self.tasks = validate_manifest(manifest)
        if set(clients) != {task["task_key"] for task in self.tasks}:
            raise ValueError("Provide exactly the declared semantic task workers")
        self.clients, self.episodes, self.split_episodes = clients, [], {k: [] for k in ("train", "validation", "test")}
        self.active = None
        for task in self.tasks:
            client = clients[task["task_key"]]
            if (client.metadata["suite"] != task["suite"] or client.metadata["task_id"] != task["task_id"]
                    or Path(client.metadata["bddl_file"]).stem != task["semantic_task"]):
                raise ValueError("Simulator worker task differs from its semantic manifest")
            states = task.get("initial_states")
            if (not states or len(set(states)) != len(states)
                    or any(type(s) is not int or not 0 <= s < client.metadata["initial_states"] for s in states)):
                raise ValueError("Declare each task's unique benchmark initial-state bank")
            for state in states:
                self.split_episodes[task["split"]].append(len(self.episodes))
                self.episodes.append((task["task_key"], state))

    @property
    def metadata(self):
        if self.active is None:
            raise RuntimeError("Select a task by resetting its episode identity first")
        return self.active.metadata

    def reset(self, initial_state, scene_seed):
        if type(initial_state) is not int or not 0 <= initial_state < len(self.episodes):
            raise ValueError("Unknown declared task/scene episode")
        task_key, benchmark_state = self.episodes[initial_state]
        self.active = self.clients[task_key]
        return self.active.reset(benchmark_state, scene_seed)

    def step(self, actions):
        if self.active is None:
            raise RuntimeError("Reset a task before stepping")
        return self.active.step(actions)


def train_task_bank_rl2(policy, bank, motor, common_encoder, *, attempts, chunk,
                       max_primitive_steps, updates, output_dir, source_contract, **kwargs):
    from .vla_rl2 import train_rl2
    maximum_horizon = max(bank.clients[task["task_key"]].metadata["horizon"]
                          for task in bank.tasks if task["split"] == "train")
    return train_rl2(policy, bank, motor,
        lambda obs: common_encoder(obs, bank.metadata["instruction"]),
        train_states=bank.split_episodes["train"],
        heldout_states=bank.split_episodes["validation"] + bank.split_episodes["test"],
        attempts=attempts, chunk=chunk, episode_horizon=maximum_horizon,
        max_primitive_steps=max_primitive_steps, updates=updates, output_dir=output_dir,
        source_contract=dict(**source_contract, semantic_tasks=bank.tasks,
            episode_map=bank.episodes, schedule="uniform declared training task/scene episodes; one task per complete trial"),
        **kwargs)


def merge_collections(manifest, output):
    from .vla_history import file_sha256, load_dataset, save_dataset
    tasks = validate_manifest(manifest)
    output = Path(output)
    collections = {obj: {split: [] for split in ("train", "validation")} for obj in ("ad", "dpt")}
    identities = {split: [] for split in ("train", "validation")}
    common, sources = None, []
    for task in tasks:
        if task["split"] == "test":
            continue
        datasets = task.get("datasets", {})
        if set(datasets) != {"ad", "dpt"}:
            raise ValueError("Every source task needs paired AD and DPT dataset paths")
        loaded, manifests = {}, {}
        for objective in ("ad", "dpt"):
            path = Path(datasets[objective]).expanduser().resolve()
            loaded[objective], manifests[objective] = load_dataset(path, expected_sha256=file_sha256(path))
            metadata = manifests[objective]["metadata"]
            contract = {key: metadata[key] for key in ("motor_sha256", "feature_contract", "noise_contract")}
            if contract["feature_contract"].get("kind") != FEATURE_KIND:
                raise ValueError("Multi-task pooling requires a common frozen VLA vision/language encoder")
            if common is None:
                common = contract
            elif common != contract:
                raise ValueError("Source tasks use different motor, feature or noise coordinate systems")
            source = metadata.get("source_prior_training_contract", {}).get("simulator", {})
            if source.get("suite") != task["suite"] or source.get("task_id") != task["task_id"]:
                raise ValueError("Dataset source benchmark task differs from the declared split")
            if Path(source.get("bddl_file", "")).stem != task["semantic_task"]:
                raise ValueError("Dataset semantic BDDL identity was not verified against the task manifest")
            if manifests[objective]["objective"] != objective:
                raise ValueError("Dataset objective label differs from its declared collection")
        assert_context_equal(loaded["ad"], loaded["dpt"])
        ad_ids = manifests["ad"]["metadata"]["trial_ids"]
        dpt_ids = manifests["dpt"]["metadata"]["trial_ids"]
        if ad_ids != dpt_ids or len(ad_ids) != len(loaded["ad"]):
            raise ValueError("Paired objectives have different trial identities")
        identities[task["split"]].extend(f"{task['task_key']}/{trial}" for trial in ad_ids)
        for objective in ("ad", "dpt"):
            collections[objective][task["split"]].extend(loaded[objective])
        sources.append(dict(task=task, manifests=manifests))
    if len(set(identities["train"] + identities["validation"])) != sum(map(len, identities.values())):
        raise ValueError("Duplicate source trial identity after semantic task aggregation")
    # Validate every source before creating any final artifact.
    output.mkdir(parents=True, exist_ok=False)
    (output / "task-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    paths = {}
    for objective in collections:
        paths[objective] = {}
        for split in collections[objective]:
            path = output / f"{objective}-{split}.npz"
            save_dataset(path, collections[objective][split], objective=objective,
                metadata=dict(**common, trial_ids=identities[split], task_manifest=manifest,
                              source_collections=sources,
                              total_source_primitive_steps=sum(source["manifests"]["ad"]["metadata"][
                                  "total_source_primitive_steps"] for source in sources),
                              split="whole semantic tasks before any training windows"))
            paths[objective][split] = str(path)
    (output / "collections.json").write_text(json.dumps(paths, indent=2) + "\n")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge_collections(json.loads(args.manifest.read_text()), args.output), indent=2))


if __name__ == "__main__":
    main()
