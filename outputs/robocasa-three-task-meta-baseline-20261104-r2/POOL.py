#!/usr/bin/env python3
"""Pool complete three-task history streams for one joint model per method."""
import argparse
import json
from pathlib import Path
import socket
import sys

TASKS = ("OpenDrawer", "CloseToasterOvenDoor", "CoffeeSetupMug")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--expected-hostname", required=True)
    a = p.parse_args()
    if sys.platform != "linux" or socket.gethostname() != a.expected_hostname:
        raise ValueError("Require allocated Linux")
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import checked_checkout
    from fluffy_bassoon.meta_rl.vla_history import file_sha256, load_dataset, save_dataset
    checked_checkout(Path(__file__).resolve().parents[2], a.source_commit)
    if a.output.exists():
        raise ValueError("Use a fresh pooled output; never overwrite a partial assembly")
    assemblies, references = {}, {}
    for task in TASKS:
        path = a.sources / task / "preparation/ASSEMBLY.json"
        binding = json.loads(path.read_text())
        if (binding.get("completed") is not True or binding.get("student_fit_ready") is not True
                or binding.get("task") != task or binding.get("source_TRAIN_controls") != 9984000
                or binding.get("provenance", {}).get("source_commit") != a.source_commit):
            raise ValueError("Require all three complete source assemblies from the same pin")
        assemblies[task] = binding
        references[task] = dict(path=str(path), sha256=file_sha256(path))
    a.output.mkdir(parents=True)
    datasets = {}
    for objective in ("ad", "dpt", "dit", "ic_iql"):
        for split, expected_streams in (("train", 112), ("validation", 16)):
            trials, ids, sources, common = [], [], {}, None
            for task in TASKS:
                item = assemblies[task]["datasets"][f"{objective}_{split}"]
                collection, manifest = load_dataset(item["path"], expected_sha256=item["sha256"])
                metadata = manifest["metadata"]
                if manifest["objective"] != objective or len(collection) != expected_streams:
                    raise ValueError("Wrong objective or per-task split size")
                identity = {k: metadata[k] for k in (
                    "motor_sha256", "feature_contract", "noise_contract", "value_contract")}
                if common is None:
                    common = identity
                if common != identity:
                    raise ValueError("Task motor, feature, action or value contracts differ")
                task_ids = metadata["trial_ids"]
                if len(task_ids) != len(collection) or len(set(task_ids)) != len(collection):
                    raise ValueError("Bind each complete chronological task stream")
                trials.extend(collection)
                ids.extend(f"{task}/{value}" for value in task_ids)
                sources[task] = dict(item, streams=len(collection))
            if len(ids) != len(set(ids)):
                raise ValueError("Pooled stream IDs must be globally unique")
            metadata = dict(common, trial_ids=ids, source_commit=a.source_commit,
                source_tasks=list(TASKS), source_datasets=sources,
                source_assemblies=references, joint_training=True,
                sampling="uniform complete streams; equal stream counts and lengths per task",
                chronology="each task/slot retains all 130 versions; no cross-task history concatenation",
                validation="original held-out slots 112..127 independently for every task")
            path = a.output / f"{objective}-{split}.npz"
            saved = save_dataset(path, trials, objective=objective, metadata=metadata)
            datasets[f"{objective}_{split}"] = dict(path=str(path), sha256=saved["sha256"])
    assembly = dict(completed=True, student_fit_ready=True, joint_training=True,
        source_tasks=list(TASKS), source_assemblies=references,
        source_TRAIN_controls=29952000, source_physical_horizons=[600],
        feature_dim=2073, action_dim=600, datasets=datasets,
        source_commit=a.source_commit, train_streams=336, validation_streams=48)
    temporary = a.output / "ASSEMBLY.json.tmp"
    temporary.write_text(json.dumps(assembly, indent=2) + "\n")
    temporary.replace(a.output / "ASSEMBLY.json")
    print(json.dumps(assembly), flush=True)


if __name__ == "__main__":
    main()
