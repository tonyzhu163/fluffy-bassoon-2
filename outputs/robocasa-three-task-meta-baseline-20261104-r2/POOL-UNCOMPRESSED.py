#!/usr/bin/env python3
"""Pool identical history arrays in uncompressed NPZ containers to avoid zlib cost."""
import argparse
import json
from pathlib import Path
import socket
import sys

TASKS = ("OpenDrawer", "CloseToasterOvenDoor", "CoffeeSetupMug")


def save_dataset(path, trials, *, objective, metadata):
    """Preserve the core schema and validation, changing only ZIP compression."""
    import numpy as np
    from fluffy_bassoon.meta_rl.vla_history import (
        DATA_FORMAT, FIELDS, file_sha256, validate_trial)
    path = Path(path)
    if path.exists() or path.with_suffix(".json").exists() or not trials or not metadata:
        raise ValueError("Require fresh dataset paths, complete trials and provenance")
    shapes = [validate_trial(trial, objective=objective) for trial in trials]
    if len({shape[1:] for shape in shapes}) != 1:
        raise ValueError("All trials must share observation/action dimensions")
    portable_metadata = json.loads(json.dumps(metadata, allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.concatenate([trial[key] for trial in trials], axis=0) for key in FIELDS}
    arrays["lengths"] = np.asarray([shape[0] for shape in shapes], dtype=np.int64)
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
    manifest = dict(format=DATA_FORMAT, objective=objective, trials=len(trials),
        transitions=sum(shape[0] for shape in shapes), observation_dim=shapes[0][1],
        noise_dim=shapes[0][2], metadata=portable_metadata, sha256=file_sha256(path))
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--learner-code", type=Path, required=True)
    p.add_argument("--pooler-commit", required=True)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--expected-hostname", required=True)
    a = p.parse_args()
    if sys.platform != "linux" or socket.gethostname() != a.expected_hostname:
        raise ValueError("Require allocated Linux")
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import checked_checkout
    from fluffy_bassoon.meta_rl.vla_history import file_sha256, load_dataset
    checked_checkout(a.learner_code, a.source_commit)
    checked_checkout(Path(__file__).resolve().parents[2], a.pooler_commit)
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
        source_commit=a.source_commit, train_streams=336, validation_streams=48,
        serialization=dict(container="uncompressed_npz", pooler_commit=a.pooler_commit,
                           learner_source_commit=a.source_commit, numerical_arrays="unchanged"))
    temporary = a.output / "ASSEMBLY.json.tmp"
    temporary.write_text(json.dumps(assembly, indent=2) + "\n")
    temporary.replace(a.output / "ASSEMBLY.json")
    print(json.dumps(assembly), flush=True)


if __name__ == "__main__":
    main()
