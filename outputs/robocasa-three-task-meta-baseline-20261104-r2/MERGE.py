#!/usr/bin/env python3
"""Merge four disjoint source shards, preserving the original stream partition."""
import argparse
import importlib.util
import json
from pathlib import Path
import socket
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--expected-hostname", required=True)
    a = p.parse_args()
    if sys.platform != "linux" or socket.gethostname() != a.expected_hostname:
        raise ValueError("Require the allocated Linux host")
    import numpy as np
    from fluffy_bassoon.meta_rl.flow_history import DirectActionCodec, export_corpus
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import checked_checkout

    code = Path(__file__).resolve().parents[2]
    checked_checkout(code, a.source_commit)
    spec = importlib.util.spec_from_file_location("prepare", Path(__file__).with_name("PREPARE-STREAM.py"))
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    raw, features, labels, tar_index, slots = {}, [], [], [], []
    reference = None
    for index in range(4):
        directory = a.output / f"shard-{index:02d}"
        shard = json.loads((directory / "SHARD.json").read_text())
        expected_slots = list(range(index * 32, (index + 1) * 32))
        if (shard.get("completed") is not True or shard["task"] != a.task
                or shard["shard_index"] != index or shard["global_slots"] != expected_slots
                or shard["expected_members"] != 3120 or shard["teacher_queries"] != 49920
                or shard["provenance"]["source_commit"] != a.source_commit):
            raise ValueError("Incomplete, misordered or incompatible shard")
        for name, item in shard["files"].items():
            if Path(item["path"]) != directory / name or prepare.sha(item["path"]) != item["sha256"]:
                raise ValueError("Shard artifact changed")
        identity = {k: v for k, v in shard["provenance"].items() if k != "raw_corpus_sha256"}
        if reference is None:
            reference = dict(provenance=identity, codec=shard["codec"])
        if reference != dict(provenance=identity, codec=shard["codec"]):
            raise ValueError("Motor, teacher, source or codec differs across shards")
        with np.load(directory / "RAW-PUBLIC-CORPUS.npz", allow_pickle=False) as archive:
            for key in archive.files:
                raw.setdefault(key, []).append(archive[key])
        with np.load(directory / "FEATURES-TEACHER-130.npz", allow_pickle=False) as archive:
            features.append(archive["features"])
            labels.append(archive["mapped_teacher_actions"])
        tar_index.extend(json.loads((directory / "TAR-INDEX.json").read_text()))
        slots.extend(shard["global_slots"])
    if slots != list(range(128)) or len(tar_index) != 12480 or len({x["name"] for x in tar_index}) != 12480:
        raise ValueError("Source coverage is not exact")
    raw = {key: np.concatenate(value, axis=0) for key, value in raw.items()}
    features, labels = np.concatenate(features), np.concatenate(labels)
    np.savez(a.output / "RAW-PUBLIC-CORPUS.npz", **raw)
    np.savez(a.output / "FEATURES-TEACHER-130.npz", features=features, mapped_teacher_actions=labels)
    prepare.write_json(a.output / "TAR-INDEX.json", sorted(tar_index, key=lambda x: x["name"]))
    provenance = dict(reference["provenance"],
                      raw_corpus_sha256=prepare.sha(a.output / "RAW-PUBLIC-CORPUS.npz"),
                      preparation_shards=4)
    codec = DirectActionCodec(tuple(reference["codec"]["mean"]), tuple(reference["codec"]["std"]))
    datasets = export_corpus(a.output / "corpus", raw, features, labels, codec, provenance=provenance)
    assembly = dict(completed=True, task=a.task, iterations=130, chunks_per_episode=12,
        source_physical_horizons=[600], datasets=datasets, provenance=provenance,
        codec=codec.record(), feature_dim=2073, action_dim=600,
        teacher_queries=199680, source_episodes=16640, source_TRAIN_controls=9984000,
        student_fit_ready=True)
    prepare.write_json(a.output / "ASSEMBLY.json", assembly)
    print(json.dumps(assembly), flush=True)


if __name__ == "__main__":
    main()
