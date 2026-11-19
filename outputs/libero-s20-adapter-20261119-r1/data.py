"""Prepare and sample causally aligned LIBERO-Pro2.0 demonstrations."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np


REVISION = "6c4c5524676815a88880a7025badd17fb724f7f4"
REPOSITORY = "datasets/zhouxueyang/LIBERO-Pro2.0-Train"
MANIFEST_HASHES = {
    "episodes.jsonl": "373a3203923638884c1a20814a7ed9e0e509e2441ffea317df894f27a11779bb",
    "files.jsonl": "04cea73b5550756a0ca41afe43c74554cc0744610e4b97fac4f230017e2ccc67",
}
CHUNK = 10
CASES = ("BASE", "S20")
RAW_FIELDS = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
              "robot0_eef_quat_xyzw", "robot0_gripper_qpos")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_lines(path):
    with path.open() as stream:
        yield from (json.loads(line) for line in stream)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def file_matches(path, row):
    return path.is_file() and path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"]


def fetch(row, path, endpoints):
    if file_matches(path, row):
        return
    if not endpoints:
        raise RuntimeError(f"Previously staged archive failed verification: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    failure = []
    for endpoint in endpoints:
        url = f"{endpoint.rstrip('/')}/{REPOSITORY}/resolve/{REVISION}/{row['path']}?download=true"
        result = subprocess.run([
            "curl", "--fail", "--silent", "--show-error", "--location", "--retry", "2",
            "--connect-timeout", "12", "--max-time", "600", "--continue-at", "-",
            "--output", str(partial), url,
        ], capture_output=True, text=True, check=False)
        if file_matches(partial, row):
            partial.replace(path)
            return
        failure.append(f"{endpoint}: curl {result.returncode}, checksum or size unverified")
        if partial.is_file() and partial.stat().st_size >= row["bytes"]:
            partial.unlink()
    raise RuntimeError(f"{row['path']}: {'; '.join(failure)}")


def partition_rows(rows):
    """Keep a physical reset wholly on one side for each task, across both timings."""
    cases_by_reset = defaultdict(set)
    for row in rows:
        cases_by_reset[row["init_state_id"]].add(row["case"])
    identity = f"{rows[0]['suite']}/{rows[0]['task']}"
    ranked = sorted(cases_by_reset, key=lambda reset: hashlib.sha256(
        f"{REVISION}/{identity}/{reset}".encode()).digest())
    train_counts = {case: sum(case in options for options in cases_by_reset.values()) for case in CASES}
    validation = set()
    for reset in ranked[: max(1, len(ranked) // 8)]:
        if all(train_counts[case] > 1 for case in cases_by_reset[reset]):
            validation.add(reset)
            for case in cases_by_reset[reset]:
                train_counts[case] -= 1
    for row in rows:
        row["partition"] = "validation" if row["init_state_id"] in validation else "train"


def prepare(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.payload.mkdir(parents=True, exist_ok=True)
    args.cache = args.cache.resolve()
    for name, expected in MANIFEST_HASHES.items():
        if sha256(args.cache / name) != expected:
            raise ValueError(f"Pinned publisher manifest changed: {name}")
    tasks = {(suite, f"task_{task}") for suite, ids in json.loads(args.splits.read_text())["train"].items()
             for task in ids}
    if args.smoke:
        tasks &= {("libero_goal", "task_4"), ("libero_goal", "task_8")}
    rows = [row for row in read_lines(args.cache / "episodes.jsonl")
            if (row["suite"], row["task"]) in tasks and row["case"] in CASES
            and row["keep_without_declared_fanout"] and row["steps"] > 1]
    groups = defaultdict(list)
    for row in rows:
        groups[row["suite"], row["task"]].append(row)
    if set(groups) != tasks or any({row["case"] for row in group} != set(CASES) for group in groups.values()):
        raise ValueError("Release coverage differs from paired fitting plan")
    for group in groups.values():
        partition_rows(group)
    expected_paths = {row["hdf5_path"] for row in rows}
    inventory = {row["path"]: row for row in read_lines(args.cache / "files.jsonl")
                 if row["path"] in expected_paths}
    if set(inventory) != expected_paths:
        raise ValueError("Every selected archive needs a publisher checksum")

    def payload_path(name):
        parts = Path(name).parts
        cached = args.cache / f"{parts[2].removeprefix('task_')}_{parts[3]}.hdf5"
        if parts[1] == "libero_goal" and parts[2] in {"task_4", "task_8"} and cached.exists():
            return cached
        return args.payload.resolve() / name

    destination = args.payload.resolve()
    local = {name: payload_path(name) for name in inventory}
    reserve = sum(row["bytes"] for name, row in inventory.items() if not local[name].exists())
    if shutil.disk_usage(args.payload).free < reserve:
        raise OSError("Insufficient free space for the selected public HDF5 files")
    probe = args.payload / f".probe-{os.getpid()}"
    probe.write_text("adapter data\n")
    probe.unlink()
    progress_path = args.output.with_name(args.output.stem + "-progress.json")
    status = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {}
        for name, row in inventory.items():
            path = local[name]
            endpoints = args.endpoint if path.is_relative_to(destination) else []
            jobs[pool.submit(fetch, row, path, endpoints)] = name
        for job in as_completed(jobs):
            name = jobs[job]
            try:
                job.result()
                status[name] = "verified"
            except RuntimeError as error:
                status[name] = str(error)
            print(json.dumps({"archive": name, "status": status[name], "complete": len(status),
                              "expected": len(inventory)}), flush=True)
            write_json(progress_path, status)
    if any(state != "verified" for state in status.values()):
        raise RuntimeError(f"Some demonstration files could not be verified. See {progress_path}")
    write_json(args.output, {
        "revision": REVISION, "manifest_hashes": MANIFEST_HASHES, "task_split_sha256": sha256(args.splits),
        "smoke": args.smoke, "archives": {name: {**inventory[name], "local_path": str(path)}
                                          for name, path in local.items()},
        "episodes": rows,
    })


class DemonstrationSampler:
    """Balance timing, task and reset before sampling a demonstrator and query position."""

    def __init__(self, index_path):
        index = json.loads(Path(index_path).read_text())
        if index["revision"] != REVISION:
            raise ValueError("Expected the pinned demonstration release")
        self.archives = index["archives"]
        self.files = {}
        self.prompts = {}
        self.groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for row in index["episodes"]:
            key = row["partition"], row["case"]
            task = row["suite"], row["task"]
            self.groups[key][task][row["init_state_id"]].append(row)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for archive in self.files.values():
            archive.close()
        self.files.clear()

    def archive(self, path):
        if path not in self.files:
            self.files[path] = h5py.File(self.archives[path]["local_path"], "r")
        return self.files[path]

    def base_prompt(self, task):
        if task not in self.prompts:
            suite, name = task
            base = self.archive(f"data/{suite}/{name}/BASE/episodes.hdf5")
            first_demo = base["data/demo_0"]
            self.prompts[task] = json.loads(first_demo.attrs["source_metadata"])["instruction"]
        return self.prompts[task]

    def sample(self, batch_size, rng, partition="train"):
        if batch_size < 1 or any(not self.groups[partition, case] for case in CASES):
            raise ValueError(f"Sampling needs a positive batch and both conditions in {partition}")
        timing = np.tile(np.arange(len(CASES), dtype=np.int32), (batch_size + 1) // 2)
        rng.shuffle(timing)
        timing = timing[:batch_size]
        raw, prompts, sample_ids = [], [], []
        actions = np.zeros((batch_size, CHUNK, 7), dtype=np.float32)
        mask = np.zeros((batch_size, CHUNK), dtype=np.bool_)
        for i, condition in enumerate(timing):
            tasks = self.groups[partition, CASES[int(condition)]]
            task = tuple(tasks)[rng.integers(len(tasks))]
            per_reset = tasks[task]
            demos = per_reset[tuple(per_reset)[rng.integers(len(per_reset))]]
            row = demos[rng.integers(len(demos))]
            group = self.archive(row["hdf5_path"])[row["demo"]]
            steps = row["steps"]
            t = int(rng.integers(max(1, steps - CHUNK)))
            end = min(t + CHUNK + 1, steps)
            action_len = end - t - 1
            actions[i, :action_len] = group["actions"][t + 1:end]
            mask[i, :action_len] = True
            raw.append({field: group["obs"][field][t] for field in RAW_FIELDS})
            prompts.append(self.base_prompt(task))
            sample_ids.append({key: row[key] for key in ("hdf5_path", "demo", "init_state_id", "seed", "case")}
                              | {"row": t, "timing": int(condition)})
        return {"raw": raw, "prompt": prompts, "actions": actions, "action_mask": mask,
                "timing": timing, "sample_ids": sample_ids}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="Verified prior demonstration setup")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Use only the four local bowl archives")
    parser.add_argument("--endpoint", action="append", default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.endpoint = args.endpoint or ["https://huggingface.co"]
    prepare(args)


if __name__ == "__main__":
    main()
