"""Read existing Baige baseline artifacts on CPU; never run a model or simulator."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch


def array(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.float().numpy() if value.dtype == torch.bfloat16 else value.numpy()
    return np.asarray(value)


def difference(left, right):
    left, right = array(left), array(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return dict(exact=False, shape_or_dtype_mismatch=True)
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return dict(exact=bool(np.array_equal(left, right)),
                maximum_absolute_delta=float(delta.max()) if delta.size else 0.,
                mean_absolute_delta=float(delta.mean()) if delta.size else 0.,
                differing_components=int(np.count_nonzero(delta)),
                total_components=int(delta.size))


def load_reference(reference, *, compressed=False):
    path = Path(reference["path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError(f"Saved reference hash mismatch: {path}")
    return (pickle.loads(gzip.decompress(raw)) if compressed else
            torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False))


def differing_fields(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        return [path for key in sorted(left.keys() | right.keys())
                for path in differing_fields(left.get(key), right.get(key), f"{prefix}.{key}".strip("."))]
    return [] if left == right else [prefix]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs", type=Path)
    args = parser.parse_args()
    started = time.time()
    runs = [args.outputs / f"robocasa-joint-steering-{arm}-scale50-20261103-r1"
            for arm in ("velocity", "kv")]
    rows, executions, identities = [], [], []
    cameras = ("main_images", "wrist_images", "extra_view_images")
    for rank in range(8):
        requests, results = [], []
        for run in runs:
            worker = run / "workers" / f"rank-{rank:02d}"
            request_path = worker / "request-000002.pt"
            request_raw = request_path.read_bytes()
            request = torch.load(io.BytesIO(request_raw), map_location="cpu", weights_only=False)
            result = torch.load(worker / "result-000002.pt", map_location="cpu", weights_only=False)
            assert result["request_sha256"] == hashlib.sha256(request_raw).hexdigest()
            assert result["completed"] and result["phase"] == "test_before"
            assert request["zero_steering"] and request["deterministic"]
            requests.append(request)
            results.append(result)
            identities.append(dict(arm=run.name, rank=rank,
                signature=result["native_model_frozen_signature"]))
        assert requests[0]["cases"] == requests[1]["cases"]
        executions.append(dict(rank=rank, velocity=results[0]["execution"],
                               kv=results[1]["execution"]))
        for left, right in zip(results[0]["episodes"], results[1]["episodes"], strict=True):
            assert left["case"] == right["case"] and left["completed"] and right["completed"]
            initials = [load_reference(episode["initial"]) for episode in (left, right)]
            chunks = [load_reference(episode["chunks"][0], compressed=True)
                      for episode in (left, right)]
            a, b = chunks
            assert a["case"] == b["case"] == left["case"]
            assert a["chunk_index"] == b["chunk_index"] == 0
            assert a["zero_steering"] and b["zero_steering"]
            snapshots = [initial["evidence"]["snapshot"] for initial in initials]
            observations = {key: difference(a["observation"][key], b["observation"][key])
                            for key in ("states", *cameras)}
            row = dict(case=left["case"], rank=rank, case_slot=left["case_slot"],
                initial_refs=[episode["initial"] for episode in (left, right)],
                first_chunk_refs=[episode["chunks"][0] for episode in (left, right)],
                physical_exact=snapshots[0]["physical"] == snapshots[1]["physical"],
                physical_differing_fields=differing_fields(snapshots[0]["physical"], snapshots[1]["physical"]),
                observations=observations,
                pixels_exact=all(observations[key]["exact"] for key in cameras),
                noise_seed_exact=a["native_noise_seed"] == b["native_noise_seed"],
                noise=difference(a["native_noise"], b["native_noise"]),
                controls_zero=[int(np.count_nonzero(array(c["control"]))) == 0 for c in chunks],
                features=difference(a["features"], b["features"]),
                requested_commands=difference(a["requested_commands"], b["requested_commands"]),
                executed_commands=difference(a["executed_commands"], b["executed_commands"]),
                outcomes={name: [left[name], right[name]]
                          for name in ("success_once", "success_at_end", "first_success_step", "horizon")})
            rows.append(row)
        print(f"Read rank {rank}: {len(rows)} paired episodes", file=__import__("sys").stderr, flush=True)
    assert len(rows) == 500 and len({(row["case"]["task"], row["case"]["native_seed"])
                                    for row in rows}) == 500
    by_task = defaultdict(lambda: dict(episodes=0, velocity_first=0, kv_first=0,
        velocity_end=0, kv_end=0, first_disagreements=0, end_disagreements=0,
        velocity_only_first=0, kv_only_first=0))
    for row in rows:
        entry = by_task[row["case"]["task"]]
        entry["episodes"] += 1
        first, end = row["outcomes"]["success_once"], row["outcomes"]["success_at_end"]
        for index, arm in enumerate(("velocity", "kv")):
            entry[f"{arm}_first"] += int(first[index])
            entry[f"{arm}_end"] += int(end[index])
        entry["first_disagreements"] += int(first[0] != first[1])
        entry["end_disagreements"] += int(end[0] != end[1])
        entry["velocity_only_first"] += int(first[0] and not first[1])
        entry["kv_only_first"] += int(first[1] and not first[0])
    summary = dict(pairs=len(rows), verified_initial_and_first_chunk_references=4 * len(rows),
        same_cases=500,
        exact_initial_physical_fingerprints=sum(row["physical_exact"] for row in rows),
        exact_initial_public_states=sum(row["observations"]["states"]["exact"] for row in rows),
        exact_native_noise_seeds=sum(row["noise_seed_exact"] for row in rows),
        exact_native_noise_tensors=sum(row["noise"]["exact"] for row in rows),
        zero_steering_pairs=sum(all(row["controls_zero"]) for row in rows),
        maximum_initial_public_state_difference=max(row["observations"]["states"]["maximum_absolute_delta"] for row in rows),
        exact_initial_pixels=sum(row["pixels_exact"] for row in rows),
        first_chunk_action_disagreements=sum(not row["requested_commands"]["exact"] for row in rows),
        action_disagreements_with_identical_pixels=sum(row["pixels_exact"] and not row["requested_commands"]["exact"] for row in rows),
        action_disagreements_with_identical_pixels_and_features=sum(row["pixels_exact"] and row["features"]["exact"] and not row["requested_commands"]["exact"] for row in rows),
        maximum_first_chunk_command_difference=max(row["requested_commands"]["maximum_absolute_delta"] for row in rows),
        maximum_initial_pixel_difference=max(row["observations"][key]["maximum_absolute_delta"] for row in rows for key in cameras),
        different_batch_schedules=sum(e["velocity"]["native_batch_sizes"] != e["kv"]["native_batch_sizes"] for e in executions),
        any_initial_lane_barrier=any(e[arm]["initial_lane_barrier"] for e in executions for arm in ("velocity", "kv")),
        first_success_outcome_disagreements=sum(v["first_disagreements"] for v in by_task.values()),
        end_success_outcome_disagreements=sum(v["end_disagreements"] for v in by_task.values()),
        new_physical_controls=0, new_optimizer_updates=0, new_GPU_model_calls=0,
        elapsed_seconds=time.time() - started)
    print(json.dumps(dict(format="saved_zero_steering_baseline_audit_v1", summary=summary,
        by_task=dict(by_task), executions=executions, identities=identities, pairs=rows), allow_nan=False))


if __name__ == "__main__":
    main()
