"""Compare every staged example to direct released OpenPI LeRobot observations and labels."""

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import socket

import numpy as np
from openpi_client import image_tools
from PIL import Image
import pyarrow.parquet as pq
from ur5e_motor import to_openpi_observation

from source_data import FINGERS, IMAGE_KEYS, POS, PublicHistory, SourceDataset


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def reference_history(actions, states, t):
    first = max(0, t - 4)
    width = np.abs(states[:, 6] - states[:, 7])
    h = np.zeros((4, 13), dtype=np.float32)
    count = t - first
    if count:
        h[-count:, :7] = actions[first:t]
        h[-count:, 7:10] = np.diff(states[first:t + 1, :3], axis=0)
        h[-count:, 10] = np.diff(width[first:t + 1])
        h[-count:, 11] = width[first + 1:t + 1]
        h[-count:, 12] = 1
    return h.flatten()


def check(root, inputs):
    manifest = json.loads((root / "manifest.json").read_text())
    ids = set()
    summaries = {}
    for split, part in manifest["splits"].items():
        ds = SourceDataset(root, split)
        assert file_hash(root / part["file"]) == part["sha256"]
        examples, pixel_arrays, state_error = 0, 0, 0.
        depths = Counter()
        per_task = Counter()
        for traj in part["trajectories"]:
            assert traj["sequence_sha256"] not in ids
            ids.add(traj["sequence_sha256"])
            p = inputs / Path(traj["parquet"]).name
            assert file_hash(p) == traj["parquet_sha256"]
            table = pq.read_table(p, use_threads=False)
            actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
            states = np.asarray(table["state"].to_pylist(), dtype=np.float32)
            for t in traj["t"]:
                row = ds[examples]
                assert (row["trajectory_id"], row["prompt"], row["t"]) == (traj["id"], traj["prompt"], t)
                assert row["actions"].dtype == row["history"].dtype == np.float32
                np.testing.assert_array_equal(row["actions"], actions[t:t + 10])
                np.testing.assert_array_equal(row["history"], reference_history(actions, states, t))
                depths[int(row["history"][12::13].sum())] += 1
                item = to_openpi_observation(row["raw"], row["prompt"])
                assert item["prompt"] == traj["prompt"]
                err = np.abs(item["observation/state"] - states[t])
                state_error = max(state_error, float(err.max()))
                np.testing.assert_array_equal(item["observation/state"].astype(np.float32), states[t])
                for raw_key, col, model_key in zip(IMAGE_KEYS, ["image", "wrist_image"],
                                                  ["observation/image", "observation/wrist_image"], strict=True):
                    with Image.open(BytesIO(table[col][t].as_py()["bytes"])) as im:
                        direct = np.array(im)
                    assert direct.dtype == row["raw"][raw_key].dtype == np.uint8
                    np.testing.assert_array_equal(row["raw"][raw_key][::-1, ::-1], direct)
                    expected = image_tools.convert_to_uint8(image_tools.resize_with_pad(direct, 224, 224))
                    np.testing.assert_array_equal(item[model_key], expected)
                    pixel_arrays += 1
                examples += 1
                per_task[row["prompt"]] += 1
        assert examples == len(ds) == part["examples"]
        assert ds[-1]["t"] == ds[len(ds) - 1]["t"]
        for invalid in (-1 - len(ds), len(ds)):
            try:
                ds[invalid]
            except IndexError:
                continue
            raise AssertionError("Out-of-bounds index accepted")
        summaries[split] = dict(examples=examples, trajectories=len(part["trajectories"]), per_task=per_task,
                                exact_action_chunks=examples, exact_history_vectors=examples,
                                exact_camera_arrays_after_stock_conversion=pixel_arrays,
                                direct_state_max_abs_error=state_error, exact_state_float32=examples,
                                history_valid_slots=dict(sorted(depths.items())), h5_sha256=part["sha256"])
    history = PublicHistory()
    empty = np.zeros(52, np.float32)
    np.testing.assert_array_equal(history.vector(), empty)
    public = {POS: np.array([.1, .2, .3]), FINGERS: np.array([.04, -.04])}
    following = {POS: np.array([.11, .18, .35]), FINGERS: np.array([.03, -.03])}
    for i in range(7):
        history.append(np.full(7, i / 10), public, following)
    snapshot = history.vector()
    np.testing.assert_array_equal(snapshot[::13], np.array([.3, .4, .5, .6], dtype=np.float32))
    history.reset()
    np.testing.assert_array_equal(history.vector(), empty)
    assert snapshot.any()
    return dict(host=socket.gethostname(), cpus=sorted(os.sched_getaffinity(0)), cuda_visible=os.environ["CUDA_VISIBLE_DEVICES"],
                dataset_root=str(root), manifest_sha256=file_hash(root / "manifest.json"),
                splits=summaries, distinct_full_trajectory_hashes=len(ids), reset_and_eviction="passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    result = check(args.dataset, args.inputs)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v["examples"] for k, v in result["splits"].items()}))
