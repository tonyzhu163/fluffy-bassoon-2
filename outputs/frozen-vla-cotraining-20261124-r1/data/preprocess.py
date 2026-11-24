"""Build the registered two-split Panda dataset from pinned RLDS and LeRobot bytes."""

import argparse
from collections import defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import struct
import subprocess
import tempfile

import h5py
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from source_data import FINGERS, HORIZON, IMAGE_KEYS, POS, QUAT, PublicHistory

HERE = Path(__file__).resolve().parent
RLDS_REV = "6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551"
LEROBOT_REV = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def records(path):
    try:
        from tfrecord.example_pb2 import Example
    except ImportError:
        from example_pb2 import Example  # standalone generated module in the Baige overlay
    with path.open("rb") as f:
        while header := f.read(12):
            length = struct.unpack("<Q", header[:8])[0]
            payload = f.read(length)
            require(len(payload) == length and len(f.read(4)) == 4, f"Truncated {path}")
            yield Example.FromString(payload).features.feature


def checked_input(inputs, spec):
    path = inputs / Path(spec["path"]).name
    require(path.stat().st_size == spec["size"] and sha256(path) == spec["sha256"],
            f"Release digest failed for {path}")
    return path


def rgb(encoded):
    with Image.open(BytesIO(encoded)) as im:
        require(im.mode == "RGB" and im.size == (256, 256), f"Unexpected RGB format {im.mode} {im.size}")
        return np.array(im)


def xyzw(axis_angle):
    v = axis_angle.astype(np.float64)
    theta = np.linalg.norm(v)
    if theta == 0:
        return np.array([0., 0., 0., 1.])
    return np.r_[v * (np.sin(theta / 2) / theta), np.cos(theta / 2)]


def starts(n, count, episode):
    # Cover the full trajectory; distribute the first decision across reset and ages 1–3.
    times = np.linspace(0, n - HORIZON, count, dtype=np.int64)
    times[0] = episode % 4
    require(np.all(np.diff(times) > 0), "Window locations overlap at an identical step")
    return times.tolist()


def public_proprio(state):
    return {POS: state[:3], FINGERS: state[6:]}


def build_trajectory(plan, f, inputs):
    path = checked_input(inputs, plan)
    table = pq.read_table(path, use_threads=False)
    n = plan["steps"]
    require(table.num_rows == n, "Episode length mismatch")
    require(table["frame_index"].to_pylist() == list(range(n)), "Noncontiguous LeRobot episode")
    require(set(table["episode_index"].to_pylist()) == {plan["episode"]}, "Episode ID mismatch")
    prompts = f["steps/language_instruction"].bytes_list.value
    require(len(prompts) == n and set(prompts) == {plan["prompt"].encode()}, "Prompt mismatch")
    for key, want in [("is_first", [1] + [0] * (n - 1)), ("is_last", [0] * (n - 1) + [1]),
                      ("is_terminal", [0] * (n - 1) + [1])]:
        require(list(f["steps/" + key].int64_list.value) == want, f"RLDS {key} mismatch")
    require(list(f["steps/reward"].float_list.value) == [0.] * (n - 1) + [1.], "Successful terminal required")
    state = np.asarray(table["state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
    require(state.shape == (n, 8) and actions.shape == (n, 7), "Unexpected controller/state dimensions")
    for arr, name in [(actions, "action"), (state, "observation/state")]:
        original = np.asarray(f["steps/" + name].float_list.value, dtype=np.float32)
        require(np.array_equal(arr.ravel(), original) and np.isfinite(arr).all(), f"RLDS {name} mismatch")
    require(np.max(np.abs(actions)) <= 1, "Out-of-range expert action")
    t_values = starts(n, plan["samples"], plan["episode"])
    rows = []
    history = PublicHistory()
    image_diffs = []
    for t in range(n):
        if t in t_values:
            row = dict(public_proprio(state[t]), actions=actions[t:t + HORIZON], history=history.vector())
            row[QUAT] = xyzw(state[t, 3:6])
            for raw_key, column in zip(IMAGE_KEYS, ("image", "wrist_image"), strict=True):
                canonical = rgb(table[column][t].as_py()["bytes"])
                # Released LeRobot PNG follows the RLDS builder's 180-degree rotation.
                row[raw_key] = np.ascontiguousarray(canonical[::-1, ::-1])
                jpeg = rgb(f["steps/observation/" + column].bytes_list.value[t]).astype(np.int16)
                aligned = np.abs(jpeg - canonical).mean()
                inverted = np.abs(jpeg - row[raw_key]).mean()
                require(aligned < inverted, "RLDS and LeRobot camera orientations disagree")
                image_diffs.append((aligned, inverted))
            rows.append(row)
        if t + 1 < n:
            history.append(actions[t], public_proprio(state[t]), public_proprio(state[t + 1]))
    meta = dict(id=f"panda/episode_{plan['episode']:06d}", prompt=plan["prompt"], suite=plan["suite"],
                episode=plan["episode"], steps=n, t=t_values, parquet=plan["path"],
                parquet_sha256=plan["sha256"], rlds_ordinal=plan["ordinal"],
                upstream_hdf5=f["episode_metadata/file_path"].bytes_list.value[0].decode(),
                sequence_sha256=hashlib.sha256(state.tobytes() + actions.tobytes()).hexdigest(),
                terminal_reward=1, is_last=True, is_terminal=True)
    return meta, rows, image_diffs


def build(inputs, tmp):
    selection = json.loads((HERE / "selection.json").read_text())
    archives = json.loads((HERE / "archives.json").read_text())
    targets = {(p["suite"], p["shard"], p["ordinal"]): p for p in selection}
    require(len(targets) == len(selection), "Repeated RLDS trajectory")
    by_split = defaultdict(list)
    image_diffs = []
    sequences = set()
    for archive in archives:
        path = checked_input(inputs, archive)
        suite = path.name.split("-")[0]
        require(suite in ("libero_object", "libero_spatial"), "Reserved suite")
        shard = int(path.name.split("-")[-3])
        for ordinal, features in enumerate(records(path)):
            plan = targets.pop((suite, shard, ordinal), None)
            if plan is None:
                continue
            meta, rows, diffs = build_trajectory(plan, features, inputs)
            require(meta["sequence_sha256"] not in sequences, "Duplicate control/state sequence")
            sequences.add(meta["sequence_sha256"])
            meta["rlds_shard"] = archive["path"]
            by_split[plan["split"]].append((meta, rows))
            image_diffs.extend(diffs)
    require(not targets, "Missing selected records")
    manifest = dict(schema="libero-panda-openpi-v1", robot="Panda", rlds_revision=RLDS_REV,
                    lerobot_revision=LEROBOT_REV, rlds_archives=archives,
                    code_revision=subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip(),
                    selection_sha256=sha256(HERE / "selection.json"), splits={},
                    excluded_suites=["libero_goal", "libero_10"], excluded_robots="all arms except Panda",
                    split_unit="whole trajectory", horizon=10, history_shape=[52],
                    rlds_jpeg_vs_lerobot_png_mean_absolute_rgb_difference=dict(
                        aligned=float(np.mean(image_diffs, axis=0)[0]), rotated=float(np.mean(image_diffs, axis=0)[1]),
                        images_compared=len(image_diffs)))
    for split, expected in [("train", 192), ("heldout", 64)]:
        items = sorted(by_split[split], key=lambda item: item[0]["episode"])
        rows = [row for _, chunk in items for row in chunk]
        require(len(rows) == expected, f"{split} example count mismatch")
        out = tmp / f"{split}.h5"
        with h5py.File(out, "x") as f:
            for key in rows[0]:
                values = np.stack([row[key] for row in rows])
                f.create_dataset(key, data=values, track_times=False)
        manifest["splits"][split] = dict(file=out.name, sha256=sha256(out), examples=len(rows),
                                         trajectories=[meta for meta, _ in items])
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    dest = args.destination
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as scratch:
        ready = Path(scratch) / "dataset"
        ready.mkdir()
        build(args.inputs, ready)
        ready.rename(dest)


if __name__ == "__main__":
    main()
