"""Score generated cutouts and oracle comparisons on unmodified native target RGB."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "appearance"))
from measure import metrics, overlay
from motor import Motor


def images(public, deploy, oracle, diagnostic):
    native = public["native"]
    yield "source_reference", oracle["source"]
    yield "no_op", native
    yield "public_ik", overlay(native, deploy["rgb"], deploy["mask"])
    yield "displaced", overlay(native, deploy["rgb"], deploy["mask"], (-128, 0))
    yield "oracle_q", overlay(native, diagnostic["rgb"], diagnostic["mask"])
    yield "oracle_q_visibility", overlay(native, diagnostic["rgb"], diagnostic["mask"] & oracle["sm"])
    yield "matched_source", overlay(native, oracle["source"], oracle["sm"])


def evaluate(work, first, stop):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    motor = Motor(work)
    root = work / "followup"
    rig = json.loads((root / "sprites/installation.json").read_text())
    rest_quat = np.asarray(rig["rest_hand_quat_xyzw"])
    (root / "actions").mkdir(exist_ok=True)
    records = []
    for i in range(first, stop):
        name = f"{i:02d}.npz"
        data = []
        for kind in ("public", "sprites", "oracle", "diagnostic"):
            with np.load(root / kind / name) as f:
                data.append(dict(f))
        public = data[0]
        # q and -q encode the same orientation but LeRobot's axis-angle input
        # is sign sensitive near pi. Use the hemisphere of the fixed source
        # rest pose to preserve the source numeric convention.
        if np.dot(public["quat"], rest_quat) < 0:
            public["quat"] = -public["quat"]
        noise = torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(9100 + i))
        cache, actions = {}, {}
        for condition, rgb in images(*data):
            digest = hashlib.sha256(rgb.tobytes()).digest()
            # If the visibility diagnostic changes no pixels, its action is
            # identical by construction. Keep an actual duplicate no-op below.
            if digest not in cache:
                cache[digest] = motor.propose([rgb, public["wrist"]], public, noise)
            actions[condition] = cache[digest]
        repeat = motor.propose([public["native"], public["wrist"]], public, noise)
        np.testing.assert_array_equal(repeat, actions["no_op"])
        assert np.all(actions["source_reference"].std(0) > 0)
        np.savez_compressed(root / "actions" / name, **actions)
        for condition, a in actions.items():
            records.append(dict(task=str(public["task"]), condition=condition,
                                **metrics(a, actions["source_reference"])))
        print(i, public["task"], flush=True)
    with (root / f"scores-{first}-{stop}.csv").open("w") as f:
        writer = csv.DictWriter(f, list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("work", type=Path)
    p.add_argument("first", type=int)
    p.add_argument("stop", type=int)
    args = p.parse_args()
    evaluate(args.work, args.first, args.stop)
