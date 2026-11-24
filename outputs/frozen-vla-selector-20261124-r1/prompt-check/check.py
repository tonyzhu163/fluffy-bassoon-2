"""Frozen SmolVLA prompt sensitivity on one cached drawer-and-bowl observation.

With the appearance run's CPU environment and cached assets, from this directory:
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false "$WORK/env/bin/python" check.py "$WORK" receipt.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

APPEARANCE = Path(__file__).resolve().parents[2] / "frozen-vla-factory-20261123-r1/appearance"
sys.path.insert(0, str(APPEARANCE))
from motor import Motor, WEIGHTS_SHA256


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def delta(a, b):
    diff = a[:5].astype(float) - b[:5]

    def rms(x):
        return float(np.sqrt(np.mean(x**2)))

    return dict(xyz_rms=rms(diff[:, :3]), rotation_rms=rms(diff[:, 3:6]),
                gripper_rms=rms(diff[:, 6]), all_channels_rms=rms(diff),
                max_abs_by_step=np.max(np.abs(diff), axis=1).tolist())


def check(work):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with np.load(work / "cases/01.npz") as case:
        obs = {k: case[k] for k in ("source", "wrist", "pos", "quat", "grip", "task")}
    original = str(obs["task"])
    assert original == "open the top drawer and put the bowl inside"
    prior = json.loads((APPEARANCE / "motor.json").read_text())["tasks"][1]
    assert original == prior["task"]
    motor = Motor(work)
    noise = torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(9101))
    assert digest(noise.numpy()) == prior["noise_sha256"]
    prompts = dict(original=original, open="open the top drawer",
                   put="put the bowl in the top drawer", repeat=original)
    actions = {k: motor.propose([obs["source"], obs["wrist"]], dict(obs, task=prompt), noise)
               for k, prompt in prompts.items()}
    assert digest(actions["original"]) == prior["source_chunk_sha256"]
    np.testing.assert_array_equal(actions["repeat"], actions["original"])
    return dict(scope="Different VLA from the primary selector. Forward-pass sensitivity only, one Panda reset frame and one noise draw. No phase, reward or rollout evaluation.",
                policy="lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d",
                weights_sha256=WEIGHTS_SHA256, device="cpu", trainable_parameters=0,
                source="cases/01.npz under appearance scratch root, LIBERO-Goal reset seed 3201",
                input_sha256={k: digest(obs[k]) for k in ("source", "wrist", "pos", "quat", "grip")},
                noise_seed=9101, noise_sha256=digest(noise.numpy()), prompts=prompts,
                chunk_shape=list(actions["original"].shape), measured_steps=[0, 1, 2, 3, 4],
                units="Unnormalized SmolVLA policy commands, each component group in its own units.",
                chunk_sha256={k: digest(a) for k, a in actions.items()},
                first_five_deltas={f"{a}_vs_{b}": delta(actions[a], actions[b]) for a, b in
                                   [("repeat", "original"), ("open", "original"),
                                    ("put", "original"), ("open", "put")]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    args.receipt.write_text(json.dumps(check(args.work), indent=2) + "\n")
