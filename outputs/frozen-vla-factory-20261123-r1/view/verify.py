"""Independently reload the checkpoint, compare every tensor and replay the first task."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from motor import Motor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    model = Motor(args.cache)
    state = model.policy.state_dict()
    with safe_open(args.cache / "policy/model.safetensors", framework="pt") as checkpoint:
        keys = checkpoint.keys()
        for k in keys:
            assert torch.equal(state[k], checkpoint.get_tensor(k)), k
    row = json.loads((args.out / "scores.json").read_text())["tasks"][0]
    noise = torch.randn((1, 50, 32), generator=torch.Generator().manual_seed(row["seed"]))
    with np.load(args.cache / "frames" / (row["key"] + ".npz")) as obs:
        replay = model.proposal(obs["agent"], obs["wrist"], obs, row["task"], noise)
    with np.load(args.out / "proposals.npz") as z:
        saved = z[row["key"] + "_" + row["baseline"]]
    np.testing.assert_array_equal(replay, saved)
    record = dict(checkpoint_tensors_compared=len(keys), checkpoint_tensor_values_equal=True,
                  strict_load=True, replay_task=row["key"], fresh_process_replay_max_abs=float(np.abs(replay - saved).max()))
    (args.out / "verification.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
