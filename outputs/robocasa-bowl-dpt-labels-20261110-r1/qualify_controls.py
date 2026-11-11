"""Verify full-precision teacher replay and compatibility with archived training targets."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from collect import CAMERAS, branch, convert_action, identity, release, reset_scene, stable_counter_regions, state
from export_targets import native_targets
from readout import read


def check(args):
    stable_counter_regions()
    args.out.mkdir()
    dest = args.out / str(args.seed)
    dest.mkdir()
    original = args.source / str(args.seed)
    for name in ("pickup.json", "pickup.npz"):
        shutil.copy2(original / name, dest / name)
    prefix = read(dest / "pickup.json")
    stats = read(args.norm)
    stats = stats.get("norm_stats", stats)
    results = []
    for side in ("left", "right"):
        branch(args.out, args.seed, side)
        with np.load(dest / f"{side}.npz") as current, np.load(original / f"{side}.npz") as old:
            commands = current["commands"]
            assert commands.dtype == np.float64 and old["commands"].dtype == np.float32
            np.testing.assert_array_equal(commands.astype(np.float32), old["commands"])
            for field in ("frames", "states", "offsets"):
                np.testing.assert_array_equal(current[field], old[field])
            new_targets, new_mask = native_targets(current, stats)
            old_targets, old_mask = native_targets(old, stats)
            np.testing.assert_array_equal(new_targets, old_targets)
            np.testing.assert_array_equal(new_mask, old_mask)
            frames, states, offsets = current["frames"], current["states"], current["offsets"]
            wrapped, obs, layout = reset_scene(args.seed)
            try:
                env = wrapped.unwrapped.env
                assert identity(env, obs) == prefix["initial"]
                with np.load(dest / "pickup.npz") as pickup:
                    for command, expected in zip(pickup["commands"], pickup["post"], strict=True):
                        obs, _, _, _, _ = wrapped.step(convert_action(command))
                        np.testing.assert_array_equal(state(obs), expected)
                assert identity(env, obs) == prefix["decision"]
                checkpoints = {int(offset): i for i, offset in enumerate(offsets)}
                for cursor in range(len(commands) + 1):
                    if cursor in checkpoints:
                        index = checkpoints[cursor]
                        np.testing.assert_array_equal(state(obs), states[index])
                        np.testing.assert_array_equal(np.stack([obs[k] for k in CAMERAS]), frames[index])
                    if cursor < len(commands):
                        obs, _, _, _, _ = wrapped.step(convert_action(commands[cursor]))
                actual = release(env, layout["bowls"])
                assert actual == read(dest / f"{side}.json")["placed"]
                results.append(dict(side=side, controls=len(commands), verified_query_boundaries=len(offsets),
                                    placed=actual, all_public_states_and_views_equal=True,
                                    float32_targets_and_masks_unchanged=True))
            finally:
                wrapped.close()
    report = dict(scene=args.seed, stored_command_dtype="float64", results=results)
    (args.out / "verified.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for field in ("source", "norm", "out"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--seed", type=int, required=True)
    check(parser.parse_args())
