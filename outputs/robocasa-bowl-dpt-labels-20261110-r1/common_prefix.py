"""Prepare an explicitly teacher-assisted handoff at the matched training decision."""
import argparse
import json
from pathlib import Path

import numpy as np

from collect import CAMERAS, convert_action, held_fruit, identity, reset_scene, stable_counter_regions, state
from export_targets import common_boundary
from readout import read


def prepare(args):
    stable_counter_regions()
    args.out.mkdir()
    records = []
    for seed in read(args.admission)["admitted"]:
        source = args.source / str(seed)
        original = read(source / "pickup.json")
        with np.load(source / "pickup.npz") as pickup, np.load(source / "left.npz") as left, np.load(source / "right.npz") as right:
            index = common_boundary(left, right)
            assert index is not None
            offset = int(left["offsets"][index])
            np.testing.assert_array_equal(left["commands"][:offset], right["commands"][:offset])
            commands = np.concatenate((pickup["commands"], left["commands"][:offset]))
            assert len(commands) < 500
            wrapped, obs, layout = reset_scene(seed)
            try:
                env = wrapped.unwrapped.env
                assert identity(env, obs) == original["initial"]
                after = []
                for i, command in enumerate(commands):
                    obs, _, _, _, _ = wrapped.step(convert_action(command))
                    current = state(obs)
                    if i < original["controls"]:
                        np.testing.assert_array_equal(current, pickup["post"][i])
                    after.append(current)
                np.testing.assert_array_equal(state(obs), left["states"][index])
                np.testing.assert_array_equal(np.stack([obs[k] for k in CAMERAS]), left["frames"][index])
                assert held_fruit(env) == original["fruit"]
                record = {**original, "controls": len(commands), "decision": identity(env, obs)}
                dest = args.out / str(seed)
                dest.mkdir()
                np.savez_compressed(dest / "pickup.npz", commands=commands, post=after)
                (dest / "pickup.json").write_text(json.dumps(record, indent=2))
                records.append(dict(scene=seed, frozen_controls=original["controls"],
                                    teacher_controls=offset, common_boundary=index,
                                    total_prefix=len(commands), public_state_and_all_views_match=True))
            finally:
                wrapped.close()
    (args.out / "preparation.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for field in ("source", "admission", "out"):
        parser.add_argument(field, type=Path)
    prepare(parser.parse_args())
