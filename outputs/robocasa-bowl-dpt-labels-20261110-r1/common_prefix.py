"""Prepare an explicitly teacher-assisted handoff at the matched training decision."""
import argparse
from itertools import islice
import json
from pathlib import Path

import numpy as np

from collect import CAMERAS, carry, convert_action, held_fruit, identity, reset_scene, stable_counter_regions, state
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
            assert len(pickup["commands"]) + offset < 500
            wrapped, obs, layout = reset_scene(seed)
            try:
                env = wrapped.unwrapped.env
                assert identity(env, obs) == original["initial"]
                after, commands = [], []
                for command, expected in zip(pickup["commands"], pickup["post"], strict=True):
                    obs, _, _, _, _ = wrapped.step(convert_action(command))
                    np.testing.assert_array_equal(state(obs), expected)
                    after.append(state(obs))
                    commands.append(command.astype(np.float64))
                assert identity(env, obs) == original["decision"]
                # The training archive rounds teacher controls to float32. Regenerate
                # the original float64 controller outputs for bit-exact physics replay.
                script = carry(env, original["fruit"], layout["bowls"][0])
                for i, (_, command) in enumerate(islice(script, offset)):
                    np.testing.assert_array_equal(command.astype(np.float32), left["commands"][i])
                    obs, _, _, _, _ = wrapped.step(convert_action(command))
                    commands.append(command.copy())
                    after.append(state(obs))
                assert len(commands) == original["controls"] + offset
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
