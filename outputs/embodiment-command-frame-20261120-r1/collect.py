"""Collect the registered H1 world-frame response tapes on the native R2 CPU runtime."""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
R2 = HERE.parent / "libero-paligemma-dualarm-20261120-r2"
FAMILIES = ("Panda", "Sawyer")
TRAIN = ("nominal", "yaw_p90", "yaw_m90", "gain065")
EVAL = ("yaw_p45", "yaw_p90_gain065")
MAPPING = {
    "nominal": (0, 1.0), "yaw_p90": (90, 1.0), "yaw_m90": (-90, 1.0),
    "gain065": (0, 0.65), "yaw_p45": (45, 1.0), "yaw_p90_gain065": (90, 0.65),
}
PROMPT = "push the plate to the front of the stove"
STEPS = 16


def blocks():
    for split, base, count, settings in (("train", 960000, 12, TRAIN), ("eval", 970000, 8, EVAL)):
        for index in range(count):
            for offset, role in ((0, "support"), (100, "query")):
                yield split, role, index, base + offset + index, settings


def probe(seed, preflight):
    if preflight:
        # The .60 X pair supplies the exact public inverse of .39 X under gain .65.
        axes = np.array([[.39, 0, 0], [0, .39, 0], [0, 0, .39], [.60, 0, 0]])
        signed = np.stack((axes, -axes), axis=1).reshape(8, 3)
        commands = np.zeros((STEPS, 7), dtype=np.float32)
        commands[:, :3] = np.repeat(signed, 2, axis=0)
        commands[:, 6] = -1
        return commands
    rng = np.random.default_rng(seed)
    return np.column_stack((rng.uniform(-.4, .4, (STEPS, 6)),
                            rng.choice([-1, 1], (STEPS, 1)))).astype(np.float32)


def mapped(commands, setting):
    yaw, gain = MAPPING[setting]
    angle = np.deg2rad(yaw)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    issued = commands.copy()
    issued[:, :3] = gain * (commands[:, :3] @ rotation.T)
    if np.max(np.abs(issued)) > 1 or not np.isfinite(issued).all():
        raise ValueError("Probe exceeds the native action bounds")
    return issued


def tape(env, flat, origin, family, setting, commands, opening, progress):
    from generated_layouts import reset_generated

    obs = reset_generated(env, family, False, flat, origin)
    reset = env.reset_info
    physics = reset["environment_reproducibility"]
    wrist = [obs["state"][:3].copy()]
    current = (physics["initial_state_fingerprint"], physics["post_settle_scene_fingerprint"],
               obs["state"].tolist())
    if opening is not None and current != opening:
        raise ValueError("Matched settings have different settled states or public openings")
    if reset["renderer"] or obs["prompt"] != PROMPT:
        raise ValueError("Expected camera-free BASE Goal 5")
    actions, rewards, success, motion = [], [], [], []
    for step, action in enumerate(mapped(commands, setting), start=1):
        packet = env.arm.step(action)
        progress["native_controls_returned"] += 1
        actual = packet["issued_action"]
        if packet["worker_step_index"] != step or not np.array_equal(actual, action):
            raise ValueError(f"Native action or step counter mismatch at control {step}")
        observed = packet["observation"]["robot0_eef_pos"]
        delta = packet["sensor_motion"][:3]
        if (observed.shape != (3,) or delta.shape != (3,) or
                not np.allclose(observed - wrist[-1], delta, rtol=0, atol=1e-7)):
            raise ValueError(f"Invalid measured world wrist displacement at control {step}")
        wrist.append(observed.copy())
        motion.append(delta.tolist())
        actions.append(actual.tolist())
        rewards.append(packet["reward"])
        success.append(packet["success"])
        if (packet["terminated"] or packet["truncated"]) and step < STEPS:
            raise ValueError(f"Native episode ended after {step}/{STEPS} controls")
    return dict(commands=commands.tolist(), motion=motion, issued_actions=actions,
                wrist_world_m=np.asarray(wrist).tolist(), rewards=rewards, success=success,
                terminated=packet["terminated"], truncated=packet["truncated"], reset=reset), current


def pulse_checks(row, nominal):
    """Use second-step measured responses from the eight two-control pulses."""
    setting = row["setting"]
    response = np.asarray(row["motion"])[1::2]
    issued = np.asarray(row["issued_actions"])[1::2, :3]
    # Independent expectations from the registered physical axes, in public +X,+Y,+Z order.
    directions = {"nominal": (1, 2, 3), "yaw_p90": (2, -1, 3),
                  "yaw_m90": (-2, 1, 3), "gain065": (1, 2, 3)}[setting]
    signed = []
    for pair, direction in enumerate((*directions, directions[0])):
        axis = abs(direction) - 1
        plus, minus = response[2 * pair:2 * pair + 2, axis] * np.sign(direction)
        signed.extend((float(plus), float(-minus)))

    # These PUBLIC pulse pairs should issue exactly the same physical +/- X commands as nominal.
    indices = {"nominal": [0, 1], "yaw_p90": [3, 2], "yaw_m90": [2, 3], "gain065": [6, 7]}[setting]
    target = np.asarray(nominal["commands"])[[1, 3], :3]
    command_error = float(np.max(np.abs(issued[indices] - target)))
    reference = np.asarray(nominal["motion"])[[1, 3]]
    # Half the +/- difference cancels constant drift. Compare in measured world metres.
    def contrast(pair):
        return (pair[0] - pair[1]) / 2

    corrected = float(np.linalg.norm(contrast(response[indices]) - contrast(reference)))
    raw = float(np.linalg.norm(contrast(response[:2]) - contrast(reference)))
    passed = min(signed) > 0 and command_error <= 1e-7 and (setting == "nominal" or corrected < raw)
    return dict(signed_displacements_m=signed, inverse_pulse_indices=indices,
                inverse_issued_max_abs_error=command_error, inverse_motion_error_m=corrected,
                raw_motion_error_m=raw, passed=passed)


def collect(out):
    sys.path.insert(0, str(R2))
    from expert import session, write_json
    from generated_layouts import make_sampler, released_fingerprints, sample_layout
    from large_panel_protocol import mobile_joints

    out.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip()
    progress = dict(revision=revision, collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    expected_tapes=256, expected_transitions=4096, completed_tapes=0,
                    native_controls_returned=0, layouts=[], preflight=[], status="running")
    write_json(out / "collection.json", progress)
    seen = {key: set() for key in ("initial_state_sha256", "scene_hash", "movable_xy_hash", "mobile_joint_xy_hash")}
    partial = out / "tapes.jsonl.partial"
    try:
        public = released_fingerprints()
        with closing(make_sampler()) as sampler, partial.open("x", buffering=1) as stream:
            for split, role, index, layout_seed, settings in blocks():
                block = dict(split=split, role=role, index=index, layout_seed=layout_seed)
                progress["current"] = block
                flat, origin = sample_layout(sampler, layout_seed, public)
                hashes = {**origin, "mobile_joint_xy_hash": mobile_joints(origin["sampled_scene"])}
                for key, values in seen.items():
                    if hashes[key] in values:
                        raise ValueError(f"Different layout blocks share {key} at seed {layout_seed}")
                    values.add(hashes[key])
                progress["layouts"].append(dict(hashes, **block, panda_flat_state=flat.tolist()))
                first = (split, role, index) == ("train", "support", 0)
                probe_seed = layout_seed + 2_000_000
                commands = probe(probe_seed, first)
                for family in FAMILIES:
                    opening = None
                    nominal = None
                    with session(family, False) as env:
                        for setting in settings:
                            identity = dict(**block, family=family, setting=setting)
                            progress["current"] = identity
                            data, opening = tape(env, flat, origin, family, setting, commands, opening, progress)
                            row = dict(**identity, **data, layout_id=origin["source_index"],
                                       probe_seed=probe_seed, probe_kind="signed_pulses" if first else "mixed",
                                       **{key: hashes[key] for key in seen})
                            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                            progress["completed_tapes"] += 1
                            if first:
                                if setting == "nominal":
                                    nominal = row
                                check = pulse_checks(row, nominal)
                                progress["preflight"].append(dict(**identity, **check))
                                if not check["passed"]:
                                    raise ValueError(f"Signed response or inverse failed for {family}/{setting}: {check}")
                write_json(out / "collection.json", progress)
                print(f"{progress['completed_tapes']}/256 tapes, {split} {role} {index}", flush=True)
        if (progress["completed_tapes"] != 256 or progress["native_controls_returned"] != 4096 or
                len(progress["preflight"]) != 8 or len(seen["scene_hash"]) != 40):
            raise RuntimeError("Incomplete registered collection")
        partial.rename(out / "tapes.jsonl")
    except Exception as exc:
        progress.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(out / "collection.json", progress)
        raise
    del progress["current"]
    progress.update(status="complete", transitions=STEPS * progress["completed_tapes"], distinct_layouts=40)
    write_json(out / "collection.json", progress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh collection directory")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Native H1 collection requires the allocated Linux simulator runtime")
    collect(args.out)
