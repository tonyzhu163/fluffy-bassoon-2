"""Collect the frozen Panda/Sawyer probe-order panel in the camera-free R2 simulator."""

import argparse
import hashlib
import json
import signal
import subprocess
import sys
from contextlib import closing
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from runpy import run_path

import numpy as np

HERE = Path(__file__).resolve().parent
H1 = HERE.parent / "embodiment-command-frame-20261120-r1" / "collect.py"
R2 = HERE.parent / "libero-paligemma-dualarm-20261120-r2"
DESIGN_BASE = "4bd494d188bfdcb757c554bb335b5c19d3bc97e6"
FAMILIES = ("Panda", "Sawyer")
SETTINGS = tuple(product((45, 90), (1.0, 0.65)))
SCHEDULES = ("blocked", "shuffled", "query")
INDICES = range(16)
STEPS = 16


def utc_now():
    return datetime.now(UTC).isoformat()


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, allow_nan=False, indent=2) + "\n")
    tmp.replace(path)


def probes(index, mixed_probe):
    blocked = np.zeros((STEPS, 7), dtype=np.float32)
    directions = [[0.39, 0], [-0.39, 0], [0, 0.39], [0, -0.39]]
    blocked[:, :2] = np.repeat(directions, 4, axis=0)
    blocked[:, 6] = -1
    order = np.random.default_rng(2991200 + index).permutation(STEPS)
    shuffled = blocked[order]
    query = mixed_probe(2991300 + index, preflight=False)
    if not np.array_equal(shuffled[np.argsort(order)], blocked) or np.array_equal(shuffled, blocked):
        raise ValueError(f"Invalid support permutation at index {index}")
    commands = {"blocked": blocked, "shuffled": shuffled, "query": query}
    for tape in commands.values():
        if tape.shape != (STEPS, 7) or tape.dtype != np.float32 or not np.isfinite(tape).all():
            raise ValueError("Invalid public command tape")
        tape.setflags(write=False)
    return commands, order


def issued_actions(commands, yaw_deg, gain):
    angle = np.deg2rad(yaw_deg)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    issued = commands.copy()
    issued[:, :3] = gain * (commands[:, :3] @ rotation.T)
    if not np.isfinite(issued).all() or np.max(np.abs(issued)) > 1:
        raise ValueError("Private mapping exceeds native bounds")
    return issued


def translation_energy(commands):
    return float(np.square(commands[:, :3], dtype=np.float64).sum())


def arm_plan(index, family, role, commands, permutation):
    for cell, (yaw, gain) in enumerate(SETTINGS):
        if role == "query":
            yield yaw, gain, "query", issued_actions(commands["query"], yaw, gain)
            continue
        schedules = ("blocked", "shuffled")
        mapped = {name: issued_actions(commands[name], yaw, gain) for name in schedules}
        if not np.array_equal(mapped["shuffled"], mapped["blocked"][permutation]):
            raise ValueError("Issued support controls differ beyond the permutation")
        if (index + FAMILIES.index(family) + cell) % 2:
            schedules = schedules[::-1]
        for schedule in schedules:
            yield yaw, gain, schedule, mapped[schedule]


def require_headless(wrapper):
    sim_env = wrapper.env
    if sim_env.has_renderer or sim_env.has_offscreen_renderer or sim_env.use_camera_obs:
        raise ValueError("Expected a camera-free simulator with both renderers disabled")


def collect_tape(env, reset_generated, flat, origin, row, opening, issued, progress, check_stop):
    obs = reset_generated(env, row["family"], False, flat, origin)
    reset = row["reset"] = env.reset_info
    require_headless(env.arm._env)
    if reset["renderer"] or obs["prompt"] != "push the plate to the front of the stove":
        raise ValueError("Unexpected renderer or task instruction")
    state = obs["state"]
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("Invalid settled public opening")
    physics = reset["environment_reproducibility"]
    row["opening_state"] = state.tolist()
    current = (physics["initial_state_fingerprint"], physics["post_settle_scene_fingerprint"],
               row["opening_state"])
    if opening is not None and current != opening:
        raise ValueError("Settled simulator state, scene or public opening differs within arm and donor")

    previous = state[:3].copy()
    row.update(wrist_world_m=[previous.tolist()], motion=[], issued_actions=[], worker_step_index=[],
               rewards=[], success=[], terminated=[], truncated=[])
    for step, action in enumerate(issued, start=1):
        check_stop()
        progress["native_calls_started"] += 1
        packet = env.arm.step(action)
        progress["native_controls_returned"] += 1
        actual = packet["issued_action"]
        native_step = packet["worker_step_index"]
        if native_step != step or not np.array_equal(actual, action):
            raise ValueError(f"Control {step}: got step {native_step}, issued {actual!r}, expected {action!r}")
        observed = packet["observation"]["robot0_eef_pos"]
        delta = packet["sensor_motion"][:3]
        if (observed.shape != (3,) or delta.shape != (3,) or not np.isfinite(observed).all()
                or not np.isfinite(delta).all()
                or not np.allclose(observed - previous, delta, rtol=0, atol=1e-7)):
            raise ValueError(f"World wrist displacement does not match consecutive observations at {step}")
        if not np.isfinite(packet["reward"]):
            raise ValueError(f"Nonfinite native reward at control {step}")
        row["worker_step_index"].append(native_step)
        row["issued_actions"].append(actual.tolist())
        row["wrist_world_m"].append(observed.tolist())
        row["motion"].append(delta.tolist())
        for name, key in (("rewards", "reward"), ("success", "success"),
                          ("terminated", "terminated"), ("truncated", "truncated")):
            row[name].append(packet[key])
        previous = observed.copy()
        if step < STEPS and (packet["terminated"] or packet["truncated"]):
            raise ValueError(f"Native episode ended at control {step}/{STEPS}")
    row["finished_utc"] = utc_now()
    return current


def collect(out, check_stop):
    if sys.platform != "linux":
        raise RuntimeError("Native collection requires the allocated Linux simulator runtime")
    out.mkdir(parents=True, exist_ok=False)
    manifest = out / "collection.json"
    partial = out / "tapes.jsonl.partial"
    progress = {
        "status": "running", "design_base": DESIGN_BASE, "started_utc": utc_now(),
        "expected_tapes": 384, "expected_transitions": 6144, "completed_tapes": 0,
        "native_calls_started": 0, "native_controls_returned": 0, "layouts": [],
    }
    write_json(manifest, progress)
    active = None
    try:
        progress["revision"] = subprocess.check_output(
            ["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip()
        progress["collector_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        progress["h1_collector_sha256"] = hashlib.sha256(H1.read_bytes()).hexdigest()
        if progress["h1_collector_sha256"] != "018e71cc0b5819c1e294d91e087ae0e3e64512d8e29fe4c122e09298c49fe537":
            raise ValueError("H1 query generator differs from the design base")
        mixed_probe = run_path(str(H1))["probe"]
        sys.path.insert(0, str(R2))
        from expert import session
        from generated_layouts import (
            make_sampler,
            released_fingerprints,
            reset_generated,
            sample_layout,
        )
        from large_panel_protocol import GEOMETRY_FIELDS, mobile_joints

        public = released_fingerprints()
        seen = {field: set() for field in GEOMETRY_FIELDS}
        completed = set()
        with partial.open("x", buffering=1) as stream, closing(make_sampler()) as sampler:
            require_headless(sampler)
            for index in INDICES:
                commands, permutation = probes(index, mixed_probe)
                for role, base in (("support", 991200), ("query", 991300)):
                    check_stop()
                    seed = base + index
                    block = {"index": index, "role": role, "layout_seed": seed}
                    progress["current"] = block
                    flat, origin = sample_layout(sampler, seed, public)
                    hashes = {field: origin[field] for field in GEOMETRY_FIELDS if field != "mobile_joint_xy_hash"}
                    hashes["mobile_joint_xy_hash"] = mobile_joints(origin["sampled_scene"])
                    for field, value in hashes.items():
                        if value in seen[field]:
                            raise ValueError(f"Duplicate {field} at donor seed {seed}")
                        seen[field].add(value)
                    progress["layouts"].append({**origin, **block, **hashes, "panda_flat_state": flat.tolist()})
                    for family in FAMILIES:
                        progress["current"] = dict(**block, family=family)
                        opening = None
                        with session(family, False) as env:
                            plan = arm_plan(index, family, role, commands, permutation)
                            for yaw, gain, schedule, issued in plan:
                                check_stop()
                                key = (index, family, (yaw, gain), schedule)
                                if key in completed:
                                    raise ValueError(f"Duplicate tape {key}")
                                identity = dict(**block, family=family, yaw_deg=yaw, gain=gain,
                                                schedule=schedule, tape_ordinal=len(completed) + 1)
                                progress["current"] = identity
                                public_commands = commands[schedule]
                                active = dict(**identity, **hashes, layout_id=origin["source_index"],
                                              started_utc=utc_now(), commands=public_commands.tolist(),
                                              probe_seed=None if schedule == "blocked" else seed + 2000000,
                                              translation_energy=translation_energy(public_commands),
                                              issued_translation_energy=translation_energy(issued))
                                if schedule == "shuffled":
                                    active["blocked_row_indices"] = permutation.tolist()
                                opening = collect_tape(env, reset_generated, flat, origin, active, opening,
                                                       issued, progress, check_stop)
                                stream.write(json.dumps(active, allow_nan=False, separators=(",", ":")) + "\n")
                                completed.add(key)
                                progress["completed_tapes"] = len(completed)
                                active = None
                    write_json(manifest, progress)
                    print(f"{len(completed)}/384 tapes through {role} index {index}", flush=True)
        expected = set(product(INDICES, FAMILIES, SETTINGS, SCHEDULES))
        if (completed != expected or len(completed) != 384 or progress["native_calls_started"] != 6144
                or progress["native_controls_returned"] != 6144 or any(len(values) != 32 for values in seen.values())):
            raise RuntimeError("Native evidence does not cover the full registered panel")
        check_stop()
        partial.rename(out / "tapes.jsonl")
    except (Exception, KeyboardInterrupt) as exc:
        progress.update(status="partial", stopped_utc=utc_now(), error=f"{type(exc).__name__}: {exc}")
        write_json(manifest, progress)
        if active is not None:
            write_json(out / "partial-tape.json", active)
        raise
    progress.pop("current")
    progress.update(status="complete", finished_utc=utc_now(), transitions=6144, distinct_layouts=32)
    write_json(manifest, progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="fresh directory for native evidence")
    args = parser.parse_args()
    stop_signal = None

    def request_stop(signum, _frame):
        nonlocal stop_signal
        stop_signal = signal.Signals(signum).name

    def check_stop():
        if stop_signal:
            raise InterruptedError(f"Received {stop_signal}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    collect(args.out, check_stop)


if __name__ == "__main__":
    main()
