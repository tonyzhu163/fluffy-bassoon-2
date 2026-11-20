"""Bind pre-action native plate labels and future wrist waypoints to successful source camera tapes."""

import argparse
import errno
import hashlib
import json
import multiprocessing
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from expert import session, sha, write_json
from recover import BODIES, PROMPT, RELEASE
from scipy.spatial.transform import Rotation

ROUTES = {"rendered_commands", "whole_gripper_push"}
LOOKAHEAD = 6
# Distances in metres and rotation-vector magnitudes in radians. Trace XY
# was rounded to four decimals by Paddle; its quantization floor is 0.071 mm.
TOL = {
    "position_m": 0.0001,
    "rotation_rad": 0.001,
    "jaw_m": 0.0001,
    "motion_xyz_m": 0.0001,
    "motion_rotation_rad": 0.001,
    "motion_jaw_m": 0.0001,
    "trace_plate_xy_m": 0.0002,
    "trace_tcp_m": 0.0002,
}


RESET_REJECTIONS = (
    "Released grip-site pose unreachable",
    "Settling moved",
    "Task already successful",
)


class Diverged(ValueError):
    pass


def check_error(audit, key, value, tick):
    value = float(value)
    if not np.isfinite(value) or value > TOL[key]:
        raise Diverged(f"t={tick} {key}={value:.9g} exceeds {TOL[key]}")
    audit[key] = max(audit.get(key, 0.0), value)


def compare_state(audit, live, expected, t):
    check_error(audit, "position_m", np.linalg.norm(live[:3] - expected[:3]), t)
    angle = (
        Rotation.from_quat(live[3:7]) * Rotation.from_quat(expected[3:7]).inv()
    ).magnitude()
    check_error(audit, "rotation_rad", angle, t)
    check_error(audit, "jaw_m", abs(live[7] - expected[7]), t)


def reset_identity(row, reset):
    body, pose, seed = (row[k] for k in ("body", "pose", "seed"))
    p = reset["environment_reproducibility"]
    expected = dict(
        **RELEASE, suite="libero_goal", task_id=5, init_state_id=pose, seed=seed
    )
    if (
        body not in BODIES
        or type(pose) is not int
        or pose not in range(32)
        or type(seed) is not int
        or seed not in range(2**32)
        or reset["model_identity"]["arm"] != body
        or any(p.get(k) != v for k, v in expected.items())
        or (p["perturbation_plan"]["case_id"], p["perturbation_plan"]["seed"])
        != ("BASE", seed)
        or not all(
            p.get(k) for k in ("released_row_fingerprint", "initial_state_fingerprint")
        )
    ):
        raise ValueError("Source receipt or released reset identity differs")
    return p


def renderer(reset):
    r = reset["renderer"]
    if len(r) != 2 or "NVIDIA" not in r[0].upper() or not r[1]:
        raise ValueError("Expected an allocated NVIDIA camera trajectory")


def numeric(a, shape):
    if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
        raise ValueError(
            f"Invalid float32 tape array, expected {shape}, found {a.shape} {a.dtype}"
        )
    return a


def read_tape(path, row):
    n = row["steps"]
    if not (
        row["done"] is True
        and row["native_success"] is True
        and row["accepted"] is True
        and 0 < n <= row["reset"]["max_steps"]
        and row["terminal_reward"] > 0
    ):
        raise ValueError("Invalid successful native receipt")
    reset_identity(row, row["reset"])
    renderer(row["reset"])
    with np.load(path, allow_pickle=False) as tape:
        if (
            any(tape[k].item() != row[k] for k in ("body", "pose", "seed"))
            or tape["native_success"].item() is not True
            or tape["prompt"].item() != PROMPT
        ):
            raise ValueError("Camera tape and successful receipt differ")
        states = numeric(tape["state"], (n, 8))
        terminal = numeric(tape["terminal_state"], (8,))
        actions = numeric(tape["actions"], (n, 7))
        motions = numeric(tape["motion"], (n, 7))
        rewards = numeric(tape["rewards"], (n,))
        states = np.vstack((states, terminal))
        if (
            np.max(abs(actions)) > 1
            or np.any(rewards[:-1])
            or rewards[-1] != row["terminal_reward"]
            or not np.allclose(np.linalg.norm(states[:, 3:7], axis=1), 1, atol=1e-5)
        ):
            raise ValueError("Invalid issued commands, quaternions or terminal reward")
    return states, actions, motions, rewards


def replay(row, data, audit):
    states, actions, motions, rewards = data
    samples = {s["t"]: s for s in row["trace"]}
    samples[len(actions) - 1] = row["last"]
    plate = []
    with session(row["body"], False) as env:
        try:
            obs = env.reset(row["pose"], row["seed"])
        except ValueError as exc:
            if not str(exc).startswith(RESET_REJECTIONS):
                raise
            raise Diverged(f"native reset rejected: {exc}") from exc
        live = reset_identity(row, env.reset_info)
        source = row["reset"]["environment_reproducibility"]
        for key in ("released_row_fingerprint", "initial_state_fingerprint"):
            if live[key] != source[key]:
                raise Diverged(f"reset {key} mismatch")
        for t, action in enumerate(actions):
            if obs["prompt"] != PROMPT:
                raise ValueError("Unexpected task instruction")
            compare_state(audit, obs["state"], states[t], t)
            scene = env.expert_scene()
            plate.append(scene["plate"])
            if t in samples:
                s = samples[t]
                check_error(
                    audit,
                    "trace_plate_xy_m",
                    np.linalg.norm(np.subtract(scene["plate"][:2], s["plate"])),
                    t,
                )
                check_error(
                    audit,
                    "trace_tcp_m",
                    np.linalg.norm(np.subtract(scene["tcp"], s["tcp"])),
                    t,
                )
            packet = env.arm.step(action)
            if (
                packet["worker_step_index"] != t + 1
                or packet["issued_action"].dtype != np.float32
                or not np.array_equal(packet["issued_action"], action)
            ):
                raise Diverged(f"t={t} native issued control/index mismatch")
            diff = packet["sensor_motion"] - motions[t]
            check_error(audit, "motion_xyz_m", np.linalg.norm(diff[:3]), t)
            check_error(audit, "motion_rotation_rad", np.linalg.norm(diff[3:6]), t)
            check_error(audit, "motion_jaw_m", abs(diff[6]), t)
            final = t == len(actions) - 1
            if (
                bool(packet["success"]) != final
                or bool(packet["terminated"] or packet["truncated"]) != final
                or float(packet["reward"]) != rewards[t]
            ):
                raise Diverged(f"t={t} native reward or endpoint mismatch")
            obs = env.observation(packet)
            audit["validated_controls"] = t + 1
        compare_state(audit, obs["state"], states[-1], len(actions))
    return plate


def camera_capture(out, original, actions, mode):
    """Instrument the original Paddle camera collector. Frames and labels share one actual transition."""
    import paddle

    plates, pre, actual = [], [], []

    @contextmanager
    def observed_session(body, camera):
        with session(body, camera) as env:
            arm = env.native.arm
            step = arm.step

            def observed_step(command):
                scene = env.expert_scene()
                plates.append(scene["plate"])
                packet = step(command)
                if packet["worker_step_index"] != len(plates) or not np.array_equal(
                    command, packet["issued_action"]
                ):
                    raise ValueError(
                        "Native camera collector issued action/index mismatch"
                    )
                actual.append(packet["issued_action"].copy())
                return packet

            public_step = env.step
            public_reset = env.reset

            def reset(*args):
                try:
                    obs = public_reset(*args)
                except ValueError as exc:
                    if not str(exc).startswith(RESET_REJECTIONS):
                        raise
                    raise Diverged(f"direct camera reset rejected: {exc}") from exc
                pre.append(obs["state"].copy())
                return obs

            def advance(command):
                result = public_step(command)
                pre.append(result[0]["state"].copy())
                return result

            with (
                patch.object(arm, "step", observed_step),
                patch.object(env, "reset", reset),
                patch.object(env, "step", advance),
            ):
                yield env

    body, pose, seed = (original[k] for k in ("body", "pose", "seed"))
    (out / body).mkdir()
    (out / "records").mkdir()
    with patch.object(paddle, "session", observed_session):
        row = paddle.trial(
            (out, pose, seed, 0, paddle.Route(**original["params"]), True),
            body=body,
            replay=actions if mode == "commands" else None,
        )
    if not row["accepted"]:
        raise Diverged(
            f"direct camera {mode}: {row['steps']} steps, terminal reward {row['terminal_reward']}"
        )
    tape = out / row["file"]
    data = read_tape(tape, row)
    if not np.array_equal(data[0], pre) or not np.array_equal(data[1], actual):
        raise ValueError("Camera capture differs from its own native trajectory")
    if (
        "reset" in original
        and reset_identity(row, row["reset"])["released_row_fingerprint"]
        != reset_identity(original, original["reset"])["released_row_fingerprint"]
    ):
        raise ValueError("Direct capture changed released layout")
    return row, tape, data, plates


def publish(out, row, path, states, plate, key, provenance):
    n = len(states) - 1
    plate = numeric(np.asarray(plate, np.float32), (n, 3))
    future = np.minimum(np.arange(n) + LOOKAHEAD, n)
    waypoint = states[future, :3]
    target = out / row["body"] / (key + ".npz")
    try:
        os.link(path, target)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        with path.open("rb") as source, target.open("xb") as dest:
            shutil.copyfileobj(source, dest, length=1024 * 1024)
    aux = out / "aux" / (key + ".npz")
    with aux.open("xb") as file:
        np.savez_compressed(
            file,
            plate_world_m=plate,
            waypoint_world_m=waypoint,
            waypoint_minus_plate_m=waypoint - plate,
            motion_m=waypoint - states[:-1, :3],
            future_state_index=future,
            pre_state=states[:-1],
        )
    digest = sha(target)
    if (
        provenance["mode"] == "camera_free_replay"
        and digest != provenance["original"]["sha256"]
    ):
        raise ValueError("Original camera archive changed during publication")
    return dict(
        row,
        file=str(target.relative_to(out)),
        sha256=digest,
        aux_file=str(aux.relative_to(out)),
        aux_sha256=sha(aux),
        waypoint_provenance=provenance,
    )


def process(job):
    out, path, row, key, fallback, expected_sha = job
    original = {
        "body": row["body"],
        "pose": row["pose"],
        "seed": row["seed"],
        "file": str(path),
    }
    info = {
        "original": original,
        "mode": "camera_free_replay",
        "replay_max": {},
        "skips": [],
    }
    digest = sha(path)
    original["sha256"] = digest
    if expected_sha != digest or row.get("sha256", digest) != digest:
        raise ValueError(f"Camera source digest changed: {path}")
    data = read_tape(path, row)
    try:
        plate = replay(row, data, info["replay_max"])
    except Diverged as exc:
        info["skips"].append(str(exc))
    else:
        return publish(out, row, path, data[0], plate, key, info)
    for mode in fallback:
        with TemporaryDirectory(dir=out) as directory:
            try:
                new, tape, captured, plate = camera_capture(
                    Path(directory), row, data[1], mode
                )
            except Diverged as exc:
                info["skips"].append(str(exc))
                continue
            info["mode"] = "direct_camera_" + mode
            return publish(out, new, tape, captured[0], plate, key, info)
    return {
        "body": row["body"],
        "pose": row["pose"],
        "seed": row["seed"],
        "accepted": False,
        "waypoint_provenance": info,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--banks", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("bank"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument(
        "--fixed-offline",
        action="store_true",
        help="omit offline rows from sweep-search indices",
    )
    p.add_argument(
        "--camera-fallback",
        choices=("none", "commands", "commands-and-route"),
        default="none",
    )
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be positive")
    fallback = {
        "none": (),
        "commands": ("commands",),
        "commands-and-route": ("commands", "route"),
    }[args.camera_fallback]
    out = args.out.resolve()
    jobs, sources, ids = [], [], set()
    for bank in sorted({b.resolve(strict=True) for b in args.banks}):
        raw = (bank / "index.json").read_bytes()
        index = json.loads(raw)
        if (index["suite"], index["case"], index["task_id"]) != (
            "libero_goal",
            "BASE",
            5,
        ) or not set(index["bodies"]) <= set(BODIES):
            raise ValueError(f"Unexpected source bank: {bank}")
        excluded = Counter()
        for row in index["attempts"]:
            if (
                row["body"] not in index["bodies"]
                or type(row["pose"]) is not int
                or row["pose"] not in range(32)
            ):
                raise ValueError(f"Out-of-split source bank: {bank}")
            if not row["accepted"] or row.get("route") not in ROUTES:
                excluded[
                    "unsuccessful" if not row["accepted"] else "other_collector"
                ] += 1
                continue
            if (
                args.fixed_offline
                and row["pose"] >= 28
                and index.get("source_constructor")
            ):
                excluded["offline_search"] += 1
                continue
            f = Path(row["file"])
            path = (bank / f).resolve(strict=True)
            if f.is_absolute() or not path.is_relative_to(bank) or path in ids:
                raise ValueError(f"Invalid or duplicate camera tape {path}")
            ids.add(path)
            # Content identity orders the bank independently of paths, CLI order and async collectors.
            tape_sha = sha(path)
            key = hashlib.sha256(
                json.dumps(
                    [row["body"], row["pose"], row["seed"], tape_sha, row["params"]],
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            jobs.append((out, path, row, key, fallback, tape_sha))
        sources.append(
            {
                "path": str(bank),
                "index_sha256": hashlib.sha256(raw).hexdigest(),
                "source_constructor": index.get("source_constructor"),
                "excluded": dict(excluded),
            }
        )
    if not jobs:
        raise ValueError(
            "No successful native Paddle camera trajectories in the supplied banks"
        )
    jobs.sort(key=lambda j: (j[2]["body"], j[2]["pose"], j[2]["seed"], j[3]))
    unique = {}
    for job in jobs:
        unique.setdefault(job[3], job)
    out.mkdir(parents=True, exist_ok=False)
    for name in (*BODIES, "aux"):
        (out / name).mkdir()
    index = {
        "suite": "libero_goal",
        "case": "BASE",
        "task_id": 5,
        "bodies": list(BODIES),
        "attempts": [],
        "status": "running",
        "label_schema": 1,
        "lookahead": LOOKAHEAD,
        "tolerances": TOL,
        "sources": sources,
        "duplicate_input_tapes": len(jobs) - len(unique),
        "revision": os.environ.get("MOTOR_CODE_REVISION"),
        "label_code_sha256": sha(Path(__file__)),
        "camera_fallback": args.camera_fallback,
        "fixed_offline": args.fixed_offline,
    }
    write_json(out / "index.json", index)
    with ProcessPoolExecutor(
        args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for row in pool.map(process, unique.values()):
            index["attempts"].append(row)
            write_json(out / "index.json", index)
            print(
                row["body"],
                row["pose"],
                row["accepted"],
                row["waypoint_provenance"]["mode"],
                row["waypoint_provenance"]["skips"],
                flush=True,
            )
    index.update(
        status="complete", accepted=sum(r["accepted"] for r in index["attempts"])
    )
    write_json(out / "index.json", index)
    return 0 if index["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
