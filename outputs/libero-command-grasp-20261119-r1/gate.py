"""Collect fresh paired frozen-OpenPI motor queries on native Goal 2."""

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
from itertools import product
import json
from multiprocessing import Pipe
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
CADENCE = 5
HORIZON = 300
CLOSE_CONTROLS = 12
PREGRASP_LOOKBACK = 8
PHASE_CONTROLS = 30
LIFT_M = 0.10
OPENPI_PIN = "215abfb217dbac7d5f1273282331b9b1866c0479"
PRO2_PIN = "e5500c690c1498eca1b7050350654964190cdf70"


def digest(array):
    a = np.asarray(array)
    return hashlib.sha256(repr((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def revision(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()


@contextmanager
def simulator(pilot):
    channel, peer = Pipe()
    env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
               PYTHONPATH=f"{pilot.PRO}:{pilot.SETUP}/simulator-source",
               LIBERO_CONFIG_PATH=str(pilot.SETUP / "libero-config"),
               LIBERO_PRO_ROOT=str(pilot.SETUP / "simulator-source"),
               MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID="0")
    try:
        process = subprocess.Popen(
            [pilot.SETUP / "env/bin/python", "-Bu", HERE / "grasp_simulator.py", str(peer.fileno()), pilot.PRO],
            env=env, pass_fds=(peer.fileno(),),
        )
    finally:
        peer.close()
    try:
        yield channel
    finally:
        channel.close()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=20)
        if process.returncode and sys.exc_info()[0] is None:
            raise RuntimeError(f"Simulator exited with code {process.returncode}")


def public_pose(raw):
    return np.concatenate([raw[key] for key in (
        "robot0_eef_pos", "robot0_eef_quat_xyzw", "robot0_gripper_qpos",
    )])


def images(item):
    return np.stack((item["observation/image"], item["observation/wrist_image"]))


def open_toward(current, target):
    rotation = Rotation.from_quat(target[3:7]) * Rotation.from_quat(current[3:7]).inv()
    delta = np.r_[(target[:3] - current[:3]) / 0.05, rotation.as_rotvec() / 0.5]
    return np.r_[0.65 * np.clip(delta, -1, 1), -1].astype(np.float32)


def check_common(expected, actual, location):
    for name, value in actual.items():
        if name in expected and expected[name] != value:
            raise ValueError(f"Unpaired {name} at {location}: {value!r} != {expected[name]!r}")
        expected[name] = value


def rollout(pilot, motor, pipe, key, folder, pairing):
    pose, sign, condition, decode, regrasp = key
    seed = 19193000 + pose
    reset = pilot.call(pipe, "reset", (pose, seed, condition, sign))
    first = reset["initial"]
    raw = first["observation"]
    prompt = reset["prompt"]
    schema = first["action_schema"]
    low, high = (np.asarray(schema[k], dtype=np.float32) for k in ("low", "high"))
    if (reset["horizon"] != HORIZON or low.shape != (7,) or high.shape != (7,)
            or not np.array_equal(low, -np.ones(7)) or not np.array_equal(high, np.ones(7))
            or schema["gripper_semantics"] != "negative_open_positive_close_zero_hold"
            or schema["translation_frame"] != "world" or schema["rotation_frame"] != "world"
            or schema["control_mode"] != "relative"):
        raise ValueError("Goal 2 needs the released 300-control relative OSC action space")
    # Index noise by physical time, including skipped motor slots during the maneuver.
    noise_bank = np.random.default_rng(seed).standard_normal((HORIZON // CADENCE, 10, 32)).astype(np.float32)
    noise_bank.setflags(write=False)
    pose_key = pilot.jax.random.PRNGKey(seed)
    physics = first["environment_reproducibility"]["initial_state_fingerprint"]
    if not physics:
        raise ValueError(f"Missing native reset fingerprint at {key}")
    identity = dict(seed=seed, prompt=prompt, initial_physics=physics,
                    initial_public={k: digest(v) for k, v in raw.items()}, noise_bank=digest(noise_bank),
                    horizon=HORIZON, action_schema=schema)
    folder.mkdir()
    np.savez_compressed(folder / "reset.npz", **raw)
    check_common(pairing, identity, key)
    states = [public_pose(raw)]
    queries, controls, telemetry = defaultdict(list), defaultdict(list), defaultdict(list)
    consecutive = 0
    close12 = start = pregrasp = None
    retreat_target = None
    for t in range(HORIZON):
        if t % CADENCE == 0:
            if regrasp and close12 is not None and start is None:
                start = t
                # state[k] is observed after k controls. First close was issued from
                # state[close12 - CLOSE_CONTROLS]. Both target poses are already public.
                pregrasp = max(0, close12 - CLOSE_CONTROLS - PREGRASP_LOOKBACK)
                retreat_target = states[t].copy()
                retreat_target[2] += LIFT_M
            recovering = start is not None and t < start + 2 * PHASE_CONTROLS
            if not recovering:
                slot = t // CADENCE
                item = pilot.observation(raw, prompt)
                noise = noise_bank[slot]
                inference_key = pilot.jax.random.fold_in(pose_key, slot)
                # The released JAX Policy splits its private key in infer().
                # Reset it by physical slot as well as supplying the paired flow noise.
                motor._rng = inference_key
                proposal = np.asarray(motor.infer(item, noise=noise)["actions"])
                if proposal.shape != (10, 7) or not np.isfinite(proposal).all():
                    raise ValueError(f"Invalid frozen motor proposal at {key}, control {t}")
                queries["control"].append(t)
                queries["rgb"].append(images(item))
                queries["state"].append(item["observation/state"])
                queries["noise"].append(noise)
                queries["jax_key"].append(np.asarray(inference_key))
                queries["proposal"].append(proposal.copy())
                if t == 0:
                    opening = dict(first_proposal=digest(proposal), first_motor_input={
                        name: digest(queries[name][0]) for name in ("rgb", "state", "noise", "jax_key")
                    })
                    check_common(pairing, opening, key)
                    identity.update(opening)
        if recovering:
            retreat = t < start + PHASE_CONTROLS
            selected = open_toward(states[-1], retreat_target if retreat else states[pregrasp])
            phase = 1 if retreat else 2
        else:
            selected = proposal[t % CADENCE]
            phase = 0
        bounded = np.clip(np.asarray(selected, dtype=np.float32), low, high)
        wire_action = bounded.copy()
        wire_action[0] *= decode
        step = pilot.call(pipe, "step", wire_action)
        native = step["native_input"]
        if native[0] != bounded[0] * sign * decode or not np.array_equal(native[1:], bounded[1:]):
            raise ValueError(f"Physical command transform mismatch at {key}, control {t + 1}")
        states.append(step["state"])
        controls["phase"].append(phase)
        controls["selected"].append(selected.copy())
        controls["wire_action"].append(wire_action)
        for field in ("reward", "success", "terminated", "truncated"):
            controls[field].append(step[field])
        for field in ("native_input", "applied"):
            telemetry[field].append(step[field])
        # Read only the public commanded gripper. The first threshold crossing is latched.
        consecutive = consecutive + 1 if wire_action[-1] > 0 else 0
        if close12 is None and consecutive == CLOSE_CONTROLS:
            close12 = t + 1
        raw = step["observation"]
        if step["terminated"] or step["truncated"]:
            break
    else:
        raise RuntimeError(f"Native horizon exhausted without a terminal record at {key}")

    diag = pilot.call(pipe, "diagnostics", None)
    if diag["case_id"] != condition:
        raise ValueError(f"Received diagnostics for the wrong native case at {key}")
    opportunities = [e for e in diag["events"] if e["event"] == "grasp_failure_window_started"]
    failures = [e for e in diag["events"] if e["event"] == "recoverable_grasp_failure"]
    n = len(states) - 1
    milestone = [int(e["step"]) + 1 for e in failures]
    if any(not 1 <= control <= n for control in milestone):
        raise ValueError(f"D17 event outside executed horizon at {key}")
    final = pilot.observation(raw, prompt)
    public = {**{f"query_{k}": np.asarray(v) for k, v in queries.items()},
              **{k: np.asarray(v) for k, v in controls.items()},
              "robot_state": np.stack(states), "final_rgb": images(final), "final_state": final["observation/state"]}
    native = {k: np.stack(v) for k, v in telemetry.items()}
    np.savez_compressed(folder / "public.npz", **public)
    np.savez_compressed(folder / "evaluator.npz", **native)
    write_json(folder / "diagnostics.json", diag)
    k = min(CADENCE, n)
    row = dict(pose=pose, wire=sign, condition=condition, decode=decode, regrasp=regrasp,
               net_x_sign=sign * decode, episode=folder.name, **identity,
               suite="libero_goal", task_id=2, released_prompt=first["language_instruction"],
               manifest_sha256=reset["manifest"], bddl_sha256=reset["bddl"], renderer=reset["renderer"],
               perturbation_plan=first["environment_reproducibility"]["perturbation_plan"],
               controls=n, queries=len(queries["control"]), native_success=bool(public["success"].any()),
               native_return=float(public["reward"].sum()), terminated=bool(step["terminated"]),
               truncated=bool(step["truncated"]), close12_control=close12, regrasp_start_after=start,
               pregrasp_state_index=pregrasp, regrasp_controls=int(np.count_nonzero(public["phase"])),
               opening_native_x=native["native_input"][:k, 0].tolist(),
               opening_wrist_dx_m=float(public["robot_state"][k, 0] - public["robot_state"][0, 0]),
               d17_opportunity=bool(opportunities) if condition == "D17" else None,
               d17_triggered=bool(failures) if condition == "D17" else None,
               d17_event_control=milestone[0] if milestone else None,
               gripper_overrides=int(np.count_nonzero(native["applied"][:, -1] != native["native_input"][:, -1])))
    # Both signs reversed and both signs nominal should reproduce each other's physical
    # traces. Compare actual fresh observations, proposals and trajectories across each pair.
    trajectory = {k: digest(v) for k, v in {**public, **native}.items() if k != "wire_action"}
    write_json(folder / "result.json", dict(row, trajectory=trajectory))
    return row, trajectory


def run(out, smoke):
    sys.path.insert(0, str(HERE.parent / "libero-bowl-decisions-20261118-r1"))
    import pilot  # Qualified policy factory, preprocessing, paths and simulator RPC.

    devices = pilot.jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("Expected one scheduler-visible CUDA device for OpenPI")
    if revision(pilot.SRC) != OPENPI_PIN or revision(pilot.PRO) != PRO2_PIN:
        raise ValueError("Frozen OpenPI or Pro2 source differs from the qualified release")
    keys = ([(30, -1, "D17", -1, 1)] if smoke
            else list(product(range(30, 36), (1, -1), ("BASE", "D17"), (1, -1), (0, 1))))
    write_json(out / "design.json", dict(
        mode=out.name, cases=keys, fields=["pose", "wire", "condition", "decode", "regrasp"],
        code_revision=revision(CODE), source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), HERE / "grasp_simulator.py", Path(pilot.__file__))},
        openpi=OPENPI_PIN, pro2=PRO2_PIN, checkpoint=str(pilot.CKPT),
        checkpoint_ready=json.loads((pilot.CKPT / "READY.json").read_text())["checkpoint"],
        norm_sha256=hashlib.sha256((pilot.CKPT / "assets/physical-intelligence/libero/norm_stats.json").read_bytes()).hexdigest(),
        policy="pi05_libero", device=str(devices[0]), cadence=CADENCE, horizon=HORIZON,
        close_controls=CLOSE_CONTROLS, pregrasp_lookback=PREGRASP_LOOKBACK, retreat_m=LIFT_M,
        phase_controls=PHASE_CONTROLS, seed_base=19193000,
    ))
    motor = pilot.policy_config.create_trained_policy(pilot.config.get_config("pi05_libero"), str(pilot.CKPT))
    completed = pairs = 0
    pair_differences = []
    identities, trajectories = defaultdict(dict), {}
    with simulator(pilot) as pipe, (out / "episodes.jsonl").open("x", buffering=1) as log:
        for key in keys:
            pose, sign, condition, decode, regrasp = key
            name = f"p{pose}-{condition}-wire{sign:+d}-decode{decode:+d}-grip{regrasp}"
            row, trace = rollout(pilot, motor, pipe, key, out / name, identities[pose])
            pair = (pose, condition, sign * decode, regrasp)
            if pair in trajectories:
                prior = trajectories.pop(pair)
                changed = [field for field in trace if trace[field] != prior[field]]
                if changed:
                    pair_differences.append(dict(pose=pose, condition=condition, net_x=sign * decode,
                                                 regrasp=regrasp, fields=changed))
                pairs += 1
            else:
                trajectories[pair] = trace
            print(json.dumps(row, allow_nan=False), file=log, flush=True)
            completed += 1
            print(f"{completed}/{len(keys)} {name} success={row['native_success']} "
                  f"controls={row['controls']} D17={row['d17_triggered']}", flush=True)
    if not smoke and (completed != 96 or pairs != 48 or trajectories):
        raise ValueError("Incomplete paired pilot")
    write_json(out / "complete.json", dict(mode=out.name, episodes=completed, equivalent_pairs_checked=pairs,
               distinct_physics=len({v["initial_physics"] for v in identities.values()}),
               trajectory_differences=pair_differences))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Parent of the fresh pilot/ or smoke/ directory")
    parser.add_argument("--smoke", action="store_true", help="One independent reversed-X/D17 episode")
    args = parser.parse_args()
    out = args.output / ("smoke" if args.smoke else "pilot")
    out.mkdir(parents=True)
    try:
        run(out, args.smoke)
    except Exception as exc:
        write_json(out / "failure.json", dict(error=repr(exc), time_unix=time.time()))
        raise


if __name__ == "__main__":
    main()
