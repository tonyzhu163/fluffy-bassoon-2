#!/usr/bin/env python3
"""Supplementary CPU unit for exact identity at final source-action issuance.

    python identity_readback.py --scored-directory /path/to/stride-5 --out identity.json

The large source report supplies the expected hash for each native tape. Rebuild
each issued prefix from its motor chunk, stopping at the recorded terminal tick.
A closed hook must bypass a residual that would raise if evaluated. An open
hook receives an artificial inward X correction in metres, capped at 1 mm of
commanded translation per control (OSC scale 0.05 m per normalized unit).
This single-axis probe exercises intervention even on saturated commands.
Native successes belong to the archived raw trajectories. This unit measures
neither the outcome of applying the correction nor learned abstention.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np


METRES_PER_UNIT = 0.05
LIMIT_M = 0.001
BODIES = ("Panda", "Sawyer")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selective_action(raw, *, enabled, residual):
    if not enabled:
        return raw.copy()
    # Immutable backing also prevents the callback from re-enabling writes.
    readonly = np.frombuffer(raw.tobytes(), dtype=raw.dtype)
    dx_m = float(residual(readonly))
    require(np.isfinite(dx_m), "Nonfinite X correction")
    output = raw.copy()
    dx = np.clip(dx_m, -LIMIT_M, LIMIT_M) / METRES_PER_UNIT
    output[0] = np.clip(float(raw[0]) + dx, -1.0, 1.0)
    return output


def forbidden_residual(_):
    raise AssertionError("Hard-off hook evaluated the residual")


def check_episode(path, row, body, stride, horizon):
    def check(ok, message):
        require(ok, f"{path}: {message}")

    payload = path.read_bytes()
    check(hashlib.sha256(payload).hexdigest() == row["tape_sha256"], "SHA-256 mismatch")
    with np.load(io.BytesIO(payload), allow_pickle=False) as tape:
        header = json.loads(tape["header"].item())
        chunks, actions = tape["chunks"], tape["actions"]
        rewards, done, success = tape["rewards"], tape["done"], tape["success"]
        native_steps = tape["native_steps"]
    check(header["arm"] == row["arm"] == body and header["layout_seed"] == row["layout_seed"]
          and header["layout_id"] == row["layout_id"], "wrong episode identity")
    n = row["steps"]
    check(type(n) is int and 0 < n <= horizon, "invalid control count")
    check(chunks.shape == ((n + stride - 1) // stride, 5, 7) and actions.shape == (n, 7)
          and chunks.dtype == actions.dtype == np.float32, "invalid action shape or dtype")
    check(np.isfinite(chunks).all() and (np.abs(chunks) <= 1).all(), "invalid motor chunk")
    check(native_steps.dtype.kind in "iu" and np.array_equal(native_steps, np.arange(1, n + 1)),
          "missing or duplicated native control")
    check(rewards.shape == done.shape == success.shape == (n,) and done.dtype == success.dtype == np.bool_
          and not done[:-1].any() and done[-1] and not rewards[:-1].any() and rewards[-1] in (0, 1)
          and np.array_equal(success, rewards == 1) and (n == horizon or success[-1]),
          "invalid native reward or termination")
    won = bool(success[-1])
    check(row["success"] is won and row["native_return"] == float(rewards.sum()), "native outcome mismatch")

    calls = 0

    def mechanical_offer(action):
        nonlocal calls
        calls += 1
        return -2 * LIMIT_M if action[0] >= 0 else 2 * LIMIT_M

    changed = 0
    max_dx_m = 0.0
    for i, issued in enumerate(actions):
        raw = chunks[i // stride, i % stride]
        original = raw.tobytes()
        off = selective_action(raw, enabled=False, residual=forbidden_residual)
        check(off.dtype == issued.dtype and off.shape == issued.shape
              and original == off.tobytes() == issued.tobytes(), f"hard-off mismatch at control {i + 1}")
        on = selective_action(raw, enabled=True, residual=mechanical_offer)
        check(raw.tobytes() == original, f"mutated raw input at control {i + 1}")
        dx_m = (float(on[0]) - float(raw[0])) * METRES_PER_UNIT
        expected_dx_m = -LIMIT_M if raw[0] >= 0 else LIMIT_M
        check(on.dtype == raw.dtype and on.shape == raw.shape and on[1:].tobytes() == raw[1:].tobytes()
              and abs(on[0]) <= 1 and abs(dx_m - expected_dx_m) <= METRES_PER_UNIT * np.finfo(np.float32).eps,
              f"incorrect bounded mechanical correction at control {i + 1}")
        changed += int(dx_m != 0)
        max_dx_m = max(max_dx_m, abs(dx_m))
    check(calls == changed == n, "open hook did not apply one nonzero offer per issued action")
    return dict(layout_seed=row["layout_seed"], tape_sha256=row["tape_sha256"], controls=n,
                native_success=won, hard_off_byte_matches=n, mechanical_on_calls=calls,
                mechanical_on_changed=changed, max_commanded_x_change_m=max_dx_m)


def readback(directory):
    source = (directory / "report.json").read_bytes()
    report = json.loads(source)
    stride, horizon = report["stride"], report["horizon"]
    require(report["status"] == "complete" and set(report["arms"]) == set(BODIES), "Incomplete source report")
    require(type(stride) is int and 1 <= stride <= 5 and horizon == 300, "Invalid native schedule")
    bodies = {}
    panel_seeds = None
    for body in BODIES:
        arm = report["arms"][body]
        rows = arm["episodes"]
        seeds = {row["layout_seed"] for row in rows}
        require(rows and len(rows) == len(seeds) == arm["total"], f"{body}: incomplete or duplicate episodes")
        require(panel_seeds is None or seeds == panel_seeds, "Source arms cover different layout seeds")
        panel_seeds = seeds
        paths = {f"{body}/{seed}.npz" for seed in seeds}
        require(paths == {p.relative_to(directory).as_posix() for p in (directory / body).glob("*.npz")},
                f"{body}: tape inventory differs from source report")
        episodes = []
        for row in rows:
            tape = f"{body}/{row['layout_seed']}.npz"
            require(row["tape"] == tape, f"Wrong source path for {tape}")
            episodes.append(check_episode(directory / tape, row, body, stride, horizon))
        wins = sum(e["native_success"] for e in episodes)
        require(wins == arm["successes"], f"{body}: native success total mismatch")
        totals = {key: sum(e[key] for e in episodes)
                  for key in ("controls", "hard_off_byte_matches", "mechanical_on_calls", "mechanical_on_changed")}
        bodies[body] = dict(episodes=len(rows), native_successes=wins, timeouts=len(rows) - wins,
                            early_successes=sum(e["native_success"] and e["controls"] < horizon for e in episodes),
                            hard_off_residual_calls=0, **totals, episode_records=episodes)
    return dict(source_report_sha256=hashlib.sha256(source).hexdigest(), stride=stride, horizon=horizon,
                checkpoint_sha256=report["checkpoint_sha256"], bodies=bodies,
                mechanical_control="Forced open with synthetic inward X offer of 2 mm, capped at 1 mm.",
                interpretation="Archived raw outcomes only. Learned abstention and corrected outcomes unmeasured.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-directory", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="New JSON result file")
    args = parser.parse_args()
    result = readback(args.scored_directory)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
