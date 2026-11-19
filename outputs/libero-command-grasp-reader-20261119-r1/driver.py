"""Collect fitting trials, freeze the readers, then evaluate on fresh pose pairs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

import live
import readers

CELLS = ((1, "BASE"), (-1, "BASE"), (1, "D17"), (-1, "D17"))
HERE = Path(__file__).resolve().parent


def save(path, payload):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def checkpoint_hashes(folder):
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob("*")) if path.is_file()}


def collect_episode(runtime, path, pose, wire, condition, decode, grip, decide=None):
    path.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".collect-") as temp:
        working = Path(temp) / path.name
        row = runtime.episode(working, pose, wire, condition, decode, grip, decide=decide)
        saved = json.loads((working / "result.json").read_text())
        if row != saved or not (working / "tape.npz").is_file():
            raise ValueError(f"Incomplete episode at {path}")
        with np.load(working / "tape.npz", allow_pickle=False) as tape:
            if row["success"] != bool(np.any(tape["rewards"] > 0)) or len(tape["rewards"]) != row["controls"]:
                raise ValueError(f"Native return/control mismatch at {path}")
        working.rename(path)
    print(path.parent.name, path.name, "pose", pose, "native", row["success"], "controls", row["controls"], flush=True)
    return row


def public_sample(folder, wire, condition):
    result = json.loads((folder / "result.json").read_text())
    with np.load(folder / "tape.npz", allow_pickle=False) as tape:
        return dict(commands=tape["commands"].copy(), states=tape["states"].copy(),
                    wire=wire, condition=condition, choice_after=result["choice_after"])


def stage(runtime, out, name, model=None):
    fitting = name == "fit"
    poses = range(0, 8) if fitting else range(16, 24)
    cells = CELLS[:3] if fitting else CELLS
    samples, cases = [], []
    totals = [0, 0]
    for support_pose in poses:
        query_pose = support_pose + (8 if fitting else 20)
        for wire, condition in cells:
            folder = out / name / f"p{support_pose:02d}_w{wire:+d}_{condition}"
            support = collect_episode(runtime, folder / "support", support_pose, wire, condition, None, 0)
            if fitting:
                samples.append(public_sample(folder / "support", wire, condition))
            query_ref = opening_ref = None
            opening_differences = []
            def query(arm, decode, grip, decide=None):
                nonlocal query_ref, opening_ref
                row = collect_episode(runtime, folder / arm, query_pose, wire, condition, decode, grip, decide)
                identity = [row[k] for k in ("initial_physics", "initial_public")]
                opening = [row[k] for k in ("first_motor_input", "first_proposal")]
                if opening_ref is not None and opening != opening_ref:
                    opening_differences.append(arm)
                if opening_ref is None:
                    opening_ref = opening
                if row["initial_physics"] == support["initial_physics"] or (query_ref is not None and identity != query_ref):
                    raise ValueError(f"Unpaired query or nonfresh reset at {folder / arm}")
                query_ref = identity
                return row
            for decode in ((wire,) if fitting else (1, -1)):
                for grip in (0, 1):
                    row = query(f"d{decode:+d}_g{grip}", decode, grip)
                    if fitting:
                        totals[grip] += int(row["success"])
            if not fitting:
                def current(commands, states):
                    return int(model.predict(commands, states, "current", 41)[1])
                query("current", None, 0, current)
            cases.append(dict(support_pose=support_pose, query_pose=query_pose, wire=wire, condition=condition,
                              opening_differences=opening_differences))
            save(out / (name + "-progress.json"), cases)
            print(f"{name} complete {len(cases)}/{len(poses) * len(cells)} cases", flush=True)
    return samples, totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(exist_ok=False, parents=True)
    source = subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"], text=True).strip()
    save(out / "design.json", dict(source=source, seeds=[41, 42, 43], fit_cells=CELLS[:3], held_out=CELLS[3],
                                   fitting_poses=list(range(8)), eval_poses=list(range(16, 24)), start=time.time()))
    with live.Runtime() as runtime:
        samples, totals = stage(runtime, out, "fit")
        weights = out / "weights"
        readers.fit(samples, weights)
        hashes = checkpoint_hashes(weights)
        if not hashes:
            raise ValueError("No frozen reader artifacts")
        fixed_grip = int(totals[1] > totals[0])
        save(out / "frozen.json", dict(fixed_grip=fixed_grip, fitting_successes_by_grip=totals,
                                      fitting_cases=len(samples), weights=hashes, time=time.time()))
        print("Fit frozen before evaluation", totals, flush=True)
        stage(runtime, out, "eval", readers.load(weights))
        if checkpoint_hashes(weights) != hashes:
            raise ValueError("Reader weights changed during evaluation")
    save(out / "complete.json", dict(source=source, fit_cases=24, eval_cases=32,
                                    fit_episodes=72, eval_episodes=192, end=time.time()))


if __name__ == "__main__":
    main()
