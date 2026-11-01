#!/usr/bin/env python
"""Finite native8 calibration or F-only pending-batch Flora service.

Uses the existing official CLI adapter and unchanged graded terminal reward.
Training accepts no simulator-monitor path. It never constructs a native env.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import time

from rlinf_flora_ppo_score import score
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import (
    FLORA_CODE, FLORA_MODEL, baseline_validation, digest, flora_jobs, sha, write_new,
)
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_runtime import read_ref
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--episodes")
    parser.add_argument("--baseline-monitor")
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("Scoring only in an allocated Linux GPU job")
    profile = json.loads(Path(args.profile).read_text())
    if profile["mode"] not in ("baseline", "train") or not 0 < profile["scheduler_seconds"] < 8 * 3600:
        raise ValueError("Exact finite calibration or F-only service required")
    read_ref(profile["actual_gpu_probe"])
    model_receipt = read_ref(profile["model_receipt"])
    model, code = Path(profile["model"]), Path(profile["official_code"])
    head = subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
    if (head != FLORA_CODE or model_receipt["model_revision"] != FLORA_MODEL
            or not model_receipt["completed"] or Path(model_receipt["model_directory"]).resolve() != model.resolve()):
        raise ValueError("Same qualified official Flora code, weights and local receipt required")
    proto = protocol(profile["task"])
    output = Path(profile["output"])
    output.mkdir(parents=True, exist_ok=False)
    if profile["mode"] == "baseline":
        if not args.episodes or not args.baseline_monitor:
            raise ValueError("Freeze the native8 bank before paired calibration")
        episodes = json.loads(Path(args.episodes).read_text())
        if len(episodes) != 8 or sha(args.episodes) != profile["frozen_native_episodes_sha256"]:
            raise ValueError("Exactly the first native-passing8 bank, no task selection by Flora")
        rows, _, receipt = score(episodes, output / "terminal.jsonl", model, code, profile["model_receipt"], proto)
        truth = {r["id"]: r["simulator_final_success"] for path in sorted(Path(args.baseline_monitor).glob("*.json"))
                 for r in [json.loads(path.read_text())]}
        if set(truth) != {episode["id"] for episode in episodes}:
            raise ValueError("Calibration video/truth identity mismatch")
        result = baseline_validation([dict(id=row["id"], expected_score=row["expected_score"],
            simulator_final_success=truth[row["id"]]) for row in rows], proto)
        write_new(output / "baseline-validation.json", dict(result, completed=True, scorer_receipt=receipt))
        return
    if args.baseline_monitor or args.episodes:
        raise ValueError("F training scorer cannot receive simulator truth or another case bank")
    gate = read_ref(profile["baseline_validation"])
    if gate.get("go") is not True or gate.get("protocol") != proto:
        raise ValueError("Same task native headroom/Flora agreement gate required")
    maximum = profile["maximum_batches"]
    if type(maximum) is not int or not 1 <= maximum <= 10:
        raise ValueError("At most ten admitted F batches")
    seen = set()
    exchange = Path(profile["F_output"])
    while time.time() < profile["deadline_unix"] - profile["closure_seconds"] and len(seen) < maximum:
        for path in sorted(exchange.glob("labels-*/request.json")):
            if str(path) in seen:
                continue
            if (path.parent / "ready.json").exists():
                seen.add(str(path))
                continue
            request = json.loads(path.read_text())
            pending = request["pending_checkpoint"]
            # The complete producing-policy/critic/Adam/RNG+rollout checkpoint
            # was saved before the request. No simulator labels are read here.
            if sha(pending["path"]) != pending["sha256"] or request["protocol"] != proto:
                raise ValueError("Pending batch bytes or native task changed")
            jobs = flora_jobs(request["episodes"], proto)
            if request["jobs"] != jobs or request["jobs_sha256"] != digest(jobs):
                raise ValueError("Fresh batch video identity mismatch")
            target = output / (path.parent.name + ".jsonl")
            rows, _, _ = score(request["episodes"], target, model, code, profile["model_receipt"], proto)
            array = target.with_suffix(".json")
            write_new(array, rows)
            write_new(path.parent / "ready.json", dict(pending_checkpoint_sha256=pending["sha256"],
                jobs_sha256=digest(jobs), code_commit=FLORA_CODE, model_revision=FLORA_MODEL,
                outputs=dict(path=str(array), sha256=sha(array))))
            seen.add(str(path))
        if (exchange / "FINAL.json").exists():
            break
        time.sleep(1)


if __name__ == "__main__":
    main()
