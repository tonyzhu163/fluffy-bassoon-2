#!/usr/bin/env python
"""Persistent native8 calibration followed by the F-only PPO reward service.

Uses the pinned official engine and unchanged graded terminal reward.
Training accepts no simulator-monitor path. It never constructs a native env.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from fluffy_bassoon.meta_rl.flora_terminal_worker import TerminalWorker
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import (
    FLORA_CODE, FLORA_MODEL, baseline_validation, digest, flora_jobs, parse_outputs, sha, write_new,
)
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_monitor import ProgressLog
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_runtime import read_ref
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import protocol


def score(worker, episodes, output, model_receipt, proto):
    jobs = flora_jobs(episodes, proto)
    for episode in episodes:
        if sha(episode["video"]["path"]) != episode["video"]["sha256"]:
            raise ValueError("Video bytes changed")
    with output.with_suffix(".jobs.jsonl").open("x") as stream:
        for row in jobs:
            stream.write(json.dumps(row) + "\n")
    rows = worker.score_jobs(jobs, output)
    labels = parse_outputs(episodes, rows, proto)
    receipt = dict(code_commit=FLORA_CODE, model_revision=FLORA_MODEL, model_receipt=model_receipt,
        completed=True, jobs_sha256=digest(jobs), video_refs_sha256=digest([e["video"] for e in episodes]),
        output=dict(path=str(output), sha256=sha(output)), training_reward_rule="(expected_score-1)/4",
        engine_process_pid=os.getpid(), model_load_count=worker.model_load_count,
        completed_batches=worker.completed_batches, engine="official persistent terminal_reward helpers")
    write_new(output.with_suffix(".receipt.json"), receipt)
    return rows, labels, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--episodes")
    parser.add_argument("--baseline-monitor")
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("Scoring only in an allocated Linux GPU job")
    profile = json.loads(Path(args.profile).read_text())
    if profile["mode"] not in ("baseline", "train", "integrated") or not 0 < profile["scheduler_seconds"] < 8 * 3600:
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
    progress = ProgressLog(output / "progress", "F", contract_sha256=digest(profile)).start()
    try:
        progress.emit(phase="official_persistent_engine_loading", model_load_count=0)
        worker = TerminalWorker(model)
        progress.emit(phase="official_persistent_engine_ready", model_load_count=1,
                      engine_process_pid=os.getpid())
        serve(profile, args, worker, output, proto, progress)
    finally:
        progress.close()


def serve(profile, args, worker, output, proto, progress):
    if profile["mode"] in ("baseline", "integrated"):
        if not args.episodes or not args.baseline_monitor:
            raise ValueError("Freeze the native8 bank before paired calibration")
        episodes = json.loads(Path(args.episodes).read_text())
        if len(episodes) != 8 or sha(args.episodes) != profile["frozen_native_episodes_sha256"]:
            raise ValueError("Exactly the first native-passing8 bank, no task selection by Flora")
        progress.emit(phase="native8_Flora_calibration", episodes_complete=8)
        rows, _, receipt = score(worker, episodes, output / "terminal.jsonl", profile["model_receipt"], proto)
        truth = {r["id"]: r["simulator_final_success"] for path in sorted(Path(args.baseline_monitor).glob("*.json"))
                 for r in [json.loads(path.read_text())]}
        if set(truth) != {episode["id"] for episode in episodes}:
            raise ValueError("Calibration video/truth identity mismatch")
        result = baseline_validation([dict(id=row["id"], expected_score=row["expected_score"],
            simulator_final_success=truth[row["id"]]) for row in rows], proto)
        write_new(output / "baseline-validation.json", dict(result, completed=True, scorer_receipt=receipt))
        progress.emit(phase="calibration_passed" if result["go"] else "calibration_failed_comparison_stopped",
                      labels_complete=8, AUROC=result["AUROC"])
        if profile["mode"] == "baseline" or not result["go"]:
            return
        # Calibration truth is used only above. Neither the training requests
        # nor the F policy receive it; the loaded engine is the same object.
        del truth, episodes, rows, result
    elif args.baseline_monitor or args.episodes:
        raise ValueError("F training scorer cannot receive simulator truth or another case bank")
    gate = (json.loads((output / "baseline-validation.json").read_text())
            if profile["mode"] == "integrated" else read_ref(profile["baseline_validation"]))
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
            progress.emit(phase="official_Flora_terminal_inference", iteration=len(seen) + 1,
                          episodes_complete=len(request["episodes"]))
            rows, _, _ = score(worker, request["episodes"], target, profile["model_receipt"], proto)
            array = target.with_suffix(".json")
            write_new(array, rows)
            write_new(path.parent / "ready.json", dict(pending_checkpoint_sha256=pending["sha256"],
                jobs_sha256=digest(jobs), code_commit=FLORA_CODE, model_revision=FLORA_MODEL,
                outputs=dict(path=str(array), sha256=sha(array))))
            seen.add(str(path))
            progress.emit(phase="complete_batch_labels_ready", labels_complete=len(rows),
                          model_load_count=worker.model_load_count, completed_batches=worker.completed_batches)
        if (exchange / "FINAL.json").exists():
            break
        if (output / "shutdown.json").exists():
            break
        time.sleep(1)


if __name__ == "__main__":
    main()
