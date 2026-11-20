"""Evaluate a frozen source-dynamics joint reader on two native UR5e Goal 5 starts."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

import adapt_eval as ev


TARGET = "UR5e"


def load_inputs(args, report):
    fit_bytes = (args.fit / "report.json").read_bytes()
    fit = json.loads(fit_bytes)
    if fit["status"] != "complete" or fit["stage"] != "source_motion_joint_diagnostic":
        raise ValueError("Expected a complete source-dynamics joint fit report")
    config = fit["config"]
    sources = set(ev.SOURCES)
    if (config["task"] != ev.TASK or sorted(fit["train_bodies"]) != sorted(sources)
            or sorted(config["bodies"]) != sorted(sources)):
        raise ValueError("Expected a Goal 5 reader trained on exactly Panda and Sawyer")
    if ({row["body"] for row in fit["proposals"]} != sources or not fit["curve"]
            or any(set(row["motion_mse"]) != sources for row in fit["curve"])):
        raise ValueError("Joint proposal receipts and per-update loss records must cover both sources")
    reader_sha = ev.digest_file(args.fit / "reader-adapter.pt")
    if reader_sha != fit["checkpoint_sha256"]:
        raise ValueError("Reader checkpoint differs from the completed joint fit")
    reader = ev.load_reader(args.fit)

    from source_control import audit

    # The same trainer lock and Panda audit are used by the producer.
    with (args.motor_output / "motor.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        audited, prompt = audit(args.motor_output / "checkpoint.pt", open_loop=False)
        motor_receipt = audited["checkpoint"]
        if motor_receipt["sha256"] != fit["motor_sha256"]:
            raise ValueError("Motor checkpoint differs from the one used during joint training")
        motor = ev.load_motor(args)
    report.update(
        fit_report_sha256=hashlib.sha256(fit_bytes).hexdigest(),
        checkpoint_sha256=reader_sha,
        motor_sha256=motor_receipt["sha256"],
        train_bodies=fit["train_bodies"],
        motor={**motor_receipt, "prompt": prompt, "train_poses": audited["train_poses"],
               "holdout_poses": audited["holdout_poses"], "source_bodies": audited["source_bodies"]},
    )
    if any(model.training or any(p.requires_grad for p in model.parameters()) for model in (reader, motor)):
        raise ValueError("Reader and motor must both be frozen in evaluation mode")
    return reader, motor


@torch.inference_mode()
def evaluate(args, report):
    reader, motor = load_inputs(args, report)
    panel = ev.PANELS[TARGET]
    path = args.out / "report.json"
    report.update(panel=panel, calibration={}, pairs=[])
    ev.write_json(path, report)
    responses = {}
    for body in ("Panda", TARGET):
        pulses = [ev.practice(args.out, body, pose, calibration=True) for pose in panel["calibration"]]
        response = ev.measured_response(pulses)
        check = ev.practice(args.out, body, panel["check"])
        report["calibration"][body] = {**response.report(check), "supports": [h.path for h in pulses]}
        responses[body] = response
        ev.write_json(path, report)
    calibrated = ev.privileged(responses["Panda"], responses[TARGET])

    for support, query, donor in zip(panel["practice"], panel["query"], ev.SOURCES, strict=True):
        correct = ev.practice(args.out, TARGET, support)
        wrong = ev.practice(args.out, donor, support)
        if not np.array_equal(correct.commands, wrong.commands):
            raise ValueError(f"Wrong-arm practice commands differ on pose {support}")
        transforms = {"raw": None, "correct": ev.learned(reader, correct), "cleared": ev.learned(reader, None),
                      "wrong": ev.learned(reader, wrong), "privileged": calibrated}
        records = {}
        report["pairs"].append({"pose": query, "practice_pose": support, "correct": correct.path,
                                "donor": donor, "wrong": wrong.path, "conditions": records})
        for condition in ev.CONDITIONS:
            opening = None if condition == "raw" else records["raw"]["opening"]
            row = ev.episode(args.out, motor, TARGET, query, condition, transforms[condition], opening)
            if row["opening"]["observation"]["prompt"] != report["motor"]["prompt"]:
                raise ValueError("Native query instruction differs from the motor training instruction")
            records[condition] = row
            ev.write_json(path, report)

    paired = [p["conditions"] for p in report["pairs"]]
    report["native_successes"] = {c: sum(r[c]["success"] for r in paired) for c in ev.CONDITIONS}
    report["terminal_rewards"] = {c: [r[c]["terminal_reward"] for r in paired] for c in ev.CONDITIONS}
    report["versus_raw"] = {c: ev.paired_change(paired, c, "raw") for c in ev.CONDITIONS if c != "raw"}
    report["correct_versus"] = {c: ev.paired_change(paired, "correct", c) for c in ("cleared", "wrong")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("motor-output", "weights", "fit", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True, help="deployed evaluator revision")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    path = args.out / "report.json"
    report = {"stage": "motion_dev", "status": "running", "body": TARGET, "task_id": ev.TASK,
              "practice_transitions": ev.K, "execute_prefix": ev.EXECUTE, "argv": sys.argv,
              "revision": args.revision, "torch": str(torch.__version__), "numpy": np.__version__}
    ev.write_json(path, report)
    try:
        evaluate(args, report)
    except Exception as exc:
        report.update(status="error", error_type=type(exc).__name__, reason=str(exc))
        ev.write_json(path, report)
        raise
    report["status"] = "complete"
    ev.write_json(path, report)
    print(path, flush=True)


if __name__ == "__main__":
    main()
