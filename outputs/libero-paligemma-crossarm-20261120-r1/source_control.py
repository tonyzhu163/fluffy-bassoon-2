"""Score the frozen Panda-only SFT feasibility control on two independent native starts."""

import argparse
import fcntl
import math
from pathlib import Path

import torch

from adapt_eval import QUALIFY, digest_file, episode, write_json
from motor import Motor, recipe


def audit(checkpoint, open_loop):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    contract = saved["contract"]
    if contract.get("source_bodies") != ["Panda"] or not contract.get("collector_index_sha256"):
        raise ValueError("Expected an indexed Panda-only motor checkpoint")
    if contract["recipe"] != recipe() or saved["step"] != 300 or contract["seed"] != 20261120:
        raise ValueError("Expected the registered motor settings and completed 300-update fit")

    splits = {"train": set(), "validation": set()}
    prompts = set()
    for row in contract["episodes"]:
        pose = int(row["pose"])
        if row["body"] != "Panda" or pose not in range(16) or row["split"] not in splits:
            raise ValueError("Unexpected demonstration in the Panda control")
        splits[row["split"]].add(pose)
        prompts.add(row["prompt"])
    train, held = splits["train"], splits["validation"]
    if (
        not train
        or not held
        or train & held
        or len(prompts) != 1
        or train != set(map(int, contract["train_poses"]))
        or held != set(map(int, contract["holdout_poses"]))
        or train | held != set(map(int, contract["all_motor_poses"]))
    ):
        raise ValueError("Inconsistent released-pose split or task instruction")
    anchors = contract["validation_anchors"]
    if any(body != "Panda" or int(pose) not in held for body, pose, _ in anchors):
        raise ValueError("Validation anchors must come from held-out Panda poses")
    best = saved["best"]
    if not best or not math.isfinite(best["mae"]):
        raise ValueError("Expected a validation-selected motor")
    selected = next(row["validation"] for row in saved["curve"] if row["step"] == best["step"])
    if selected["mae"] != best["mae"] or selected["axis_mae_by_body"].keys() != {"Panda"}:
        raise ValueError("Selected update disagrees with the Panda holdout receipt")

    report = {
        "study": "Panda-only SFT feasibility control",
        "source_bodies": ["Panda"],
        "task": "BASE libero_goal/5",
        "query_poses": list(QUALIFY),
        "train_poses": sorted(train),
        "holdout_poses": sorted(held),
        "checkpoint": {
            "sha256": digest_file(checkpoint),
            "completed_step": saved["step"],
            "selected_step": best["step"],
            "base_repo": contract["weights"]["repo"],
            "base_revision": contract["weights"]["revision"],
            "collector_index_sha256": contract["collector_index_sha256"],
        },
        "episodes": [],
        "native_successes": 0,
        "status": "running",
    }
    if open_loop:
        report["open_loop_holdout"] = {
            "axes": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
            "selected_step_axis_mae": selected["axis_mae_by_body"]["Panda"],
            "anchors": selected["anchors_by_body"]["Panda"],
        }
    return report, prompts.pop()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("motor-output", "weights", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--open-loop", action="store_true", help="include selected-step holdout axis MAE")
    parser.add_argument("--revision", help="deployed code revision from the launcher")
    args = parser.parse_args()
    # The trainer takes an exclusive lock while replacing checkpoints. Audit and load the same bytes.
    with (args.motor_output / "motor.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        report, prompt = audit(args.motor_output / "checkpoint.pt", args.open_loop)
        motor = Motor.load(args.motor_output, args.weights)
    args.out.mkdir(parents=True, exist_ok=False)
    report["revision"] = args.revision
    path = args.out / "report.json"
    write_json(path, report)
    for pose in QUALIFY:
        row = episode(args.out, motor, "Panda", pose, "raw")
        if row["opening"]["observation"]["prompt"] != prompt:
            raise ValueError("Native query instruction differs from demonstration instruction")
        report["episodes"].append(
            {
                key: row[key]
                for key in ("pose", "seed", "success", "steps", "native_return", "terminal_reward", "tape")
            }
        )
        report["native_successes"] += int(row["success"])
        if len(report["episodes"]) == len(QUALIFY):
            report["status"] = "complete"
        write_json(path, report)


if __name__ == "__main__":
    main()
