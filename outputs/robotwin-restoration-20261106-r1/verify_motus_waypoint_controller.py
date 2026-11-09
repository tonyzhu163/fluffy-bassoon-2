"""Paired Motus SFT verification of the repaired take_action waypoint controller.

Evaluates motus-robotics/pi0.5_robotwin2 on the 16 precommitted source DEV scenes
(MOTUS-DEV-CASES.json) with waypoint_controller=True at (K=5, H=192) and (K=10, H=384),
comparing directly against the legacy gen_sparse_reward_data baseline (2/16 = 12.5%).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal, json_metrics, sha, write
from fluffy_bassoon.rl.robotwin_batch import (
    FastMeasuredSimulator,
    RoboTwinBatchRuntime,
    accounted_controller_source,
)
from fluffy_bassoon.rl.robotwin_motor import FrozenRoboTwinMotor
from fluffy_bassoon.rl.robotwin_noise import model_config
from fluffy_bassoon.rl.robotwin_production_process import ProductionSimulatorPool
from qualify_motor import MOTOR, Memory

ROOT = Path(__file__).resolve().parent


def run_profile(profile, cases, root, local, journal, motor_noise_seed=20261108):
    name = profile["name"]
    horizon = int(profile["horizon"])
    num_steps = int(profile["num_steps"])
    waypoint_controller = bool(profile["waypoint_controller"])

    norm = MOTOR / "assets/pi0.5_clean_randomize_joint_training/norm_stats.json"
    from rlinf.models import get_model

    model = get_model(
        model_config(
            os.environ["REPO_PATH"],
            MOTOR,
            norm,
            action_horizon=32,
            num_steps=num_steps,
        )
    ).cuda()
    motor = FrozenRoboTwinMotor(model)

    options = accounted_controller_source(
        os.environ["ROBOTWIN_PATH"],
        local / name / "controller",
        waypoint_controller=waypoint_controller,
    )
    pool = ProductionSimulatorPool(
        2,
        rlinf_root=os.environ["REPO_PATH"],
        robotwin_root=os.environ["ROBOTWIN_PATH"],
        output_dir=local / name / "simulators",
        task_episode_limits={"place_phone_stand": horizon},
        final_observation_only=False,
        task_config_name="demo_clean",
        _simulator_factory=FastMeasuredSimulator,
        worker_options=options,
    )
    runtime = RoboTwinBatchRuntime(motor, pool, seed=motor_noise_seed)
    entry = dict(
        name=name,
        waypoint_controller=waypoint_controller,
        controller_sha256=options["controller_sha256"],
        num_steps=num_steps,
        horizon=horizon,
        episodes=[],
        started_unix=time.time(),
    )
    try:
        num_pairs = len(cases) // 2
        for pair in range(num_pairs):
            selected = cases[pair * 2 : pair * 2 + 2]
            receipts = runtime.reset(
                "place_phone_stand",
                selected,
                case_ids=[f"{name}/case-{seed}" for seed in selected],
            )
            episodes = [
                dict(
                    case=int(seed),
                    success=False,
                    complete=False,
                    decisions=0,
                    delivered_targets=0,
                    physics_steps=0,
                )
                for seed in selected
            ]
            entry["episodes"].extend(episodes)
            while any(runtime._active):
                base, features, metadata = runtime.infer(fixed_native_batch=True)
                requested = len(metadata["active_lanes"]) * 32
                cursor = journal.begin(requested, phase=name)
                transitions = runtime.step(base)
                journal.finish(cursor, requested)
                for transition in transitions:
                    lane = transition["native_batch"]["row"]
                    episode = episodes[lane]
                    episode["decisions"] += 1
                    episode["success"] |= bool(transition["audit"]["success"])
                    exec_info = transition["audit"].get("execution", {})
                    episode["delivered_targets"] += int(
                        exec_info.get("delivered_targets", 0)
                    )
                    episode["physics_steps"] += int(
                        exec_info.get("physics_steps", 0)
                    )
                    episode["complete"] = bool(
                        transition["terminated"] or transition["truncated"]
                    )
            assert all(ep["complete"] for ep in episodes)
            entry["completed_episodes"] = len(entry["episodes"])
            entry["successes_so_far"] = sum(
                1 for ep in entry["episodes"] if ep["success"]
            )
            write(root / f"{name}-progress.json", entry)
        motor.check_frozen()
    finally:
        runtime.close()
        del model
        torch.cuda.empty_cache()

    entry["ended_unix"] = time.time()
    entry["elapsed_seconds"] = entry["ended_unix"] - entry["started_unix"]
    entry["successes"] = sum(1 for ep in entry["episodes"] if ep["success"])
    entry["total_episodes"] = len(entry["episodes"])
    entry["success_rate"] = entry["successes"] / max(1, entry["total_episodes"])
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=16)
    args = parser.parse_args()

    assert torch.cuda.device_count() >= 1
    torch.cuda.set_device(0)

    dev_cases_data = json.loads((ROOT / "MOTUS-DEV-CASES.json").read_bytes())
    cases = dev_cases_data["seeds"][: args.max_cases]
    assert len(cases) % 2 == 0 and len(cases) == len(set(cases))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.local_dir.mkdir(parents=True, exist_ok=True)

    journal = ActionJournal(
        args.output_dir / "action-accounting",
        rank=0,
        world=1,
        attempt="waypoint-verify01",
        maximum_actions=100000,
    )

    profiles = [
        dict(
            name="waypoint_k5_h192",
            waypoint_controller=True,
            num_steps=5,
            horizon=192,
        ),
        dict(
            name="waypoint_k10_h384",
            waypoint_controller=True,
            num_steps=10,
            horizon=384,
        ),
    ]

    report = dict(
        status="running",
        task="place_phone_stand",
        cases=cases,
        legacy_gen_sparse_reward_data_dev16_reference=dict(
            successes=2, total=16, success_rate=0.125
        ),
        profiles={},
        started_unix=time.time(),
    )
    write(args.output_dir / "RESULT.json", report)

    with Memory(torch) as memory:
        for profile in profiles:
            entry = run_profile(
                profile,
                cases,
                args.output_dir,
                args.local_dir,
                journal,
            )
            report["profiles"][profile["name"]] = entry
            write(args.output_dir / "RESULT.json", report)

    report["memory"] = memory.result()
    report["charged_requested_targets"] = journal.charged()
    report["ended_unix"] = time.time()
    report["status"] = "completed"
    write(args.output_dir / "RESULT.json", report)
    print(json.dumps(json_metrics(report), indent=2))


if __name__ == "__main__":
    main()
