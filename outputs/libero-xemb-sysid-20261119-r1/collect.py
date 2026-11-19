"""Collect active-probe and passive-attempt1 support archives across train and dev splits."""

import argparse
import json
from pathlib import Path

import numpy as np

from fluffy_bassoon.xemb.embodiment import (
    ACTIVE_PROBE_BURSTS,
    EVAL_EMBODIMENTS,
    TRAIN_EMBODIMENTS,
    jaw_width_from_qpos,
)
from motor_iface import (
    SPLITS_PATH,
    array_digest,
    load_frozen_pi05,
    run_episode,
    to_openpi_observation,
    write_json,
)
from sim_worker import XEmbSimulator


def run_active_probe_episode(simulator, case: dict, spec, prompt=None):
    """Execute the 15-control (3-chunk) active probe without requiring a VLA policy."""
    sim_spec, initial = simulator.call("create", dict(case, embodiment=spec.name))
    raw = initial["observation"]
    released_prompt = initial["language_instruction"]
    prompt = released_prompt if prompt is None else prompt
    bounds = initial["action_schema"]
    low = np.asarray(bounds["low"], dtype=np.float32)
    high = np.asarray(bounds["high"], dtype=np.float32)

    k = len(ACTIVE_PROBE_BURSTS)
    img_before = np.zeros((k, 64, 64, 3), dtype=np.uint8)
    img_after = np.zeros((k, 64, 64, 3), dtype=np.uint8)
    issued_mean = np.zeros((k, 7), dtype=np.float32)
    delta_eef = np.zeros((k, 3), dtype=np.float32)
    delta_jaw = np.zeros((k, 2), dtype=np.float32)
    valid = np.zeros((k,), dtype=bool)

    frame0_img = np.asarray(initial["agentview_64"], dtype=np.uint8)
    frame0_state = to_openpi_observation(raw, prompt)["observation/state"].astype(np.float32)
    cur_img = frame0_img
    cur_pos = np.asarray(raw["robot0_eef_pos"], dtype=np.float32).copy()
    cur_jaw = float(jaw_width_from_qpos(raw["robot0_gripper_qpos"]))

    controls = 0
    for idx, burst in enumerate(ACTIVE_PROBE_BURSTS):
        bounded = np.clip(burst, low, high).astype(np.float32)
        step = simulator.call("chunk", bounded)
        n = int(step["steps"])
        controls += n
        next_img = np.asarray(step["agentview_64"], dtype=np.uint8)
        next_pos = np.asarray(step["positions"][-1], dtype=np.float32)
        next_jaw = float(jaw_width_from_qpos(step["gripper_positions"][-1]))
        img_before[idx] = cur_img
        img_after[idx] = next_img
        issued_mean[idx] = bounded[:n].mean(axis=0)
        delta_eef[idx] = next_pos - cur_pos
        delta_jaw[idx] = np.array([next_jaw - cur_jaw, next_jaw], dtype=np.float32)
        valid[idx] = True
        cur_img, cur_pos, cur_jaw = next_img, next_pos, next_jaw
        if step["terminated"] or step["truncated"]:
            break

    trace = dict(
        img_before=img_before,
        img_after=img_after,
        issued_mean=issued_mean,
        delta_eef=delta_eef,
        delta_jaw=delta_jaw,
        valid=valid,
        frame0_img=frame0_img,
        frame0_state=frame0_state,
    )
    meta = dict(
        physics=initial["environment_reproducibility"]["initial_state_fingerprint"],
        prompt=prompt,
        controls=controls,
        renderer=initial["renderer"],
    )
    return meta, trace


def build_schedule(panel: str):
    splits = json.loads(SPLITS_PATH.read_text())
    if panel == "smoke":
        train_tasks = {"libero_goal": [4]}
        dev_tasks = {"libero_goal": [1]}
        train_starts = (0,)
        dev_starts = (11,)
        train_embs = TRAIN_EMBODIMENTS[:2]
        eval_embs = (EVAL_EMBODIMENTS[0], EVAL_EMBODIMENTS[4], EVAL_EMBODIMENTS[8])
    else:
        train_tasks = splits["development_scaling"]["task_subsets"]["8"]
        dev_tasks = splits["development"]
        train_starts = (0, 1, 2, 3)
        dev_starts = (11,)
        train_embs = TRAIN_EMBODIMENTS
        eval_embs = EVAL_EMBODIMENTS

    entries = []
    for split_name, tasks, starts, embs, seed_base in (
        ("train", train_tasks, train_starts, train_embs, 20100000),
        ("dev", dev_tasks, dev_starts, eval_embs, 20200000),
    ):
        for suite_idx, (suite, task_ids) in enumerate(sorted(tasks.items())):
            for task_id in task_ids:
                for init_id in starts:
                    seed = seed_base + 10_000 * suite_idx + 100 * int(task_id) + int(init_id)
                    for spec in embs:
                        entries.append((split_name, suite, int(task_id), int(init_id), seed, spec))
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", choices=("smoke", "full"), default="full")
    parser.add_argument("--modes", type=str, default="active_probe,passive_attempt1")
    parser.add_argument("--passive-max-controls", type=int, default=120)
    parser.add_argument("--allow-mesa", action="store_true")
    args = parser.parse_args()

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    for m in modes:
        if m not in ("active_probe", "passive_attempt1"):
            raise ValueError(f"Unsupported collection mode {m!r}")
    if args.passive_max_controls % 5 != 0 or args.passive_max_controls < 5:
        raise ValueError("passive_max_controls must be a positive multiple of 5")
    max_passive_chunks = args.passive_max_controls // 5

    policy = load_frozen_pi05() if "passive_attempt1" in modes else None
    args.output.mkdir(parents=True, exist_ok=True)
    tapes_dir = args.output / "tapes"
    tapes_dir.mkdir(parents=True, exist_ok=True)

    schedule = build_schedule(args.panel)
    trials = []
    prompts = {}

    with XEmbSimulator(allow_mesa=args.allow_mesa) as simulator:
        robots_info = simulator.call("inspect_robots")
        write_json(args.output / "inspect-robots.json", robots_info)

        for idx, (split_name, suite, task_id, init_id, seed, spec) in enumerate(schedule):
            case = dict(suite=suite, task_id=task_id, variant="BASE", init_state_id=init_id, seed=seed)
            key = (suite, task_id)
            prompt = prompts.get(key)

            row = dict(
                index=idx,
                split=split_name,
                suite=suite,
                task_id=task_id,
                init_state_id=init_id,
                seed=seed,
                embodiment=spec.name,
                regime=spec.regime,
                yaw_deg=float(spec.yaw_deg),
                xy_scale=float(spec.xy_scale),
                gripper_sign=int(spec.gripper_sign),
            )

            if "active_probe" in modes:
                act_meta, act_trace = run_active_probe_episode(simulator, case, spec, prompt=prompt)
                prompts.setdefault(key, act_meta["prompt"])
                prompt = prompts[key]
                act_path = f"tapes/{idx:04d}-{split_name}-active.npz"
                np.savez_compressed(args.output / act_path, **act_trace)
                row.update(
                    physics=act_meta["physics"],
                    frame0_img_sha256=array_digest(act_trace["frame0_img"]),
                    frame0_state_sha256=array_digest(act_trace["frame0_state"]),
                    active_path=act_path,
                    active_controls=act_meta["controls"],
                    active_delta_jaw=act_trace["delta_jaw"][:, 0].tolist(),
                )

            if "passive_attempt1" in modes:
                pas_rec, pas_trace = run_episode(
                    simulator,
                    policy,
                    case,
                    spec,
                    prompt=prompt,
                    yaw_hat=0.0,
                    scale_hat=1.0,
                    grip_hat=1,
                    max_controls=args.passive_max_controls,
                    record_chunks=True,
                    max_record_chunks=max_passive_chunks,
                )
                prompts.setdefault(key, pas_rec["reset"]["prompt"])
                if "physics" in row and row["physics"] != pas_rec["reset"]["physics"]:
                    raise ValueError(f"Physics reset mismatch between active and passive on {row}")
                pas_path = f"tapes/{idx:04d}-{split_name}-passive.npz"
                np.savez_compressed(args.output / pas_path, **pas_trace)
                row.update(
                    physics=pas_rec["reset"]["physics"],
                    frame0_img_sha256=array_digest(pas_trace["frame0_img"]),
                    frame0_state_sha256=array_digest(pas_trace["frame0_state"]),
                    passive_path=pas_path,
                    passive_controls=pas_rec["controls"],
                    passive_success=pas_rec["success"],
                )

            trials.append(row)
            if (idx + 1) % 20 == 0 or idx + 1 == len(schedule):
                print(json.dumps(dict(collected=idx + 1, total=len(schedule), last=spec.name)), flush=True)

    # Audit: frame-0 img & state must be bitwise identical across embodiments for each reset pose.
    grouped = {}
    for row in trials:
        k = (row["split"], row["suite"], row["task_id"], row["init_state_id"])
        grouped.setdefault(k, []).append(row)
    for k, group in grouped.items():
        if len({r["physics"] for r in group}) != 1:
            raise ValueError(f"Physics fingerprint differed across embodiments on {k}")
        if len({r["frame0_img_sha256"] for r in group}) != 1:
            raise ValueError(f"Frame-0 RGB differed across embodiments on {k}")
        if len({r["frame0_state_sha256"] for r in group}) != 1:
            raise ValueError(f"Frame-0 proprio differed across embodiments on {k}")

    archive = dict(
        panel=args.panel,
        modes=list(modes),
        passive_max_controls=args.passive_max_controls,
        frame0_bitwise_identical_verified=True,
        reset_groups=len(grouped),
        trials=trials,
    )
    write_json(args.output / "archive.json", archive)
    print(json.dumps(dict(status="complete", trials=len(trials), reset_groups=len(grouped))), flush=True)


if __name__ == "__main__":
    main()
