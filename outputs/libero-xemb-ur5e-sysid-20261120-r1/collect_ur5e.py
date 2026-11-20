"""Collect active-probe and passive-attempt1 archives: 7-DoF Panda/Sawyer train -> held-out 6-DoF UR5e dev."""

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
from ur5e_motor import (
    SPLITS_PATH,
    array_digest,
    load_frozen_pi05,
    run_episode,
    to_openpi_observation,
    write_json,
)
from ur5e_sim_worker import UR5eSimulator


MODE_PATH = {"active_probe": "active_path", "passive_attempt1": "passive_path"}
CHUNK_SIZE = len(ACTIVE_PROBE_BURSTS[0])
SUPPORT_BUDGET = sum(map(len, ACTIVE_PROBE_BURSTS))
TRAIN_ARMS = ("Panda", "Sawyer")
DEV_ARM = "UR5e"


def chunk_counts(valid, controls):
    """Five-control chunks can end early only in the final executed chunk."""
    if type(controls) is not int or valid.dtype != np.bool_ or valid.ndim != 1 or not 0 < controls <= len(valid) * CHUNK_SIZE:
        raise ValueError("Invalid support mask or control count")
    counts = np.clip(controls - np.arange(len(valid)) * CHUNK_SIZE, 0, CHUNK_SIZE)
    if not np.array_equal(valid, counts > 0):
        raise ValueError("Recorded controls disagree with valid chunk prefix")
    return counts


def issued_exposure(data):
    """Integrated issued action in units of a full chunk, for analytic regression."""
    fraction = data["chunk_steps"][..., None].astype(np.float32) / CHUNK_SIZE
    return data["issued_mean"] * fraction


def frame0_identity(physics, trace):
    return dict(
        physics=physics,
        frame0_img_sha256=array_digest(trace["frame0_img"]),
        frame0_state_sha256=array_digest(trace["frame0_state"]),
    )


def read_support(archive_dir: Path, row: dict, mode: str):
    """Read public prefix for training, audit, and live inference."""
    controls = row["active_controls" if mode == "active_probe" else "passive_controls"]
    expected_dof = 6 if row["arm"] == "UR5e" else 7
    with np.load(archive_dir / row[MODE_PATH[mode]], allow_pickle=False) as tape:
        valid = tape["valid"]
        counts = chunk_counts(valid, controls)
        expected = {
            "img_before": (64, 64, 3),
            "img_after": (64, 64, 3),
            "issued_mean": (7,),
            "delta_eef": (3,),
            "delta_jaw": (2,),
            "delta_joint": (expected_dof,),
        }
        data = {}
        k = len(ACTIVE_PROBE_BURSTS)
        for name, shape in expected.items():
            value = tape[name]
            dtype = np.uint8 if name.startswith("img_") else np.float32
            if value.shape != (len(valid), *shape) or value.dtype != dtype:
                raise ValueError(f"Invalid {name} shape or dtype on row {row['index']}: {value.shape}")
            if not np.isfinite(value[:k][valid[:k]]).all():
                raise ValueError(f"Nonfinite observed {name} on row {row['index']}")
            data[name] = value[:k].copy()
        for name, shape, dtype in (("frame0_img", (64, 64, 3), np.uint8), ("frame0_state", (8,), np.float32)):
            value = tape[name]
            if (
                value.shape != shape
                or value.dtype != dtype
                or not np.isfinite(value).all()
                or array_digest(value) != row[name + "_sha256"]
            ):
                raise ValueError(f"Reset mismatch for {name} on row {row['index']}")
            data[name] = value
        if len(valid) < k or not np.array_equal(data["frame0_img"], data["img_before"][0]):
            raise ValueError("Support prefix is short or starts after reset")
        data["valid"] = valid[:k]
        data["chunk_steps"] = counts[:k]
    return data


def run_active_probe_episode(simulator, case: dict, spec, prompt=None):
    """Execute the 15-control (3-chunk) active probe without requiring a VLA policy."""
    _, initial = simulator.call("create", dict(case, embodiment=spec.name))
    raw = initial["observation"]
    released_prompt = initial["language_instruction"]
    if prompt is not None and prompt != released_prompt:
        raise ValueError(f"Unpaired support prompt on {case}")
    prompt = released_prompt
    bounds = initial["action_schema"]
    low = np.asarray(bounds["low"], dtype=np.float32)
    high = np.asarray(bounds["high"], dtype=np.float32)
    dof = int(initial["model_identity"]["joint_count"])

    k = len(ACTIVE_PROBE_BURSTS)
    img_before = np.zeros((k, 64, 64, 3), dtype=np.uint8)
    img_after = np.zeros((k, 64, 64, 3), dtype=np.uint8)
    issued_mean = np.zeros((k, 7), dtype=np.float32)
    delta_eef = np.zeros((k, 3), dtype=np.float32)
    delta_jaw = np.zeros((k, 2), dtype=np.float32)
    delta_joint = np.zeros((k, dof), dtype=np.float32)
    valid = np.zeros((k,), dtype=bool)

    frame0_img = np.asarray(initial["agentview_64"], dtype=np.uint8)
    frame0_state = to_openpi_observation(raw, prompt)["observation/state"].astype(np.float32)
    cur_img = frame0_img
    cur_pos = np.asarray(raw["robot0_eef_pos"], dtype=np.float32).copy()
    cur_jaw = float(jaw_width_from_qpos(raw["robot0_gripper_qpos"]))
    cur_joint = np.asarray(raw["robot0_joint_pos"], dtype=np.float32).copy()

    controls = 0
    step = None
    for idx, burst in enumerate(ACTIVE_PROBE_BURSTS):
        bounded = np.clip(burst, low, high).astype(np.float32)
        step = simulator.call("chunk", bounded)
        n = int(step["steps"])
        if not 0 < n <= len(bounded) or (n < len(bounded) and not (step["terminated"] or step["truncated"])):
            raise ValueError("Simulator returned an invalid active probe length")
        controls += n
        next_img = np.asarray(step["agentview_64"], dtype=np.uint8)
        next_pos = np.asarray(step["positions"][-1], dtype=np.float32)
        next_jaw = float(jaw_width_from_qpos(step["gripper_positions"][-1]))
        next_joint = np.asarray(step["joint_positions"][-1], dtype=np.float32)
        img_before[idx] = cur_img
        img_after[idx] = next_img
        issued_mean[idx] = bounded[:n].mean(axis=0)
        delta_eef[idx] = next_pos - cur_pos
        delta_jaw[idx] = np.array([next_jaw - cur_jaw, next_jaw], dtype=np.float32)
        delta_joint[idx] = next_joint - cur_joint
        valid[idx] = True
        cur_img, cur_pos, cur_jaw, cur_joint = next_img, next_pos, next_jaw, next_joint
        if step["terminated"] or step["truncated"]:
            break

    trace = dict(
        img_before=img_before,
        img_after=img_after,
        issued_mean=issued_mean,
        delta_eef=delta_eef,
        delta_jaw=delta_jaw,
        delta_joint=delta_joint,
        valid=valid,
        joint_count=np.int32(dof),
        frame0_img=frame0_img,
        frame0_state=frame0_state,
    )
    meta = dict(
        arm=str(initial["model_identity"]["arm"]),
        model_class=str(initial["model_identity"]["model_class"]),
        joint_count=dof,
        physics=initial["environment_reproducibility"]["initial_state_fingerprint"],
        prompt=prompt,
        controls=controls,
        renderer=initial["renderer"],
        terminated=bool(step["terminated"]),
        truncated=bool(step["truncated"]),
    )
    return meta, trace


def build_schedule(panel: str):
    splits = json.loads(SPLITS_PATH.read_text())
    if panel == "smoke":
        train_tasks = {"libero_goal": [5]}
        dev_tasks = {"libero_goal": [1]}
        train_starts = (0,)
        dev_starts = (11,)
        train_embs = TRAIN_EMBODIMENTS[:2]
        eval_embs = (EVAL_EMBODIMENTS[0], EVAL_EMBODIMENTS[4], EVAL_EMBODIMENTS[8])
    elif panel == "full":
        train_tasks = splits["development_scaling"]["task_subsets"]["8"]
        dev_tasks = splits["development"]
        train_starts = (0, 1)
        dev_starts = (11,)
        train_embs = TRAIN_EMBODIMENTS
        eval_embs = EVAL_EMBODIMENTS
    else:
        raise ValueError(f"Unknown collection panel {panel!r}")

    entries = []
    for split_name, arms, tasks, starts, embs, seed_base in (
        ("train", TRAIN_ARMS, train_tasks, train_starts, train_embs, 20600000),
        ("dev", (DEV_ARM,), dev_tasks, dev_starts, eval_embs, 20700000),
    ):
        official = "train" if split_name == "train" else "development"
        for arm_idx, arm in enumerate(arms):
            for suite, task_ids in sorted(tasks.items()):
                suite_idx = sorted(splits[official]).index(suite)
                for task_id in task_ids:
                    if (
                        task_id not in splits[official][suite]
                        or task_id in splits["test"][suite]
                        or task_id in splits["development" if official == "train" else "train"][suite]
                    ):
                        raise ValueError(f"Task outside registered {split_name} split: {suite}/{task_id}")
                    for init_id in starts:
                        seed = seed_base + 100_000 * arm_idx + 10_000 * suite_idx + 100 * int(task_id) + int(init_id)
                        for spec in embs:
                            entries.append((split_name, arm, suite, int(task_id), int(init_id), seed, spec))
    return entries


def validate_archive(archive: dict) -> int:
    """Verify complete panel, strict train/dev morphology disjointness, and bitwise frame-0 pairing."""
    schedule = build_schedule(archive["panel"])
    expected = {
        (split, arm, suite, task, init, spec.name): (seed, spec)
        for split, arm, suite, task, init, seed, spec in schedule
    }
    modes = archive["modes"]
    if not modes or len(set(modes)) != len(modes) or set(modes) - MODE_PATH.keys():
        raise ValueError("Invalid archive modes")
    seen_paths = set()
    resets = {}
    budgets = {"active_probe": SUPPORT_BUDGET, "passive_attempt1": archive["passive_max_controls"]}
    if budgets["passive_attempt1"] < SUPPORT_BUDGET or budgets["passive_attempt1"] % CHUNK_SIZE:
        raise ValueError("Passive collection must cover the matched support budget")

    for row in archive["trials"]:
        if row["split"] == "train" and row["arm"] not in TRAIN_ARMS:
            raise ValueError(f"Forbidden training arm {row['arm']!r}")
        if row["split"] == "dev" and row["arm"] != DEV_ARM:
            raise ValueError(f"Forbidden dev arm {row['arm']!r}")
        expected_dof = 6 if row["arm"] == "UR5e" else 7
        if int(row["joint_count"]) != expected_dof:
            raise ValueError(f"Unexpected joint_count={row['joint_count']} on {row['arm']}")
        group = tuple(row[k] for k in ("split", "arm", "suite", "task_id", "init_state_id"))
        key = (*group, row["embodiment"])
        if key not in expected:
            raise ValueError(f"Unexpected or repeated support trial {key}")
        seed, spec = expected.pop(key)
        if (row["seed"], row["regime"], row["yaw_deg"], row["xy_scale"], row["gripper_sign"]) != (
            seed,
            spec.regime,
            spec.yaw_deg,
            spec.xy_scale,
            spec.gripper_sign,
        ):
            raise ValueError(f"Support seed or factor mismatch on {key}")
        identity = tuple(row[k] for k in ("physics", "frame0_img_sha256", "frame0_state_sha256"))
        if resets.setdefault(group, identity) != identity:
            raise ValueError(f"Embodiments differ at reset on {group}")
        for mode in modes:
            path = row[MODE_PATH[mode]]
            if path in seen_paths:
                raise ValueError(f"Reused support path {path}")
            seen_paths.add(path)
            controls = row["active_controls" if mode == "active_probe" else "passive_controls"]
            if not 0 < controls <= budgets[mode]:
                raise ValueError(f"Invalid {mode} controls on {key}")
    if expected:
        raise ValueError(f"Incomplete support panel, missing {len(expected)} rows")
    return len(resets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", choices=("smoke", "full"), default="full")
    parser.add_argument("--modes", type=str, default="active_probe,passive_attempt1")
    parser.add_argument("--passive-max-controls", type=int, default=SUPPORT_BUDGET)
    parser.add_argument("--allow-mesa", action="store_true")
    args = parser.parse_args()

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    if not modes or len(set(modes)) != len(modes) or set(modes) - MODE_PATH.keys():
        parser.error("Supply one or both distinct support modes")
    if args.passive_max_controls < SUPPORT_BUDGET or args.passive_max_controls % CHUNK_SIZE:
        parser.error("Passive cap must cover the active budget in complete chunks")
    max_passive_chunks = args.passive_max_controls // CHUNK_SIZE
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tapes").mkdir()
    policy = load_frozen_pi05() if "passive_attempt1" in modes else None

    schedule = build_schedule(args.panel)
    trials = []
    prompts = {}
    reset_identities = {}

    with UR5eSimulator(allow_mesa=args.allow_mesa) as simulator:
        for idx, (split_name, arm, suite, task_id, init_id, seed, spec) in enumerate(schedule):
            case = dict(arm=arm, suite=suite, task_id=task_id, variant="BASE", init_state_id=init_id, seed=seed)
            key = (suite, task_id)
            prompt = prompts.get(key)
            dof = 6 if arm == "UR5e" else 7

            row = dict(
                index=idx,
                split=split_name,
                arm=arm,
                joint_count=dof,
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
                chunk_counts(act_trace["valid"], act_meta["controls"])
                act_path = f"tapes/{idx:04d}-{split_name}-{arm}-active.npz"
                np.savez_compressed(args.output / act_path, **act_trace)
                row.update(
                    model_class=act_meta["model_class"],
                    **frame0_identity(act_meta["physics"], act_trace),
                    active_terminated=act_meta["terminated"],
                    active_truncated=act_meta["truncated"],
                    active_path=act_path,
                    active_controls=act_meta["controls"],
                    active_delta_jaw=act_trace["delta_jaw"][:, 0].tolist(),
                    active_delta_eef_norm_mm=(np.linalg.norm(act_trace["delta_eef"], axis=-1) * 1000.0).tolist(),
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
                identity = frame0_identity(pas_rec["reset"]["physics"], pas_trace)
                if "physics" in row and any(row[k] != v for k, v in identity.items()):
                    raise ValueError(f"Active/passive frame-0 or physics mismatch on {case}")
                chunk_counts(pas_trace["valid"], pas_rec["controls"])
                if pas_rec["controls"] < args.passive_max_controls and not (
                    pas_rec["terminated"] or pas_rec["truncated"] or pas_rec["controls"] == pas_rec["reset"]["horizon"]
                ):
                    raise ValueError(f"Unexplained short passive attempt on {case}")
                pas_path = f"tapes/{idx:04d}-{split_name}-{arm}-passive.npz"
                np.savez_compressed(args.output / pas_path, **pas_trace)
                row.update(
                    model_class=pas_rec["reset"]["model_class"],
                    **identity,
                    passive_terminated=pas_rec["terminated"],
                    passive_truncated=pas_rec["truncated"],
                    passive_path=pas_path,
                    passive_controls=pas_rec["controls"],
                    passive_success=pas_rec["success"],
                    passive_noise_sha256=pas_rec["reset"]["noise_sha256"],
                    passive_first_proposal_sha256=pas_rec["opening_proposal_sha256"],
                )

            group = (split_name, arm, suite, task_id, init_id)
            identity = tuple(
                row.get(k)
                for k in (
                    "physics",
                    "frame0_img_sha256",
                    "frame0_state_sha256",
                    "passive_noise_sha256",
                    "passive_first_proposal_sha256",
                )
            )
            if reset_identities.setdefault(group, identity) != identity:
                raise ValueError(f"Unmatched reset or passive opening on {group}/{spec.name}")
            trials.append(row)
            if (idx + 1) % 20 == 0 or idx + 1 == len(schedule):
                print(
                    json.dumps(dict(collected=idx + 1, total=len(schedule), arm=arm, last=spec.name)),
                    flush=True,
                )

    archive = dict(
        panel=args.panel,
        train_arms=list(TRAIN_ARMS),
        dev_arm=DEV_ARM,
        modes=list(modes),
        passive_max_controls=args.passive_max_controls,
        matched_reader_controls=SUPPORT_BUDGET,
        trials=trials,
    )
    archive["reset_groups"] = validate_archive(archive)
    archive["frame0_bitwise_identical_verified"] = True
    write_json(args.output / "archive.json", archive)
    print(json.dumps(dict(archive_trials=len(trials), reset_groups=archive["reset_groups"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
