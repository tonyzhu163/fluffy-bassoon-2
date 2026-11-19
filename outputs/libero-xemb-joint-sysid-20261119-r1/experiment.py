"""Replay the seven fixed native supports and export public-sensor predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fluffy_bassoon.xemb.embodiment import ACTIVE_PROBE_BURSTS, circular_yaw_error_deg, jaw_width_from_qpos
from joint_reader import fit, predict
from motor_iface import write_json
from sim_worker import XEmbSimulator

ARCHIVE_SHA = "832029f23f298c4edb0a5caf6c0da634660230acc6cb380d3a44689910a97767"
CASES = [
    (0, "train", 5, 0, 20110500, "nom"), (7, "train", 5, 1, 20110501, "nom"),
    (14, "train", 7, 0, 20110700, "nom"), (21, "train", 7, 1, 20110701, "nom"),
    (28, "dev", 1, 11, 20210111, "fam_nom"), (29, "dev", 1, 11, 20210111, "comp_180_lo_inv"),
    (30, "dev", 1, 11, 20210111, "cont_45_nom"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(simulator, row, archive, out):
    identity = {k: row[k] for k in ("suite", "task_id", "init_state_id", "seed", "embodiment")}
    _, reset = simulator.call("create", dict(identity, variant="BASE"))
    physics = reset["environment_reproducibility"]["initial_state_fingerprint"]
    if physics != row["physics"] or "llvmpipe" not in reset["renderer"].lower():
        raise ValueError(f"Startup or CPU renderer mismatch on {identity}")
    records = [reset["observation"]]
    rewards = []
    for burst in ACTIVE_PROBE_BURSTS:
        step = simulator.call("chunk", burst)
        if step["steps"] != 5 or step["terminated"] or step["truncated"]:
            raise ValueError(f"Incomplete probe at {identity}")
        records.append(step["observation"])
        rewards.extend(step["rewards"])
    if row["active_controls"] != len(rewards) or row["active_terminated"] or row["active_truncated"]:
        raise ValueError("Original pilot support length differs")

    def field(name):
        return np.stack([raw[name] for raw in records])

    pos = field("robot0_eef_pos")
    fingers = field("robot0_gripper_qpos")
    width = jaw_width_from_qpos(fingers)
    trace = dict(q=field("robot0_joint_pos"), qvel=field("robot0_joint_vel"),
                 wrist_local=pos, fingers=fingers, delta_eef=np.diff(pos, axis=0),
                 delta_jaw=np.stack((np.diff(width), width[1:]), axis=-1),
                 issued_mean=ACTIVE_PROBE_BURSTS.mean(axis=1), issued=ACTIVE_PROBE_BURSTS.reshape(15, 7),
                 rewards=np.asarray(rewards))
    index = row["index"]
    np.savez(out / f"sensors-{index:02d}.npz", **trace)
    path = archive / row["active_path"]
    with np.load(path, allow_pickle=False) as pilot:
        if not np.array_equal(trace["issued_mean"], pilot["issued_mean"]) or not np.all(pilot["valid"]):
            raise ValueError("Pilot command mismatch")
        pairs = dict(frame0_wrist=(pos[0], pilot["frame0_state"][:3]),
                     frame0_fingers=(fingers[0], pilot["frame0_state"][-2:]),
                     delta_eef=(trace["delta_eef"], pilot["delta_eef"]),
                     delta_jaw=(trace["delta_jaw"], pilot["delta_jaw"]))
        max_error_mm = {k: float(1000 * np.max(np.abs(a - b))) for k, (a, b) in pairs.items()}
    record = dict(**identity, archive_index=index, split=row["split"], physics=physics,
                  renderer=reset["renderer"], renderer_vendor=reset["renderer_vendor"], controls=len(rewards),
                  tape_sha256=sha(path), original_tape=row["active_path"], replay_max_error_mm=max_error_mm)
    write_json(out / f"replay-{index:02d}.json", record)
    if not all(np.isfinite(v) and v <= 0.01 for v in max_error_mm.values()):
        raise ValueError(f"Public sensor replay mismatch on {identity}: {max_error_mm}")
    print(f"Verified pilot row {index} ({identity['embodiment']}), 15 controls", flush=True)
    return trace


def run(archive, out):
    if sha(archive / "archive.json") != ARCHIVE_SHA:
        raise ValueError("Pilot archive changed")
    manifest = json.loads((archive / "archive.json").read_text())
    if manifest["panel"] != "poc" or manifest["matched_reader_controls"] != 15:
        raise ValueError("Expected first pilot active support")
    rows = []
    for expected in CASES:
        r = manifest["trials"][expected[0]]
        fields = ("index", "split", "task_id", "init_state_id", "seed", "embodiment")
        if tuple(r[k] for k in fields) != expected or r["suite"] != "libero_goal":
            raise ValueError("Invalid archive row")
        rows.append(r)

    with XEmbSimulator(allow_mesa=True) as sim:
        train = [collect(sim, r, archive, out) for r in rows[:4]]
        calibration = fit(train)
        write_json(out / "calibration.json", dict(fitting_rows=[r["index"] for r in rows[:4]], **calibration))
        predictions = []
        for r in rows[4:]:
            trace = collect(sim, r, archive, out)
            pred = predict(trace, calibration)
            predictions.append(dict(suite=r["suite"], task_id=r["task_id"], variant="BASE",
                                    support_init_state_id=r["init_state_id"], support_seed=r["seed"],
                                    support_archive_index=r["index"], support_controls=15,
                                    query_init_state_id=13, embodiment=r["embodiment"], **pred))
    pred_file = out / "predictions.json"
    write_json(pred_file, dict(arm="analytic_public_joint_sysid", archive_sha256=ARCHIVE_SHA, rows=predictions))
    errors = []
    methods = ("yaw_deg", "joint_raw_yaw_deg", "wrist_yaw_deg", "wrist_referenced_yaw_deg")
    for pred, row in zip(json.loads(pred_file.read_text())["rows"], rows[4:], strict=True):
        truth = {k: row[k] for k in ("yaw_deg", "xy_scale", "gripper_sign")}
        err = {k: circular_yaw_error_deg(pred[k], truth["yaw_deg"]) for k in methods}
        errors.append(dict(index=row["index"], embodiment=row["embodiment"], truth=truth, yaw_abs_error_deg=err,
                           scale_abs_error=abs(pred["xy_scale"] - truth["xy_scale"]),
                           grip_correct=pred["gripper_sign"] == truth["gripper_sign"]))
    passes = all(r["yaw_abs_error_deg"]["yaw_deg"] < 20 for r in errors)
    for shifted in errors[1:]:
        e = shifted["yaw_abs_error_deg"]
        passes &= e["yaw_deg"] < min(e["wrist_yaw_deg"], e["wrist_referenced_yaw_deg"])
    write_json(out / "scores.json", dict(predictions_sha256=sha(pred_file), rows=errors,
               mean_abs_yaw_error_deg={k: float(np.mean([r["yaw_abs_error_deg"][k] for r in errors]))
                                       for k in methods}, strong_recovery=bool(passes)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    run(args.archive, args.out)
