"""0-GPUh CPU Gate 0 audit for held-out 6-DoF UR5e cross-morphology visuomotor SysID."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from fluffy_bassoon.xemb.embodiment import (
    ACTIVE_PROBE_BURSTS,
    circular_yaw_error_deg,
    get_embodiment_spec,
    jaw_width_from_qpos,
)
from fluffy_bassoon.xemb.joint_audit import evaluate_cross_morphology_joint_jacobian
from fluffy_bassoon.xemb.readers import analytic_proprio_sysid
from ur5e_motor import SPLITS_PATH, array_digest, public_digest, write_json
from ur5e_sim_worker import UR5eSimulator


DEV_EMBODIMENTS = ("fam_nom", "fam_yaw90", "comp_180_lo_inv", "cont_45_nom")
FIT_CHECK_TASKS = (("libero_goal", 5), ("libero_spatial", 2))


def run_probe(sim: UR5eSimulator, *, arm: str, suite: str, task_id: int, init_state_id: int, seed: int, emb: str):
    spec = get_embodiment_spec(emb)
    _, initial = sim.call(
        "create",
        dict(
            arm=arm,
            suite=suite,
            task_id=task_id,
            variant="BASE",
            init_state_id=init_state_id,
            seed=seed,
            embodiment=spec.name,
        ),
    )
    raw = initial["observation"]
    model_id = initial["model_identity"]
    dof = int(model_id["joint_count"])
    k = len(ACTIVE_PROBE_BURSTS)
    low = np.asarray(initial["action_schema"]["low"], dtype=np.float32)
    high = np.asarray(initial["action_schema"]["high"], dtype=np.float32)

    cur_img = np.asarray(initial["agentview_64"], dtype=np.uint8)
    cur_pos = np.asarray(raw["robot0_eef_pos"], dtype=np.float32).copy()
    cur_jaw = float(jaw_width_from_qpos(raw["robot0_gripper_qpos"]))
    cur_q = np.asarray(raw["robot0_joint_pos"], dtype=np.float32).copy()

    issued_mean = np.zeros((k, 7), dtype=np.float32)
    delta_eef = np.zeros((k, 3), dtype=np.float32)
    delta_jaw = np.zeros((k, 2), dtype=np.float32)
    delta_q = np.zeros((k, dof), dtype=np.float32)
    valid = np.zeros((k,), dtype=bool)
    rgb_l1 = []

    for idx, burst in enumerate(ACTIVE_PROBE_BURSTS):
        bounded = np.clip(burst, low, high).astype(np.float32)
        step = sim.call("chunk", bounded)
        n = int(step["steps"])
        next_img = np.asarray(step["agentview_64"], dtype=np.uint8)
        next_pos = np.asarray(step["positions"][-1], dtype=np.float32)
        next_jaw = float(jaw_width_from_qpos(step["gripper_positions"][-1]))
        next_q = np.asarray(step["joint_positions"][-1], dtype=np.float32)
        issued_mean[idx] = bounded[:n].mean(axis=0)
        delta_eef[idx] = next_pos - cur_pos
        delta_jaw[idx] = np.array([next_jaw - cur_jaw, next_jaw], dtype=np.float32)
        delta_q[idx] = next_q - cur_q
        valid[idx] = True
        rgb_l1.append(float(np.mean(np.abs(next_img.astype(np.float32) - cur_img.astype(np.float32)))))
        cur_img, cur_pos, cur_jaw, cur_q = next_img, next_pos, next_jaw, next_q
        if step["terminated"] or step["truncated"]:
            break

    frame0 = dict(
        physics=initial["environment_reproducibility"]["initial_state_fingerprint"],
        public=public_digest(raw),
        agentview_64=array_digest(initial["agentview_64"]),
        prompt=initial["language_instruction"],
    )
    return dict(
        arm=arm,
        model_class=model_id["model_class"],
        joint_count=dof,
        embodiment=spec.name,
        yaw_deg=float(spec.yaw_deg),
        xy_scale=float(spec.xy_scale),
        gripper_sign=int(spec.gripper_sign),
        frame0=frame0,
        issued_mean=issued_mean,
        delta_eef=delta_eef,
        delta_jaw=delta_jaw,
        delta_q=delta_q,
        valid=valid,
        sum_eef_mm=float(np.sum(np.linalg.norm(delta_eef[valid], axis=-1)) * 1000.0),
        max_jaw_mm=float(np.max(np.abs(delta_jaw[valid, 0])) * 1000.0),
        mean_rgb_l1=float(np.mean(rgb_l1)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-state-id", type=int, default=13)
    args = parser.parse_args()

    started = time.monotonic()
    splits = json.loads(SPLITS_PATH.read_text())
    dev_tasks = [(s, int(t)) for s, ids in sorted(splits["development"].items()) for t in ids]
    if len(dev_tasks) != 8:
        raise ValueError(f"Expected 8 development tasks, got {len(dev_tasks)}")

    dev_rows = []
    ur5e_issued, ur5e_dq, ur5e_deef, ur5e_valid, ur5e_true_yaw = [], [], [], [], []
    panda_nom_dq, panda_nom_deef, panda_nom_valid = [], [], []
    panda_test_issued, panda_test_dq, panda_test_deef, panda_test_valid, panda_test_yaw = [], [], [], [], []
    sawyer_dofs = []

    with UR5eSimulator(allow_mesa=True) as sim:
        for idx, (suite, tid) in enumerate(dev_tasks):
            seed = 20500000 + 100 * idx + tid
            by_emb = {}
            for emb in DEV_EMBODIMENTS:
                by_emb[emb] = run_probe(
                    sim, arm="UR5e", suite=suite, task_id=tid, init_state_id=args.init_state_id, seed=seed, emb=emb
                )
            ref_f0 = by_emb["fam_nom"]["frame0"]
            frame0_equal = all(by_emb[e]["frame0"] == ref_f0 for e in DEV_EMBODIMENTS)
            dof_ok = all(
                by_emb[e]["joint_count"] == 6 and by_emb[e]["model_class"] in ("MountedUR5e", "OnTheGroundUR5e")
                for e in DEV_EMBODIMENTS
            )
            # Local proprioception yaw-invariance across scale=1.0 embodiments (fam_nom, fam_yaw90, cont_45_nom).
            nom_xy = by_emb["fam_nom"]["delta_eef"][:, :2]
            max_yaw_local_diff_m = max(
                float(np.max(np.abs(by_emb[e]["delta_eef"][:, :2] - nom_xy)))
                for e in ("fam_yaw90", "cont_45_nom")
            )
            min_eef_mm = min(by_emb[e]["sum_eef_mm"] for e in DEV_EMBODIMENTS)
            min_jaw_mm = min(by_emb[e]["max_jaw_mm"] for e in DEV_EMBODIMENTS)
            min_rgb_l1 = min(by_emb[e]["mean_rgb_l1"] for e in DEV_EMBODIMENTS)
            excitation_ok = min_eef_mm > 15.0 and min_jaw_mm > 10.0 and min_rgb_l1 > 0.5

            dev_rows.append(
                dict(
                    task=f"{suite}-{tid}",
                    model_class=by_emb["fam_nom"]["model_class"],
                    joint_count=by_emb["fam_nom"]["joint_count"],
                    frame0_bitwise_equal=frame0_equal,
                    dof_ok=dof_ok,
                    max_yaw_local_diff_m=max_yaw_local_diff_m,
                    local_proprio_yaw_invariant=max_yaw_local_diff_m < 0.01,
                    min_sum_eef_mm=round(min_eef_mm, 3),
                    min_max_jaw_mm=round(min_jaw_mm, 3),
                    min_mean_rgb_l1=round(min_rgb_l1, 3),
                    excitation_ok=excitation_ok,
                )
            )
            for e in DEV_EMBODIMENTS:
                ur5e_issued.append(by_emb[e]["issued_mean"])
                ur5e_dq.append(by_emb[e]["delta_q"])
                ur5e_deef.append(by_emb[e]["delta_eef"])
                ur5e_valid.append(by_emb[e]["valid"])
                ur5e_true_yaw.append(by_emb[e]["yaw_deg"])

        # Source arm 7-DoF checks (Panda + Sawyer) on 2 fitting tasks.
        for idx, (suite, tid) in enumerate(FIT_CHECK_TASKS):
            seed = 20510000 + 100 * idx + tid
            for emb in ("nom", "yaw90"):
                p_rec = run_probe(sim, arm="Panda", suite=suite, task_id=tid, init_state_id=0, seed=seed, emb=emb)
                s_rec = run_probe(sim, arm="Sawyer", suite=suite, task_id=tid, init_state_id=0, seed=seed, emb=emb)
                sawyer_dofs.append(s_rec["joint_count"])
                if emb == "nom":
                    panda_nom_dq.append(p_rec["delta_q"])
                    panda_nom_deef.append(p_rec["delta_eef"])
                    panda_nom_valid.append(p_rec["valid"])
                panda_test_issued.append(p_rec["issued_mean"])
                panda_test_dq.append(p_rec["delta_q"])
                panda_test_deef.append(p_rec["delta_eef"])
                panda_test_valid.append(p_rec["valid"])
                panda_test_yaw.append(p_rec["yaw_deg"])

    # Joint-Jacobian audit: Panda (7-DoF) -> Panda (7-DoF) vs Panda (7-DoF) -> UR5e (6-DoF).
    p_dq = np.stack(panda_nom_dq)
    p_deef = np.stack(panda_nom_deef)
    p_val = np.stack(panda_nom_valid)

    same_arm_audit = evaluate_cross_morphology_joint_jacobian(
        p_dq,
        p_deef,
        p_val,
        np.stack(panda_test_issued),
        np.stack(panda_test_dq),
        np.stack(panda_test_deef),
        np.stack(panda_test_valid),
    )
    cross_arm_audit = evaluate_cross_morphology_joint_jacobian(
        p_dq,
        p_deef,
        p_val,
        np.stack(ur5e_issued),
        np.stack(ur5e_dq),
        np.stack(ur5e_deef),
        np.stack(ur5e_valid),
    )

    same_arm_strict_err = float(
        np.mean([circular_yaw_error_deg(yh, yt) for yh, yt in zip(same_arm_audit["strict_yaw_deg"], panda_test_yaw)])
    )
    ur5e_zero_pad_err = float(
        np.mean([circular_yaw_error_deg(yh, yt) for yh, yt in zip(cross_arm_audit["zero_padded_yaw_deg"], ur5e_true_yaw)])
    )
    ur5e_self_cal_err = float(
        np.mean(
            [circular_yaw_error_deg(yh, yt) for yh, yt in zip(cross_arm_audit["self_calibrated_local_yaw_deg"], ur5e_true_yaw)]
        )
    )
    proprio_yaws, _, _ = analytic_proprio_sysid(
        np.stack(ur5e_issued),
        np.stack(ur5e_deef),
        np.zeros((len(ur5e_issued), len(ACTIVE_PROBE_BURSTS), 2), dtype=np.float32),
        np.stack(ur5e_valid),
    )
    ur5e_proprio_err = float(
        np.mean([circular_yaw_error_deg(float(yh), yt) for yh, yt in zip(proprio_yaws, ur5e_true_yaw)])
    )

    predicates = dict(
        all_8_dev_ur5e_6dof=all(r["dof_ok"] for r in dev_rows),
        all_8_dev_frame0_bitwise_equal=all(r["frame0_bitwise_equal"] for r in dev_rows),
        all_8_dev_local_proprio_yaw_invariant=all(r["local_proprio_yaw_invariant"] for r in dev_rows),
        ur5e_analytic_proprio_yaw_blind=ur5e_proprio_err >= 60.0,
        all_8_dev_excitation_ok=all(r["excitation_ok"] for r in dev_rows),
        source_arms_7dof=same_arm_audit["source_dof"] == 7 and all(d == 7 for d in sawyer_dofs),
        joint_jacobian_inapplicable_to_ur5e=(
            not cross_arm_audit["applicable"] and cross_arm_audit["reason"] == "joint_dim_mismatch:7!=6"
        ),
    )
    passed = all(predicates.values())

    summary = dict(
        gate0_passed=passed,
        elapsed_s=round(time.monotonic() - started, 2),
        init_state_id=args.init_state_id,
        predicates=predicates,
        joint_jacobian_audit=dict(
            same_arm_panda_applicable=same_arm_audit["applicable"],
            same_arm_panda_mean_yaw_err_deg=round(same_arm_strict_err, 3),
            cross_arm_ur5e_applicable=cross_arm_audit["applicable"],
            cross_arm_ur5e_reason=cross_arm_audit["reason"],
            cross_arm_ur5e_zero_padded_mean_yaw_err_deg=round(ur5e_zero_pad_err, 3),
            cross_arm_ur5e_self_cal_local_mean_yaw_err_deg=round(ur5e_self_cal_err, 3),
            cross_arm_ur5e_analytic_proprio_mean_yaw_err_deg=round(ur5e_proprio_err, 3),
        ),
        dev_rows=dev_rows,
    )
    write_json(args.output / "gate0-summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
