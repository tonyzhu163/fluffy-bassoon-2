"""Recover world grip-site displacement from the measured joints on the original passive tapes."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


STORAGE = Path("/mowenjie/fluffy-bassoon-2")
CAMPAIGN = "libero-xemb-ur5e-scale-20261120-r1"
ARCHIVE = STORAGE / "outputs" / CAMPAIGN / "gpu09test/test_live"
SOURCE = STORAGE / "setups" / CAMPAIGN / "source-5dd5fa91"
COLLECTOR = "5dd5fa914bbbfe25b9cf4ca5fa504c8a472e997b"
CROSSARM = Path("outputs/libero-crossarm-20261120-r1")
SPLITS = Path("outputs/libero-meta-learning-20261117-r1/SPLITS.json")
SIM_FILES = {
    CROSSARM / "arm_env.py": "16ec2d3cd367e975a65d5be49ce4a897ba56621889a0a3502a5a47d7cafff0eb",
    CROSSARM / "sim_reset.py": "010c2ef9d71e268cd81470a6aa6640e0e78d3151b4e95b5f5408c8831ca8eb10",
    Path("src/fluffy_bassoon/xemb/embodiment.py"): "e0ed86b5fc85bbde7b4dd95b0ec2b520b792325d0e6f135465132cc2dc99c41c",
    SPLITS: "9b363ba704b628b34839a854e19118e7dd7958241e4b28530fa40241cc2cde2f",
}


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(source):
    """Use the collecting revision's reset, stock model, action semantics and split."""
    for name, digest in SIM_FILES.items():
        if file_sha(source / name) != digest:
            raise ValueError(f"Collector source mismatch: {name}")
    sim = STORAGE / "setups/libero-meta-learning-20261117-r1"
    for key, value in {
        "CROSSARM_RELEASE": STORAGE / "sources/libero-pro2-e5500c69",
        "LIBERO_PRO_ROOT": sim / "simulator-source", "LIBERO_CONFIG_PATH": sim / "libero-config",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_EGL_DEVICE_ID": "0", "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
    }.items():
        os.environ.setdefault(key, str(value))
    os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", LIBGL_ALWAYS_SOFTWARE="1")
    sys.path[:0] = [str(source / CROSSARM), str(source / "src/fluffy_bassoon/xemb"),
                    os.environ["CROSSARM_RELEASE"], os.environ["LIBERO_PRO_ROOT"]]


def panel(source, archive, selected):
    from embodiment import EVAL_EMBODIMENTS

    split = json.loads((source / SPLITS).read_text())["test"]
    tasks = [(suite, tid) for suite in sorted(split) for tid in split[suite]]
    if len(tasks) != 8 or len(EVAL_EMBODIMENTS) != 12:
        raise ValueError("Official UR5e panel changed")
    for shard in selected:
        suite, tid = tasks[shard]
        case = dict(shard=shard, suite=suite, task_id=tid, init_state_id=11,
                    seed=31800000 + 10000 * sorted(split).index(suite) + 100 * tid + 11)
        base = archive / f"shard-{shard}"
        meta = json.loads((base / "support-meta.json").read_text())
        rows = []
        for idx, (entry, spec) in enumerate(zip(meta, EVAL_EMBODIMENTS, strict=True)):
            if entry["embodiment"] != spec.name:
                raise ValueError(f"Manifest order mismatch: {case} slot {idx}")
            row = dict(index=12 * shard + idx, embodiment=spec.name, physics=entry["physics"])
            for mode in ("active", "passive"):
                rel = Path("support") / f"{idx:02d}-{spec.name}-{mode}.npz"
                if Path(entry[f"{mode}_npz"]).parts[-3:] != (base.name, *rel.parts):
                    raise ValueError(f"Unexpected tape identity: {entry}")
                row[mode] = base / rel
            rows.append(row)
        if len({r["physics"] for r in rows}) != 1:
            raise ValueError(f"Support reset changed within shard {shard}")
        yield case, rows


def tape(path):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    shapes = dict(issued_mean=(3, 7), delta_joint=(3, 6), delta_eef=(3, 3), delta_jaw=(3, 2),
                  frame0_state=(8,), img_before=(3, 64, 64, 3), img_after=(3, 64, 64, 3))
    for key, shape in shapes.items():
        arr = data[key]
        if arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError(f"Invalid {key} in {path}")
    if (data["joint_count"].item() != 6 or data["valid"].dtype != np.bool_
            or not np.array_equal(data["valid"], np.ones(3, bool))
            or np.max(np.abs(data["issued_mean"])) > 1
            or not np.array_equal(data["frame0_img"], data["img_before"][0])
            or not np.array_equal(data["img_after"][:-1], data["img_before"][1:])):
        raise ValueError(f"Incomplete or inconsistent UR5e tape: {path}")
    return data


def reset(arm, case):
    return arm.reset(init_state_id=case["init_state_id"], seed=case["seed"])


def motor_state(raw):
    # Same float32 quaternion-to-axis-angle operation as the collecting revision.
    quat = raw["robot0_eef_quat_xyzw"]
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    rot = quat[:3] * 2.0 * np.arccos(w) / den if den > 1e-10 else np.zeros(3)
    return np.concatenate((raw["robot0_eef_pos"], rot, raw["robot0_gripper_qpos"])).astype(np.float32)


class GripSite:
    """Evaluate the measured arm configurations on scratch data for the actual mounted UR5e."""

    def __init__(self, arm):
        import mujoco

        sim = arm._env.sim
        robot = arm._env.robots[0]
        self.model = sim.model._model
        self.site = robot.eef_site_id
        self.indices = list(robot._ref_joint_pos_indexes)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = sim.data.qpos
        self.data.mocap_pos[:] = sim.data._data.mocap_pos
        self.data.mocap_quat[:] = sim.data._data.mocap_quat
        # Only these six measured coordinates may move the rigid grip site.
        chain = []
        body = int(self.model.site_bodyid[self.site])
        while body:
            if self.model.body_mocapid[body] >= 0:
                raise ValueError("Movable mocap body on UR5e chain")
            start = int(self.model.body_jntadr[body])
            count = int(self.model.body_jntnum[body])
            chain[0:0] = list(range(start, start + count))
            body = int(self.model.body_parentid[body])
        if (len(chain) != 6 or list(self.model.jnt_qposadr[chain]) != self.indices
                or not np.all(self.model.jnt_type[chain] == mujoco.mjtJoint.mjJNT_HINGE)):
            raise ValueError("World grip site does not depend exclusively on the six ordered arm hinges")

    def position(self, q):
        import mujoco

        self.data.qpos[self.indices] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site].copy()

    def archived_positions(self, q0, delta_q):
        increments = np.vstack((np.zeros(6), np.asarray(delta_q, np.float64)))
        q = np.asarray(q0, np.float64) + np.cumsum(increments, axis=0)
        return np.stack([self.position(v) for v in q])


def qualify(case, rows, arm, first, site):
    """Independent reset and encoder check via the paired nominal active tape with known primitives."""
    from embodiment import ACTIVE_PROBE_BURSTS

    nominal = rows[0]
    if nominal["embodiment"] != "fam_nom":
        raise ValueError("The reset qualification requires the nominal active sibling")
    active = tape(nominal["active"])
    raw = first["observation"]
    q0 = raw["robot0_joint_pos"].copy()
    state0 = motor_state(raw)
    fp = first["environment_reproducibility"]["initial_state_fingerprint"]
    actual_q = [q0]
    actual_p = [raw["robot0_eef_pos"].copy()]
    np.testing.assert_allclose(state0, active["frame0_state"], atol=2e-6, rtol=0)
    np.testing.assert_array_equal(ACTIVE_PROBE_BURSTS.mean(axis=1), active["issued_mean"])
    n = 0
    for burst in ACTIVE_PROBE_BURSTS:
        for command in burst:
            step = arm.step(command)
            n += 1
            if step["worker_step_index"] != n or step["terminated"] or step["truncated"]:
                raise ValueError(f"Active reference ended unexpectedly at {n} for {case}")
        actual_q.append(step["observation"]["robot0_joint_pos"].copy())
        actual_p.append(step["observation"]["robot0_eef_pos"].copy())
    dq_error = np.max(np.abs(np.diff(actual_q, axis=0) - active["delta_joint"]))
    native_error = np.max(np.abs(np.diff(actual_p, axis=0) - active["delta_eef"]))
    # A few float32 units cover numerical differences across x86 hosts.
    if dq_error > 2e-6 or native_error > 1e-6:
        raise ValueError(f"Known-command trajectory disagrees with archived nominal active tape on {case}: "
                         f"max joint error {dq_error} rad, max public TCP error {native_error} m, "
                         f"reset fingerprint match {fp == nominal['physics']}")
    fk_delta = np.diff(site.archived_positions(q0, active["delta_joint"]), axis=0)
    return dict(**case, expected_physics=nominal["physics"], actual_physics=fp,
                reset_fingerprint_matches=fp == nominal["physics"], active_reference_sha256=file_sha(nominal["active"]),
                reset_qualification=("bitwise_physics_and_numeric_active_replay" if fp == nominal["physics"]
                                     else "numeric_frame0_and_active_replay_only"),
                q0_rad=q0.tolist(), frame0_state=state0.tolist(), active_controls=n,
                active_max_abs_dq_error_rad=float(dq_error), active_max_abs_tcp_delta_error_m=float(native_error),
                active_fk_vs_public_delta_error_mm=(1000 * np.linalg.norm(fk_delta - active["delta_eef"], axis=1)).tolist())


def align(issued_mean, world_delta):
    """Unweighted, zero-intercept planar Procrustes with separate excitation diagnostics."""
    a = np.asarray(issued_mean, np.float64)[:, :2]
    b = np.asarray(world_delta, np.float64)[:, :2]
    dot = float(np.sum(a * b))
    cross = float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
    strength = float(np.hypot(dot, cross))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    defined = strength > 16 * np.finfo(float).eps * norm
    return dict(yaw_hat_deg=float(np.degrees(np.arctan2(cross, dot))) if defined else None,
                dot=dot, cross=cross, cmd_xy_l2=float(np.linalg.norm(a)),
                world_xy_path_mm=float(1000 * np.linalg.norm(b, axis=1).sum()),
                aligned_correlation=strength / norm if norm else None)


def scored(row, issued, pos):
    from embodiment import circular_yaw_error_deg, get_embodiment_spec

    delta = np.diff(pos, axis=0)
    result = align(issued, delta)
    # Evaluation labels enter only after alignment.
    spec = get_embodiment_spec(row["embodiment"])
    yaw = result["yaw_hat_deg"]
    return dict(**result, regime=spec.regime, true_yaw_deg=spec.yaw_deg,
                yaw_error_deg=circular_yaw_error_deg(yaw, spec.yaw_deg) if yaw is not None else None,
                world_xyz_m=pos.tolist(), delta_world_m=delta.tolist(), issued_mean=issued.tolist())


def summary(rows):
    groups = defaultdict(list)
    for r in rows:
        for key in ("overall", r["regime"], f'{r["suite"]}/{r["task_id"]}'):
            groups[key].append(r)
        if r["regime"] in ("compositional_ood", "continuous_angle_ood"):
            groups["ood_combined"].append(r)
    out = {}
    for name, group in groups.items():
        errors = [r["yaw_error_deg"] for r in group if r["yaw_error_deg"] is not None]
        out[name] = dict(n=len(group), n_identified=len(errors),
                         yaw_mae_deg=float(np.mean(errors)) if errors else None,
                         min_world_xy_path_mm=min(r["world_xy_path_mm"] for r in group))
    return out


def write_json(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def arguments(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    p.add_argument("--source", type=Path, default=SOURCE, help="Collecting revision's source checkout")
    p.add_argument("--shards", type=int, choices=range(8), nargs="+", default=list(range(8)))
    return p


def main():
    p = arguments(__doc__)
    p.add_argument("--output", type=Path, required=True, help="Fresh JSON result path")
    args = p.parse_args()
    if args.output.exists() or len(set(args.shards)) != len(args.shards):
        p.error("Use a fresh output path and unique shard indices")
    setup(args.source)
    from arm_env import ArmEnv

    report = dict(status="running", collector_commit=COLLECTOR, archive=str(args.archive),
                  method="same_archived_passive_joint_fk", qualifications=[], rows=[])
    write_json(args.output, report)
    try:
        for case, entries in panel(args.source, args.archive, args.shards):
            with ArmEnv("UR5e", case["suite"], case["task_id"]) as arm:
                first = reset(arm, case)
                site = GripSite(arm)
                receipt = qualify(case, entries, arm, first, site)
                report["qualifications"].append(receipt)
                q0 = first["observation"]["robot0_joint_pos"]
                for entry in entries:
                    z = tape(entry["passive"])
                    np.testing.assert_allclose(receipt["frame0_state"], z["frame0_state"], atol=2e-6, rtol=0)
                    positions = site.archived_positions(q0, z["delta_joint"])
                    fit = scored(entry, z["issued_mean"], positions)
                    delta = np.diff(positions, axis=0)
                    # These rotation-invariant diagnostics never enter the fit.
                    xy_norm_error = np.abs(np.linalg.norm(delta[:, :2], axis=1)
                                           - np.linalg.norm(z["delta_eef"][:, :2], axis=1))
                    row = dict(**case, index=entry["index"], embodiment=entry["embodiment"], **fit,
                               archive_npz=str(entry["passive"]), archive_sha256=file_sha(entry["passive"]),
                               delta_joint_rad=z["delta_joint"].tolist(), xy_norm_error_mm=(1000 * xy_norm_error).tolist(),
                               z_error_mm=(1000 * np.abs(delta[:, 2] - z["delta_eef"][:, 2])).tolist())
                    if entry["embodiment"] in ("fam_nom", "fam_grip_inv"):
                        row["zero_yaw_public_xyz_delta_error_mm"] = (1000 * np.linalg.norm(
                            delta - z["delta_eef"], axis=1)).tolist()
                    report["rows"].append(row)
            write_json(args.output, report)
            print(f"Qualified shard {case['shard']}, {len(report['rows'])} archived predictions", flush=True)
        report.update(status="complete", summary=summary(report["rows"]))
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(args.output, report)


if __name__ == "__main__":
    main()
