"""Compare archived Kinova3 policy state with source arms. CPU, archived records only."""
import glob
import json
from pathlib import Path
import re

import numpy as np

OUT = Path(__file__).resolve().parents[1]
VIEW = OUT / "frozen-vla-source-view-20261123-r1/evidence/cpu01"
POSE = OUT / "frozen-transfer-pose-20261124-r1/evidence/replay"
CELLS = OUT / "xemb-stage2c-cpu-20261121-r1/evidence"


def rotvec(quat):
    """The packer's rotation vector (ur5e_motor.to_openpi_observation, raw sign)."""
    quat = np.asarray(quat, dtype=np.float64)
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    return quat[:3] * 2.0 * np.arccos(w) / den if den > 1e-10 else np.zeros(3)


def quat_of(matrix):
    """xyzw quaternion of a rotation matrix with x >= 0 (the source branch)."""
    m = np.asarray(matrix, dtype=np.float64)
    k = np.array([
        [m[0, 0] - m[1, 1] - m[2, 2], 0, 0, 0],
        [m[0, 1] + m[1, 0], m[1, 1] - m[0, 0] - m[2, 2], 0, 0],
        [m[0, 2] + m[2, 0], m[1, 2] + m[2, 1], m[2, 2] - m[0, 0] - m[1, 1], 0],
        [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], m[0, 0] + m[1, 1] + m[2, 2]],
    ]) / 3.0
    _, vectors = np.linalg.eigh(k, UPLO="L")
    q = vectors[:, -1]
    return q if q[0] >= 0 else -q


def matrix_of(quat):
    x, y, z, w = np.asarray(quat, dtype=np.float64) / np.linalg.norm(quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


def span(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(min=values.min(axis=0).round(4).tolist(), max=values.max(axis=0).round(4).tolist(),
                mean=values.mean(axis=0).round(4).tolist())


def main():
    result = {}

    # 1. Kinova3 public state, the only archived frames that hold the reported quaternion and finger readings.
    frames = []
    for path in sorted(VIEW.glob("s1*-t*.npz")):
        z = np.load(path)
        layout, control = map(int, re.match(r"s(\d+)-t(\d+)", path.name).groups())
        quat = z["robot0_eef_quat_xyzw"]
        replay = np.load(POSE / f"libero_goal-3-{layout}-Kinova3.npz")
        frames.append(dict(
            layout=layout, control=control, eef_pos=z["robot0_eef_pos"].round(4).tolist(),
            quat_xyzw=quat.round(4).tolist(), gripper_qpos=z["robot0_gripper_qpos"].round(4).tolist(),
            rotvec_raw_sign=rotvec(quat).round(4).tolist(), rotvec_source_sign=rotvec(-quat if quat[0] < 0 else quat).round(4).tolist(),
            # Does the reported quaternion describe the replay's grip-site matrix R?
            angle_to_replay_R_deg=round(angle(matrix_of(quat), replay["R"][control]), 3) if control < len(replay["R"]) else None,
            pos_to_replay_m=round(float(np.linalg.norm(z["robot0_eef_pos"] - replay["public_p"][control])), 5)
            if control < len(replay["p"]) else None,
        ))
    raw = np.array([f["rotvec_raw_sign"] for f in frames])
    # robot0_eef_quat is the hand link; the replay matrix R is the grip site. Both arms carry the same
    # PandaGripper, so hand = site @ C with one fixed C. Estimate C on Kinova3 and check that it is constant.
    offsets = []
    for path in sorted(VIEW.glob("s1*-t*.npz")):
        z = np.load(path)
        layout, control = map(int, re.match(r"s(\d+)-t(\d+)", path.name).groups())
        offsets.append(np.load(POSE / f"libero_goal-3-{layout}-Kinova3.npz")["R"][control].T @ matrix_of(z["robot0_eef_quat_xyzw"]))
    u, _, vt = np.linalg.svd(np.mean(offsets, axis=0))
    fixed = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    result["kinova3_public_frames"] = dict(
        source=str(VIEW.relative_to(OUT)), frames=frames,
        quat_x_negative=int(sum(f["quat_xyzw"][0] < 0 for f in frames)), count=len(frames),
        rotvec_raw_sign=span(raw), rotvec_source_sign=span([f["rotvec_source_sign"] for f in frames]),
        gripper_qpos=span([f["gripper_qpos"] for f in frames]),
        hand_from_site=dict(matrix=fixed.round(4).tolist(), rotvec=rotvec(quat_of(fixed)).round(4).tolist(),
                            max_frame_deviation_deg=round(max(angle(fixed, c) for c in offsets), 3)),
        state_change_l2_raw_vs_source=span(np.linalg.norm(raw - [f["rotvec_source_sign"] for f in frames], axis=1)[:, None]),
    )

    # 2. Kinova3 against Panda along archived native rollouts (grip-site position and matrix, both arms).
    pairs = []
    for path in sorted(POSE.glob("*-Kinova3.npz")):
        name = path.name[:-len("-Kinova3.npz")]
        k, p = np.load(path), np.load(POSE / f"{name}-Panda.npz")
        hand = {arm: [quat_of(r @ fixed) for r in z["R"]] for arm, z in (("Kinova3", k), ("Panda", p))}
        rv = {arm: np.array([rotvec(q) for q in hand[arm]]) for arm in hand}
        pairs.append(dict(
            start=name, controls=dict(Kinova3=len(k["p"]), Panda=len(p["p"])),
            native_reward=dict(Kinova3=float(k["native_reward"].sum()), Panda=float(p["native_reward"].sum())),
            reset_pos_kinova3_minus_panda_mm=((k["public_p"][0] - p["public_p"][0]) * 1000).round(2).tolist(),
            reset_rotation_difference_deg=round(angle(k["R"][0], p["R"][0]), 2),
            reset_rotvec_source_sign=dict(Kinova3=rv["Kinova3"][0].round(4).tolist(), Panda=rv["Panda"][0].round(4).tolist()),
            position_range=dict(Kinova3=span(k["public_p"]), Panda=span(p["public_p"])),
            rotvec_source_sign_range=dict(Kinova3=span(rv["Kinova3"]), Panda=span(rv["Panda"])),
            # The same Kinova3 rotations packed with the sign Kinova3 reported on all ten archived frames (x < 0).
            # Inferred for the controls between those frames.
            kinova3_rotvec_reported_sign_range=span([rotvec(-q) for q in hand["Kinova3"]]),
            hand_quat_x_min=dict(Kinova3=round(float(min(q[0] for q in hand["Kinova3"])), 4),
                                 Panda=round(float(min(q[0] for q in hand["Panda"])), 4)),
        ))
    result["kinova3_vs_panda_rollouts"] = dict(source=str(POSE.relative_to(OUT)), starts=pairs)

    # 3. Reset tool pose and jaw opening per arm on one matched scene (libero_goal/1 layout 13).
    cells = {}
    for path in sorted(CELLS.glob("*-BASE/cell.json")):
        cell = json.loads(path.read_text())
        tool = cell["support"]["tool"]
        cell["support"]["reset"]["alignment"] = {key: value for key, value in cell["support"]["reset"]["alignment"].items()
                                                 if key.endswith(("_m", "_rad"))}
        cells[cell["arm"]] = dict(
            gripper=cell["support"]["model"]["gripper"], tcp_site=cell["support"]["model"]["tcp_site"],
            tcp_xyz=np.round(tool["tcp_xyz"], 4).tolist(), jaw_opening=round(tool["jaw_opening"], 5),
            settle=cell["support"]["reset"]["alignment"],
            jaw_opening_over_15_controls=span([c["jaw_opening"] for c in cell["controls"]]),
            matrix=tool["tcp_matrix"],
        )
    for arm, cell in cells.items():
        if arm != "Kinova3" and "Kinova3" in cells:
            cell["kinova3_minus_this_mm"] = ((np.array(cells["Kinova3"]["tcp_xyz"]) - cell["tcp_xyz"]) * 1000).round(2).tolist()
            cell["rotation_difference_to_kinova3_deg"] = round(angle(np.array(cells["Kinova3"]["matrix"]), np.array(cell["matrix"])), 2)
    for cell in cells.values():
        del cell["matrix"]
    result["reset_tool_by_arm"] = dict(source=str(CELLS.relative_to(OUT)), arms=cells)

    # Fallback constant for `state-dropped` when the checkpoint statistics cannot be read.
    panda = [np.load(POSE / f"libero_goal-3-{layout}-Panda.npz") for layout in (14, 15)]
    reference = np.r_[
        np.concatenate([z["public_p"] for z in panda]).mean(axis=0),
        np.concatenate([[rotvec(quat_of(r @ fixed)) for r in z["R"]] for z in panda]).mean(axis=0),
        np.mean([f["gripper_qpos"] for f in frames], axis=0),
    ]
    (Path(__file__).parent / "source_reference.json").write_text(json.dumps(dict(
        state_mean=reference.round(6).tolist(),
        position_and_rotation="mean over the 406 controls of the archived Panda native successes on libero_goal/3 layouts 14 and 15, "
                              "rotation as the packer's vector of the hand quaternion on the source branch",
        fingers="mean of the ten archived Kinova3 frames; both arms carry PandaGripper and no Panda finger reading is archived",
    ), indent=1) + "\n")
    result["archived_source_reference"] = reference.round(4).tolist()

    result["not_archived"] = (
        "No record in the repository holds a source arm's reported quaternion or finger qpos as numbers; "
        "source records keep SHA-256 digests of them. The source sign is taken from the checkpoint statistic in "
        "frozen-vla-factory-20261123-r1/state/RESULT.md (minimum rotation-vector X +0.353 rad) and FINDINGS line 117."
    )
    (Path(__file__).parent / "measurement.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


if __name__ == "__main__":
    r = main()
    k = r["kinova3_public_frames"]
    print("Kinova3 frames", k["count"], "quat x<0:", k["quat_x_negative"], "hand_from_site", k["hand_from_site"])
    print(" raw", k["rotvec_raw_sign"], "\n src", k["rotvec_source_sign"], "\n grip", k["gripper_qpos"], k["state_change_l2_raw_vs_source"])
    print(" replay check", max(f["pos_to_replay_m"] for f in k["frames"]))
    for s in r["kinova3_vs_panda_rollouts"]["starts"]:
        print(s["start"], s["native_reward"], s["reset_pos_kinova3_minus_panda_mm"], s["reset_rotation_difference_deg"],
              s["reset_rotvec_source_sign"], s["hand_quat_x_min"])
        for arm in ("Kinova3", "Panda"):
            print("   rv ", arm, s["rotvec_source_sign_range"][arm]["min"], s["rotvec_source_sign_range"][arm]["max"])
        print("   rv reported sign", s["kinova3_rotvec_reported_sign_range"]["min"], s["kinova3_rotvec_reported_sign_range"]["max"])
    for arm, c in r["reset_tool_by_arm"]["arms"].items():
        print(arm, c)
