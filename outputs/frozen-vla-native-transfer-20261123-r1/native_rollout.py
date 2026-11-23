"""Matched frozen-policy trials and a privileged source-trajectory control."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from panel import cases, key, save
from ur5e_motor import load_frozen_pi05, run_episode
from ur5e_sim_worker import UR5eSimulator
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec


class Trace:
    """Record the public physical trajectory and optionally transform the issued action."""
    def __init__(self, sim, matrix=None):
        self.sim = sim
        self.matrix = matrix
        self.chunks = []
        self.first = None
        self.obs = None

    def call(self, op, value):
        if op == "create":
            spec, first = self.sim.call(op, value)
            self.first = first
            self.obs = first["observation"]
            self.chunks.clear()
            return spec, first
        if op != "chunk":
            raise ValueError(op)
        issued = value.copy()
        if self.matrix is not None:
            issued[:, :3] = np.clip(value[:, :3] @ self.matrix, -1.0, 1.0)
        step = self.sim.call(op, issued)
        following = step["observation"]
        n = step["steps"]
        self.chunks.append(dict(cmd=issued[:n].tolist(), before=self.obs["robot0_eef_pos"].tolist(),
                                xyz=np.asarray(step["positions"]).tolist(),
                                q0=self.obs["robot0_eef_quat_xyzw"].tolist(),
                                q1=following["robot0_eef_quat_xyzw"].tolist(),
                                jaw=np.asarray(step["gripper_positions"]).tolist()))
        self.obs = following
        return step


def source_tracking(sim, target, source, horizon):
    """Bounded task-space feedback on a privileged successful source path, no query rewards as inputs."""
    _, first = sim.call("create", dict(target, embodiment="nom"))
    observed = first["observation"]
    scale = np.array(first["model_identity"]["osc_output_max"], dtype=float)
    references = []
    for chunk in source:
        n = len(chunk["cmd"])
        quats = Slerp([0, n], Rotation.from_quat([chunk["q0"], chunk["q1"]]))(np.arange(n)).as_quat()
        references.extend(zip(chunk["cmd"], [chunk["before"], *chunk["xyz"][:-1]], quats))
    held_grip = references[-1][0][-1]
    hold = ([0.0] * 6 + [held_grip], source[-1]["xyz"][-1], source[-1]["q1"])
    native_rewards = []
    tracking = []
    for i in range(horizon):
        nominal, desired, desired_q = references[i] if i < len(references) else hold
        pos_err = np.subtract(desired, observed["robot0_eef_pos"])
        ori_err = (Rotation.from_quat(desired_q) * Rotation.from_quat(observed["robot0_eef_quat_xyzw"]).inv()).as_rotvec()
        issued = np.array(nominal, dtype=np.float32)
        issued[:6] = np.clip(issued[:6] + np.r_[pos_err, ori_err] / scale, -1, 1)
        step = sim.call("chunk", issued[None, :])
        observed = step["observation"]
        native_rewards.extend(step["rewards"])
        tracking.append([float(np.linalg.norm(pos_err)), float(np.linalg.norm(ori_err))])
        if step["terminated"] or step["truncated"]:
            break
    if not (step["terminated"] or step["truncated"]):
        raise ValueError("Tracking failed to reach a native terminal condition")
    return dict(success=bool(step["success"]), terminated=bool(step["terminated"]), truncated=bool(step["truncated"]),
                controls=len(native_rewards), native_rewards=native_rewards, native_return=sum(native_rewards),
                tracking_error_m_rad=tracking, source_controls=len(references),
                reset_physics=first["environment_reproducibility"]["initial_state_fingerprint"],
                prompt=first["language_instruction"])


def check_native(record):
    rewards = record["native_rewards"]
    if (not rewards or len(rewards) != record["controls"] or any(r not in (0, 1) for r in rewards)
            or bool(max(rewards)) != record["success"] or not (record["terminated"] or record["truncated"])):
        raise ValueError("Invalid native reward tape or termination")


def raw(sim, policy, case, matrix=None):
    trace = Trace(sim, matrix)
    record = run_episode(trace, policy, case, get_embodiment_spec("nom"), canonicalize_quat=False)
    check_native(record)
    first = trace.first
    if first["model_identity"]["arm"] != case["arm"] or first["renderer_vendor"] != "NVIDIA Corporation":
        raise ValueError("Unexpected arm or EGL renderer")
    return dict(case=case, record=record, chunks=trace.chunks, model=first["model_identity"],
                full_reset=first["environment_reproducibility"])


def verify_pair(source, target):
    for field in ("scene_fingerprint", "released_row_fingerprint", "bddl_sha256", "perturbation_plan_fingerprint"):
        if source["full_reset"][field] != target["full_reset"][field]:
            raise ValueError(f"Cross-arm setup mismatch in {field}")
    for field in ("prompt", "noise_sha256", "horizon", "action_schema"):
        if source["record"]["reset"][field] != target["record"]["reset"][field]:
            raise ValueError(f"Cross-arm contract mismatch in {field}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--matrices", type=Path)
    args = parser.parse_args()
    mapping = json.loads(args.matrices.read_text()) if args.matrices else None
    policy = load_frozen_pi05()
    from unittest.mock import patch
    # Robosuite validates GL ordinals against CUDA mask strings. The render-only child
    # receives an EGL device resolved from the scheduler allocation before this call.
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        simulator = UR5eSimulator()
    with simulator as sim:
        for target in cases("Kinova3"):
            stem = key(target)
            source = raw(sim, policy, dict(target, arm="Panda"))
            save(args.out / f"{stem}-Panda.json", source)
            identity = raw(sim, policy, target)
            verify_pair(source, identity)
            save(args.out / f"{stem}-Kinova3.json", identity)
            if source["record"]["success"]:
                tracking = source_tracking(sim, target, source["chunks"], source["record"]["reset"]["horizon"])
                check_native(tracking)
                if tracking["reset_physics"] != identity["record"]["reset"]["physics"]:
                    raise ValueError("Mismatched tracker reset")
            else:
                tracking = dict(skipped=True, reason="No successful source reference on this query")
            save(args.out / f"{stem}-tracking.json", tracking)
            if mapping is not None:
                row = mapping[f"Kinova3/{target['suite']}/{target['task_id']}"]
                support = row["support"]
                if (support["arm"] != target["arm"] or support["task"] != f"{target['suite']}/{target['task_id']}"
                        or support["start"] != 11 or support["controls"] != 15
                        or support["physics"] == identity["record"]["reset"]["physics"]):
                    raise ValueError("Invalid separation of support and query")
                for field in ("matrix", "planar_matrix"):
                    matrix = np.array(row[field], dtype=np.float32)
                    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                        raise ValueError(field)
                    adapted = raw(sim, policy, target, matrix)
                    for name in ("physics", "public", "noise_sha256"):
                        if adapted["record"]["reset"][name] != identity["record"]["reset"][name]:
                            raise ValueError(f"Target reset mismatch in {name}")
                    if adapted["record"]["opening_proposal_sha256"] != identity["record"]["opening_proposal_sha256"]:
                        raise ValueError("Target opening frozen action changed before adaptation")
                    adapted["calibration"] = row
                    adapted["matrices_sha256"] = hashlib.sha256(args.matrices.read_bytes()).hexdigest()
                    save(args.out / f"{stem}-{field}.json", adapted)
            print(f"{stem} source={source['record']['success']} target={identity['record']['success']} "
                  f"tracker={tracking.get('success')}", flush=True)


if __name__ == "__main__":
    main()
