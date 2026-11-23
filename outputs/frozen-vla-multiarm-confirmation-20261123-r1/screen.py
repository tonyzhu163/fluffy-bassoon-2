"""Layout-14 contract and checks shared by the CPU qualifier and live native runner."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from native_rollout import check_native
from panel import SUITES, key, save

ARMS = ("IIWA", "UR5e")
TASKS = (("libero_10", 5), ("libero_10", 8), ("libero_goal", 3), ("libero_goal", 6),
         ("libero_object", 1), ("libero_object", 8), ("libero_spatial", 0), ("libero_spatial", 8))
METHODS = {"raw": None, "planar": "planar_matrix", "full": "matrix"}
MATRIX_SHA = "20a9d6800e10404ee23a542c70a3ba1fb077daeccd56cf7fea93555c7add64cf"
SOURCE_SHA = "acf6150ecd7937733d77516cabfaca4a7648531009e3d7c3264c025170279f73"
HISTORY_SHA = "90acca7d4b5c872626000a55a06adf9f1758fa3086ec94c54b1d7ad1cfb5dc17"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def locked_json(path, digest):
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == digest, f"Changed input {path}")
    return json.loads(payload)


def cases(arm):
    require(arm in ARMS, arm)
    for suite, task_id in TASKS:
        seed = 41800000 + 10000 * SUITES.index(suite) + 100 * task_id + 14
        yield dict(arm=arm, suite=suite, task_id=task_id, init_state_id=14, variant="BASE", seed=seed)


def historical(arm):
    rows = locked_json(Path(os.environ["NATIVE_HISTORICAL"]), HISTORY_SHA)["rows"]
    rows = [r for r in rows if r["panel"] == arm.lower() + "-pi05" and r["layout"] == 13]
    require(len(rows) == 8 and {(r["suite"], r["task"]) for r in rows} == set(TASKS), "layout-13 coverage")
    return {(r["suite"], r["task"]): r["attempts"]["raw"]["success"] for r in rows}


def calibrations(arm):
    directory = Path(os.environ["NATIVE_MOTION_DIR"])
    locked_json(directory / "source-fit.json", SOURCE_SHA)
    mapping = locked_json(directory / "matrices.json", MATRIX_SHA)
    rows = {}
    for case in cases(arm):
        suite, task = case["suite"], case["task_id"]
        row = mapping[f"{arm}/{suite}/{task}"]
        sup = row["support"]
        expected_seed = 31800000 + 10000 * SUITES.index(suite) + 100 * task + 11
        require((sup["arm"], sup["task"], sup["start"], sup["controls"], sup["seed"]) ==
                (arm, f"{suite}/{task}", 11, 15, expected_seed), "support identity or budget")
        require(row["source_fit_sha256"] == SOURCE_SHA, "source prior")
        require(hashlib.sha256(Path(sup["path"]).read_bytes()).hexdigest() == sup["sha256"], "support tape")
        for field in ("planar_matrix", "matrix"):
            matrix = np.asarray(row[field], dtype=np.float32)
            require(matrix.shape == (3, 3) and np.isfinite(matrix).all(), field)
        rows[key(case)] = row
    return rows


class JointChecked:
    """Validate the native model's physical joint width at reset and on each control."""
    def __init__(self, sim, arm):
        self.sim = sim
        self.arm = arm
        self.width = None

    def call(self, op, value):
        result = self.sim.call(op, value)
        if op == "create":
            spec, first = result
            model = first["model_identity"]
            require(spec["embodiment"] == "nom" and spec["variant"] == "BASE" and
                    spec["arm"] == model["arm"] == self.arm, "native arm or command interface")
            self.width = len(model["joint_names"])
            require(model["joint_count"] == self.width == len(model["joint_links"]), "compiled joints")
            require(first["action_schema"]["shape"] == [7], "OSC command width")
            obs = first["observation"]
        else:
            require(op == "chunk", op)
            joints = np.asarray(result["joint_positions"])
            require(joints.shape == (result["steps"], self.width) and np.isfinite(joints).all(),
                    "per-control joint tape width")
            obs = result["observation"]
        for field in ("robot0_joint_pos", "robot0_joint_vel"):
            require(obs[field].shape == (self.width,) and np.isfinite(obs[field]).all(), field)
        return result


def check_case(obj, case, *, terminal=True):
    require(obj["case"] == case, "unexpected query")
    model, full, rec = obj["model"], obj["full_reset"], obj["record"]
    reset = rec["reset"]
    require((model["arm"], model["controller"], model["gripper"], model["control_freq"]) ==
            (case["arm"], "OSC_POSE", "PandaGripper", 20), "native physical interface")
    require(model["joint_count"] == reset["joint_count"] == len(model["joint_names"]) and
            reset["arm"] == case["arm"] and reset["worker_step_index"] == 0, "native joint identity")
    require(all(full[k] == case[k] for k in ("suite", "task_id", "init_state_id", "seed")) and
            full["perturbation_plan"]["case_id"] == "BASE" and
            reset["physics"] == full["initial_state_fingerprint"], "native reset identity")
    require(rec["inferred"] == {"yaw_deg": 0., "xy_scale": 1., "gripper_sign": 1}, "nominal command interface")
    require(sum(len(c["cmd"]) for c in obj["chunks"]) == rec["controls"], "command tape length")
    if terminal:
        check_native(rec)
        require(reset["renderer_vendor"] == "NVIDIA Corporation" and sum(rec["native_rewards"]) == rec["native_return"],
                "NVIDIA native return")
        require(0 < rec["controls"] <= reset["horizon"] and
                (rec["success"] or rec["controls"] == reset["horizon"]), "native termination or horizon")


def check_pair(raw, obj, row, method):
    require(method in ("planar", "full"), method)
    require(raw["case"] == obj["case"] and raw["full_reset"] == obj["full_reset"] and raw["model"] == obj["model"],
            "within-arm physical reset")
    rec, ref = obj["record"], raw["record"]
    require(rec["reset"] == ref["reset"] and rec["opening_proposal_sha256"] == ref["opening_proposal_sha256"],
            "within-arm raw cameras, motor input, prompt, noise or opening")
    require(row["support"]["physics"] != rec["reset"]["physics"], "support/query physics overlap")
    # Only the opening is a shared proposal; later closed-loop proposals may diverge.
    n = min(len(raw["chunks"][0]["cmd"]), len(obj["chunks"][0]["cmd"]))
    expected = np.array(raw["chunks"][0]["cmd"][:n], dtype=np.float32)
    expected[:, :3] = np.clip(expected[:, :3] @ np.asarray(row[METHODS[method]], dtype=np.float32), -1, 1)
    issued = np.array(obj["chunks"][0]["cmd"][:n], dtype=np.float32)
    require(np.allclose(issued[:, :3], expected[:, :3], rtol=1e-6, atol=1e-7) and
            np.array_equal(issued[:, 3:], expected[:, 3:]), "opening translation, rotation or gripper")
