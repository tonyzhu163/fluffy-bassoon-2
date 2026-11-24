"""Locked inputs, source-only cut, and release identities."""
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "frozen-transfer-demonstration-20261124-r1"
PROGRESS = HERE.parent / "frozen-transfer-progress-20261124-r1"
sys.path.insert(0, str(OLD))
sys.path.insert(1, str(PROGRESS))
from contract import IDENTITY, MANIFEST_SHA, RELEASE_SHA, SHARED, WEIGHT_SHA, cases, checked, digest, require, save
from readback import fp, native_commands
from study import sources as drawer_sources

LOCK = json.loads((HERE / "registration.json").read_bytes())
TASKS = LOCK["tasks"]
ROWS = tuple(range(20, 26))
MODES = ("tuned", "switch", "source", "corrected")


def query(task, row):
    return dict(arm="Kinova3", suite=task["suite"], task_id=task["task_id"],
                variant="BASE", init_state_id=row, seed=task["seed_base"] + row)


def source_tapes():
    drawer, _ = drawer_sources()
    tapes = {"drawer": drawer["IIWA34"]}
    task = TASKS["moka"]
    data = checked((HERE.parent / task["archive"]).read_bytes(), task["archive_sha256"])
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        src = json.loads(checked(archive.extractfile(task["member"]).read(), task["member_sha256"]))
    rec = src["native"]
    require(src["case"]["query"] == task["source_query"] and src["mode"] == "raw" and not src["fixture"]
            and src["origin"]["pin"] == task["source_pin"] and src["weights_sha256"] == WEIGHT_SHA
            and rec["success"] and rec["terminated"] and rec["native_return"] == 1 and rec["inferred"] == IDENTITY
            and rec["released_prompt"] == task["prompt"] and src["model"]["gripper"] == "PandaGripper",
            "moka donor identity")
    values = [a for chunk in src["chunks"] for a in chunk["sent"]]
    actions = np.asarray(values, np.float32)
    require(actions.shape == (rec["controls"], 7) and np.array_equal(actions, values)
            and np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1, "moka issued controls")
    close = int(np.flatnonzero(actions[:, 6] > 0)[0])
    require(close == task["first_positive"] and actions[0, 6] == task["initial_grip"] and
            (actions[:close, 6] < 0).all() and len(actions) == task["source_controls"], "moka cut")
    tapes["moka"] = actions[:5 * (close // 5)].copy()
    for key, tape in tapes.items():
        require(len(tape) == TASKS[key]["cut"] and digest(tape.tobytes()) == TASKS[key]["prefix_bytes_sha256"],
                f"{key} frozen prefix")
    return tapes


def supports(check=False):
    all_cases = cases("Kinova3", verify_tapes=check)
    return {key: next(x["support"] for x in all_cases if
                      (x["query"]["suite"], x["query"]["task_id"]) == (t["suite"], t["task_id"]))
            for key, t in TASKS.items()}


def assets():
    import torch

    checked((HERE / "PLAN.md").read_bytes(), LOCK["plan_sha256"])
    for name, sha in LOCK["predecessor_files"].items():
        checked((OLD / name).read_bytes(), sha)
    old_cpu = SHARED / OLD.name / "cpu02"
    require((old_cpu / "exit.txt").read_text().strip() == "0", "predecessor Linux dry-run")
    checked((old_cpu / "qualified.json").read_bytes(), LOCK["predecessor_files"]["evidence/cpu02/qualified.json"])
    banks = json.loads(checked((PROGRESS / "evidence/release-banks.json").read_bytes(), LOCK["release_contract_sha256"]))
    release = Path(os.environ["CROSSARM_RELEASE"])
    manifest = json.loads(checked((release / "assets/task_manifest.json").read_bytes(), MANIFEST_SHA))
    hasher = runpy.run_path(str(release / "libero_remote/reproducibility.py"))["numerical_state_fingerprint"]
    rows = {}
    for key, t in TASKS.items():
        b = t["bank"]
        require(b in banks["tasks"], "registered bank")
        entry, = (x for x in manifest["tasks"] if (x["suite"], x["base_task_id"], x["case_id"])
                  == (t["suite"], t["task_id"], "BASE"))
        require(entry["max_steps"] == b["native_horizon"] and entry["bddl_sha256"] == b["bddl_sha256"], "manifest")
        checked((release / "assets" / entry["bddl_path"]).read_bytes(), b["bddl_sha256"])
        installed = Path(b["state_bank"])
        released = release / "simulator-source/libero/libero/init_files" / t["suite"] / installed.name
        checked(installed.read_bytes(), b["bank_sha256"])
        checked(released.read_bytes(), b["bank_sha256"])
        states = torch.load(released, map_location="cpu", weights_only=False)
        require(list(states.shape) == b["shape"] and hasher(states[31]) == b["row31_fingerprint"], "bank anchor")
        rows[key] = {str(i): hasher(states[i]) for i in ROWS}
        require(len(set(rows[key].values())) == len(ROWS), "repeated release row")
    return rows
