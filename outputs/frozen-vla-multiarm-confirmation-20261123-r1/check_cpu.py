"""Read back real simulator receipts against the deployed screen and the pinned shared runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from native_rollout import verify_pair
from screen import ARMS, MATRIX_SHA, SOURCE_SHA, METHODS, calibrations, cases, check_case, check_pair, historical, key, require, save


def read(path):
    return json.loads(path.read_bytes())


def main(out, dest, launch_pin):
    require((out / "exit.txt").read_text().strip() == "0", "CPU exit")
    cpu_pin, host, started = (out / "launch.txt").read_text().split()
    summaries = {a: read(out / f"{a}-qualified.json") for a in ARMS}
    require(sorted(i for r in summaries.values() for i in r["cpus"]) == [38, 39], "CPU core budget")
    histories = {a: historical(a) for a in ARMS}
    calibrations_by_arm = {a: calibrations(a) for a in ARMS}
    tasks, files = [], {}
    for pair in zip(*(cases(a) for a in ARMS), strict=True):
        by_arm = {}
        receipts = {}
        for case in pair:
            arm = case["arm"]
            path = out / f"{arm}-{key(case)}.json"
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            obj = read(path)
            paired = (case["suite"], case["task_id"]) in (("libero_10", 8), ("libero_goal", 6))
            require(set(obj) == (set(METHODS) if paired else {"raw"}), "CPU method coverage")
            widths = set()
            row = calibrations_by_arm[arm][key(case)]
            for method, result in obj.items():
                check_case(result, case, terminal=False)
                rec = result["record"]
                require(rec["controls"] == len(rec["native_rewards"]) == (10 if paired else 1), "CPU tape length")
                require("llvmpipe" in rec["reset"]["renderer"].lower(), "software rendering")
                require(row["support"]["physics"] != rec["reset"]["physics"], "support/query overlap")
                if method != "raw":
                    check_pair(obj["raw"], result, row, method)
                widths.add(result["model"]["joint_count"])
            require(len(widths) == 1, "within-arm dimension mismatch")
            r = obj["raw"]
            receipts[arm] = dict(model=r["model"]["model_class"], joint_count=widths.pop(),
                    xml_sha256=r["model"]["xml_sha256"], support_sha256=row["support"]["sha256"],
                    historical_layout13_raw=histories[arm][(case["suite"], case["task_id"])],
                    resets=len(obj), controls=sum(v["record"]["controls"] for v in obj.values()))
            by_arm[arm] = r
        # Across robots check only fields actually shared by the design.
        verify_pair(by_arm["IIWA"], by_arm["UR5e"])
        require(receipts["IIWA"]["xml_sha256"] != receipts["UR5e"]["xml_sha256"], "same robot model")
        tasks.append(dict(suite=pair[0]["suite"], task_id=pair[0]["task_id"], layout=14, seed=pair[0]["seed"],
                          arms=receipts, shared_scene=by_arm["IIWA"]["full_reset"]["scene_fingerprint"],
                          shared_noise=by_arm["IIWA"]["record"]["reset"]["noise_sha256"]))
    save(dest, dict(status="passed", scope="Baige CPU simulator wiring only", cpu_pin=cpu_pin,
                    launch_pin=launch_pin, host=host, started=started,
                    cores={a: s["cpus"] for a, s in summaries.items()},
                    source_fit_sha256=SOURCE_SHA, matrices_sha256=MATRIX_SHA, tasks=tasks, evidence_sha256=files))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cpu_dir", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("launch_pin")
    args = p.parse_args()
    actual = subprocess.check_output(["git", "-C", os.environ["MULTIARM_LAUNCH_CHECKOUT"], "rev-parse", "HEAD"], text=True)
    require(actual.strip() == args.launch_pin, "deployed launch checkout mismatch")
    main(args.cpu_dir, args.output, args.launch_pin)
