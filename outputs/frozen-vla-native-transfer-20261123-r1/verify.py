"""Independent compact readback for the launched single-GPU native transfer experiment."""
import hashlib
import json
import sys
from pathlib import Path


def read(p):
    return json.loads(p.read_bytes())


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def native(record):
    rewards = record["native_rewards"]
    require(len(rewards) == record["controls"] and all(r in (0, 1) for r in rewards), "native reward tape")
    require(sum(rewards) == record["native_return"] and bool(max(rewards)) == record["success"], "terminal reward")
    require(record["terminated"] or record["truncated"], "terminal flags")
    return {"success": record["success"], "controls": record["controls"], "return": record["native_return"]}


def main(p):
    probe = read(p / "gpu.json")
    require(probe["status"] == "passed" and len(probe["devices"]) == 1, "single GPU admission")
    rows, hashes = [], {}
    for suite, tid in (("libero_goal", 3), ("libero_10", 8), ("libero_goal", 6)):
        for layout in (14, 15):
            key = f"{suite}-{tid}-{layout}"
            runs = {}
            for label in ("Panda", "Kinova3", "matrix", "planar_matrix", "tracking"):
                path = p / f"{key}-{label}.json"
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                runs[label] = read(path)
            raw = runs["Kinova3"]
            controls = {}
            for method, obj in runs.items():
                if method == "tracking":
                    if obj.get("skipped"):
                        require(not runs["Panda"]["record"]["success"], "tracker skip without source failure")
                        controls[method] = None
                    else:
                        controls[method] = native(obj)
                        require(runs["Panda"]["record"]["success"], "tracker used failed source")
                        require(obj["reset_physics"] == raw["record"]["reset"]["physics"], "tracking reset")
                    continue
                rec = obj["record"]
                controls[method] = native(rec)
                arm = "Panda" if method == "Panda" else "Kinova3"
                case = obj["case"]
                require((case["arm"], case["suite"], case["task_id"], case["init_state_id"], case["variant"]) ==
                        (arm, suite, tid, layout, "BASE"), "case ID")
                require(obj["model"]["arm"] == arm and obj["model"]["joint_count"] == 7 and
                        obj["model"]["controller"] == "OSC_POSE" and obj["model"]["gripper"] == "PandaGripper", "physical schema")
                require(all(obj["full_reset"][k] == raw["full_reset"][k] for k in
                            ("bddl_sha256", "released_row_fingerprint", "scene_fingerprint", "perturbation_plan_fingerprint")), "scene")
                require(all(rec["reset"][k] == raw["record"]["reset"][k] for k in
                            ("horizon", "noise_sha256", "prompt", "action_schema")), "pair inputs")
                require(sum(len(c["cmd"]) for c in obj["chunks"]) == rec["controls"], "native action tape")
                if method in ("matrix", "planar_matrix"):
                    require(all(rec["reset"][k] == raw["record"]["reset"][k] for k in
                                ("physics", "public", "motor_frame0")), "target observations")
                    require(rec["opening_proposal_sha256"] == raw["record"]["opening_proposal_sha256"], "frozen opening")
                    sup = obj["calibration"]["support"]
                    require(sup["start"] == 11 and sup["controls"] == 15 and
                            sup["physics"] != rec["reset"]["physics"], "target support")
            require(runs["Panda"]["model"]["xml_sha256"] != raw["model"]["xml_sha256"], "distinct arms")
            rows.append(dict(suite=suite, task_id=tid, init_state=layout, results=controls,
                paired_opening_sha256={m: runs[m]["record"]["opening_proposal_sha256"]
                                       for m in ("Kinova3", "matrix", "planar_matrix")},
                common_target_motor_input=raw["record"]["reset"]["motor_frame0"],
                common_target_noise_sha256=raw["record"]["reset"]["noise_sha256"]))
    totals = {m: sum(r["results"][m]["success"] for r in rows if r["results"][m] is not None)
              for m in rows[0]["results"]}
    for k, pair in read(p / "summary.json")["contrasts"].items():
        first, second = k.split("_minus_")
        diff = [int(r["results"][first]["success"]) - int(r["results"][second]["success"])
                for r in rows if r["results"][first] is not None and r["results"][second] is not None]
        require((pair["n"], pair["gains"], pair["losses"]) == (len(diff), diff.count(1), diff.count(-1)),
                "runner summary reconciliation")
    device = probe["devices"][0]
    return dict(launch=(p / "launch.txt").read_text().strip(), exit=(p / "exit.txt").read_text().strip(),
                gpu=device["cuda_name"], gpu_pci=device["pci_bus_id"], rows=rows, successes=totals, sha256=hashes)


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=2, allow_nan=False))
