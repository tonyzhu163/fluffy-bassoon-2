"""Report per-layout counts, every waypoint, and actual/reenacted drawer outcomes."""
import argparse
import csv
import gzip
import json
from pathlib import Path


def read(path):
    if not path.exists():
        path = path.with_suffix(path.suffix + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


def flags(fit, phase):
    kinds = {c["kind"] for c in fit["contacts"]}
    ok = fit["pose_ok"]
    arm_clear = ok and not kinds.intersection(("arm_scene", "arm_self_or_base"))
    strict = ok and kinds == (set() if phase == "approach" else {"intended_finger"})
    return ok, arm_clear, strict


def write_csv(out, name, rows):
    with (out / name).open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def tables(inputs, output):
    waypoints, cases, replay = [], [], []
    for task in ("goal3", "long8", "goal6"):
        for layout in (14, 15):
            stem = f"{task}-{layout}"
            d = read(inputs / (stem+".json"))
            checked = read(inputs / (stem+"-readback.json"))
            relaxed = read(inputs / (stem+"-tilt.json"))
            by_key = {(r["arm"],r["donor"],r["body"],r["phase"],r["mode"]): r for r in checked["rows"]}
            by_tilt = {(r["donor"],r["body"],r["phase"]): r["fits"] for r in relaxed}
            case_rows = []
            for donor in d["demonstrations"]:
                for w in donor["waypoints"]:
                    for arm in ("Panda", "Kinova3"):
                        for mode, fits in w["search"][arm].items():
                            phase = w["phase"]
                            result = by_key[arm,donor["demo"],w["body"],phase,mode]
                            assert result["candidates_verified"] == len(fits)
                            outcomes = [flags(s, phase) for s in fits]
                            poses, clear, strict = zip(*outcomes)
                            qualified_contacts = sorted({c["body1"]+" / "+c["body2"]
                                for s in fits if s["pose_ok"] for c in s["contacts"]
                                if c["kind"] in ("arm_scene","arm_self_or_base")})
                            tilts = by_tilt.get((donor["demo"],w["body"],phase), []) if (arm,mode)==("Kinova3","recorded") else []
                            tilts = [s for s in tilts if s["position_ok"] and s["min_limited_margin_rad"] >= -1e-10]
                            best = min(tilts, key=lambda s: s["angle_deg"], default=None)
                            row = dict(case=stem, donor=donor["demo"], frame=w["frame"], body=w["body"], phase=phase,
                                       arm=arm, orientation=mode, starts=len(fits), solved=sum(poses),
                                       no_arm_collision=sum(clear), contact_compatible=sum(strict),
                                       reset_seed_solved=int(poses[0]), reset_seed_compatible=int(strict[0]),
                                       solved_arm_contact_pairs=" | ".join(qualified_contacts),
                                       min_position_bound_mm=1000*result["bound"]["lower_bound_position_error_m"]
                                       if "bound" in result else "",
                                       best_position_preserving_tilt_deg=best["angle_deg"] if best else "",
                                       tilted_contacts=" | ".join(sorted({c["kind"] for c in best["contacts"]})) if best else "")
                            case_rows.append(row)
            waypoints.extend(case_rows)
            for arm in ("Panda","Kinova3"):
                for mode in ("recorded","half_turn"):
                    subset = [r for r in case_rows if (r["arm"],r["orientation"])==(arm,mode)]
                    assert len(subset) == sum(len(x["waypoints"]) for x in d["demonstrations"])
                    cases.append(dict(case=stem, arm=arm, orientation=mode, sampled_waypoints=len(subset),
                                      pose_witness=sum(r["solved"]>0 for r in subset),
                                      pose_and_no_arm_collision=sum(r["no_arm_collision"]>0 for r in subset),
                                      contact_compatible=sum(r["contact_compatible"]>0 for r in subset),
                                      reset_seed_contact_compatible=sum(r["reset_seed_compatible"] for r in subset),
                                      has_both_colliding_and_arm_clear_ik=sum(
                                          0 < r["no_arm_collision"] < r["solved"] for r in subset),
                                      excluded_by_bound=sum(r["min_position_bound_mm"]!="" and
                                                            r["min_position_bound_mm"]>1 for r in subset)))
    for layout in (14,15):
        for arm in ("Panda","Kinova3"):
            r = read(inputs / f"raw-drawer-{layout}-{arm}.json")
            rows = r["frames"]
            first_drift = next((f["step"] for f in rows if f["tcp_replay_error_m"]>.001 or
                                f["finger_replay_max_error_m"]>.001 or f.get("hand_replay_error_rad",0)>.01), None)
            obj = dict(layout=layout, arm=arm, native_success=int(r["native_success"]),
                       native_controls=r["native_controls"], replay_success=int(r["success"]), replay_controls=r["controls"],
                       max_tcp_drift_mm=1000*r["max_tcp_delta_m"], max_finger_drift_mm=1000*r["max_jaw_coord_delta_m"],
                       max_boundary_hand_drift_rad=r["max_boundary_angle_rad"],
                       first_exceeds_reset_tolerance=first_drift or "", reward_tape_exact=r["reward_tape_equal"],
                       max_drawer_travel_mm=1000*max(abs(f["drawer_q"]-r["drawer_initial"]) for f in rows),
                       max_bowl_lift_mm=1000*max(f["bowl_xyz"][2]-r["bowl_xyz_initial"][2] for f in rows),
                       arm_collision_frames=sum(any(c["kind"] in ("arm_scene","arm_self_or_base") for c in f["contacts"]) for f in rows),
                       min_joint_limit_margin_rad=min(f["min_limited_margin_rad"] for f in rows))
            for name, target in (("drawer","wooden_cabinet_1_cabinet_top"),("bowl","akita_black_bowl_1_main"),("plate","plate_1_main")):
                steps=[f["step"] for f in rows if any("finger" in c["body1"] and c["body2"]==target for c in f["contacts"])]
                obj[name+"_first_finger_step"] = steps[0] if steps else ""
                obj[name+"_finger_frames"] = len(steps)
            replay.append(obj)
    write_csv(output, "cases.csv", cases)
    write_csv(output, "waypoints.csv", waypoints)
    write_csv(output, "drawer-replay.csv", replay)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evidence", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(exist_ok=True, parents=True)
    tables(a.evidence, a.output)
