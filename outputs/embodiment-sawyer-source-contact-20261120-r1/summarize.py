"""Independently recount direct plate contacts from archived native geom pair masks on Linux."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np


GROUPS = ("hand", "palm", "left_finger", "right_finger", "arm")


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")


def verify_trace(path, record):
    geom = record["geometry"]
    n = geom["controls"]
    nsub = geom["substeps_per_control"]
    catalog = {g["id"]: g["category"] for g in geom["collision_geoms"]}
    fields = geom["summary"]
    observed_pairs = defaultdict(set)
    with np.load(path, allow_pickle=False) as z:
        assert len(z["time_s"]) == n+1 and np.all(np.diff(z["time_s"]) > 0)
        for key in ("plate_xyz_m", "goal_xyz_m", "eef_xyz_m"):
            assert z[key].shape == (n+1, 3) and np.isfinite(z[key]).all()
        distances = np.linalg.norm((z["plate_xyz_m"] - z["goal_xyz_m"])[:, :2], axis=1)
        assert np.allclose(distances, z["plate_goal_xy_m"], rtol=0., atol=1e-12)
        pairs = z["plate_pairs"]
        assert pairs.ndim == 2 and pairs.shape[1] == 4
        masks = {g: [0]*(n+1) for g in GROUPS}
        for t, plate, other, mask in pairs.tolist():
            role = catalog[other]
            assert 0 <= t <= n and catalog[plate] == "plate" and 0 < mask < (1 << (nsub+1))
            if role in GROUPS:
                masks[role][t] |= mask
                observed_pairs[role].add((plate, other))
                if role in GROUPS[1:4]:
                    masks["hand"][t] |= mask
        solver = z["solver_distance_xy_m"]
        assert solver.shape == (n, nsub if geom["sampling"] == "all_native_substeps_and_boundaries" else 0)
        for group, series in masks.items():
            stats = fields[group]
            active = [t for t, mask in enumerate(series) if mask]
            assert stats["observed"] == bool(active)
            assert stats["contact_controls"] == sum(bool(m) for m in series[1:])
            assert stats["boundary_contact_controls"] == sum(bool(m >> nsub) for m in series[1:])
            if geom["sampling"] == "all_native_substeps_and_boundaries":
                assert stats["active_substeps"] == sum((m % (1 << nsub)).bit_count() for m in series[1:])
            if active:
                first = stats["first_touch"]
                mask = series[active[0]]
                bit = (mask & -mask).bit_length() - 1
                assert first["control"] == active[0] and first["substep"] == (bit+1 if bit < nsub else None)
                assert abs(stats["final_progress_since_first_xy_m"] - (first["distance_xy_m"]-distances[-1])) < 1e-12
                t = active[0]
                start_distance = solver[t-1, bit] if bit < nsub else distances[t]
                assert abs(start_distance - first["distance_xy_m"]) < 1e-12
                spans = [distances[t:]]
                if solver.shape[1]:
                    spans.extend([solver[t-1, bit:] if t and bit < nsub else [], solver[t:].ravel()])
                least = min(np.min(x) for x in spans if len(x))
                assert abs(stats["best_progress_since_first_xy_m"] - (start_distance-least)) < 1e-12
            else:
                assert stats["first_touch"] is None and stats["final_progress_since_first_xy_m"] is None
    return {g: sorted(v) for g, v in observed_pairs.items()}, float(distances[0]), float(distances[-1])


def describe(values):
    if not values:
        return None
    return dict(n=len(values), median=median(values), q25=float(np.quantile(values, .25)),
                q75=float(np.quantile(values, .75)), min=min(values), max=max(values))


def main(root):
    report = read(root / "report.json")
    assert report["status"] == "finished" and [e["layout_seed"] for e in report["episodes"]] == list(range(740000, 740050))
    cases = []
    for row in report["episodes"]:
        case = {k: row[k] for k in ("layout_id", "layout_seed", "status", "record")}
        case["source"] = row.get("archived")
        case["record_sha256"] = sha(root / row["record"])
        if row["status"] == "validated":
            record = read(root / row["record"])
            geo = record["geometry"]
            v = record["physics_validation"]
            tape = case["source"]
            assert v["done"] and not v["violations"] and v["terminal_reward"] == tape["native_return"]
            assert v["success"] == tape["success"] and v["steps"] == geo["controls"] == tape["steps"]
            assert sha(root / row["trace"]) == row["trace_sha256"]
            pairs, opening, final = verify_trace(root / row["trace"], record)
            case.update(sampling=geo["sampling"], geometry=geo["summary"], goal_site=geo["goal_site"],
                        actual_robot_pairs=pairs, initial_goal_distance_xy_m=opening, final_goal_distance_xy_m=final,
                        steps=v["steps"], reward=v["terminal_reward"], exact_opening=v["opening"]["exact_physics"],
                        exact_opening_public=v["opening"]["exact_public_state"], opening_errors=v["opening"]["max_errors"],
                        trajectory_errors=v["max_errors"], exact_motion_steps=v["exact_motion_steps"],
                        exact_replans=v["replan_state_hash_matches"], replans=v["replans_checked"],
                        trace=row["trace"], trace_sha256=row["trace_sha256"])
        else:
            case["reason"] = row.get("reason")
        cases.append(case)
    groups = {}
    for success in (True, False):
        rows = [c for c in cases if c["source"] and c["source"]["success"] == success]
        valid = [c for c in rows if c["status"] == "validated"]
        touched = [c for c in valid if c["geometry"]["hand"]["observed"]]
        hand = [c["geometry"]["hand"] for c in touched]
        groups["successful" if success else "failed"] = dict(
            archived=len(rows), validated=len(valid), touched=len(touched), without_observed_touch=len(valid)-len(touched),
            touched_by_group={g: sum(c["geometry"][g]["observed"] for c in valid) for g in GROUPS},
            endpoint_only_count=sum(bool(c["geometry"]["hand"]["reset_contact"] or c["geometry"]["hand"]["boundary_contact_controls"]) for c in valid),
            first_control=describe([h["first_touch"]["control"] for h in hand]),
            contact_controls=describe([h["contact_controls"] for h in hand]),
            active_duration_s=describe([h["sampled_solver_duration_s"] for h in hand if h["sampled_solver_duration_s"] is not None]),
            first_goal_distance_m=describe([h["first_touch"]["distance_xy_m"] for h in hand]),
            final_goal_distance_m=describe([c["final_goal_distance_xy_m"] for c in touched]),
            net_progress_m=describe([h["final_progress_since_first_xy_m"] for h in hand]),
            best_progress_m=describe([h["best_progress_since_first_xy_m"] for h in hand]),
            positive_net_progress=sum(h["final_progress_since_first_xy_m"] > 0 for h in hand))
    result = dict(counts=report["counts"], by_original_native_outcome=groups,
                  original_report=report["source"].get("report_sha256"), report_sha256=sha(root/"report.json"), cases=cases)
    write(root / "cases.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "cases"}, allow_nan=False), flush=True)


if __name__ == "__main__":
    if sys.platform != "linux" or not Path("/sys/fs/cgroup/cpu.max").is_file():
        sys.exit("Use the allocated Linux job")
    main(Path(sys.argv[1]))
