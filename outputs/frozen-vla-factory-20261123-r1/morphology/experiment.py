"""Offline planar response screen. Run from any directory with NumPy installed."""
from collections import defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]
BASE = "757a44cd4374833cad790852df8b87ddf412e638"
FIT_TASKS = ("libero_10/5", "libero_10/8")
ARMS = ("UR5e", "IIWA", "Kinova3")
PANELS = (*ARMS, "UR5e_pi0")
TURN = np.array([[0.0, 1.0], [-1.0, 0.0]])  # positive 90 degrees on row vectors
RECEIPTS = {}


def read(path):
    raw = path.read_bytes()
    RECEIPTS[str(path.relative_to(OUTPUTS.parent))] = hashlib.sha256(raw).hexdigest()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


@dataclass
class Tape:
    u: np.ndarray
    xy: np.ndarray
    sha: str
    seed: int


def read_panels():
    cross = OUTPUTS / "symmeta-fk-crossarm-20261123-r1"
    specs = read(cross / "QUERY_CONFIG.json")
    files = {"IIWA": "gpu09iiwa", "Kinova3": "gpu06kinova3", "UR5e_pi0": "gpu08pi0fix"}
    data = {}
    skipped = defaultdict(list)
    jobs = {}
    slots = None
    for panel in PANELS:
        if panel == "UR5e":
            export = read(OUTPUTS / "symmeta-camera-command-20261123-r1/evidence/world.json.gz")
            assert all(q["active_controls"] == 15 and q["active_max_abs_dq_error_rad"] <= 2e-6
                       and q["active_max_abs_tcp_delta_error_m"] <= 1e-6 for q in export["qualifications"])
            jobs[panel] = {"collector_commit": export["collector_commit"], "archive": export["archive"]}
        else:
            code = files[panel]
            export = read(cross / f"evidence/{code}-cases.json.gz")
            config = specs["panels"][code]
            assert config["checkpoint_ready"] == "verified"
            assert export["arm"] == panel.split("_")[0]
            assert export["policy"] == ("pi0" if panel == "UR5e_pi0" else "pi05")
            jobs[panel] = {k: config[k] for k in ("launch", "checkpoint", "weight_transport_manifest_sha256")}
        sequence = [r["embodiment"] for r in export["rows"]]
        if slots is None:
            slots = sequence[:12]
        assert len(set(slots)) == 12 and sequence == slots * 8
        result = {}
        for row in export["rows"]:
            task = f'{row["suite"]}/{row["task_id"]}'
            if not row.get("fk_available", True):
                skipped[panel].append(task)
                continue
            if panel == "UR5e":
                u, y = row["issued_mean"], row["delta_world_m"]
                sha, seed = row["archive_sha256"], row["seed"]
                assert row["init_state_id"] == 11
            else:
                u, y = row["issued_xy"], row["world_delta_xy_m"]
                sha, seed = row["support_sha256"], row["support_seed"]
                assert row["support_layout"] == 11 and row["query_layout"] == 13
            u, y = np.asarray(u)[:, :2], np.asarray(y)[:, :2]
            assert u.shape == y.shape == (3, 2) and np.isfinite(u).all() and np.isfinite(y).all()
            assert np.abs(u).max() <= 1 and len(sha) == 64
            # Stationarity pairing via archive slot; no index or setting metadata in fit functions.
            slot = row["index"] % 12
            assert slot != 0 or row["embodiment"] == "fam_nom"
            key = (task, slot)
            assert key not in result
            result[key] = Tape(u, y, sha, seed)
        assert len(export["rows"]) == 96
        data[panel] = result
    names = [t.sha for records in data.values() for t in records.values()]
    assert len(names) == len(set(names))
    return data, {p: sorted(set(t)) for p, t in skipped.items()}, jobs


def source_templates():
    directory = OUTPUTS / "libero-crossarm-20261120-r1/evidence/gpu01/pilot"
    config = read(directory / "config.json")
    report = read(directory / "responses.json")
    assert config["source_bodies"] == ["Panda", "Sawyer"] and config["target_body"] == "UR5e"
    assert config["fit_tasks"] == [["libero_goal", 4], ["libero_goal", 5]]
    maps = {}
    for arm, item in report.items():
        fit = item["fit"]["target"]
        expected = {f"libero_goal-{task}-i{pose}-s{20261120 + 100*task + pose}"
                    for task in (4, 5) for pose in (0, 1)}
        assert set(fit["episode_ids"]) == expected and fit["rows"] == 288 and fit["response_rank"] == 6
        # Primitive response (m/control) to five-control displacement.
        maps[arm] = 5 * np.array(fit["cartesian_gain"])[:2, :2]
    return (maps["Panda"] + maps["Sawyer"]) / 2, maps["UR5e"]


def solve(x, y):
    coef, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != x.shape[1]:
        raise ValueError("Singular input design")
    return coef


def morphology(shared, nominal):
    """Scalar normalization of source template, then one supported input direction."""
    x, y = nominal.u, nominal.xy
    base_motion = x @ shared
    scale = np.sum(base_motion * y) / np.sum(base_motion**2)
    base = scale * shared
    direction = np.linalg.svd(x, full_matrices=False)[2][0]
    exposure = x @ direction
    correction = exposure @ (y - x @ base) / (exposure @ exposure)
    low_rank = base + np.outer(direction, correction)
    return low_rank, solve(x, y), base


def register(template, x, y):
    """Fit a scaled rotation before an arm-response map, using unlabelled practice."""
    design = np.column_stack(((x @ template).ravel(), (x @ TURN @ template).ravel()))
    a, b = solve(design, y.ravel())
    return (a * np.eye(2) + b * TURN) @ template


def error_mm(prediction, outcome):
    return float(1000 * np.linalg.norm(prediction - outcome, axis=1).mean())


def run_folds(data, shared, active_target):
    reads = defaultdict(set)
    errors = defaultdict(lambda: defaultdict(list))
    nominal_errors = defaultdict(lambda: defaultdict(list))
    for panel in PANELS:
        records = data[panel]
        for nominal_task, support_task in itertools.permutations(FIT_TASKS):
            target = data[panel.split("_")[0]][nominal_task, 0]
            low_rank, full, scaled_source = morphology(shared, target)
            templates = {"isotropic": np.eye(2), "source_shared": shared,
                         "rank_one": low_rank, "arm_specific": full}
            if panel.startswith("UR5e"):
                templates["active_ur5e"] = active_target
            for slot in range(12):
                support = records[support_task, slot]
                other = records[support_task, (slot + 1) % 12]
                maps = {name: register(template, support.u, support.xy) for name, template in templates.items()}
                maps["unrestricted_support"] = solve(support.u, support.xy)
                maps["wrong_support"] = register(np.eye(2), other.u, other.xy)
                # Extra cost diagnostic: spend both episodes at the unknown interface.
                second = records[nominal_task, slot]
                u2, y2 = np.vstack((support.u, second.u)), np.vstack((support.xy, second.xy))
                maps["two_support_isotropic"] = register(np.eye(2), u2, y2)
                maps["two_support_unrestricted"] = solve(u2, y2)
                reads["morphology"].add(target.sha)
                reads["unknown_interface"].add(support.sha)
                for (task, query_slot), query in records.items():
                    if query_slot != slot or task in FIT_TASKS:
                        continue
                    assert task not in ("libero_goal/4", "libero_goal/5")
                    assert query.seed not in (support.seed, target.seed)
                    reads["score"].add(query.sha)
                    key = (panel, task, slot)
                    # All model matrices fixed before exposing the scored episode's commands.
                    for name, matrix in maps.items():
                        errors[key][name].append(error_mm(query.u @ matrix, query.xy))
                    if slot == 0:
                        for name, template in (templates | {"scaled_source": scaled_source}).items():
                            if name != "isotropic":
                                nominal_errors[key][name].append(error_mm(query.u @ template, query.xy))
    assert not reads["score"] & (reads["unknown_interface"] | reads["morphology"])
    assert all(len(fold) == 2 for row in errors.values() for fold in row.values())
    counts = {name: len(hashes) for name, hashes in reads.items()}
    counts["unique_fit_or_support"] = len(reads["unknown_interface"] | reads["morphology"])
    return errors, nominal_errors, counts


def summarize(errors, panels, tasks):
    by_task = {}
    for task in sorted(tasks):
        rows = [row for (p, t, _), row in errors.items() if p in panels and t == task]
        by_task[task] = {k: np.mean([r[k] for r in rows])
                         for k in sorted(set.intersection(*(set(r) for r in rows)))}
    metrics = {k: float(np.mean([t[k] for t in by_task.values()])) for k in next(iter(by_task.values()))}
    intervals = {}
    # Enumerate n^n paired whole-task draws exactly for this small panel.
    n = len(by_task)
    draws = np.array(list(itertools.product(range(n), repeat=n)))
    for control in ("source_shared", "arm_specific", "isotropic", "scaled_source"):
        if control not in metrics:
            continue
        paired = np.array([t["rank_one"] - t[control] for t in by_task.values()])
        lo, hi = np.percentile(paired[draws].mean(axis=1), [2.5, 97.5])
        intervals[control] = [paired.mean(), lo, hi]
    return {"mm": metrics, "rank_one_minus_control_mm_mean_lo95_hi95": intervals,
            "task_mean_mm": by_task}


def design_reference(data):
    stats = {}
    for panel, records in data.items():
        stats[panel] = {}
        for group, tapes in (("calibration_tasks", [t for (name, _), t in records.items() if name in FIT_TASKS]),
                             ("all_qualified", list(records.values()))):
            singular = np.array([np.linalg.svd(t.u, compute_uv=False) for t in tapes])
            stats[panel][group] = {
                "n": len(tapes), "condition_min_median_max": np.percentile(singular[:, 0] / singular[:, 1], [0, 50, 100]).tolist(),
                "median_fraction_command_energy_on_minor_axis": float(np.median(singular[:, 1]**2 / (singular**2).sum(axis=1))),
                "median_mm_per_unit_weak_axis_slope_change_for_1mm_frobenius_noise": float(np.median(1 / singular[:, 1])),
            }
    active = OUTPUTS / "libero-xemb-joint-sysid-20261119-r1/cpu01"
    conditions = []
    for path in sorted(active.glob("sensors-*.npz")):
        RECEIPTS[str(path.relative_to(OUTPUTS.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as z:
            assert z["issued"].shape == (15, 7)
            np.testing.assert_allclose(z["issued"].reshape(3, 5, 7).mean(axis=1), z["issued_mean"])
            conditions.append(float(np.linalg.cond(z["issued_mean"][:, :2].astype(float))))
    assert len(conditions) == 7
    stats["Panda_active_pulse_reference"] = {"episodes": 7, "condition_range": [min(conditions), max(conditions)]}
    return stats


def main():
    data, skipped, jobs = read_panels()
    shared, active = source_templates()
    errors, nominal, counts = run_folds(data, shared, active)
    tasks = {p: {t for panel, t, _ in errors if panel == p} for p in PANELS}
    common = set.intersection(*(tasks[p] for p in ARMS))
    scores = {}
    for name, cube in (("transfer", errors), ("nominal_no_interface_support", nominal)):
        scores[name] = {"pi05_common": summarize(cube, ARMS, common)}
        for p in PANELS:
            scores[name][p] = summarize(cube, [p], tasks[p])
    result = {"baseline_revision": BASE, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "job": {"location": "session Linux CPU", "BLAS_threads_requested": 1, "GPU_hours": 0,
                      "new_motor_calls": 0, "new_env_steps": 0, "numpy_version": np.__version__},
              "control_budget": {"source_active_Panda": 288, "source_active_Sawyer": 288,
                                 "optional_active_UR5e": 288, "reference_only_Panda_probes": 7 * 15,
                                 "target_nominal_per_deployment": 15, "target_unknown_per_setting": 15,
                                 "historical_passive_calibration_controls_up_to": counts["unique_fit_or_support"] * 15,
                                 "historical_passive_score_controls_up_to": counts["score"] * 15},
              "counts_unique_tapes": counts, "unqualified_tasks": skipped,
              "evaluation_episodes_per_panel": {p: len(tasks[p])*12 for p in PANELS},
              "primary_common_tasks": sorted(common), "calibration_tasks": FIT_TASKS,
              "morphology_inputs_pi05": {p: [[t, data[p][t, 0].sha] for t in FIT_TASKS] for p in ARMS},
              "archive_provenance": jobs, "design": design_reference(data),
              "metrics": scores, "input_sha256": RECEIPTS}
    path = HERE / "metrics.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(scores["transfer"]["pi05_common"], indent=2))


if __name__ == "__main__":
    main()
