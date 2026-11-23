"""Score sensor fusion on the committed NPZ-derived motion vectors. CPU / NumPy only."""

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


LANE = Path(__file__).resolve().parent
EVIDENCE = LANE.parents[1] / "symmeta-camera-command-20261123-r1" / "evidence"
BASE = "757a44cd4374833cad790852df8b87ddf412e638"
TAPE_SET = "8729fa68343e2a7a0cc6471ab25e7c707555a866e682a3dc999079a7b9dc3554"
READER = "c12bace5cb391499d6abd7cc5767f94b1519de7d77940efb0cfcb71ca0f686a4"
METHODS = ("visual", "world", "equal", "fusion")
BOOT_SEED = 41
BOOT_DRAWS = 20000


def require(test, message):
    if not test:
        raise ValueError(message)


def case_id(row):
    return f"{row['suite']}/{row['task_id']}/{row['embodiment']}"


def keyed(rows, key):
    table = {key(row): row for row in rows}
    require(len(table) == len(rows), "duplicate case identity")
    return table


def wrapped(deg):
    return (deg + 180) % 360 - 180


def complex_xy(vectors):
    xy = np.asarray(vectors, dtype=float)
    require(xy.ndim == 2 and xy.shape[0] == 3 and xy.shape[1] >= 2, "invalid chunk shape")
    return xy[:, 0] + 1j * xy[:, 1]


def angle_and_variance(votes):
    """Each complex vote is weight * conjugate(command_xy) * motion_xy."""
    # Precision-only check, relative to the sum of individual vote magnitudes.
    # A zero resultant has no identifiable heading. Do not assign it a yaw.
    total = votes.sum()
    leave_one = total - votes
    all_sums = np.append(leave_one, total)
    require(np.all(np.abs(all_sums) > 16 * np.finfo(float).eps * np.abs(votes).sum()),
            "full or leave-one-out yaw undefined")
    heading = total / abs(total)
    residual = np.angle(leave_one * heading.conjugate())
    variance = (len(votes) - 1) / len(votes) * np.square(residual - residual.mean()).sum()
    return heading, variance


def predict(command, visual_xy, visual_weights, world_delta):
    """No case identity, camera condition, target label or other episode as input."""
    cmd = complex_xy(command).conjugate()
    weights = np.asarray(visual_weights, dtype=float)
    require(weights.shape == (3,) and np.all(weights >= 0), "invalid visual chunk weights")
    visual, vv = angle_and_variance(cmd * complex_xy(visual_xy) * weights)
    world, wv = angle_and_variance(cmd * complex_xy(world_delta))
    # Inverse-variance weighting expressed without division by either variance.
    alpha = wv / (vv + wv) if vv + wv > 0 else 0.5
    equal = (visual + world) / 2
    fused = alpha * visual + (1 - alpha) * world
    require(min(abs(equal), abs(fused)) > 16 * np.finfo(float).eps, "antipodal fusion undefined")
    angles = np.angle([visual, world, equal, fused], deg=True)
    return angles, float(alpha)


def fit_case(row, support, world, offset=0):
    vectors = row["vectors"]
    delta = np.asarray(world["delta_world_m"])[:, :2]
    if offset:
        # Explicit synthetic frame error on the very same measured deltas.
        rotation = np.exp(1j * np.deg2rad(offset))
        rotated = complex_xy(delta) * rotation
        delta = np.stack([rotated.real, rotated.imag], axis=1)
    angles, alpha = predict(support["issued_mean"], vectors["model_visual_xy"],
                            vectors["model_native_weights"], delta)
    archived = np.array([row["yaw_deg"]["model_native"], world["yaw_hat_deg"] + offset])
    parity = np.abs(wrapped(angles[:2] - archived))
    # Frozen visual reference is float32. Matches the original scorer's direct-readout tolerance.
    require(np.all(parity <= 1e-4), f"single-sensor reconstruction mismatch {parity}")
    return angles, alpha, parity


def check_receipts(archive, visual, world):
    require(set(visual) == set(world) and len(world) == 96, "invalid official support panel")
    require(archive["status"] == "complete", "incomplete world-control export")
    require(set(r["index"] for r in world.values()) == set(range(96)), "missing scheduled index")
    resets = keyed(archive["qualifications"], lambda r: r["shard"])
    require(set(resets) == set(range(8)), "missing active qualification")
    for sid, s in visual.items():
        w, c = world[sid], s["case"]
        q = resets[w["shard"]]
        require(c["id"] == case_id(c) == sid and c["index"] == w["index"] and
                c["shard"] == w["shard"] and c["physics"] == q["expected_physics"] and
                c["support_init_state_id"] == w["init_state_id"] == 11 and
                c["support_seed"] == w["seed"] == q["seed"] and
                q["active_max_abs_dq_error_rad"] == q["active_max_abs_tcp_delta_error_m"] == 0,
                "reset, seed or case mismatch")
        require(s["npz"]["sha256"] == w["archive_sha256"] and
                s["npz"]["path"] == w["archive_npz"], "FK/vision passive tape mismatch")
        require(s["issued_mean"] == w["issued_mean"] and s["controls"] == 15 and
                s["selected_chunk_indices"] == [0, 1, 2], "support budget or command mismatch")
        require(np.array_equal(np.diff(w["world_xyz_m"], axis=0), w["delta_world_m"]),
                "world displacements differ from recorded endpoints")
    hashes = [w["archive_sha256"] for w in sorted(world.values(), key=lambda x: x["index"])]
    require(len(set(hashes)) == 96, "non-unique tape digest")
    digest = hashlib.sha256("".join(f"{h}\n" for h in hashes).encode("ascii")).hexdigest()
    require(digest == TAPE_SET, "official tape-set digest mismatch")


def summarize(rows):
    tasks = sorted({r["task"] for r in rows})
    grouped = [[r["error"] for r in rows if r["task"] == t] for t in tasks]
    require(len({len(group) for group in grouped}) == 1, "unbalanced within-task panel")
    task_mae = np.array([np.mean(group, axis=0) for group in grouped])
    errors = np.array([r["error"] for r in rows])
    means = task_mae.mean(axis=0)
    summary = dict(n_supports=len(rows), n_tasks=len(tasks), mae_deg=dict(zip(METHODS, means)),
                   median_visual_fraction=float(np.median([r["alpha"] for r in rows])),
                   mean_visual_fraction=float(np.mean([r["alpha"] for r in rows])),
                   task_mae_deg={task: dict(zip(METHODS, values)) for task, values in zip(tasks, task_mae)})
    paired = {}
    rng = np.random.default_rng(BOOT_SEED)
    indices = rng.integers(len(tasks), size=(BOOT_DRAWS, len(tasks))) if len(tasks) > 2 else None
    for name in ("visual", "world", "equal"):
        j = METHODS.index(name)
        effect = task_mae[:, -1] - task_mae[:, j]
        stats = dict(mean_deg=effect.mean(), supports_lower_error=int(np.sum(errors[:, -1] < errors[:, j])),
                     tasks_lower_error=int(np.sum(effect < 0)))
        if indices is not None:
            stats["task_bootstrap_ci95_deg"] = np.quantile(effect[indices].mean(axis=1), [.025, .975]).tolist()
        paired[name] = stats
    summary["fusion_minus_control"] = paired
    return summary


def main(output):
    file_sha = {}

    def load(name):
        raw = (EVIDENCE / f"{name}.json.gz").read_bytes()
        file_sha[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(gzip.decompress(raw))

    roll, world, moderate, replay = (load(n) for n in ("roll", "world", "moderate", "replay"))
    original = keyed(roll["supports"], lambda x: x["case"]["id"])
    fk = keyed(world["rows"], case_id)
    check_receipts(world, original, fk)
    require(roll["checks"]["same_runtime_passed"] and moderate["zoom"] == 1.4, "view export mismatch")
    require(roll["support_set_sha256"] == moderate["support_set_sha256"] == TAPE_SET,
            "optical panels refer to different supports")

    groups = defaultdict(list)
    max_parity = np.zeros(2)

    def record(view, angle, row, s, w, offset=0):
        require(row["id"] == case_id(row) == case_id(w) and row["index"] == w["index"],
                "view case mismatch")
        # IDs and target labels are kept by the evaluator. Only fit_case's numeric
        # sensor fields reach predict. No fitting against another case or view.
        predictions, weight, parity = fit_case(row, s, w, offset)
        max_parity[:] = np.maximum(max_parity, parity)
        require(row["true_yaw_deg"] == w["true_yaw_deg"] and row["regime"] == w["regime"],
                "scoring labels inconsistent")
        err = abs(wrapped(predictions - row["true_yaw_deg"]))
        groups[f"{view}_{angle:+d}"].append(dict(task=f"{w['suite']}/{w['task_id']}",
                                                 regime=w["regime"], error=err, alpha=weight))

    for name, report, angles in (("full_roll", roll, (0, -90, 90)),
                                 ("zoom_roll", moderate, (0, -15, 15, -30, 30))):
        require(report["sources"]["checkpoint"]["sha256"] == READER, "checkpoint mismatch")
        support_rows = keyed(report["supports"], lambda s: s["case"]["id"] if "case" in s else s["id"])
        require(set(support_rows) == set(original), "missing view support")
        rows = keyed(report["rows"], lambda r: (r["id"], r["roll_deg"]))
        require(set(rows) == {(sid, a) for sid in original for a in angles}, "view grid mismatch")
        for (sid, angle), row in rows.items():
            s, w = original[sid], fk[sid]
            require(support_rows[sid]["npz"] == s["npz"] and
                    support_rows[sid]["selected_chunk_indices"] == [0, 1, 2], "view NPZ mismatch")
            if name == "full_roll":
                require(row["checkpoint_sha256"] == READER and row["support_sha256"] == w["archive_sha256"],
                        "view row hash mismatch")
            record(name, angle, row, s, w)
            if name == "full_roll" and angle == 0:
                for offset in (-30, 30):
                    record("synthetic_world_bias", offset, row, s, w, offset)

    require(replay["status"] == "complete" and replay["method"] == "fresh_held_means_physical_replay",
            "replay type mismatch")
    fresh = keyed(replay["rows"], case_id)
    require(len(fresh) == 8, "orbit case count mismatch")
    for az in (-30, 0, 30):
        name = f"azimuth{az:+d}"
        report = load(name)
        require(report["sources"]["checkpoint"]["sha256"] == READER and report["checks"]["passed"],
                "orbital checkpoint or readout mismatch")
        supports = keyed(report["supports"], lambda r: r["case"]["id"])
        rows = keyed(report["rows"], lambda r: r["id"])
        require(set(supports) == set(rows) == set(fresh), "orbit panel mismatch")
        for sid, w in fresh.items():
            s, r = supports[sid], rows[sid]
            c = s["case"]
            require(c["camera_condition"] == name and c["physics"] == w["physics"] and
                    c["chunk_steps"] == [5, 5, 5] and s["selected_chunk_indices"] == [0, 1, 2] and
                    c["sha256"] == s["npz"]["sha256"] == r["support_sha256"] and
                    c["index"] == w["index"] and r["checkpoint_sha256"] == READER and
                    w["archive_sha256"] == fk[sid]["archive_sha256"] and
                    s["issued_mean"] == w["issued_mean"] and
                    np.array_equal(w["primitive_issued"], np.repeat(w["issued_mean"], 5, axis=0)),
                    "fresh view/trajectory identity mismatch")
            record("physical_azimuth", az, r, s, w)

    summaries = {}
    for condition, rows in sorted(groups.items()):
        held = [r for r in rows if r["regime"] in ("compositional_ood", "continuous_angle_ood")]
        if condition.startswith("physical"):
            require(len(held) == len(rows) == 8, "orbit denominator mismatch")
            summaries[condition] = {"ood": summarize(held)}
        else:
            require((len(held), len(rows)) == (64, 96), "official denominator mismatch")
            summaries[condition] = {"ood": summarize(held), "all": summarize(rows)}
    receipt = dict(base_revision=BASE, input_gz_sha256=file_sha, reader_sha256=READER,
                   source_npz_set_sha256=TAPE_SET, controls_per_support=15,
                   original_supports=96, original_tasks=8, replays=8, replay_tasks=2,
                   gpu_hours=0, motor_updates=0, bootstrap_seed=BOOT_SEED,
                   bootstrap_draws=BOOT_DRAWS,
                   max_reconstruction_difference_deg=dict(zip(METHODS[:2], max_parity)))
    output.write_text(json.dumps({"receipt": receipt, "conditions": summaries}, indent=2,
                                 allow_nan=False) + "\n")
    print(f"Qualified 96 original supports and 8 replays. Scored {len(summaries)} conditions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=LANE / "metrics.json")
    main(parser.parse_args().output)
