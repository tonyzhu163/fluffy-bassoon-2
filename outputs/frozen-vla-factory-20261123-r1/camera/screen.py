"""Offline post-readout camera correction on archived, paired UR5e supports."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "symmeta-camera-command-20261123-r1" / "evidence"
BASE = "757a44cd4374833cad790852df8b87ddf412e638"
CHECKPOINT = "c12bace5cb391499d6abd7cc5767f94b1519de7d77940efb0cfcb71ca0f686a4"
SUPPORT_SET = "8729fa68343e2a7a0cc6471ab25e7c707555a866e682a3dc999079a7b9dc3554"
BOOTSTRAP_SEED = 20261123


def load(name):
    with gzip.open(DATA / f"{name}.json.gz", "rt") as archive:
        return json.load(archive)


def task(row):
    return f"{row['suite']}/{row['task_id']}"


def wrap(degrees):
    return (degrees + 180) % 360 - 180


def estimate_roll(flow_xy, world_delta, projection):
    """Fit observed-to-upright planar rotation in (column, row) image coordinates."""
    observed = np.asarray(flow_xy)
    expected = np.asarray(world_delta) @ projection
    dot = (observed * expected).sum()
    cross = (observed[:, 0] * expected[:, 1] - observed[:, 1] * expected[:, 0]).sum()
    if np.hypot(cross, dot) <= np.finfo(float).eps * np.linalg.norm(observed) * np.linalg.norm(expected):
        raise ValueError("Calibration motion does not identify a rotation")
    return float(np.rad2deg(np.arctan2(cross, dot)))


def relative_azimuth(extrinsics, mount_key):
    reference = np.asarray(extrinsics["0"]["camera_to_world"])
    current = np.asarray(extrinsics[mount_key]["camera_to_world"])
    relative = current @ reference.T
    np.testing.assert_allclose(relative[:, 2], [0, 0, 1], atol=1e-10)
    return float(np.rad2deg(np.arctan2(relative[1, 0], relative[0, 0])))


def predictions(native_yaw, known_angle, estimated_angle=None):
    values = {"raw": native_yaw, "known": native_yaw + known_angle,
              "opposite_sign": native_yaw - known_angle}
    if estimated_angle is not None:
        values["estimated"] = native_yaw + estimated_angle
    return values


def score(row, view, predictions_deg):
    # The command-interface label enters only at scoring.
    errors = {name: abs(wrap(yaw - row["true_yaw_deg"])) for name, yaw in predictions_deg.items()}
    return task(row), row["index"], view, errors


def summarize(rows):
    # Views are repeated measures. Reduce within support before weighting tasks equally.
    episodes = defaultdict(list)
    for task_id, index, _, errors in rows:
        episodes[task_id, index].append(list(errors.values()))
    tasks = defaultdict(list)
    for (task_id, _), errors in episodes.items():
        tasks[task_id].append(np.mean(errors, axis=0))
    names = list(rows[0][-1])
    assert names[0] == "raw" and all(list(errors) == names for *_, errors in rows)
    by_task = np.array([np.mean(values, axis=0) for values in tasks.values()])
    gains = by_task[:, [0]] - by_task
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(len(tasks), size=(20000, len(tasks)))
    intervals = np.quantile(gains[draws].mean(axis=1), [0.025, 0.975], axis=0)
    return {
        "tasks": len(tasks), "supports": len(episodes), "view_rows": len(rows),
        "mae_deg": dict(zip(names, np.round(by_task.mean(axis=0), 4).tolist(), strict=True)),
        "raw_minus_corrected_deg": {
            name: {"mean": round(float(gains[:, i].mean()), 4),
                   "task_bootstrap_95": intervals[:, i].round(4).tolist(),
                   "by_task": dict(zip(tasks, gains[:, i].round(4).tolist(), strict=True))}
            for i, name in enumerate(names) if name != "raw"
        },
    }


def panels(rows):
    angles = sorted({view for _, _, view, _ in rows})
    groups = {str(angle): [r for r in rows if r[2] == angle] for angle in angles}
    for angle in angles:
        if angle > 0:
            groups[f"abs{angle}"] = [r for r in rows if abs(r[2]) == angle]
    return {name: summarize(group) for name, group in groups.items()}


def roll_screen(data, world):
    assert data["support_set_sha256"] == SUPPORT_SET
    assert data["sources"]["checkpoint"]["sha256"] == CHECKPOINT
    supports = {s.get("id") or s["case"]["id"]: s for s in data["supports"]}
    rows = data["rows"]
    angles = data["rolls_deg"]
    assert len({(r["id"], r["roll_deg"]) for r in rows}) == len(rows) == 96 * len(angles)
    assert len(supports) == 96 and set(angles) == {r["roll_deg"] for r in rows}
    for r in rows:
        s, w = supports[r["id"]], world[r["index"]]
        assert s["npz"]["sha256"] == w["archive_sha256"]
        if "support_sha256" in r:
            assert r["support_sha256"] == s["npz"]["sha256"]
        assert r["id"] == f"{task(w)}/{w['embodiment']}" and task(w) == task(r)
        assert s["selected_chunk_indices"] == [0, 1, 2]

    ood = [r for r in rows if r["regime"] in {"compositional_ood", "continuous_angle_ood"}]
    assert len(ood) == 64 * len(angles)
    all_known = [score(r, r["roll_deg"], predictions(r["yaw_deg"]["model_native"], r["roll_deg"]))
                 for r in ood]
    result = {"all_tasks_known": panels(all_known), "donor_sensitivity": {}}
    projection = np.asarray(data["sources"]["calibration"]["projection"])
    # One calibration episode at each mount, chosen by archive order before scoring.
    for donor in range(0, 96, 12):
        donor_task = task(world[donor])
        calibration = [r for r in rows if r["index"] == donor]
        assert len(calibration) == len(angles) and world[donor]["embodiment"] == "fam_nom"
        # roll_deg groups observations from the same physical setup. Its numeric
        # value is withheld from estimate_roll, including when calibrating at zero.
        estimated = {
            r["roll_deg"]: estimate_roll(r["vectors"]["hs_image_motion_px"],
                                         world[donor]["delta_world_m"], projection)
            for r in calibration
        }
        heldout = [r for r in ood if task(r) != donor_task]
        assert len(heldout) == 56 * len(angles)
        scored = []
        for r in heldout:
            mount = r["roll_deg"]
            pred = predictions(r["yaw_deg"]["model_native"], mount, estimated[mount])
            scored.append(score(r, mount, pred))
        if donor == 0:
            result["fixed_donor"] = {"task": donor_task, "archive_index": donor,
                                     "estimated_roll_deg": estimated, "scores": panels(scored)}
        changed = [r for r in scored if r[2] != 0]
        summary = summarize(changed)
        result["donor_sensitivity"][donor_task] = {
            "estimated_roll_deg": {v: round(a, 4) for v, a in estimated.items()},
            "changed_raw_minus_estimated_deg": summary["raw_minus_corrected_deg"]["estimated"]["mean"],
            "changed_estimated_mae_deg": summary["mae_deg"]["estimated"],
        }
    return result


def orbit_screen():
    replay = {r["index"]: r for r in load("replay")["rows"]}
    scored = []
    for angle in (-30, 0, 30):
        data = load(f"azimuth{angle:+d}")
        assert data["sources"]["checkpoint"]["sha256"] == CHECKPOINT
        supports = {s["case"]["index"]: s for s in data["supports"]}
        assert {r["index"] for r in data["rows"]} == set(replay) == set(supports)
        assert len(data["rows"]) == len(replay) == 8
        for r in data["rows"]:
            w, s = replay[r["index"]], supports[r["index"]]
            assert s["case"]["physics"] == w["physics"] and s["case"]["id"] == r["id"]
            assert r["id"] == f"{task(w)}/{w['embodiment']}"
            assert r["support_sha256"] == s["npz"]["sha256"] and s["controls"] == 15
            np.testing.assert_allclose(s["issued_mean"], w["issued_mean"], atol=0, rtol=0)
            angles = [relative_azimuth(boundary, str(angle)) for boundary in w["camera_extrinsics"]]
            np.testing.assert_allclose(angles, angles[0], atol=1e-10, rtol=0)
            scored.append(score(r, angle, predictions(r["yaw_deg"]["model_native"], angles[0])))
    return panels(scored)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    world = {r["index"]: r for r in load("world")["rows"]}
    metrics = {
        "base_revision": BASE, "reader_checkpoint_sha256": CHECKPOINT,
        "support_set_sha256": SUPPORT_SET, "bootstrap_seed": BOOTSTRAP_SEED,
        "full_frame_image_roll": roll_screen(load("roll"), world),
        "zoom_1.4_image_roll": roll_screen(load("moderate"), world),
        "physical_orbit_known_azimuth": orbit_screen(),
    }
    args.output.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
