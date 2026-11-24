"""Support-camera calibration and paired open-loop scoring of frozen pi0.5 commands."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from fluffy_bassoon.xemb.embodiment import (
    apply_embodiment_action, circular_yaw_error_deg, get_embodiment_spec, retarget_action_from_motor,
)
from fluffy_bassoon.xemb.readers import analytic_proprio_sysid

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]
CAMERA = OUTPUTS / "symmeta-camera-command-20261123-r1/evidence"
TAPES = OUTPUTS / "libero-xemb-friction-dev-20261119-r1/cpu01"
PIN = "757a44cd4374833cad790852df8b87ddf412e638"
MOTOR = "/mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1"
CHECKPOINT = "c12bace5cb391499d6abd7cc5767f94b1519de7d77940efb0cfcb71ca0f686a4"
REPLAY = "db666ac91680b69b38c04e2dfc997ca3080a9ee6eab2991f9d367f29e884894a"
VIEWS = {-30: "43f5a1c0477f43c7d8fae4bf98129e9e09e03b8d258714cac04d5b4486412688",
         0: "aba3f6160289397da7fed65d651dee062a20e7f062bde5a2d60c8b0aa52e08f8",
         30: "d97dd0c9e79a18189ac08964af9ffe89689f498885bac06c65ab66769d0f5702"}
INDICES = (8, 9, 10, 11, 80, 81, 82, 83)
ARMS = ("symmeta", "symmeta_camera", "hs", "hs_camera", "unadapted", "oracle")
SOURCE_GAIN = 0.054141017027535904  # nominal passive source fit in camera run's RUNTIME.md


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(path, expected):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, f"Digest mismatch at {path}")
    return raw


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def calibrated_yaw(yaw, reference, current):
    # Camera-to-world rotations. For a world-Z orbital move, C1 C0^T is Rz(delta).
    relative = np.asarray(current) @ np.asarray(reference).T
    np.testing.assert_allclose(relative.T @ relative, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(relative[:, 2], [0, 0, 1], atol=1e-8)
    np.testing.assert_allclose(np.linalg.det(relative), 1, atol=1e-8)
    delta = np.degrees(np.arctan2(relative[1, 0], relative[0, 0]))
    return float((yaw + delta + 180) % 360 - 180)


def nuisance(issued, delta_eef, delta_jaw):
    _, scale, grip = analytic_proprio_sysid(
        issued, delta_eef, delta_jaw, np.ones(3, bool), nominal_gain_m_per_cmd=SOURCE_GAIN)
    return scale, grip


def panel():
    replay = json.loads(gzip.decompress(checked(CAMERA / "replay.json.gz", REPLAY)))
    require(replay["status"] == "complete", "Partial orbital replay")
    result = []
    for angle, digest in VIEWS.items():
        scores = json.loads(gzip.decompress(checked(CAMERA / f"azimuth{angle:+d}.json.gz", digest)))
        require(scores["sources"]["checkpoint"]["sha256"] == CHECKPOINT, "Unexpected reader")
        rows = zip(scores["rows"], scores["supports"], replay["rows"], INDICES, strict=True)
        for readout, support, trajectory, index in rows:
            case = support["case"]
            identity = ("suite", "task_id", "embodiment", "index")
            require(all(readout[k] == case[k] == trajectory[k] for k in identity)
                    and index == case["index"] and case["physics"] == trajectory["physics"]
                    and case["chunk_steps"] == [5, 5, 5] and support["controls"] == 15
                    and support["issued_mean"] == trajectory["issued_mean"]
                    and readout["support_sha256"] == support["npz"]["sha256"] == case["sha256"]
                    and readout["checkpoint_sha256"] == CHECKPOINT, "Unmatched physical replay")
            frames = trajectory["camera_extrinsics"]
            cameras = [frames[0][str(a)]["camera_to_world"] for a in (0, angle)]
            predictions = {}
            for arm, source in (("symmeta", "model_native"), ("hs", "hs_native")):
                raw = readout["yaw_deg"][source]
                predictions[arm] = raw
                predictions[arm + "_camera"] = calibrated_yaw(raw, *cameras)
            result.append(dict(suite=case["suite"], task_id=case["task_id"], index=index, angle=angle,
                               embodiment=case["embodiment"], support=support["npz"], physics=case["physics"],
                               issued=support["issued_mean"], delta_eef=support["delta_eef"], yaws=predictions))
    return result


def corrections(row, scale, grip):
    # Labels enter here solely to instantiate the privileged control / forward evaluation map.
    spec = get_embodiment_spec(row["embodiment"])
    return {**{arm: (yaw, scale, grip) for arm, yaw in row["yaws"].items()},
            "unadapted": (0.0, 1.0, 1), "oracle": (spec.yaw_deg, spec.xy_scale, spec.gripper_sign)}


def command_tapes():
    manifest = json.loads(checked(TAPES / "sources.json",
                                  "6608a41e37056d9ae89627566a7a3f8da0b2cec133a61eb6d284ba6e84203f65"))
    require(manifest["design"]["checkpoint"] == MOTOR, "Wrong frozen command bank")
    tapes = {}
    for path in sorted(TAPES.glob("*-default-ordinary.npz")):
        name = path.name.removesuffix("-ordinary.npz")
        with np.load(path, allow_pickle=False) as npz:
            commands = npz["issued"]
        digest = hashlib.sha256(str((commands.shape, commands.dtype.str)).encode() + commands.tobytes())
        require(digest.hexdigest() == manifest["sources"][name]["actions_sha256"], f"Wrong commands: {name}")
        tapes[name] = np.clip(commands.astype(np.float32), -1, 1)
    require(len(tapes) == 8, "Incomplete motor bank")
    return tapes


def contrasts(rows, value):
    """Equal-size clusters. Enumerate all four resamples of two entire support tasks."""
    result = {}
    for first, second in (("symmeta", "symmeta_camera"), ("hs", "hs_camera"),
                          ("hs_camera", "symmeta_camera")):
        by_task = {}
        for task in sorted({row["task"] for row in rows}):
            pairs = [row[value][first] - row[value][second] for row in rows if row["task"] == task]
            by_task[task] = float(np.mean(pairs))
        a = list(by_task.values())
        resamples = [(x + y) / 2 for x in a for y in a]
        result[first + "_minus_" + second] = dict(mean=float(np.mean(a)), task_effects=by_task,
                                                   task_bootstrap_95=np.quantile(
                                                       resamples, [.025, .975], method="inverted_cdf").tolist())
    return result


def screen():
    tapes = command_tapes()
    rows = []
    for row in panel():
        spec = get_embodiment_spec(row["embodiment"])
        # Only XY is scored below. The jaw field is absent from these compact receipts.
        # Its value affects only gripper sign and cannot affect XY or inferred scale.
        scale, _ = nuisance(row["issued"], row["delta_eef"], np.zeros((3, 2)))
        yaw_mae, xy = {}, {}
        for arm, (yaw, g, sign) in corrections(row, scale, 1).items():
            yaw_mae[arm] = circular_yaw_error_deg(yaw, spec.yaw_deg)
            errors = []
            for commands in tapes.values():
                issued = retarget_action_from_motor(commands, yaw, g, sign, -1, 1)
                physical = apply_embodiment_action(issued, spec, -1, 1)
                errors.append(np.linalg.norm(physical[:, :2] - commands[:, :2], axis=1).mean(dtype=np.float64))
            xy[arm] = float(np.mean(errors))
        rows.append(dict(index=row["index"], task=f"{row['suite']}/{row['task_id']}", angle=row["angle"],
                         yaw_mae=yaw_mae, xy_error=xy))
    groups = {}
    for label, views in (("upright", [0]), ("negative", [-30]), ("positive", [30]), ("moved", [-30, 30])):
        group = [r for r in rows if r["angle"] in views]
        groups[label] = {}
        for field in ("yaw_mae", "xy_error"):
            groups[label][field] = {arm: float(np.mean([r[field][arm] for r in group])) for arm in ARMS}
            groups[label][field + "_paired"] = contrasts(group, field)
    return dict(revision=PIN, type="offline physical-camera readouts crossed with open-loop frozen command tapes",
                new_native_episodes=0, support_trajectories=8, support_tasks=2, controls_per_support=15,
                commands_per_development_tape={k: len(v) for k, v in tapes.items()},
                groups=groups, rows=rows)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    write_json(cli.parse_args().output, screen())
