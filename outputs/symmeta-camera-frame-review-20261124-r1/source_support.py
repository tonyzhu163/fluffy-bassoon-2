"""Read the original source and historical physical support, without fitting or rollouts."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/mowenjie/fluffy-bassoon-2/outputs")
PR7 = ROOT / "xemb-icml-campaign-20261121-r1/gpu12scale/stage3/diverse_shards"
CAL = ROOT / "symmeta-omitted-chunk-20261124-r1/data01"
ORBIT = ROOT / "symmeta-camera-command-20261123-r1/orbitjob01/replays"
CENTER = 31.5
FOCAL = 32 / np.tan(np.pi / 8)
GRID = np.array([[CENTER, 0, CENTER], [0, CENTER, CENTER], [0, 0, 1.0]])


def read(path):
    return json.loads(path.read_bytes())


def signature(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).digest()


def frame_keys(before, after):
    assert np.array_equal(before[1:3], after[:2])
    frames = np.concatenate((before[:1], after[:3]))
    return signature(frames[0]), signature(frames)


def source_overlap():
    whole, openings, contexts = set(), set(), set()
    for shard in range(8):
        folder = PR7 / f"shard-{shard}"
        records = read(folder / "meta.json")
        with np.load(folder / "diverse.npz", allow_pickle=False) as z:
            bef, aft = z["img_before"][:, :3].copy(), z["img_after"][:, :3].copy()
        for b, a, r in zip(bef, aft, records, strict=True):
            op, seq = frame_keys(b, a)
            openings.add(op)
            whole.add(seq)
            contexts.add((r["arm"], r["suite"], int(r["task_id"]), int(r["init_state_id"])))
    meta = {r["case_id"]:r for r in read(CAL / "source-membership.json")}
    counts, by_arm, by_task = Counter(), Counter(), Counter()
    with np.load(CAL / "reader-cal.npz", allow_pickle=False) as z:
        for key, b, a in zip(z["case_id"], z["img_before"], z["img_after"], strict=True):
            op, seq = frame_keys(b, a)
            r = meta[str(key)]
            suite, tid = r["task"].split("/")
            context = (r["arm"], suite, int(tid), int(r["init_state_id"]))
            counts["candidate"] += 1
            if seq in whole:
                counts["duplicate_four_frames"] += 1
                continue
            counts["retained"] += 1
            counts["retained_with_fit_reset_RGB"] += int(op in openings)
            counts["retained_with_fit_arm_task_layout"] += int(context in contexts)
            by_arm[r["arm"]] += 1
            by_task[r["task"]] += 1
    return dict(counts=counts, heldout_by_arm=by_arm, heldout_by_task=by_task,
                unique_fit_openings=len(openings), unique_fit_sequences=len(whole))


def norm_h(h):
    n = np.linalg.solve(GRID, h @ GRID)
    return n / n[2, 2]


def plane_projection(center, world_to_camera, k, z):
    # Matrix acting on the world plane coordinates (X, Y, 1).
    return k @ np.column_stack((world_to_camera[:, :2], world_to_camera @
                              (np.array([0, 0, z])-center)))


def source_plane(angle):
    a = np.deg2rad(angle)
    ctr = np.array([np.cos(a), np.sin(a), 1.0]) / np.sqrt(2)
    d = -ctr
    x = np.cross(d, [0, 0, 1])
    x /= np.linalg.norm(x)
    k = np.array([[FOCAL, 0, CENTER], [0, FOCAL, CENTER], [0, 0, 1.]])
    return plane_projection(ctr, np.stack((x, np.cross(d, x), d)), k, 0.)


def source_view(roll, orbit):
    a = np.deg2rad(roll)
    rot = np.eye(3)
    rot[:2, :2] = [[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]
    return rot @ norm_h(source_plane(orbit) @ np.linalg.inv(source_plane(0)))


def physical_plane(pose, z):
    k = np.array([[-FOCAL, 0, CENTER], [0, FOCAL, CENTER], [0, 0, 1.]])
    r = np.diag([1., -1., -1.]) @ np.asarray(pose["camera_to_world"]).T
    return plane_projection(np.asarray(pose["center_m"]), r, k, z)


def measures(matrices):
    h = np.stack(matrices)
    # The image center is (0,0,1) in this coordinate system.
    shift = h[:, :2, 2] * CENTER
    centered = (h - np.eye(3)).reshape(-1, 9)
    return dict(count=len(h), min_center_shift_xy_px=shift.min(0).tolist(),
                max_center_shift_xy_px=shift.max(0).tolist(),
                min_center_shift_norm_px=float(np.linalg.norm(shift, axis=1).min()),
                max_center_shift_norm_px=float(np.linalg.norm(shift, axis=1).max()),
                denominator_gradient_ranges_unit_xy=[h[:, 2, :2].min(0).tolist(),
                                                      h[:, 2, :2].max(0).tolist()],
                singular_values_about_identity=np.linalg.svd(centered, compute_uv=False).tolist())


def geometry():
    train = [(0, 0)] + [(i*s, 0) for i in (12,35,60,80) for s in (-1,1)]
    train += [(roll, orbit) for roll in (0,-35,35) for orbit in (-18,18)]
    val = [(0,0),(-60,0),(60,0),(-25,0),(25,0),(0,-24),(0,24)]
    physical = []
    for row in read(ORBIT / "replay.json")["rows"]:
        ext = row["camera_extrinsics"][0]
        z = row["orbital_pivot_m"][2]
        p0 = physical_plane(ext["0"], z)
        for sign in ("-30", "30"):
            physical.append(norm_h(physical_plane(ext[sign], z) @ np.linalg.inv(p0)))
    return dict(training=measures([source_view(*p) for p in train]),
                validation=measures([source_view(*p) for p in val]),
                original_physical=measures(physical),
                synthesis="Registered 15 positive-focal synthetic views; physical signed image K, fixed reset plane.")


if __name__ == "__main__":
    print(json.dumps(dict(overlap=source_overlap(), calibration=geometry()), indent=2))
