"""Read sealed native Goal 5 arrays on one CPU; emit compact JSON to stdout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import h5py
import numpy as np
import ml_dtypes
import tensorstore as ts

MODES = ("stock", "panda", "mixed")
AXES = ("dx", "dy", "dz", "rx", "ry", "rz", "grip")
STATES = ("physics", "robot0_eef_pos", "robot0_eef_quat_xyzw",
          "robot0_gripper_qpos", "robot0_joint_pos", "robot0_joint_vel")
UNITS = np.tile(np.array([1] * 7 + [100] * 5 + [1], np.float32), 4)
MOTION_TIMES = (6, 10, 25, 50, 100, 300)
NORM = "assets/physical-intelligence/libero/norm_stats.json"


def require(ok, context):
    if not ok:
        raise ValueError(context)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def digest(a, head=False):
    a = np.ascontiguousarray(a)
    prefix = (str((str(a.dtype), a.shape)) + "\0" if head else repr((a.shape, a.dtype.str))).encode()
    return hashlib.sha256(prefix + a.tobytes()).hexdigest()


def history(actions, xyz, fingers, t):
    end = np.float32(xyz[max(0, t - 4):t + 1])
    jaw = np.abs(np.diff(np.float32(fingers[max(0, t - 4):t + 1]), axis=1))
    rows = np.column_stack((actions[max(0, t - 4):t], np.diff(end, axis=0),
                            np.diff(jaw, axis=0), jaw[1:], np.ones(len(end) - 1)))
    return np.pad(rows, ((5 - len(end), 0), (0, 0))).astype(np.float32).reshape(52)


def offset(head, h):
    return (np.tanh((h * UNITS) @ head["history_in"]) @ head["history_out"]).reshape(-1, 10, 32)


def source(base, data_seal, heads, q, scale):
    tapes = base / "data/gpu01/tapes"
    dataset = base / "data/dataset-v1"
    replay = {}
    for layout in data_seal["layouts"]:
        for body in ("Panda", "Sawyer"):
            key = (body, layout["layout_seed"])
            path = tapes / f"{key[1]}-{body}.h5"
            info = read(path.with_suffix(".json"))
            with h5py.File(path, "r") as f:
                a, p, g, h, reward = (f[k][:] for k in
                                      ("actions", "robot0_eef_pos", "robot0_gripper_qpos", "history", "rewards"))
            require(sha(Path(info["archive"])) == info["archive_sha256"], f"archived teacher {key}")
            with np.load(info["archive"], allow_pickle=False) as original:
                require(np.array_equal(a, original["actions"]), f"teacher/issued controls {key}")
            require(info["exact_issued_prefix"] and info["native_success"] and reward[-1] == 1
                    and not reward[:-1].any() and np.abs(a).max() <= 1, f"teacher terminal/bounds {key}")
            require(np.array_equal(h, [history(a, p, g, t) for t in range(len(a) + 1)]),
                    f"source history {key}")
            replay[key] = (a, p, g, h, info)
    splits = {}
    early = {}
    members = list(data_seal["variants"].values()) + list(data_seal["diagnostics"].values())
    for entry in members:
        variant = entry["root"]
        index = dataset / variant / "manifest.json"
        require(sha(index) == entry["manifest_sha256"], f"source manifest {variant}")
        manifest = read(index)
        for split, metadata in manifest["splits"].items():
            starts = []
            count = 0
            with h5py.File(dataset / variant / metadata["file"], "r") as f:
                for traj in metadata["trajectories"]:
                    a, p, g, h, _ = replay[traj["body"], traj["layout_seed"]]
                    with h5py.File(tapes / f"{traj['layout_seed']}-{traj['body']}.h5", "r") as native:
                        for t in traj["t"]:
                            require(t + 10 <= len(a) and np.array_equal(f["actions"][count], a[t:t + 10])
                                    and np.array_equal(f["history"][count], h[t])
                                    and np.array_equal(f["robot0_eef_pos"][count], p[t])
                                    and np.array_equal(f["robot0_gripper_qpos"][count], g[t])
                                    and np.array_equal(f["robot0_eef_quat_xyzw"][count],
                                                       native["robot0_eef_quat_xyzw"][t]),
                                    f"packed source row {variant}/{split}/{count}")
                            count += 1
                            starts.append(t)
            require(count == metadata["examples"], f"source examples {variant}/{split}")
            splits[f"{variant}/{split}"] = {
                "windows": count, "t_mod_5": np.bincount(np.array(starts) % 5, minlength=5).tolist(),
                "history_ages": np.bincount(np.minimum(starts, 4), minlength=5).tolist(),
            }
    for body in ("Panda", "Sawyer"):
        group = [r for (b, _), r in replay.items() if b == body]
        a = np.concatenate([r[0] for r in group])
        h = np.array([r[3][4] for r in group])
        commands = np.array([r[0][4] for r in group])
        early[body] = {
            "source_opening_t": 4,
            "layouts": len(group),
            "teacher_controls": len(a),
            "gripper_unique": np.unique(a[:, -1]).tolist(),
            "teacher_command_min": a.min(axis=0).tolist(),
            "teacher_command_max": a.max(axis=0).tolist(),
            "teacher_fraction_outside_checkpoint_q01_q99":
                ((a < q["q01"]) | (a > q["q99"])).mean(axis=0).tolist(),
            "early_gripper_positive": int((commands[:, 6] > 0).sum()),
            "mean_early_issued_command": h.reshape(-1, 4, 13)[:, :, :7].mean(axis=(0, 1)).tolist(),
            "opening_width_m": summarize(h[:, -2]),
            "direct_decode_term_axes": {
                mode: summarize((-offset(heads[mode], h)[:, 0, :7] * scale)) for mode in heads
            },
        }
    measured_fit = read(base / "fit/gpu03/source-after.json")
    return {
        "teacher_tapes": len(replay), "teacher_controls": sum(len(r[0]) for r in replay.values()),
        "all_rows_native_and_causal": True, "splits": splits, "source_history": early,
        "heldout_source_fit": measured_fit,
    }


def summarize(x):
    x = np.asarray(x, np.float64)
    rng = np.random.default_rng(20261124)
    sample = rng.integers(0, len(x), size=(10000, len(x)))
    return {
        "n": len(x), "mean": x.mean(axis=0).tolist(), "median": np.median(x, axis=0).tolist(),
        "q25_q75": np.quantile(x, (0.25, 0.75), axis=0).tolist(),
        "mean_bootstrap_95": np.quantile(x[sample].mean(axis=1), (0.025, 0.975), axis=0).tolist(),
        "min": x.min(axis=0).tolist(), "max": x.max(axis=0).tolist(),
    }


def bound_proposal(call):
    return np.clip(call["proposed"], -1, 1).astype(np.float32)


def case_read(path, manifest, original, heads, scale, goal):
    seal = read(path / "complete.json")
    require(sha(path / "complete.json") == original["complete_sha256"], f"{path} seal")
    rng = np.random.default_rng(seal["noise_seed"])
    noise = rng.standard_normal((60, 10, 32)).astype(np.float32)
    packets, arrays = {}, {}
    for mode in MODES:
        j = path / f"{mode}.json"
        packet = packets[mode] = read(j)
        require(sha(j) == seal["receipts"][mode] and packet["tape_sha256"] == original["trajectory_sha256"][mode]
                and packet["controls"] == original["controls"][mode], f"{path}/{mode} readback binding")
        with h5py.File(path / f"{mode}.h5", "r") as f:
            a = f["issued_action"][:]
            require(np.array_equal(f["worker_step_index"][:], np.arange(1, len(a) + 1)),
                    f"{path}/{mode} control indices")
            states = [f[name][:] for name in STATES]
            reward = int(f["reward"][-1])
        require(len(a) == packet["controls"] and np.isfinite(a).all() and np.abs(a).max() <= 1,
                f"{path}/{mode} action shape/bounds")
        arrays[mode] = (a, states, reward)
        require(len(packet["calls"]) == (len(a) + 4) // 5, f"{path}/{mode} replan count")
        for replan, call in enumerate(packet["calls"]):
            t, n = call["t"], call["issued_count"]
            require(t == replan * 5 and n == min(5, len(a) - t) and call["noise"] == digest(noise[replan])
                    and np.array_equal(bound_proposal(call)[:n], a[t:t + n])
                    and np.array_equal(call["history"], history(a, states[1], states[3], t)),
                    f"{path}/{mode} control and causal history {t}")
    stock, reference, _ = arrays["stock"]
    pairs = {}
    for mode in ("panda", "mixed"):
        a, states, _ = arrays[mode]
        joined = min(len(a), len(stock))
        differences = np.flatnonzero(np.any(a[:joined] != stock[:joined], axis=-1))
        require(differences.size > 0, f"{path}/{mode} no first divergence")
        t = int(differences[0])
        require(t >= 5 and all(np.array_equal(s[:t + 1], r[:t + 1])
                               for s, r in zip(states, reference, strict=True))
                and packets[mode]["reset"] == packets["stock"]["reset"],
                f"{path}/{mode} divergent opening")
        c, b = packets[mode]["calls"][t // 5], packets["stock"]["calls"][t // 5]
        require(all(c[k] == b[k] for k in ("t", "observation", "noise", "history"))
                and packets[mode]["calls"][0]["proposed"] == packets["stock"]["calls"][0]["proposed"]
                and c["route"]["hlo_sha256"] in original["compiled_history"], f"{path}/{mode} unequal input")
        h = np.float32(c["history"])
        slots = h.reshape(4, 13)
        aligned = slots.copy()
        aligned[:, 6] = aligned[:, 12]
        valid_only = np.zeros_like(aligned)
        valid_only[:, 12] = aligned[:, 12]
        actual, grip_aligned, validity_only = offset(
            heads[mode], np.stack((h, aligned.ravel(), valid_only.ravel()))
        )
        direct = -actual[:, :7] * scale
        first_alias_term = -(actual - grip_aligned)[0, :7] * scale
        post_opening_closes = np.flatnonzero(a[5:, -1] > 0)
        pairs[mode] = {
            "first_unequal_t_zero_based": t, "first_delta": (a[t] - stock[t]).tolist(),
            "replan_issued_delta": (bound_proposal(c)[:5] - bound_proposal(b)[:5]).tolist(),
            "replan_decoded_delta": (np.array(c["proposed"]) - b["proposed"]).tolist(),
            "first_gripper_sign_change": bool(np.sign(a[t, -1]) != np.sign(stock[t, -1])),
            "any_5_gripper_sign_change": bool(np.any(np.sign(bound_proposal(c)[:5, -1]) !=
                                                      np.sign(bound_proposal(b)[:5, -1]))),
            "first_positive_grip_t_zero_based": int(post_opening_closes[0]) + 5 if post_opening_closes.size else None,
            "direct_first_decode_term": direct[0].tolist(),
            "direct_grip_alias_delta_first": first_alias_term.tolist(),
            "max_abs_grip_alias_offset_delta_decoded_5": float(np.max(np.abs((actual - grip_aligned)[:5, :7] * scale))),
            "max_abs_validity_only_offset_delta_decoded_5": float(np.max(np.abs((actual - validity_only)[:5, :7] * scale))),
        }
    mixed = packets["mixed"]
    d = mixed["diagnostic"]
    require(d["t"] == 5 and d["learned"] == mixed["calls"][1]["proposed"]
            and d["stock"] == packets["stock"]["calls"][1]["proposed"]
            and d["zero_route"]["hlo_sha256"] == mixed["calls"][1]["route"]["hlo_sha256"],
            f"{path} diagnostic binding")
    proposals = {m: np.array(p["calls"][1]["proposed"]) for m, p in packets.items()}
    proposals["zero"] = np.array(d["zero_add"])
    clipped = {m: np.clip(p, -1, 1) for m, p in proposals.items()}
    zero = {
        rel: {
            "first_issued_delta": (clipped[left][0] - clipped[right][0]).tolist(),
            "max_decoded_10": float(np.max(np.abs(proposals[left] - proposals[right]))),
            "max_clipped_5": float(np.max(np.abs(clipped[left][:5] - clipped[right][:5]))),
        } for rel, left, right in (("zero_minus_stock", "zero", "stock"), ("mixed_minus_zero", "mixed", "zero"))
    }
    donor = np.array(manifest["donor_state"])
    plate = manifest["sampled_scene"]["joints"]["plate_1_joint0"]
    xyz = np.array(plate["qpos"])
    nq = packets["stock"]["reset"]["info"]["model_identity"]["model_nq"]
    candidates = [i for i in range(1, nq - 5) if np.array_equal(donor[i:i + 7], xyz)]
    require(plate["type"] == 0 and plate["body"] == "plate_1_main" and len(candidates) == 1
            and np.array_equal(xyz[:3], manifest["movable_xyz_m"]["plate_1"]), f"{path} plate joint map")
    i, = candidates
    require(np.linalg.norm(reference[0][0, i:i + 2] - xyz[:2]) <= 0.002, f"{path} settled plate XY")
    trajectories = {}
    for mode, (a, s, reward) in arrays.items():
        p = s[0][:, i:i + 3]
        e = s[1]
        stop = np.minimum(MOTION_TIMES, len(a))
        trajectories[mode] = {
            "controls": len(a), "success": reward,
            "first_issued": a[5].tolist(),
            "outside_bounds_by_axis_5": (np.abs(proposals[mode][:5]) > 1).sum(axis=0).tolist(),
            "outside_bounds_by_axis_after_opening": (np.abs(np.concatenate([
                np.asarray(c["proposed"])[:c["issued_count"]] for c in packets[mode]["calls"][1:]
            ])) > 1).sum(axis=0).tolist(),
            "positive_gripper_controls": int((a[5:, -1] > 0).sum()),
            "plate_net_xy_mm": (1000 * np.linalg.norm(p[stop, :2] - p[5, :2], axis=1)).tolist(),
            "plate_goal_center_progress_mm": (1000 * (np.linalg.norm(p[5, :2] - goal)
                                                - np.linalg.norm(p[stop, :2] - goal, axis=1))).tolist(),
            "ee_delta_xyz_mm": (1000 * (e[stop] - e[5])).tolist(),
            "ee_plate_xy_gap_mm": (1000 * np.linalg.norm(e[stop, :2] - p[stop, :2], axis=1)).tolist(),
        }
    return {
        "case_id": seal["case_id"], "layout_seed": seal["layout_seed"], "plate_flat_qpos_offset": i,
        "shared_t5_width_m": float(np.abs(np.diff(reference[3][5]))[0]),
        "shared_t5_gripper_history": stock[1:5, 6].tolist(),
        "shared_t5_mean_history_command": np.array(packets["stock"]["calls"][1]["history"]).reshape(4, 13)[:, :7].mean(axis=0).tolist(),
        "arm": trajectories, "paired": pairs, "compiled_diagnostic": zero,
    }


def aggregate(rows):
    out = {}
    for mode in MODES:
        a = [r["arm"][mode] for r in rows]
        b = {}
        for key in ("first_issued", "plate_net_xy_mm", "plate_goal_center_progress_mm", "ee_delta_xyz_mm", "ee_plate_xy_gap_mm"):
            samples = np.array([v[key] for v in a])
            b[key] = summarize(samples)
            if mode != "stock":
                b[f"{key}_minus_stock"] = summarize(samples - np.array([r["arm"]["stock"][key] for r in rows]))
        for k in ("outside_bounds_by_axis_5", "outside_bounds_by_axis_after_opening", "positive_gripper_controls", "success"):
            b[k] = np.sum([v[k] for v in a], axis=0).tolist()
        b["first_5_positive_grip_layouts"] = sum(v["first_issued"][-1] > 0 for v in a)
        b["eligible_controls_after_opening"] = sum(v["controls"] - 5 for v in a)
        b["layouts_reaching_each_motion_time"] = [sum(v["controls"] >= t for v in a) for t in MOTION_TIMES]
        if mode != "stock":
            pair = [r["paired"][mode] for r in rows]
            b["first_unequal_t_counts"] = {str(t): sum(v["first_unequal_t_zero_based"] == t for v in pair)
                                           for t in sorted({p["first_unequal_t_zero_based"] for p in pair})}
            b["first_absolute_axis_delta"] = summarize(np.abs([v["first_delta"] for v in pair]))
            b["first_axis_delta_positive"] = (np.array([v["first_delta"] for v in pair]) > 0).sum(axis=0).tolist()
            b["first_gripper_sign_changes"] = sum(p["first_gripper_sign_change"] for p in pair)
            b["any_5_gripper_sign_changes"] = sum(p["any_5_gripper_sign_change"] for p in pair)
            closes = [p["first_positive_grip_t_zero_based"] for p in pair if p["first_positive_grip_t_zero_based"] is not None]
            b["first_positive_grip_t_zero_based"] = summarize(closes) if closes else None
            b["replan_max_clipped_5"] = summarize(np.max(np.abs([v["replan_issued_delta"] for v in pair]), axis=(1, 2)))
            b["replan_max_decoded_10"] = summarize(np.max(np.abs([v["replan_decoded_delta"] for v in pair]), axis=(1, 2)))
            for k in ("direct_first_decode_term", "direct_grip_alias_delta_first",
                      "max_abs_grip_alias_offset_delta_decoded_5", "max_abs_validity_only_offset_delta_decoded_5"):
                b[k] = summarize([v[k] for v in pair])
        out[mode] = b
    for rel in ("zero_minus_stock", "mixed_minus_zero"):
        out[rel] = {k: summarize([r["compiled_diagnostic"][rel][k] for r in rows])
                    for k in rows[0]["compiled_diagnostic"][rel]}
    out["shared_t5_width_m"] = summarize([r["shared_t5_width_m"] for r in rows])
    grip = np.array([r["shared_t5_gripper_history"] for r in rows])
    out["shared_t5_negative_grip_history_slots"] = int((grip < 0).sum())
    out["shared_t5_grip_history_slots"] = int(grip.size)
    out["shared_t5_mean_history_command"] = summarize([r["shared_t5_mean_history_command"] for r in rows])
    return out


def projection_digest(checkpoint):
    spec = {
        "driver": "zarr", "kvstore": {"driver": "ocdbt", "base": f"file://{checkpoint}/params/"},
        "path": "params.action_out_proj.kernel.value",
    }
    kernel = ts.open(spec, open=True).result().read().result().astype(ml_dtypes.bfloat16)
    return {"shape": list(kernel.shape), "bf16_sha256": digest(kernel, head=True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--openpi", type=Path, required=True)
    parser.add_argument("--bddl", type=Path, required=True)
    args = parser.parse_args()
    require(len(os.sched_getaffinity(0)) == 1 and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "require one Linux CPU and no visible CUDA")
    base, fit = args.base, args.base / "fit/gpu03"
    checkpoint = args.checkpoint
    family = read(fit / "fit.json")
    hashes = family["base"]
    require(sha(checkpoint / "READY.json") == hashes["checkpoint_manifest"]
            and sha(checkpoint / NORM) == hashes["normalization"], "base identity")
    for entry in read(checkpoint / "READY.json")["files"]:
        require((checkpoint / entry["name"]).stat().st_size == entry["bytes"], "checkpoint file size")
    source_file = args.openpi / "src/openpi/models/pi0.py"
    git = ["git", "-C", str(args.openpi)]
    require(subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() == hashes["openpi_revision"]
            and subprocess.call([*git, "diff", "--quiet", "HEAD", "--", "src", "packages"]) == 0,
            "pinned OpenPI checkout")
    q = read(checkpoint / NORM)["norm_stats"]["actions"]
    scale = (np.array(q["q99"]) - q["q01"] + 1e-6) / 2
    data_seal = base / "data/dataset-v1/seal.json"
    require(sha(data_seal) == family["data_seal_sha256"], "source seal")
    audit = base / "eval/gpu01s3/audit/readback-64.json"
    inventory = read(audit)
    frozen = inventory["frozen"]
    require(sha(fit / "fit.json") == frozen["files"]["fit.json"]
            and frozen["base"] == hashes and frozen["resident_stock"] == family["resident_stock"]
            and frozen["deployed_stock"] == family["deployed_stock"], "frozen deployment identity")
    projection = projection_digest(checkpoint)
    require(projection["bf16_sha256"] == frozen["deployed_kernel"], "original checkpoint/deployed projection")
    heads = {}
    for mode in MODES[1:]:
        f = fit / mode / "head.npz"
        require(sha(f) == frozen["files"][f"{mode}/head.npz"], f"export {mode}")
        with np.load(f, allow_pickle=False) as archive:
            heads[mode] = dict(archive)
        require({k: digest(v, head=True) for k, v in heads[mode].items()} == frozen["resident_heads"][mode],
                f"resident exported arrays {mode}")
    manifest = base / "eval/prepare01/panel/manifest.json"
    cases = read(manifest)["cases"]
    bddl = args.bddl.read_text()
    bounds = re.search(r"\(stove_front_region\s+\(:target main_table\)\s+\(:ranges\s*\(\s*\(([^()]+)\)", bddl)
    require(bounds and "(And (On plate_1 main_table_stove_front_region))" in bddl, "plate goal geometry")
    rectangle = np.array([float(v) for v in bounds.group(1).split()]).reshape(2, 2)
    target_center = rectangle.mean(axis=0)
    bddl_sha = sha(args.bddl)
    rows = []
    require([x["case_id"] for x in inventory["audits"]] == list(range(64)), "closed 64-case inventory")
    for prior in inventory["audits"]:
        i = prior["case_id"]
        row = cases[i]
        path = Path(prior["pfs"])
        require(row["index"] == i and row["bddl_sha256"] == bddl_sha
                and path == base / f"eval/gpu01s{i % 4}/shard{i % 4}/case-{i:05}", f"case {i} lineage")
        rows.append(case_read(path, row, prior, heads, scale, target_center))
    output = {
        "cpu_affinity": list(os.sched_getaffinity(0)), "gpu": 0,
        "native_action_axes": AXES, "motion_control_counts": MOTION_TIMES,
        "goal_rectangle_world_xy_m": rectangle.tolist(),
        "input_sha256": {"source_seal": sha(data_seal), "native_readback": sha(audit), "manifest": sha(manifest),
                         "bddl": bddl_sha, "openpi_pi0_py": sha(source_file), **frozen["files"], **hashes},
        "frozen_stock_array_sha256": frozen["resident_stock"],
        "base_checkpoint_projection": projection,
        "action_q01": q["q01"], "action_q99": q["q99"], "decode_slope": scale.tolist(),
        "source": source(base, read(data_seal), heads, q, scale), "target": aggregate(rows),
        "cases": rows,
    }
    print(json.dumps(output, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
