"""Score immutable raw nominal policy queries against independent passive supports."""
import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec, retarget_action_from_motor

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
DATA = Path("/mowenjie/fluffy-bassoon-2/outputs")
NATIVE = "frozen-vla-multiarm-confirmation-20261123-r1"
READERS = "symmeta-perceiver-retry-20261123-r1"
MOTION = "frozen-vla-motion-models-20261123-r1"
CONTACT = "frozen-transfer-object-20261124-r1"
PANELS = {"UR5e": "gpu09test", "IIWA": "gpu09iiwa"}
TASKS = [("libero_10", 5), ("libero_10", 8), ("libero_goal", 3), ("libero_goal", 6),
         ("libero_object", 1), ("libero_object", 8), ("libero_spatial", 0), ("libero_spatial", 8)]
WINDOWS = {"all": slice(None), "first15": slice(0, 15), "after15": slice(15, None),
           "first25": slice(0, 25), "after25": slice(25, None)}
SEEDS = [41, 42, 43]


def require(condition, message):
    if not condition:
        raise ValueError(str(message))


def checked(path, digest):
    data = path.read_bytes()
    require(sha256(data).hexdigest() == digest, path)
    return data


def write(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def geometry(xy, weights, axis, mm):
    if not len(xy):
        return None
    n = int(np.sum(weights))
    energy = float(np.sum(weights * np.sum(xy ** 2, axis=1)))
    weak = float(np.sum(weights * (xy @ axis) ** 2))
    return dict(controls=n, vectors=len(xy), xy_energy=energy, weak_energy=weak,
                fraction=weak / energy if energy else None,
                rms_xy=float(np.sqrt(energy / n)), rms_weak=float(np.sqrt(weak / n)),
                rms_xy_mm=float(mm * np.sqrt(energy / n)), rms_weak_mm=float(mm * np.sqrt(weak / n)))


def spectrum(support):
    _, s, vh = np.linalg.svd(support, full_matrices=False)
    require(s[0] > s[1] and s[0] > 0, "Weak direction is undefined")
    return s, vh[-1]


def load_case(arm, suite, task, idx, panel, refs, matrices):
    key = f"{suite}/{task}"
    ref = refs["arms"][arm]["tasks"][key]
    path = DATA / NATIVE / f"gpu01-{arm}" / f"{suite}-{task}-14-raw.json"
    digest = ref["methods"]["raw"]["receipt_sha256"]
    obj = json.loads(checked(path, digest))
    case, model, rec, reset = obj["case"], obj["model"], obj["record"], obj["full_reset"]
    seed = 41800000 + (idx // 2) * 10000 + task * 100 + 14
    require(case == dict(arm=arm, suite=suite, task_id=task, init_state_id=14, variant="BASE", seed=seed)
            and obj["method"] == "raw" and rec["inferred"] == dict(yaw_deg=0., xy_scale=1., gripper_sign=1), case)
    require(all(reset[k] == v for k, v in ref["released"].items())
            and rec["reset"]["physics"] == reset["initial_state_fingerprint"] == ref["physical_reset"]
            and rec["reset"]["noise_sha256"] == ref["noise_sha256"]
            and rec["opening_proposal_sha256"] == ref["methods"]["raw"]["opening_proposal_sha256"], key)
    schema = rec["reset"]["action_schema"]
    require(model["arm"] == rec["reset"]["arm"] == arm and model["model_class"] == ref["model_class"]
            and model["joint_count"] == ref["joint_count"] and model["gripper"] == "PandaGripper"
            and schema["controller"] == model["controller"] == "OSC_POSE" and schema["control_mode"] == "relative"
            and schema["control_freq"] == model["control_freq"] == 20 and schema["shape"] == [7]
            and schema["low"] == [-1.] * 7 and schema["high"] == [1.] * 7, (arm, schema))
    scale = model["osc_output_max"][:2]
    require(scale[0] == scale[1] and scale[0] > 0, scale)
    preds = [p for p in panel["rows"] if (p["suite"], p["task_id"], p["embodiment"]) == (suite, task, "fam_nom")]
    require([p["seed"] for p in preds] == SEEDS, key)
    sup = matrices[f"{arm}/{key}"]["support"]
    require(sup["start"] == 11 and sup["controls"] == 15 and sup["arm"] == arm and sup["task"] == key
            and sup["seed"] == seed - 10000003 and sup["physics"] != ref["physical_reset"], sup)
    for p in preds:
        require((p["tape_sha256"], p["tape"], p["support_physics"], p["support_seed"], p["support_layout"])
                == (ref["support_sha256"], sup["path"], sup["physics"], sup["seed"], 11)
                and p["tape_sha256"] == sup["sha256"] and p["query"]["arm"] == arm, p["index"])
    spec = get_embodiment_spec("fam_nom")
    require((spec.yaw_deg, spec.xy_scale, spec.gripper_sign) == (0., 1., 1), spec)
    with np.load(io.BytesIO(checked(Path(sup["path"]), sup["sha256"])), allow_pickle=False) as z:
        mask = z["valid"]
        require(mask.dtype == bool and mask[:3].all() and not mask[3:].any()
                and int(z["joint_count"].item()) == model["joint_count"], sup["path"])
        support = z["issued_mean"][mask, :2].astype(float)
    chunks = [np.asarray(c["cmd"], dtype=np.float32) for c in obj["chunks"]]
    require(all(x.shape == (5, 7) for x in chunks[:-1]) and chunks[-1].shape[1] == 7
            and 0 < len(chunks[-1]) <= 5, "chunk length")
    query = np.concatenate(chunks)
    prefix = np.ascontiguousarray(query[:5])
    prefix_digest = sha256(repr((prefix.shape, prefix.dtype.str)).encode() + prefix.tobytes()).hexdigest()
    require(prefix_digest == ref["methods"]["raw"]["issued_first_five_sha256"], (key, "opening controls"))
    require(len(query) == rec["controls"] == ref["methods"]["raw"]["controls"] and len(query) > 25
            and len(query) <= rec["reset"]["horizon"]
            and np.isfinite(query).all() and np.isfinite(support).all()
            and np.abs(query).max() <= 1 and np.abs(support).max() <= 1, key)
    info = dict(arm=arm, task=key, query_path=str(path), query_sha256=digest, case=case, support=sup,
                model_class=model["model_class"], model_xml_sha256=model["xml_sha256"],
                body_joints=model["joint_names"], query_physics=ref["physical_reset"], released=ref["released"],
                noise_sha256=ref["noise_sha256"], opening_proposal_sha256=rec["opening_proposal_sha256"],
                horizon=rec["reset"]["horizon"], controls=len(query), action_schema=schema,
                mm_per_unit=scale[0] * 1000, frozen_motor=refs["arms"][arm]["runtime"]["openpi_checkpoint"])
    return support, query, preds, info


def phase_masks(actions, info, contacts):
    close = actions[:, 6] > 0
    idx = np.flatnonzero(close)
    first = int(idx[0]) if len(idx) else len(close)
    times = np.arange(len(close))
    name = f"target/{info['arm']}-{info['task'].replace('/', '-')}-tape14-layout14"
    line = json.load(contacts.extractfile(name + ".lineage.json"))
    truth = json.load(contacts.extractfile(name + ".truth.json"))
    require(line["spec"] == info["case"] and line["original"] and line["layout"] == 14
            and line["model"]["arm"] == info["arm"] and line["input_sha256"] == info["query_sha256"]
            and line["source_controls"] == len(actions) and line["seed"] == info["case"]["seed"]
            and all(line["reset"][k] == v for k, v in info["released"].items()), name)
    carry = np.zeros(len(close), dtype=bool)
    for row in truth:
        end = row["control"]
        require(5 <= end <= min(len(actions), line["cpu_controls"]) and end % 5 == 0
                and close[end-5:end].all(), (name, end))
        held = set.intersection(*(set(e["carry"]) for e in row["endpoints"]))
        require(len(row["endpoints"]) == 5 and bool(held) == (row["y"] == 1), (name, end))
        if held:
            carry[end-5:end] = True
    fidelity = {k: line[k] for k in ("cpu_controls", "source_controls", "tcp_delta_mm", "jaw_delta_mm")}
    fidelity.update(reward_tape_equal=line["cpu_rewards"] == line["original_rewards"],
                    eligible_closed_blocks=len(truth), carry_blocks=int(carry.sum() // 5),
                    first_close_control=first + 1 if len(idx) else None)
    return dict(approach=times < first, close_hold=close, reopen=(times >= first) & ~close, replay_carry=carry), fidelity


def fitted_errors(query, preds, mm):
    low, high = np.full(7, -1, dtype=np.float32), np.ones(7, dtype=np.float32)
    oracle = retarget_action_from_motor(query, 0., 1., 1, low, high)
    np.testing.assert_array_equal(oracle, query)
    rows = []
    for p in preds:
        for reader in ("symmeta", "tuned"):
            est = p[reader]
            corrected = retarget_action_from_motor(query, est["yaw_deg"], est["xy_scale"], est["gripper_sign"], low, high)
            diff = corrected[:, :2].astype(float) - oracle[:, :2]
            rows.append(dict(seed=p["seed"], reader=reader, estimate=est,
                             rms_error_mm=float(mm * np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))),
                             rel_rms_error=float(np.linalg.norm(diff) / np.linalg.norm(oracle[:, :2].astype(float)))))
    return rows


def score_case(data, axes, index, contacts):
    support, action, preds, info = data
    mm = info["mm_per_unit"]
    xy = action[:, :2].astype(float)
    s, v = spectrum(support)
    support_stats = geometry(support, np.full(3, 5), v, mm)
    np.testing.assert_allclose(support_stats["fraction"], s[1] ** 2 / (s @ s))
    chunk_stats, step_stats = {}, {}
    for name, selection in WINDOWS.items():
        window = xy[selection]
        blocks = [window[i:i+5] for i in range(0, len(window), 5)]
        means = np.stack([b.mean(axis=0) for b in blocks])
        weights = np.array([len(b) for b in blocks])
        chunk_stats[name] = geometry(means, weights, v, mm)
        step_stats[name] = geometry(window, np.ones(len(window)), v, mm)
        if name == "all":
            full_means, full_weights = means, weights
    # The time cuts are multiples of the archive's five-control chunk period.
    for group in (chunk_stats, step_stats):
        for split in (15, 25):
            for field in ("xy_energy", "weak_energy", "controls"):
                np.testing.assert_allclose(group[f"first{split}"][field] + group[f"after{split}"][field], group["all"][field])
        for window in group.values():
            window["share_of_query_xy_energy"] = window["xy_energy"] / group["all"]["xy_energy"]
    other = [geometry(full_means, full_weights, axis, mm)["fraction"] for i, axis in enumerate(axes) if i != index]
    rot = np.array([[0., -1.], [1., 0.]])
    rotated_axis = spectrum(support @ rot)[1]
    reexpressed = geometry(full_means @ rot, full_weights, rotated_axis, mm)["fraction"]
    np.testing.assert_allclose(reexpressed, chunk_stats["all"]["fraction"], atol=1e-14)
    masks, fidelity = phase_masks(action, info, contacts)
    phases = {k: geometry(xy[mask], np.ones(mask.sum()), v, mm) for k, mask in masks.items()}
    for name, phase in phases.items():
        if phase:
            phase["share_of_query_xy_energy"] = phase["xy_energy"] / step_stats["all"]["xy_energy"]
            if name == "replay_carry":
                block_means = xy[masks[name]].reshape(-1, 5, 2).mean(axis=1)
                phase["chunk_metric"] = geometry(block_means, np.full(len(block_means), 5), v, mm)
    return dict(**info, support_xy=support.tolist(), singular_values=s.tolist(), weak_axis=v.tolist(),
                singular_ratio=float(s[1] / s[0]), support_stats=support_stats, chunks=chunk_stats, steps=step_stats,
                wrong_task_fractions=other, isotropic_fraction=0.5, rotated_frame_fraction=reexpressed,
                phases=phases, replay_fidelity=fidelity, fitted=fitted_errors(action, preds, mm))


def estimate(values):
    a = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(a):
        return None
    draws = np.random.default_rng(1124).choice(a, (20000, len(a)))
    return dict(tasks=len(a), mean=float(a.mean()), ci95=np.quantile(draws.mean(axis=1), [.025, .975]).tolist(),
                positive=int(np.sum(a > 0)))


def task_metrics(r):
    sup = r["support_stats"]
    out = dict(support_fraction=sup["fraction"], support_rms_xy_mm=sup["rms_xy_mm"],
               support_rms_weak_mm=sup["rms_weak_mm"], support_xy_energy=sup["xy_energy"])
    for group in ("chunks", "steps"):
        for window, value in r[group].items():
            for field in ("fraction", "rms_xy_mm", "rms_weak_mm", "xy_energy", "share_of_query_xy_energy"):
                out[f"{group}_{window}_{field}"] = value[field]
        for h in (15, 25):
            out[f"{group}_late_minus_early_{h}"] = r[group][f"after{h}"]["fraction"] - r[group][f"first{h}"]["fraction"]
            out[f"{group}_late_minus_support_{h}"] = r[group][f"after{h}"]["fraction"] - sup["fraction"]
    out["primary_query_minus_support"] = r["chunks"]["all"]["fraction"] - sup["fraction"]
    out["wrong_task_fraction"] = float(np.mean(r["wrong_task_fractions"]))
    out["matched_minus_wrong"] = r["chunks"]["all"]["fraction"] - out["wrong_task_fraction"]
    for k, phase in r["phases"].items():
        for field in ("fraction", "rms_weak_mm", "rms_xy_mm", "share_of_query_xy_energy"):
            out[f"{k}_{field}"] = phase[field] if phase else None
    carry = r["phases"]["replay_carry"]
    out["carry_chunk_minus_support"] = carry["chunk_metric"]["fraction"] - sup["fraction"] if carry else None
    for reader in ("symmeta", "tuned"):
        out[f"{reader}_rms_error_mm"] = float(np.mean([f["rms_error_mm"] for f in r["fitted"] if f["reader"] == reader]))
    out["tuned_minus_symmeta_mm"] = out["tuned_rms_error_mm"] - out["symmeta_rms_error_mm"]
    return out


def source_residual(refs):
    src = json.loads(checked(CODE / MOTION / "evidence/source-fit.json", refs["source_fit_sha256"]))
    errors, commands = [], []
    for row in src["sources"]:
        with np.load(io.BytesIO(checked(Path(row["path"]), row["sha256"])), allow_pickle=False) as z:
            mask = z["valid"]
            require(mask[:3].all() and not mask[3:].any(), row["path"])
            x = z["issued_mean"][mask, :3].astype(float)
            y = z["delta_eef"][mask].astype(float)
        design = np.column_stack((x, np.vstack((np.zeros(3), x[:-1]))))
        errors.append(y - design @ np.asarray(src["prior"]))
        commands.append(x)
    e = np.concatenate(errors)
    require(e.shape == (576, 3) and src["counts"] == {"Panda": 96, "Sawyer": 96}, "source panel")
    ratio = float(np.sum(e**2) / np.sum(np.asarray(commands)**2))
    np.testing.assert_allclose(ratio, src["residual_power"], rtol=1e-10)
    return dict(n_supports=len(errors), n_chunks=len(e), training_source_sha256=refs["source_fit_sha256"],
                residual_power_per_unit_command_energy=ratio,
                xy_rms_mm=float(1000 * np.sqrt(np.mean(np.sum(e[:, :2]**2, axis=1)))),
                xyz_rms_mm=float(1000 * np.sqrt(np.mean(np.sum(e**2, axis=1)))))


def main(out):
    refs = json.loads((CODE / NATIVE / "evidence/production-native.json").read_bytes())
    matrices = json.loads(checked(CODE / MOTION / "evidence/matrices.json", refs["matrix_file_sha256"]))
    contract = json.loads((CODE / READERS / "evidence/prepare01pro/contract.json").read_bytes())
    contact_bytes = checked(CODE / CONTACT / "evidence/trajectories.tar.gz",
                            "56bed093a4f48f5f293f342541878dd18929fd1c9f64093d7a2005d5ffa52f44")
    scored, summary = [], {}
    with tarfile.open(fileobj=io.BytesIO(contact_bytes), mode="r:gz") as contacts:
        for arm, panel_id in PANELS.items():
            panel_path = DATA / READERS / "prepare01pro" / panel_id / "panel.json"
            panel = json.loads(checked(panel_path, contract["panels"][panel_id]))
            require(panel["runtime"]["arm"] == arm and panel["runtime"]["policy"] == "pi05", panel_id)
            batch = [load_case(arm, suite, task, i, panel, refs, matrices) for i, (suite, task) in enumerate(TASKS)]
            axes = [spectrum(data[0])[1] for data in batch]
            rows = [score_case(data, axes, i, contacts) for i, data in enumerate(batch)]
            metrics = [task_metrics(r) for r in rows]
            for r, m in zip(rows, metrics, strict=True):
                r["task_metrics"] = m
            scored.extend(rows)
            summary[arm] = {key: estimate([m[key] for m in metrics]) for key in metrics[0]}
    write(out / "tasks.json", scored)
    write(out / "summary.json", dict(arms=summary, source_residual=source_residual(refs),
                                     query_collector_pin=refs["executable"], openpi_pin=refs["openpi"],
                                     libero_pro_pin=refs["libero_pro"], bootstrap_seed=1124, draws=20000))
    print(json.dumps({a: v["primary_query_minus_support"] for a, v in summary.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    main(parser.parse_args().out)
