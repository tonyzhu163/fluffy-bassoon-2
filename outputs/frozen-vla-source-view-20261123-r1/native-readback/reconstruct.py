"""Read-only CPU reconstruction. Run on Baige with the existing OpenPI Python and the PFS run as argv[1]."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

RUN = Path(sys.argv[1])
GPU = RUN / "gpu01"
CPU = RUN / "cpu01"
MODES = ("raw", "control", "panda")
STEPS = (0, 30, 45, 60, 90)
CAMS = ("agentview", "robot0_eye_in_hand")
FIELDS = ("observation/image", "observation/wrist_image", "observation/state")
PROMPT = "open the top layer of the drawer and put the bowl inside"


def load(path):
    return json.loads(path.read_text())


def check(test, label):
    if not test:
        raise ValueError(label)


def digest(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(repr((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def case(layout):
    return dict(arm="Kinova3", suite="libero_goal", task_id=3, variant="BASE",
                init_state_id=layout, seed=41810300 + layout)


def noise_bank(layout):
    rng = np.random.default_rng(41810300 + layout + 817117)
    return rng.standard_normal((60, 10, 32)).astype(np.float32)


def outside(a, b):
    return sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))


def geometry(row, source):
    timing = row.get("timing", row.get("exposure"))
    check(max(timing["reconstruction"].values()) <= 1e-12, "camera exposure reconstruction")
    for name in (("control", "panda") if source else ("control",)):
        g = row[name]
        check(g["admissible"] and g["task_body_count"] == 15 and g["jaw_error_m"] == 0
              and max(g["scene_max_m"], g["scene_max_rad"]) <= 1e-12, name + " geometry")
    camera = row.get("cameras", row.get("images"))
    check(set(camera) == set(CAMS), "camera labels")
    for c in camera.values():
        check(c["raw_vs_full_control"]["changed"] == 0, "full Kinova render")
        if source:
            check(c["raw_vs_sham"]["changed"] == c["clean_plate"]["changed"] == c["clean_depth_max"]
                  == c["changed_outside_union"] == c["visible_depth_behind_px"] == 0, "camera isolation")
    if source:
        p = row["panda"]
        fit = p["attempts"][-1]
        check(fit["pose_ok"] and fit["min_limited_margin_rad"] >= 0 and not fit["arm_collision"]
              and not any(c["kind"] in ("arm_scene", "arm_self_or_base") for c in fit["contacts"]), "admitted pose")
        return dict(external_pixels=camera[CAMS[0]]["sham_vs_swap"]["changed"],
                    wrist_pixels=camera[CAMS[1]]["sham_vs_swap"]["changed"],
                    tcp_position_error_m=p["tcp_error_m"], tcp_rotation_error_rad=p["tcp_error_rad"],
                    min_joint_margin_rad=fit["min_limited_margin_rad"])


def frame_inputs(layout, step, record):
    path = CPU / f"s{layout}-t{step:03}.npz"
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    geom = geometry(record, True)
    with np.load(path, allow_pickle=False) as z:
        q = z["robot0_eef_quat_xyzw"].astype(np.float64)
        w = np.clip(q[3], -1., 1.)
        den = np.sqrt(1. - w * w)
        rot = q[:3] * 2. * np.arccos(w) / den if den > 1e-10 else np.zeros(3)
        state = np.concatenate((z["robot0_eef_pos"], rot, z["robot0_gripper_qpos"]))
        inputs = {m: {FIELDS[2]: digest(state)} for m in MODES}
        pixels = {}
        for cam, field, key in zip(CAMS, FIELDS[:2], ("external_pixels", "wrist_pixels")):
            native = {m: z[f"{m}_{cam}"] for m in MODES}
            check(all(a.dtype == np.uint8 and a.shape == (256, 256, 3) for a in native.values()), "RGB shape")
            check(np.array_equal(native["raw"], native["control"]), "raw vs control RGB")
            diff = np.any(native["panda"] != native["control"], axis=-1)
            check(int(diff.sum()) == geom[key], "camera pixel count")
            imgs = {m: np.asarray(Image.fromarray(a[::-1, ::-1]).resize((224, 224), Image.Resampling.BILINEAR))
                    for m, a in native.items()}
            pixels[cam] = int(np.any(imgs["control"] != imgs["panda"], axis=-1).sum())
            for m in MODES:
                inputs[m][field] = digest(imgs[m])
    check(inputs["raw"] == inputs["control"] and outside(inputs["control"], inputs["panda"]) == [FIELDS[0]],
          "selective motor image intervention")
    noise = digest(noise_bank(layout)[step // 5])
    cpu = load(RUN / "cpu03" / f"motor-input-{layout}-{step:03}.json")
    check(cpu["hashes"] == inputs and cpu["noise"] == noise, "qualified motor input and noise")
    return dict(npz_sha256=file_hash, input_hashes=inputs, noise=noise, pixels_224=pixels, geometry=geom)


def proposal(layout, step, inputs):
    path = GPU / f"proposals-{layout}-{step:03}.json"
    if not path.exists():
        return None
    r = load(path)
    check((r["layout"], r["step"], r["instruction"]) == (layout, step, PROMPT)
          and r["sha256"] == inputs["npz_sha256"] and r["input_hashes"] == inputs["input_hashes"]
          and r["noise"] == inputs["noise"], "proposal provenance")
    actions = {m: np.array(a, dtype=np.float32) for m, a in r["actions"].items()}
    check(set(actions) == set(MODES) | {"raw_repeat"} and
          all(a.shape == (10, 7) and np.isfinite(a).all() for a in actions.values()), "action labels or shape")
    metrics = {}
    for left, right in (("control", "raw"), ("raw_repeat", "raw"), ("panda", "control"), ("panda", "raw")):
        a, b = actions[left], actions[right]
        d = a.astype(float) - b.astype(float)
        metric = dict(first_action=d[0].tolist(), rms_by_dimension=np.sqrt(np.mean(d ** 2, axis=0)).tolist(),
                      max_abs=float(np.abs(d).max()), first_five_bounded_l2=float(
                          np.linalg.norm(np.clip(a[:5].astype(float), -1, 1) - np.clip(b[:5].astype(float), -1, 1))))
        key = f"{left}-minus-{right}"
        if key in r["differences"]:
            for field, value in metric.items():
                np.testing.assert_allclose(value, r["differences"][key][field], atol=1e-7, rtol=5e-7)
        metrics[key] = metric
    return dict(action_sha256={k: digest(v) for k, v in actions.items()}, differences=metrics)


def branch(layout, mode, bank):
    path = GPU / f"native-{layout}-{mode}.json"
    calls_path = GPU / f"calls-{layout}-{mode}.json"
    calls = load(calls_path) if calls_path.exists() else []
    for j, c in enumerate(calls):
        check(c["prompt"] == PROMPT and c["noise"] == digest(bank[j]) and set(c["input"]) == set(FIELDS)
              and all(len(h) == 64 for h in [*c["input"].values(), c["proposal"]]), "call inputs/noise")
    traces_path = GPU / f"views-{layout}-{mode}.jsonl"
    traces = [json.loads(line) for line in traces_path.read_text().splitlines()] if traces_path.exists() else []
    admitted, invalid = [], []
    for j, t in enumerate(traces):
        check((t["step"], t["layout"], t["mode"]) == (j * 5, layout, mode), "view trace label")
        if "invalid" in t:
            invalid.append(dict(step=t["step"], reason=t["invalid"]))
        else:
            checked = geometry(t, mode == "panda")
            admitted.append(checked)
    out = dict(status="missing", calls=len(calls), valid_render_boundaries=len(admitted), invalid=invalid)
    if mode == "panda" and admitted:
        out["within_branch_geometry_ranges"] = {k: [min(a[k] for a in admitted), max(a[k] for a in admitted)]
                                                for k in admitted[0]}
    if not path.exists():
        if invalid:
            out["status"] = "censored"
        elif calls or traces:
            out["status"] = "incomplete"
        return out, None, calls
    r = load(path)
    reset, n, rewards = r["reset"], r["controls"], r["native_rewards"]
    check(r["case"] == case(layout) and r["mode"] == mode and reset["arm"] == "Kinova3"
          and reset["model_class"] == "MountedKinova3" and reset["joint_count"] == 7
          and reset["worker_step_index"] == 0 and reset["prompt"] == r["released_prompt"] == PROMPT
          and reset["renderer_vendor"] == "NVIDIA Corporation" and reset["horizon"] == 300
          and reset["noise_seed"] == case(layout)["seed"] + 817117 and reset["noise_sha256"] == digest(bank),
          "reset identity, renderer, horizon, labels or noise")
    check(reset["action_schema"] == dict(shape=[7], dtype="float32", controller="OSC_POSE", control_mode="relative",
                                        control_freq=20, low=[-1.] * 7, high=[1.] * 7)
          and r["inferred"] == dict(yaw_deg=0., xy_scale=1., gripper_sign=1), "native motor interface")
    check(0 < n <= 300 and n == len(rewards) and len(calls) == math.ceil(n / 5)
          and (r["terminated"] or r["truncated"]) and (not r["truncated"] or n == 300)
          and sum(rewards) == r["native_return"] and all(v in (0., 1.) for v in rewards)
          and bool(max(rewards)) == r["success"], "reward, terminal flags or coverage")
    check(reset["motor_frame0"] == calls[0]["input"] and r["opening_proposal_sha256"] == calls[0]["proposal"],
          "opening motor input/output")
    check(not invalid and (mode == "raw" or len(traces) == len(calls)), "render coverage")
    out.update(status="terminal", controls=n, native_return=sum(rewards), final_reward=rewards[-1],
               peak_reward=max(rewards), rewarded_controls=[j+1 for j, v in enumerate(rewards) if v],
               success=r["success"], terminated=r["terminated"], truncated=r["truncated"],
               reset_physics=reset["physics"], scene=reset["scene_fingerprint"], opening_proposal=calls[0]["proposal"])
    return out, r, calls


def native(layout):
    bank = noise_bank(layout)
    rows, calls, out = {}, {}, {}
    for m in MODES:
        out[m], rows[m], calls[m] = branch(layout, m, bank)
    ref = rows["raw"]
    checks = {}
    if rows["control"] and ref:
        control = rows["control"]
        equal = (control["reset"] == ref["reset"] and calls["control"] == calls["raw"]
                 and all(control[k] == ref[k] for k in ("controls", "native_rewards", "success", "terminated", "truncated")))
        checks["raw_control_entire_loop_equal"] = equal
        check(equal, "raw vs target-render negative control")
    if calls["panda"] and calls["raw"]:
        ap, ar = calls["panda"][0], calls["raw"][0]
        check(outside(ap["input"], ar["input"]) == [FIELDS[0]], "opening Panda camera, wrist and state")
        checks["panda_opening_proposal_changed"] = ap["proposal"] != ar["proposal"]
        joint = list(zip(calls["panda"], calls["raw"]))
        checks["common_noise_calls"] = sum(p["noise"] == r["noise"] for p, r in joint)
        checks["common_length"] = len(joint)
        checks["first_divergent_motor_state_step"] = next((5*j for j, (p, r) in enumerate(joint)
                                                          if p["input"][FIELDS[2]] != r["input"][FIELDS[2]]), None)
    if rows["panda"] and ref:
        p, r = rows["panda"]["reset"], ref["reset"]
        check(outside(p["public"], r["public"]) == ["agentview_image"]
              and outside(p["motor_frame0"], r["motor_frame0"]) == [FIELDS[0]]
              and all(p[k] == r[k] for k in r if k not in ("public", "motor_frame0", "reader_frame0")),
              "Panda matched reset")
        checks["matched_public_reset_except_external_rgb"] = True
    paired = all(rows.values())
    return dict(layout=layout, branches=out, checks=checks, paired=paired,
                panda_minus_raw_return=(out["panda"]["native_return"] - out["raw"]["native_return"]) if paired else None)


def main():
    frames, scores = [], []
    for layout in (14, 15):
        meta = load(CPU / f"layout-{layout}.json")
        check(meta["case"] == case(layout) and meta["instruction"] == PROMPT
              and tuple(r["step"] for r in meta["frames"]) == STEPS, "frame list")
        for row in meta["frames"]:
            step = row["step"]
            inputs = frame_inputs(layout, step, row)
            frames.append(dict(layout=layout, step=step, inputs=inputs, proposal=proposal(layout, step, inputs)))
    for layout in (16, 17):
        scores.append(native(layout))
    path = GPU / "native-pairs.json"
    if path.exists():
        expected = [dict(layout=s["layout"], paired=s["paired"], outcomes={
            m: b["success"] for m, b in s["branches"].items() if b["status"] == "terminal"}) for s in scores]
        check(load(path) == expected, "producer pairing/labels vs reconstructed episodes")
    print(json.dumps(dict(frames=frames, native=scores), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
