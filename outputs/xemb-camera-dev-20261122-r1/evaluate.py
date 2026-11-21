"""Matched BASE/S15 development panel with the frozen Stage 2 active readers."""

import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np

import eval_stage2a_live_ur5e as stage
from fluffy_bassoon.xemb.embodiment import circular_yaw_error_deg
from ur5e_motor import NOISE_SEED_OFFSET, array_digest, public_digest, run_episode, write_json
from ur5e_sim_worker import UR5eSimulator

CAMERAS = ("BASE", "S15")
TRANSFORMS = ("cont_45_nom", "cont_m45_075")
LAYOUTS = ((11, 13), (12, 14))
METHODS = ("ours_equiv_homog_active", "dense_flow_dis_active", "nominal", "privileged_oracle")
VIEWS = ("agentview_image", "robot0_eye_in_hand_image")
PHYSICS = ("released_row_fingerprint", "scene_fingerprint", "post_settle_scene_fingerprint",
           "initial_state_fingerprint")
ARCHIVE_SHA = "37f062958cb63b962ceefb760f0592fdb120da966369ee308b509e2d82449076"
BDDL_SHA = {"BASE": "7a6bb627da6e87025c1cfdd19fff5f8a2c873226734393657505dde2401144a6",
            "S15": "f5b6d5c0511cab56f88bc80859c6a0ddc4d78aa23144591744c04636d71b61e7"}


def load_readers(archive, checkpoint, out):
    raw = archive.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA:
        raise ValueError("Unexpected Stage 2 source archive")
    params, cfg, meta = stage.load_v2_checkpoint(checkpoint)
    if (meta["archive_sha256"], meta["tag"], meta["mode"], meta["seed"], cfg.model_type,
        cfg.visual_mode, cfg.use_dz_parallax) != (
            ARCHIVE_SHA, "equiv_flow_homog_nodz", "active_probe", 41,
            "equiv_flow_homography", "dis_lumaflow", False):
        raise ValueError("Unexpected Stage 2 checkpoint metadata")
    # Same subset, order, and numerical routines as fit_source_calibrators' active branch.
    rows = [r for r in json.loads(raw)["trials"] if r["split"] == "train" and r["embodiment"] == "nom"]
    if {r["arm"] for r in rows} != {"Panda", "Sawyer"}:
        raise ValueError("Expected nominal Panda/Sawyer source calibration")
    data, _ = stage.load_mode_split(archive.parent, rows, "active_probe", "train")
    valid, delta = data["valid"], data["delta_eef"]
    gain = stage.fit_nominal_gain(data["issued_mean"], delta, valid)
    projection = stage.fit_dense_flow_projection(data["img_before"], data["img_after"], delta, valid, method="dis")
    write_json(out / "readers.json", dict(archive=str(archive), archive_sha256=ARCHIVE_SHA,
               checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               tag=meta["tag"], mode=meta["mode"], seed=41, source_indices=[r["index"] for r in rows],
               nominal_gain=float(gain), dis_projection=projection.tolist()))
    return params, cfg, gain, projection


def predict(readers, trace, spec):
    params, cfg, gain, projection = readers
    batch = {k: np.asarray(v)[None] for k, v in trace.items()}
    before, after, valid = (batch[k] for k in ("img_before", "img_after", "valid"))
    if before.dtype != np.uint8 or after.dtype != np.uint8 or not valid.all():
        raise ValueError("Expected native Stage 2 uint8 frames and three full probe bursts")
    batch["chunk_steps"] = np.where(valid, stage.CHUNK_SIZE, 0).astype(np.int32)
    batch["dis_flow"] = stage.compute_dis_flow_maps(before, after, valid)
    y, s, g = stage.predict_embodiment_v2(params, cfg, batch)
    issued = stage.issued_exposure(batch)
    _, scale, grip = stage.analytic_proprio_sysid(issued, batch["delta_eef"], batch["delta_jaw"], valid,
                                                 nominal_gain_m_per_cmd=gain)
    motion = stage.dense_flow_image_motion(before, after, valid, method="dis")
    yaw = stage.analytic_visual_yaw(issued, motion, batch["delta_eef"][..., 2], valid, projection)
    values = ((y[0], s[0], g[0]), (yaw[0], scale[0], grip[0]), (0., 1., 1),
              (spec.yaw_deg, spec.xy_scale, spec.gripper_sign))
    return {method: dict(yaw_hat=float(yh), scale_hat=float(sh), grip_hat=int(gh),
                         abs_circular_yaw_error_deg=circular_yaw_error_deg(yh, spec.yaw_deg))
            for method, (yh, sh, gh) in zip(METHODS, values, strict=True)}


def native_start(first, requested):
    repro = first["environment_reproducibility"]
    plan = repro["perturbation_plan"]
    variant = requested["variant"]
    op, targets = ("none", []) if variant == "BASE" else ("camera_transform", ["agentview", "robot0_eye_in_hand"])
    if (plan["case_id"], plan["seed"], plan["operator"], plan["targets"], repro["bddl_sha256"]) != (
            variant, requested["seed"], op, targets, BDDL_SHA[variant]):
        raise ValueError(f"Wrong native perturbation for {requested}: {plan}")
    if (any(repro[k] != requested[k] for k in ("suite", "task_id", "init_state_id", "seed"))
            or first["model_identity"]["arm"] != "UR5e" or first["worker_step_index"] != 0):
        raise ValueError(f"Wrong native reset for {requested}")
    return dict(repro=repro, public=public_digest(first["observation"]), prompt=first["language_instruction"])


class ObservedSimulator:
    """Capture the actual native create and only the commands consumed by each chunk."""

    def __init__(self, simulator):
        self.simulator = simulator

    def call(self, op, value):
        result = self.simulator.call(op, value)
        if op == "create":
            self.start = native_start(result[1], value)
            self.controls = []
        elif op == "chunk":
            self.controls.extend(np.asarray(value)[:result["steps"]].tolist())
        return result


class OpeningCaptured(Exception):
    """End the zero-control reference after the first unmodified policy inference."""


class OpeningPolicy:
    def __init__(self, policy, expected=None):
        self.policy = policy
        self._model = policy._model
        self.expected = expected
        self.opening = None

    def infer(self, item, *, noise):
        result = self.policy.infer(item, noise=noise)
        if self.opening is None:
            self.opening = array_digest(np.asarray(result["actions"], dtype=np.float32))
            if self.expected is None:
                raise OpeningCaptured
            if self.expected != self.opening:
                raise ValueError("Opening proposal mismatch before first control")
        return result


def check_camera_pair(base, shifted):
    for key in PHYSICS:
        if base["repro"][key] != shifted["repro"][key]:
            raise ValueError(f"Unpaired BASE/S15 {key}")
    if base["prompt"] != shifted["prompt"] or base["public"].keys() != shifted["public"].keys():
        raise ValueError("Unpaired BASE/S15 observation schema or prompt")
    for key, digest in base["public"].items():
        changed = shifted["public"][key] != digest
        if changed != (key in VIEWS):
            raise ValueError(f"Unexpected BASE/S15 frame-zero change for {key}")
    if not set(VIEWS) <= base["public"].keys():
        raise ValueError("Missing external or wrist frame")


def prepare(sim, policy, readers, row, path):
    spec = stage.get_embodiment_spec(row["embodiment"])
    meta, trace = stage.run_active_probe_episode(sim, row["support_case"], spec)
    row["support"] = dict(start=sim.start, meta=meta, issued_controls=sim.controls)
    np.savez_compressed(path.with_suffix(".npz"), **trace)
    write_json(path, row)
    if meta["controls"] != 15 or len(sim.controls) != 15:
        raise ValueError("Incomplete active support")
    row["predictions"] = predict(readers, trace, spec)
    write_json(path, row)

    recorder = OpeningPolicy(policy)
    try:
        run_episode(sim, recorder, row["query_case"], spec, prompt=meta["prompt"],
                    check_reset=lambda reset: row.update(query_reset=reset))
    except OpeningCaptured:
        pass
    if sim.controls or recorder.opening is None:
        raise ValueError("Reference inference must consume zero controls")
    row.update(query_start=sim.start, opening_sha256=recorder.opening)
    write_json(path, row)


def verify_panel(rows):
    for base, shifted in zip(rows[::2], rows[1::2], strict=True):
        check_camera_pair(base["support"]["start"], shifted["support"]["start"])
        check_camera_pair(base["query_start"], shifted["query_start"])
        if base["support"]["issued_controls"] != shifted["support"]["issued_controls"]:
            raise ValueError("Probe commands changed across cameras")
        if base["query_reset"]["noise_sha256"] != shifted["query_reset"]["noise_sha256"]:
            raise ValueError("Noise bank changed across cameras")
    for key in PHYSICS:
        support = {r["support"]["start"]["repro"][key] for r in rows}
        query = {r["query_start"]["repro"][key] for r in rows}
        if support & query:
            raise ValueError(f"Support/query collision for {key}")


def rollout(sim, policy, row, method):
    case = row["query_case"]
    pred = row["predictions"][method]
    reference = row["query_reset"]

    def check(reset):
        native = sim.start["repro"]
        original = row["query_start"]["repro"]
        differences = stage.reset_differences(reference, reset)
        for key in PHYSICS:
            if native[key] != original[key]:
                differences[key] = dict(expected=original[key], observed=native[key])
        if differences or reset["noise_seed"] != case["seed"] + NOISE_SEED_OFFSET:
            raise ValueError(f"Unpaired native query for {method} on {case}: {differences}")

    record = run_episode(sim, OpeningPolicy(policy, row["opening_sha256"]), case,
                         stage.get_embodiment_spec(row["embodiment"]), prompt=reference["prompt"],
                         yaw_hat=pred["yaw_hat"], scale_hat=pred["scale_hat"], grip_hat=pred["grip_hat"],
                         check_reset=check)
    if (not (record["terminated"] or record["truncated"]) or record["controls"] != len(sim.controls)
            or record["opening_proposal_sha256"] != row["opening_sha256"]):
        raise ValueError("Incomplete or unpaired native outcome")
    return dict(**record, issued_controls=sim.controls)


def paired(left, right):
    return dict(pairs=len(left), wins=sum(a and not b for a, b in zip(left, right, strict=True)),
                losses=sum(b and not a for a, b in zip(left, right, strict=True)))


def summary(rows):
    report = {"by_camera": {}, "S15_vs_BASE": {}, "matched_yaw_error_gap_change_deg": {}}
    for camera in CAMERAS:
        group = [r for r in rows if r["camera"] == camera]
        successes = {m: [r["episodes"][m]["success"] for r in group] for m in METHODS}
        report["by_camera"][camera] = dict(
            n=len(group), successes={m: sum(values) for m, values in successes.items()},
            mean_abs_circular_yaw_error_deg={
                m: float(np.mean([r["predictions"][m]["abs_circular_yaw_error_deg"] for r in group]))
                for m in METHODS},
            paired={f"{a}_vs_{b}": paired(successes[a], successes[b])
                    for a, b in itertools.combinations(METHODS, 2)})
    for m in METHODS:
        vals = {c: [r["episodes"][m]["success"] for r in rows if r["camera"] == c] for c in CAMERAS}
        report["S15_vs_BASE"][m] = paired(vals["S15"], vals["BASE"])
    # Difference of absolute errors (EquivFlow minus DIS), S15 minus BASE.
    changes = report["matched_yaw_error_gap_change_deg"]
    for base, shifted in zip(rows[::2], rows[1::2], strict=True):
        gaps = [r["predictions"][METHODS[0]]["abs_circular_yaw_error_deg"]
                - r["predictions"][METHODS[1]]["abs_circular_yaw_error_deg"] for r in (base, shifted)]
        changes[f"{base['query_case']['init_state_id']}-{base['embodiment']}"] = gaps[1] - gaps[0]
    report["mean_yaw_error_gap_change_deg"] = float(np.mean(list(changes.values())))
    return report


def main(out, archive, checkpoint):
    readers = load_readers(archive, checkpoint, out)
    policy = stage.load_frozen_policy("pi05")
    rows = []
    with UR5eSimulator() as worker:
        sim = ObservedSimulator(worker)
        for (support, query), transform, camera in itertools.product(LAYOUTS, TRANSFORMS, CAMERAS):
            case = dict(arm="UR5e", suite="libero_goal", task_id=1, variant=camera)
            truth = stage.get_embodiment_spec(transform)
            row = dict(id=f"{query}-{transform}-{camera}", camera=camera, embodiment=transform,
                       truth=dict(yaw_deg=truth.yaw_deg, xy_scale=truth.xy_scale, gripper_sign=truth.gripper_sign),
                       support_case=dict(case, init_state_id=support, seed=31810100 + support),
                       query_case=dict(case, init_state_id=query, seed=41810100 + query), episodes={})
            if not rows:
                # Discard the first native create, as in the Stage 2 recovery launcher.
                sim.call("create", dict(row["support_case"], embodiment=transform))
            prepare(sim, policy, readers, row, out / (row["id"] + ".json"))
            rows.append(row)
        verify_panel(rows)
        write_json(out / "admitted.json", dict(pin=os.environ["CAMERA_PIN"], cases=[r["id"] for r in rows]))
        for row in rows:
            for method in METHODS:
                row["episodes"][method] = rollout(sim, policy, row, method)
                write_json(out / (row["id"] + ".json"), row)
                print(row["id"], method, row["episodes"][method]["success"], flush=True)
    write_json(out / "summary.json", summary(rows))


if __name__ == "__main__":
    main(*(Path(arg) for arg in sys.argv[1:]))
