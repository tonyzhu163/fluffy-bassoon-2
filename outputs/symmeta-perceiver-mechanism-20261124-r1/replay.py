"""Replay the shipped Perceivers on paired interventions on the locked 96 supports."""
import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import time

import cv2
import flax
import jax
import numpy as np

from fluffy_bassoon.xemb.baselines import compute_dis_flow_maps
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
from fluffy_bassoon.xemb.equiv_v2 import _vis_inputs_v2, predict_embodiment_v2
from fluffy_bassoon.xemb.wider_family import relabel, rot2_batch

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = Path("/mowenjie/fluffy-bassoon-2")
RETRY = "symmeta-perceiver-retry-20261123-r1"
TRAIN = ROOT / "outputs" / RETRY / "train02pro"
SOURCE_PIN = "ba3c41e25575d07d7bd8bbfaae2797e7cf9dc0a9"
PRED_SHA = "c8420c2cb7d0cb38e6f0d73b3a20ce47997dbef7657b652d45eaaa8988eb6c0b"
sys.path.insert(0, str(CODE / "outputs/symmeta-camera-command-20261123-r1"))
sys.path.insert(0, str(CODE / "outputs/xemb-icml-campaign-20261121-r1"))
from score import SUPPORT_SET_SHA, load_prefix, official_panel, roll_pixels, set_digest
from eval_stage2a_live_ur5e import load_v2_checkpoint

SEEDS = (41, 42, 43)
# name, view, true action-map increment, expected relative-direction response,
# issued XY frame increment, local Cartesian XY frame increment (applied as R(-d)).
CONDITIONS = [
    ("original", "upright", 0, 0, 0, 0),
    ("command+90", "upright", 90, 90, 90, 0),
    ("command-90", "upright", -90, -90, -90, 0),
    ("family+90", "upright", 90, 90, 90, 90),
    ("family-90", "upright", -90, -90, -90, -90),
    ("local+90", "upright", 0, 0, 0, 90),
    ("local-90", "upright", 0, 0, 0, -90),
    ("image+90", "ccw", 0, -90, 0, 0),
    ("image-90", "cw", 0, 90, 0, 0),
    ("command180", "upright", 180, 180, 180, 0),
    ("reverse", "reverse", 0, 180, 0, 0),
    ("reverse_command180", "reverse", 180, 0, 180, 0),
    ("inverse", "restored", 0, 0, 0, 0),
]


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.write("\n")


def wrap(x):
    return (x + 180) % 360 - 180


def rotate_local(data, d):
    out = data.copy()
    x, y = data[..., 0], data[..., 1]
    if d == 90:
        out[..., 0], out[..., 1] = y, -x
    elif d == -90:
        out[..., 0], out[..., 1] = -y, x
    elif d == 180:
        out[..., :2] = -data[..., :2]
    else:
        raise ValueError(d)
    return out


def verify_relabels(batch, labels):
    gain, yaw = labels["xy_scale"], labels["yaw_deg"]
    source = gain[:, None, None] * rot2_batch(yaw)
    issued = batch["issued_mean"]
    original_world = np.einsum("bij,bkj->bki", source, issued[..., :2])
    max_residual = 0.0
    for d in (90, -90, 180):
        target = dict(yaw_deg=yaw+d, shape=gain[:, None, None]*np.eye(2),
                      z_gain=gain, gripper_sign=labels["gripper_sign"])
        ref = relabel(dict(batch, **labels), target)
        candidate = rotate_local(issued, d)
        for transformed, native in [(candidate, ref["issued_mean"]),
                                    (rotate_local(batch["delta_eef"], d), ref["delta_eef"])]:
            # The exact sign/swap is compared to native float32 vectors after float64 trig/solve.
            np.testing.assert_allclose(transformed, native, rtol=np.finfo(np.float32).eps,
                                       atol=8*np.finfo(np.float64).eps)
        actual = np.einsum("bij,bkj->bki", gain[:, None, None]*rot2_batch(yaw+d), candidate[..., :2])
        np.testing.assert_allclose(actual, original_world, rtol=0, atol=16*np.finfo(float).eps)
        max_residual = max(max_residual, float(np.max(np.abs(actual-original_world))))
    return max_residual


def flow_view(batch):
    # Match the shipped exporter, which computes DIS separately on each support.
    flows = [compute_dis_flow_maps(b, a, v) for b, a, v in
             zip(batch["img_before"], batch["img_after"], batch["valid"], strict=True)]
    return dict(batch, dis_flow=np.stack(flows))


def get_views(batch):
    upright = flow_view(batch)
    clockwise = roll_pixels(batch, -90)
    counterclockwise = roll_pixels(batch, 90)
    restored = roll_pixels(counterclockwise, -90)
    restored["issued_mean"] = rotate_local(rotate_local(batch["issued_mean"], 90), -90)
    for name, source in batch.items():
        require(source.tobytes() == restored[name].tobytes(), f"Inverse input: {name}")
        require(np.array_equal(source, roll_pixels(clockwise, 90)[name]), f"Other inverse: {name}")
    restored = flow_view(restored)
    require(upright["dis_flow"].tobytes() == restored["dis_flow"].tobytes(), "Inverse DIS")
    orig_sensory = np.asarray(_vis_inputs_v2(upright, "dis_lumaflow")[0])
    inv_sensory = np.asarray(_vis_inputs_v2(restored, "dis_lumaflow")[0])
    require(orig_sensory.tobytes() == inv_sensory.tobytes(), "Inverse nine-channel input")
    reverse = dict(batch, img_before=batch["img_after"], img_after=batch["img_before"])
    return dict(upright=upright, cw=flow_view(clockwise), ccw=flow_view(counterclockwise),
                reverse=flow_view(reverse), restored=restored)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--qualify", action="store_true", help="Run the first 12 recordings only")
    args = ap.parse_args()
    start = time.monotonic()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    cores = sorted(os.sched_getaffinity(0))
    require(cores == [38, 39] and jax.default_backend() == "cpu", "CPU allocation")
    cases, panel = official_panel(ROOT)
    require(len(cases) == 96, "Expected official panel")
    pred_file = TRAIN / "predictions.json"
    require(digest(pred_file) == PRED_SHA, "Shipped predictions changed")
    saved = [r for r in json.loads(pred_file.read_bytes())["rows"] if r["panel"] == "gpu09test"]
    refs = {(r["index"], r["seed"]): r for r in saved}
    require(len(refs) == len(saved) == 288 and set(refs) == {(i,s) for i in range(96) for s in SEEDS},
            "Archived prediction keys")
    paths = {}
    readers = {}
    receipt = CODE / "outputs" / RETRY / "evidence/train02pro/checkpoints.sha256"
    for line in receipt.read_text().splitlines():
        want, name = line.split()
        path = TRAIN / name
        require(digest(path) == want, f"Checkpoint hash: {name}")
        p, cfg, meta = load_v2_checkpoint(path)
        seed = meta["seed"]
        require(meta["tag"] == "perceiver_cont_warm" and meta["mode"] == "passive_attempt1"
                and meta["steps"] == 2500 and cfg.model_type == "perceiver_icba"
                and cfg.visual_mode == "dis_lumaflow", "Unexpected reader")
        paths[seed] = dict(path=str(path), sha256=want, seed=seed, config=asdict(cfg),
                           source_lock_sha256=meta["source_lock_sha256"])
        readers[seed] = (p, cfg)
    require(set(readers) == set(SEEDS), "Reader seeds")
    rows, supports = [], []
    max_world_residual = 0.0
    spec_fields = ("yaw_deg", "xy_scale", "gripper_sign")
    panel_cases = cases[:12] if args.qualify else cases
    for offset in range(0, len(panel_cases), 12):
        chunk_cases = panel_cases[offset:offset+12]
        tapes = []
        specs = [get_embodiment_spec(c["embodiment"]) for c in chunk_cases]
        for case, spec in zip(chunk_cases, specs, strict=True):
            tape, tape_file, idx = load_prefix(case, official=True)
            for seed in SEEDS:
                ref = refs[case["index"], seed]
                require(tape_file["sha256"] == ref["tape_sha256"] and ref["regime"] == spec.regime and
                        all(ref[k] == case[k] for k in ("index", "suite", "task_id", "embodiment")),
                        f"Wrong archived row for {case['id']}")
            supports.append(dict(index=case["index"], id=case["id"], task_id=case["task_id"], suite=case["suite"],
                                 embodiment=spec.name, regime=spec.regime, true_yaw_deg=spec.yaw_deg,
                                 selected_chunks=idx, valid=tape["valid"][0].tolist(), **tape_file))
            tapes.append(tape)
        batch = {k: np.concatenate([b[k] for b in tapes]) for k in tapes[0]}
        labels = {k: np.array([getattr(s, k) for s in specs]) for k in spec_fields}
        max_world_residual = max(max_world_residual, verify_relabels(batch, labels))
        views = get_views(batch)
        layout = {k: dict(dtype=str(v.dtype), shape_per_trace=list(v.shape[1:])) for k, v in views["upright"].items()}
        for condition, view_name, truth_shift, dir_shift, cmd, local in CONDITIONS:
            view = dict(views[view_name])
            if cmd:
                view["issued_mean"] = rotate_local(view["issued_mean"], cmd)
            if local:
                view["delta_eef"] = rotate_local(view["delta_eef"], local)
            # frame0_img is archived and checked during image inversion. The shipped reader has no reset-image token.
            inputs = {k: v for k,v in view.items() if k != "frame0_img"}
            for seed in SEEDS:
                yaw, gain, sign = predict_embodiment_v2(*readers[seed], inputs)
                for i, (y, g, s) in enumerate(zip(yaw, gain, sign, strict=True)):
                    case = chunk_cases[i]
                    r = refs[case["index"], seed]
                    rows.append(dict(index=case["index"], suite=case["suite"], task_id=case["task_id"],
                                     embodiment=case["embodiment"], regime=r["regime"], tape_sha256=r["tape_sha256"],
                                     seed=seed, condition=condition, true_yaw_deg=wrap(specs[i].yaw_deg + truth_shift),
                                     expected_shift_deg=dir_shift, yaw_deg=float(y), xy_scale=float(g), gripper_sign=int(s),
                                     **({"archived": r["tuned"]} if condition == "original" else {})))
        print(f"{offset+len(tapes)}/{len(panel_cases)} supports, {len(rows)} predictions, {time.monotonic()-start:.1f}s", flush=True)
    original = {(r["index"], r["seed"]): r for r in rows if r["condition"] == "original"}
    for r in rows:
        base = original[r["index"], r["seed"]]
        shift = wrap(r["yaw_deg"] - base["yaw_deg"])
        error = abs(wrap(r["yaw_deg"] - r["true_yaw_deg"]))
        r.update(shift_deg=shift, error_deg=error, error_change_deg=error-abs(wrap(base["yaw_deg"]-base["true_yaw_deg"])),
                 tracking_residual_deg=abs(wrap(shift-r["expected_shift_deg"])))
        if r["condition"] == "inverse":
            require(all(r[k] == base[k] for k in ("yaw_deg", "xy_scale", "gripper_sign")), "Inverse prediction")
    resource_use = resource.getrusage(resource.RUSAGE_SELF)
    checks = dict(max_world_command_residual=max_world_residual,
                  support_set_sha256=set_digest(r["sha256"] for r in supports),
                  counts_by_regime=dict(Counter(r["regime"] for r in supports)),
                  original_seed_trace_pairs=len(original), all_inverse_inputs_dis_nine_channels_outputs_exact=True,
                  max_archived_yaw_diff_deg=max(abs(wrap(r["yaw_deg"]-r["archived"]["yaw_deg"])) for r in original.values()),
                  max_archived_gain_diff=max(abs(r["xy_scale"]-r["archived"]["xy_scale"]) for r in original.values()),
                  archived_gripper_mismatches=sum(r["gripper_sign"] != r["archived"]["gripper_sign"] for r in original.values()))
    if not args.qualify:
        require(checks["support_set_sha256"] == SUPPORT_SET_SHA and sum("ood" in s["regime"] for s in supports)==64,
                "Complete paired support set")
    runtime = dict(backend=jax.default_backend(), jax=jax.__version__, flax=flax.__version__, numpy=np.__version__,
                   opencv=cv2.__version__, python=platform.python_version(), host=socket.gethostname(), cores=cores,
                   wall_s=time.monotonic()-start, system_cpu_s=resource_use.ru_stime, user_cpu_s=resource_use.ru_utime,
                   peak_rss_kib=resource_use.ru_maxrss, batch_traces=12, gpu_hours=0)
    revision = subprocess.check_output(["git","-C",str(CODE),"rev-parse","HEAD"], text=True).strip()
    write(args.out / "sources.json", dict(revision=revision, train_pin=SOURCE_PIN, pr7_recipe="perceiver_cont_warm",
          archive=dict(path=str(pred_file), sha256=PRED_SHA), checkpoints=paths, plan_sha256=digest(HERE/"PLAN.md"),
          panel=panel, inputs=layout, supports=supports, checks=checks, runtime=runtime))
    write(args.out / "predictions.json", rows)
    print(json.dumps(dict(checks=checks, runtime=runtime)))


if __name__ == "__main__":
    main()
