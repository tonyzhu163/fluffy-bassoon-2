r"""Score centered 1.4x views of the official passive supports at 0, ±15 and ±30 degrees.

Run on Baige Linux from the deployed checkout with a fresh report path:

    data=/mowenjie/fluffy-bassoon-2
    run=outputs/symmeta-camera-command-20261123-r1
    overlay=$data/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay
    PYTHONPATH="$overlay:$PWD/src" PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        /mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python -Bu "$run/moderate_roll.py" \
        --data-root "$data" --output "$data/$run/moderate01/scores.json"
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import map_coordinates

import score
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec


SIZE = 64
ZOOM = 1.4
ANGLES = (0, 15, -15, 30, -30)


def source_grid(angle, zoom):
    """Inverse map for a counterclockwise displayed roll, in source (row, col) order."""
    center = (SIZE - 1) / 2
    y, x = (np.indices((SIZE, SIZE), dtype=np.float64) - center) / zoom
    radians = np.deg2rad(angle)
    c, s = np.cos(radians), np.sin(radians)
    coords = center + np.stack((c * y + s * x, c * x - s * y))
    if zoom == ZOOM:
        # At the last pixel center the outward bilinear tap has zero weight.
        tol = SIZE * np.finfo(np.float64).eps
        if coords.min() < -tol or coords.max() > SIZE - 1 + tol:
            raise ValueError(f"Zoomed {angle}° view exceeds the source image")
        np.clip(coords, 0, SIZE - 1, out=coords)
    return coords


def sample(plane, grid, dtype, fill=0.0):
    return map_coordinates(plane, grid, output=dtype, order=1, mode="grid-constant",
                           cval=fill, prefilter=False)


def warp(batch, grid, dtype=np.uint8):
    view = dict(batch)
    for key in score.IMAGES:
        images = batch[key]
        if images.shape[-3:] != (SIZE, SIZE, 3):
            raise ValueError(f"Unexpected {key} shape {images.shape}")
        # Treat batch, chunk and channel as independent planes.
        planes = np.moveaxis(images, -1, -3)
        result = np.stack([sample(p, grid, dtype) for p in planes.reshape(-1, SIZE, SIZE)])
        view[key] = np.ascontiguousarray(np.moveaxis(result.reshape(planes.shape), -3, -1))
    return view


def geometry():
    cropped = {a: source_grid(a, ZOOM) for a in ANGLES}
    full = {a: source_grid(a, 1.0) for a in ANGLES}
    checks = {}
    for a in ANGLES:
        # Zero on genuine pixels, one in the fill. Nonnegative bilinear weights
        # make *any* fill contribution observable without subtracting from one.
        border = sample(np.zeros((SIZE, SIZE)), full[a], np.float64, fill=1)
        restored = sample(border, cropped[-a], np.float64, fill=1)
        if np.any(restored != 0):
            raise ValueError(f"Border fill reaches the upright {a}° resampling control")
        checks[a] = dict(source_min_px=float(cropped[a].min()), source_max_px=float(cropped[a].max()),
                         control_max_border_weight=float(restored.max()))
    return cropped, full, checks


def score_row(case, spec, angle, scored, zero):
    yaw, vectors, _ = scored
    error = {m: abs(score.wrap(v - spec.yaw_deg)) for m, v in yaw.items()}
    return dict(id=case["id"], index=case["index"], suite=case["suite"], task_id=case["task_id"],
                embodiment=spec.name, regime=spec.regime, true_yaw_deg=spec.yaw_deg,
                true_scale=spec.xy_scale, true_grip=spec.gripper_sign, roll_deg=angle, yaw_deg=yaw,
                abs_error_deg=error, shift_from_zero_deg={m: score.wrap(v - zero[m]) for m, v in yaw.items()},
                error_change_from_zero_deg={m: err - abs(score.wrap(zero[m] - spec.yaw_deg))
                                            for m, err in error.items()},
                matched_model_minus_hs_error_deg=error["model_matched"] - error["hs_matched"], vectors=vectors)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("/mowenjie/fluffy-bassoon-2"))
    parser.add_argument("--output", required=True, type=Path, help="Fresh JSON report path")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
    import cv2

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    cropped, full, checks = geometry()
    params, cfg, projection, sources = score.frozen_inputs(args.data_root)
    cases, panel = score.official_panel(args.data_root)
    rows, controls, supports = [], [], []
    for case in cases:
        batch, identity, indices = score.load_prefix(case, official=True)
        supports.append(dict(id=case["id"], npz=identity, selected_chunk_indices=indices))
        spec = get_embodiment_spec(case["embodiment"])
        upright = warp(batch, cropped[0])
        zero = score.score_view(upright, params, cfg, projection)
        for angle in ANGLES:
            rolled = zero if angle == 0 else score.score_view(warp(batch, cropped[angle]), params, cfg, projection)
            rows.append(score_row(case, spec, angle, rolled, zero[0]))
            intermediate = warp(batch, full[angle], np.float64)
            restored = warp(intermediate, cropped[-angle])
            pixel_mae = {k: float(np.abs(restored[k].astype(np.int16) - upright[k]).mean()) for k in score.IMAGES}
            if angle == 0 and any(pixel_mae.values()):
                raise ValueError(f"Zero-roll control changed pixels for {case['id']}")
            control = zero if angle == 0 else score.score_view(restored, params, cfg, projection)
            controls.append(dict(**score_row(case, spec, angle, control, zero[0]), pixel_mae_from_zero=pixel_mae))
        if len(supports) % 12 == 0:
            print(f"Scored {len(supports)}/{len(cases)} supports", flush=True)

    digest = score.set_digest(s["npz"]["sha256"] for s in supports)
    if digest != score.SUPPORT_SET_SHA:
        raise ValueError(f"Official support set differs from the frozen set: {digest}")
    report = dict(panel=panel, sources=sources, supports=supports, support_set_sha256=digest,
                  invocation=dict(cwd=str(Path.cwd()), argv=sys.orig_argv, pythonpath=os.environ.get("PYTHONPATH")),
                  code_files=[score.read_file(Path(p).resolve())[1] for p in (__file__, score.__file__)],
                  zoom=ZOOM, rolls_deg=ANGLES, positive_roll="counterclockwise in displayed image",
                  resampling_control="Full-size +roll in float64, then -roll with 1.4x zoom to uint8",
                  geometry_by_angle=checks, rows=rows, summary=score.summarize(rows),
                  resampling_control_rows=controls, resampling_control_summary=score.summarize(controls))
    encoded = json.dumps(report, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(encoded + "\n")


if __name__ == "__main__":
    main()
