"""Score input-only optical-roll correction with a frozen SmolVLA and paired noise."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from motor import Motor, POLICY_REV, VLM_REV, WEIGHT_SHA

CAMERAS = ("agent", "wrist")


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def feature(image):
    return image.reshape(16, 16, 16, 16, 3).mean(axis=(1, 3))


def estimate(image, template):
    # Only this camera's current image and fit-set mean enter the estimator.
    distances = [np.mean((np.rot90(feature(image), k) - template) ** 2) for k in (0, -1, 1)]
    return (0, -1, 1)[np.argmin(distances)]


def rms(a):
    return float(np.sqrt(np.mean(np.square(a, dtype=np.float64))))


def evaluate_task(motor, obs, record, templates, seed, repeat):
    noise = torch.randn((1, 50, 32), generator=torch.Generator().manual_seed(seed))
    nominal = [obs[k] for k in CAMERAS]
    task = record["task"]
    chunks, inputs = {}, {}

    def propose(views):
        # Reuse identical conditions only within this task and noise draw.
        key = tuple(sha(im) for im in views)
        if key not in inputs:
            name = f"p{len(inputs)}"
            chunks[name] = motor.proposal(*views, obs, task, noise)
            inputs[key] = name
        return inputs[key]

    baseline = propose(nominal)
    checks = dict(seed=seed, noise_sha256=sha(noise.numpy()), baseline=baseline,
                  nominal_temporal_std=chunks[baseline].std(axis=0).tolist())
    assert np.any(chunks[baseline].std(axis=0) > 0)
    if repeat:
        duplicate = motor.proposal(*nominal, obs, task, noise)
        checks["repeat_max_abs"] = float(np.abs(duplicate - chunks[baseline]).max())
        np.testing.assert_array_equal(chunks[baseline], duplicate)
        fresh_noise = torch.randn(noise.shape, generator=torch.Generator().manual_seed(seed + 10000))
        fresh = motor.proposal(*nominal, obs, task, fresh_noise)
        checks["fresh_noise_nominal_rmse"] = rms(fresh - chunks[baseline])
        assert checks["fresh_noise_nominal_rmse"] > 0
    rows = []
    for camera, deg in [(None, 0)] + [(k, d) for k in CAMERAS for d in (-90, 90)]:
        views = list(nominal)
        if camera:
            views[CAMERAS.index(camera)] = obs[f"{camera}{deg:+d}"]
        turns = [estimate(im, templates[k]) for k, im in zip(CAMERAS, views, strict=True)]
        fixed = [np.rot90(im, k) for im, k in zip(views, turns, strict=True)]
        perfect = [np.rot90(im, deg // 90) if name == camera else im
                   for name, im in zip(CAMERAS, views, strict=True)]
        ref = nominal[CAMERAS.index(camera)] if camera else nominal[0]
        idx = CAMERAS.index(camera) if camera else 0
        rgb_error = lambda im: float(np.abs(im.astype(np.int16) - ref).mean())
        if camera:
            # This independent camera/row convention check uses truth only for the oracle.
            assert rgb_error(perfect[idx]) < rgb_error(np.rot90(views[idx], -deg // 90))
        ids = {label: propose(pair) for label, pair in (("raw", views), ("estimated", fixed), ("oracle", perfect))}
        rows.append(dict(camera=camera, physical_roll_deg=deg, estimated_image_corrections_deg=[90*k for k in turns],
                         raw_rgb_mae=rgb_error(views[idx]), oracle_rgb_mae=rgb_error(perfect[idx]),
                         chunks=ids, rmse={k:rms(chunks[v]-chunks[baseline]) for k,v in ids.items()}))
    return checks, rows, chunks


def load_checked(frames, rec):
    with np.load(frames / (rec["key"] + ".npz"), allow_pickle=False) as f:
        obs = dict(f)
    assert {k:sha(v) for k,v in obs.items()} == rec["sha256"]
    return obs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    frames = args.cache / "frames"
    records = json.loads((frames / "frames.json").read_text())
    fit = [r for r in records if r["key"].startswith("fit-")]
    dev = [r for r in records if r["key"].startswith("development-")]
    assert len(fit) == len(dev) == 8 and not ({r["task"] for r in fit} & {r["task"] for r in dev})
    calibration = [load_checked(frames, r) for r in fit]
    templates = {k:np.mean([feature(z[k]) for z in calibration], axis=0) for k in CAMERAS}
    torch.manual_seed(23)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(6)
    motor = Motor(args.cache)
    result = dict(policy="lerobot/smolvla_libero", revision=POLICY_REV, vlm_revision=VLM_REV,
                  weights_sha256=WEIGHT_SHA, total_parameters=sum(p.numel() for p in motor.policy.parameters()),
                  trainable_parameters=sum(p.numel() for p in motor.policy.parameters() if p.requires_grad),
                  eval_mode=not motor.policy.training, precision=sorted({str(p.dtype) for p in motor.policy.parameters()}),
                  device=str(next(motor.policy.parameters()).device), threads=torch.get_num_threads(),
                  action_shape=[50,7], calibration=[r["key"] for r in fit], tasks=[])
    args.out.mkdir(parents=True, exist_ok=True)
    proposals = {}
    for index, rec in enumerate(dev):
        obs = load_checked(frames, rec)
        checks, rows, chunks = evaluate_task(motor, obs, rec, templates, 231100 + index, repeat=index == 0)
        result["tasks"].append(dict(key=rec["key"], task=rec["task"], **checks, rows=rows))
        proposals.update({rec["key"] + "_" + k:v for k,v in chunks.items()})
        print(rec["key"], len(chunks), "distinct paired proposals", flush=True)
    np.savez_compressed(args.out / "proposals.npz", **proposals)
    (args.out / "scores.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
