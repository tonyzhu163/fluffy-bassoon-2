"""Run paired action chunks and write per-task contrasts (no executed actions)."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from motor import Motor, WEIGHTS_SHA256


def sham_offset(union):
    # Use one rigid translation for both silhouettes. Stay inside the image
    # without clipping or wraparound, and avoid both original arms.
    y, x = np.nonzero(union)
    dys = (0, -int(y.min()), int(union.shape[0] - y.max() - 1))
    dxs = (0, -int(x.min()), int(union.shape[1] - x.max() - 1))
    offsets = [(dy, dx) for dy in dys for dx in dxs]
    offset = min(offsets, key=lambda p: np.count_nonzero(np.roll(union, p, axis=(0, 1)) & union))
    assert not np.any(np.roll(union, offset, axis=(0, 1)) & union)
    return offset


def overlay(background, source, source_mask, shift=(0, 0)):
    mask = np.roll(source_mask, shift, axis=(0, 1))
    cutout = np.roll(source, shift, axis=(0, 1))
    return np.where(mask[..., None], cutout, background)


def views(record):
    source, target = record["source"], record["target"]
    sm, tm = record["source_mask"], record["target_mask"]
    shift = sham_offset(sm | tm)
    yield "source", source
    yield "target", target
    yield "native_target", record["native"]
    for name, rgb, mask in [("source", source, sm), ("target", target, tm)]:
        sham = np.roll(mask, shift, axis=(0, 1))
        assert mask.sum() == sham.sum()
        yield f"{name}_masked", np.where(mask[..., None], np.uint8(128), rgb)
        yield f"{name}_sham", np.where(sham[..., None], np.uint8(128), rgb)
    # Privileged source-visible pixels only. Leave target-only protrusions
    # and all disoccluded target background in place.
    yield "source_overlay", overlay(target, source, sm)
    yield "sham_overlay", overlay(target, source, sm, shift)


CONTRASTS = {
    "raw": ("target", "source"), "uncropped_native": ("native_target", "source"),
    "masked_pair": ("target_masked", "source_masked"),
    "sham_pair": ("target_sham", "source_sham"),
    "target_mask_to_nominal": ("target_masked", "source"),
    "target_sham_to_nominal": ("target_sham", "source"),
    "source_mask_to_nominal": ("source_masked", "source"),
    "source_sham_to_nominal": ("source_sham", "source"),
    "source_overlay_to_nominal": ("source_overlay", "source"),
    "sham_overlay_to_nominal": ("sham_overlay", "source"),
    "identity": ("repeat", "source"), "new_noise": ("resampled", "source"),
}


def metrics(a, b):
    delta = (a - b).astype(np.float64)

    def rms(x):
        return float(np.sqrt(np.mean(np.square(x))))

    return {"first_xyz_rms": rms(delta[0, :3]), "xyz_rms": rms(delta[:, :3]),
            "rotation_rms": rms(delta[:, 3:6]), "gripper_rms": rms(delta[:, 6]),
            "action_rms": rms(delta), "max_abs": float(abs(delta).max())}


def run(root, out):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    motor = Motor(root)
    cases = sorted((root / "cases").glob("*.npz"))
    assert len(cases) == 10
    scores, audit = [], []
    for i, case in enumerate(cases):
        with np.load(case) as data:
            record = dict(data)
        seed = 9100 + i
        gen = torch.Generator().manual_seed(seed)
        noise = torch.randn(1, 50, 32, generator=gen)

        def propose(image, n):
            return motor.propose([image, record["wrist"]], record, n)

        actions = {key: propose(rgb, noise) for key, rgb in views(record)}
        nominal = overlay(record["source"], record["source"], record["source_mask"])
        np.testing.assert_array_equal(nominal, record["source"])
        actions["repeat"] = propose(nominal, noise)
        actions["resampled"] = propose(record["source"], torch.randn(1, 50, 32, generator=gen))
        np.testing.assert_array_equal(actions["repeat"], actions["source"])
        assert np.all(actions["source"].std(axis=0) > 0)
        archive = root / "actions"
        archive.mkdir(exist_ok=True)
        np.savez_compressed(archive / case.name, **actions)
        task = str(record["task"])
        audit.append(dict(task=task, noise_seed=seed,
                          noise_sha256=hashlib.sha256(noise.numpy().tobytes()).hexdigest(),
                          first_action=actions["source"][0].tolist(),
                          temporal_std=actions["source"].std(axis=0).tolist(),
                          source_chunk_sha256=hashlib.sha256(actions["source"].tobytes()).hexdigest(),
                          sham_dy_dx=sham_offset(record["source_mask"] | record["target_mask"])))
        for name, (a, b) in CONTRASTS.items():
            scores.append(dict(task=task, contrast=name, **metrics(actions[a], actions[b])))
        print(i, task, scores[-len(CONTRASTS)], flush=True)
    with (out / "scores.csv").open("w") as f:
        writer = csv.DictWriter(f, list(scores[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(scores)
    receipt = dict(n_parameters=sum(p.numel() for p in motor.policy.parameters()),
                   trainable=sum(p.numel() for p in motor.policy.parameters() if p.requires_grad),
                   modules_in_train_mode=sum(m.training for m in motor.policy.modules()),
                   weights_sha256=WEIGHTS_SHA256, device="cpu", dtype="float32",
                   raw_output_shape=[1, 50, 7], inference_steps=motor.policy.config.num_steps,
                   tasks=audit)
    (out / "motor.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    run(args.root, args.output)
