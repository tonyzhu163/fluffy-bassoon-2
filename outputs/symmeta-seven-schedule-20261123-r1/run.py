"""Stage 1 Perceiver schedule control and frozen-checkpoint comparison on official UR5e supports."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import jax
import numpy as np
import optax

HERE = Path(__file__).resolve().parent
CAMPAIGN = "xemb-icml-campaign-20261121-r1"
SCALE = "libero-xemb-ur5e-scale-20261120-r1"
sys.path.insert(0, str(HERE.parent / CAMPAIGN))

import stage1_offline_gate as stage1
from eval_stage2a_live_ur5e import load_v2_checkpoint
from eval_stage3_scaling import load_ur5e_official_test_batch
from paired import BOOT_SEED, DRAWS, render, summarize, trace_metrics

ROOT = Path("/mowenjie/fluffy-bassoon-2/outputs")
MODE = "passive_attempt1"
TAG = "perceiver_icba_dis"
CFG = stage1.ReaderV2Config(model_type="perceiver_icba", visual_mode="dis_lumaflow", max_chunks=32)
SEEDS = (41, 42, 43)
STEPS = 3500
BATCH = 64
ARCHIVE_SHA = "37f062958cb63b962ceefb760f0592fdb120da966369ee308b509e2d82449076"
SOURCE_SET_SHA = "b3d0de8949ba43bfe7da57b644656da456d0fe45a85c77e9e43234b4a05612f0"
TEST_SET_SHA = "8729fa68343e2a7a0cc6471ab25e7c707555a866e682a3dc999079a7b9dc3554"
ORIGINAL_SHA = {
    41: "fa34e48fe12238c9f38bc68f4394690189fc6a4463de88b3486673c5e395cacb",
    42: "dc7f8c507baee04333556479a52c1f55386c45ff772c7c945f11c66e76079553",
    43: "32fbe7d438011873f243a31d1a3880313c6ddd9c621ffe4b4caf46568f21f841",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_set(digests, expected):
    digest = hashlib.sha256("".join(f"{d}\n" for d in digests).encode("ascii")).hexdigest()
    if digest != expected:
        raise ValueError(f"Tape contents or order changed: {digest} != {expected}")
    return digest


def stage1_inputs():
    archive = ROOT / SCALE / "gpu01/archive/archive.json"
    if sha(archive) != ARCHIVE_SHA:
        raise ValueError(f"Wrong Stage 1 archive: {archive}")
    trials = json.loads(archive.read_bytes())["trials"]
    train, rows = stage1.load_mode_split(archive.parent, trials, MODE, "train")
    dev, dev_rows = stage1.load_mode_split(archive.parent, trials, MODE, "dev")
    source_sha = check_set((sha(archive.parent / r["passive_path"]) for r in rows), SOURCE_SET_SHA)
    arms = Counter(r["arm"] for r in rows)
    shifts = Counter(r["embodiment"] for r in rows)
    if (arms != {"Panda": 672, "Sawyer": 672} or sorted(shifts.values()) != [192] * 7
            or train["valid"].shape != (1344, 3) or not train["valid"].all()):
        raise ValueError("Unexpected source arms, shifts, or support lengths")
    for data in (train, dev):
        data["dis_flow"] = stage1.compute_dis_flow_maps(data["img_before"], data["img_after"], data["valid"])
    nominal = [i for i, r in enumerate(rows) if r["embodiment"] == "nom"]
    gain = stage1.fit_nominal_gain(train["issued_mean"][nominal], train["delta_eef"][nominal],
                                   train["valid"][nominal])
    donor = stage1.make_donor_permutation(dev_rows, np.random.default_rng(20261121))
    info = dict(archive=str(archive), archive_sha256=ARCHIVE_SHA, ordered_source_sha256=source_sha,
                source_arms=dict(arms), source_shifts=dict(shifts), n_dev=len(dev_rows),
                nominal_gain=float(gain))
    return train, dev, dev_rows, donor, gain, info


def official_inputs():
    panel = ROOT / SCALE / "gpu09test/test_live"
    test, rows = load_ur5e_official_test_batch(panel)
    shards = [json.loads((panel / f"shard-{s}/support-meta.json").read_bytes()) for s in range(8)]
    for i, (row, sup) in enumerate(zip(rows, (x for shard in shards for x in shard), strict=True)):
        tape = Path(sup["passive_npz"])
        expected = panel / f"shard-{i // 12}/support/{i % 12:02d}-{row['embodiment']}-passive.npz"
        if row["index"] != i or sup["embodiment"] != row["embodiment"] or tape != expected:
            raise ValueError(f"Official test identity mismatch at {i}")
        row.update(passive_npz=str(tape), support_sha256=sha(tape))
    check_set((r["support_sha256"] for r in rows), TEST_SET_SHA)
    if (test["valid"].shape != (96, 3) or not test["valid"].all()
            or Counter(r["regime"] for r in rows) != dict.fromkeys(stage1.EVAL_REGIMES, 32)):
        raise ValueError("Expected 96 complete recorded supports with 32 traces per regime")
    return test, rows


def score_checkpoint(path, seed, steps, gain, test, rows, expected_sha=None):
    digest = sha(path)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"Original checkpoint mismatch: {path}")
    params, cfg, meta = load_v2_checkpoint(path)
    signature = (meta["tag"], meta["mode"], meta["seed"], meta["steps"], meta["archive_sha256"])
    if (signature != (TAG, MODE, seed, steps, ARCHIVE_SHA) or cfg != CFG or meta["nominal_gain"] != gain
            or params["perceiver"]["pos"].shape != (1, 32, CFG.perceiver_width)):
        raise ValueError(f"Stage 1 checkpoint metadata mismatch: {path}, {signature}, {cfg}")
    pred = stage1.predict_embodiment_v2(params, cfg, test)
    factors = trace_metrics(rows, pred)
    # Qualify the custom reporting fields against the trainer's published factor metrics.
    reference = stage1.score_predictions(rows, *([r[k] for r in rows] for k in ("yaw_deg", "xy_scale", "gripper_sign")), *pred)
    for group, ref in reference.items():
        selected = [v for r, v in zip(rows, factors, strict=True) if group == "overall" or group == r["regime"]
                    or (group == "ood_combined" and r["regime"] != "familiar")]
        for name, key, agg, decimals in (("yaw_err_deg", "yaw_mae_deg", np.mean, 2),
                                         ("yaw_med_deg", "yaw_mae_deg", np.median, 2),
                                         ("scale_rel_err", "scale_rel_err", np.mean, 4),
                                         ("grip_acc", "grip_acc", np.mean, 4),
                                         ("all_correct", "all_correct", np.mean, 4)):
            if round(float(agg([v[key] for v in selected])), decimals) != ref[name]:
                raise ValueError(f"Stage 1 scoring disagreement on {group}/{name}")
    return factors, dict(path=str(path), sha256=digest, steps=steps, seed=seed, config=asdict(cfg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New result directory")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cpu-qualification", action="store_true", help="Two real updates per seed and full evaluation")
    args = parser.parse_args()
    out = args.output
    devices = jax.devices()
    backend = "cpu" if args.cpu_qualification else "gpu"
    if any(d.platform != backend for d in devices) or (backend == "gpu" and len(devices) != 1):
        raise RuntimeError(f"Expected {backend} execution, got {devices}")
    out.mkdir(parents=False, exist_ok=False)
    steps = 2 if args.cpu_qualification else STEPS
    print(f"Reading passive Stage 1 and official test tapes on {devices}", flush=True)
    train, dev, dev_rows, donor, gain, source = stage1_inputs()
    test, rows = official_inputs()
    sched = optax.warmup_cosine_decay_schedule(0.0, 3e-4, 200, STEPS, end_value=1.5e-5)
    values = {str(i): float(sched(i)) for i in (0, 1, 199, 200, STEPS - 1, STEPS)}
    print(f"Source {source['source_arms']}, official test {len(rows)}, LR by Optax count {values}", flush=True)
    contract = dict(revision=args.revision, cpu_qualification=args.cpu_qualification, hostname=platform.node(),
                    devices=[str(d) for d in devices], versions={k: version(k) for k in ("jax", "jaxlib", "optax", "flax", "numpy")},
                    model=TAG, mode=MODE, config=asdict(CFG), seeds=SEEDS, batch_size=BATCH,
                    actual_updates_per_seed=steps, schedule_steps=STEPS, schedule_lr=values, source=source,
                    test_panel=str(ROOT / SCALE / "gpu09test/test_live"), ordered_test_sha256=TEST_SET_SHA,
                    bootstrap=dict(unit="suite/task_id", fixed_paired_seeds=SEEDS, draws=DRAWS, seed=BOOT_SEED))
    stage1.write_json(out / "contract.json", contract)
    paired, checkpoints = [], []
    for seed in SEEDS:
        name = f"{TAG}-{MODE}-seed{seed}.msgpack"
        original = ROOT / CAMPAIGN / "gpu03v2/stage1/checkpoints" / name
        a, orig = score_checkpoint(original, seed, STEPS, gain, test, rows, ORIGINAL_SHA[seed])
        print(f"Seed {seed}: starting {steps} candidate updates, batch {BATCH}", flush=True)
        audit = stage1.train_and_audit_v2(
            MODE, TAG, CFG, train, dev, dev_rows, donor, seeds=(seed,), steps=steps, batch_size=BATCH,
            ckpt_dir=out / "checkpoints", nominal_gain=gain, archive_sha=ARCHIVE_SHA, lr_schedule=sched,
        )
        stage1.write_json(out / f"dev-audit-{seed}.json", audit["seeds"][0])
        b, cand = score_checkpoint(out / "checkpoints" / name, seed, steps, gain, test, rows)
        seed_rows = [dict(**r, reader_seed=seed, original=av, candidate=bv) for r, av, bv in zip(rows, a, b, strict=True)]
        stage1.write_json(out / f"paired-cases-{seed}.json", seed_rows)
        checkpoints.append(dict(seed=seed, original=orig, candidate=cand))
        paired.extend(seed_rows)
        print(f"Seed {seed}: checkpoint exported, reloaded and scored on {len(seed_rows)} paired supports", flush=True)
    results = summarize(paired, SEEDS)
    stage1.write_json(out / "summary.json", dict(**contract, checkpoints=checkpoints, panels=results))
    if not args.cpu_qualification:
        (out / "RESULT.md").write_text(render(results))
    print(f"Wrote {out / 'summary.json'} ({len(paired)} trace-seed pairs)", flush=True)


if __name__ == "__main__":
    main()
