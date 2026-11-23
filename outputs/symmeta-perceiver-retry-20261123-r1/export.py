"""Refit PR7's three warm-schedule readers and export paired passive predictions."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time

from flax import serialization
import jax
import numpy as np

import eval_wider_family as pr7
from eval_stage2a_live_ur5e import load_v2_checkpoint
from eval_stage3_scaling import build_collection_plan, build_test_plan
from fluffy_bassoon.xemb.baselines import compute_dis_flow_maps
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
from fluffy_bassoon.xemb.equiv_v2 import predict_embodiment_v2

HERE = Path(__file__).resolve().parent
CODE_OUT = Path(pr7.__file__).resolve().parents[1]
OUT = Path("/mowenjie/fluffy-bassoon-2/outputs")
CAMPAIGN = "xemb-icml-campaign-20261121-r1"
SCALE = "libero-xemb-ur5e-scale-20261120-r1"
LOCKED = OUT / SCALE / "gpu09test/test_live"
DIVERSE = OUT / CAMPAIGN / "gpu12scale/stage3/diverse_shards"
CHECKPOINTS = OUT / CAMPAIGN / "gpu03v2/stage1/checkpoints"
REFERENCE = OUT / "symmeta-wider-family-20261123-r1/gpu01w454/wider-family-summary.json"
REFERENCE_SHA = "02f10b464a91abc616ac600c2be025550d1b8f85350228c8645155a19fc7db18"
TAG = "perceiver_cont_warm"
PAPER = "equiv_v2_dis_dz"
# Last minibatch loss, at the four decimals retained in gpu01w454/run.log.
PR7_LOGGED_LOSS = {41: 0.0240, 42: 0.0193, 43: 0.0197}
PANELS = ("gpu09test", "gpu09iiwa", "gpu06kinova3", "gpu08pi0fix")
SEEDS = (41, 42, 43)
IDENTITY = ("index", "suite", "task_id", "embodiment")


def require(test, message):
    if not test:
        raise ValueError(str(message))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")
    tmp.replace(path)


def check_sources(lock_path):
    hashes = read_json(lock_path)["sha256"]
    shards = sorted(DIVERSE.glob("shard-*"))
    require(shards == [DIVERSE / f"shard-{i}" for i in range(8)], "PR7 shard order")
    paths = [s / "diverse.npz" for s in shards] + [paper_path(seed) for seed in SEEDS]
    require({p.relative_to(OUT).as_posix() for p in paths} == hashes.keys(), "PR7 input inventory")
    for path in paths:
        require(sha(path.read_bytes()) == hashes[path.relative_to(OUT).as_posix()], path)
    plan = build_collection_plan()
    for i, shard in enumerate(shards):
        require(read_json(shard / "meta.json") == plan[i::8], f"PR7 source order in {shard}")
    require(Counter(p["arm"] for p in plan) == {"Panda": 733, "Sawyer": 733}, "PR7 source arms")
    ref = REFERENCE.read_bytes()
    require(sha(ref) == REFERENCE_SHA, "PR7 score file changed")
    return json.loads(ref)["rows"]


def paper_path(seed):
    return CHECKPOINTS / f"{PAPER}-passive_attempt1-seed{seed}.msgpack"


def tape_evidence(panel):
    if panel == "gpu09test":
        path = CODE_OUT / "symmeta-camera-command-20261123-r1/evidence/roll.json.gz"
        with gzip.open(path, "rt") as f:
            return [dict(s["case"], support_sha256=s["npz"]["sha256"]) for s in json.load(f)["supports"]]
    path = CODE_OUT / f"symmeta-fk-crossarm-20261123-r1/evidence/{panel}-cases.json.gz"
    with gzip.open(path, "rt") as f:
        evidence = json.load(f)
    require(evidence["panel"] == panel, path)
    return evidence["rows"]


def load_panel(panel, plan):
    base = LOCKED if panel == "gpu09test" else OUT / CAMPAIGN / panel / "live"
    n_chunks = 3 if panel == "gpu09test" else 24
    valid = np.arange(n_chunks) < pr7.SUPPORT_CHUNKS
    refs = tape_evidence(panel)
    require(len(refs) == len(plan) == 96, panel)
    rows, tapes = [], []
    for shard in range(8):
        folder = base / f"shard-{shard}"
        meta = read_json(folder / "support-meta.json")
        require(len(meta) == 12, folder)
        for slot, support in enumerate(meta):
            i = 12 * shard + slot
            case, ref = plan[i], refs[i]
            identity = {k: case[k] for k in IDENTITY}
            require(i == case["index"] and identity == {k: ref[k] for k in IDENTITY}, (panel, identity))
            name = case["embodiment"]
            path = folder / "support" / f"{slot:02d}-{name}-passive.npz"
            require(support["embodiment"] == name and Path(support["passive_npz"]) == path
                    and support["physics"] == ref["physics"], path)
            raw = path.read_bytes()
            digest = sha(raw)
            require(digest == ref["support_sha256"], f"Wrong passive tape: {path}")
            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                tape = {k: z[k] for k in pr7.CHUNK_KEYS if k != "dis_flow"}
                require(tape["valid"].dtype == np.bool_ and np.array_equal(tape["valid"], valid), path)
                if "chunk_steps" in z:
                    require(np.array_equal(z["chunk_steps"], valid * 5), path)
            # Stage 2A _load_npz computes DIS on one tape with the native temporal axis.
            tape["dis_flow"] = compute_dis_flow_maps(tape["img_before"], tape["img_after"], tape["valid"])
            tapes.append(tape)
            rows.append(dict(panel=panel, **identity, regime=case["regime"], tape_sha256=digest))
    return rows, {k: np.stack([t[k] for t in tapes]) for k in pr7.CHUNK_KEYS}


def locked_archive_predictions(plan):
    path = CODE_OUT / "symmeta-paired-statistics-20261123-r1/sources.csv"
    by_seed = {s: [] for s in SEEDS}
    with path.open() as f:
        refs = [r for r in csv.DictReader(f) if r["panel"] == "ur5e-pi05"]
    require([int(r["task_order"]) for r in refs] == list(range(8)), path)
    for i, r in enumerate(refs):
        for files, digest, seeds in (("seed41_and_hs", "sha256_41_hs", (41,)),
                                     ("seeds42_43", "sha256_42_43", (42, 43))):
            data = Path(r[files]).read_bytes()
            require(sha(data) == r[digest], r[files])
            shard = json.loads(data)
            require(shard["shard_idx"] == i and len(shard["rows"]) == 12, r[files])
            for case, row in zip(plan[i*12:(i+1)*12], shard["rows"], strict=True):
                require(all(case[k] == row[k] for k in (*IDENTITY, "regime")), case)
                for seed in seeds:
                    arm = "ours_equiv_v2_passive" + ("" if seed == 41 else f"_s{seed}")
                    by_seed[seed].append(row["arms"][arm])
    return by_seed


def as_predictions(params, cfg, batch, per_case=False):
    # Stage 2A ran the paper reader on each tape separately. PR7 ran a 96 × 3 batch.
    n = len(batch["valid"])
    size = 1 if per_case else n
    values = [predict_embodiment_v2(params, cfg, {k: v[i:i+size] for k, v in batch.items()})
              for i in range(0, n, size)]
    yaw, scale, sign = (np.concatenate(v) for v in zip(*values, strict=True))
    require(np.isfinite(yaw).all() and np.isfinite(scale).all() and (scale > 0).all()
            and np.isin(sign, (-1, 1)).all(), "Invalid reader output")
    return [dict(yaw_deg=float(y), xy_scale=float(s), gripper_sign=int(g))
            for y, s, g in zip(yaw, scale, sign, strict=True)]


def restore(path, params, cfg, seed, steps, loss, test, source_lock_sha):
    before = as_predictions(params, cfg, test)
    payload = dict(params=params, reader_v2_config=asdict(cfg), mode="passive_attempt1", tag=TAG,
                   seed=seed, steps=steps, batch_size=32, loss=loss, source_lock_sha256=source_lock_sha)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(serialization.to_bytes(payload))
    tmp.replace(path)
    p, c, meta = load_v2_checkpoint(path)
    require(c == cfg and meta["seed"] == seed and meta["steps"] == steps, path)
    after = as_predictions(p, c, test)
    require(before == after, f"Factor predictions changed after checkpoint restore: {path}")
    return p, c, after


def scores(yaw, est_op, test, regimes):
    true = (test["xy_scale"].astype(np.float64)[:, None, None] * pr7.rot2_batch(test["yaw_deg"])).astype(np.float32)
    result = pr7.summarize(true, est_op)
    errs = np.array([pr7.circular_yaw_error_deg(float(y), float(t)) for y, t in zip(yaw, test["yaw_deg"], strict=True)])
    ood = np.array(regimes) != "familiar"
    require(ood.sum() == 64 and len(ood) == 96, "Locked UR5e split")
    return dict(**result, yaw_mae_deg=float(errs.mean()), yaw_mae_ood_deg=float(errs[ood].mean()))


def compare_pr7(params, cfg, pred, model, seed, test, regimes, refs):
    ref, = (r for r in refs if (r["model"], r["seed"], r["condition"], r["chunks"])
            == (model, seed, "recorded", 3))
    # Use the original batched, jitted scorer to compare to the original figures.
    op, yaw = pr7.predict_ops(params, "factors", cfg, test)
    score = scores(yaw, op, test, regimes)
    archived = {k: ref[k] for k in score}
    native_yaw = np.array([p["yaw_deg"] for p in pred])
    native_scale = np.array([p["xy_scale"] for p in pred])
    exported = scores(native_yaw, native_scale[:, None, None] * pr7.rot2_batch(native_yaw), test, regimes)
    exported["correct_grips"] = int(np.sum(np.array([p["gripper_sign"] for p in pred]) == test["gripper_sign"]))
    return dict(pr7_scorer=score, exported=exported, archived=archived,
                difference={k: score[k] - v for k, v in archived.items()},
                same_published_ood_yaw=round(score["yaw_mae_ood_deg"], 2) == round(ref["yaw_mae_ood_deg"], 2))


def archive_residual(pred, ref):
    return dict(
        max_circular_yaw_difference_deg=max(pr7.circular_yaw_error_deg(p["yaw_deg"], r["yaw_hat"])
                                            for p, r in zip(pred, ref, strict=True)),
        max_scale_difference=max(abs(p["xy_scale"] - r["scale_hat"]) for p, r in zip(pred, ref, strict=True)),
        gripper_mismatches=sum(p["gripper_sign"] != r["grip_hat"] for p, r in zip(pred, ref, strict=True)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--output", type=Path)
    parser.add_argument("--input-lock", type=Path, default=HERE / "export-inputs.json")
    args = parser.parse_args()
    if args.preflight:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and jax.default_backend() == "cpu", "CPU isolation")
    else:
        probe = read_json(args.output / "gpu.json")
        require(probe["status"] == "passed" and probe["hostname"] == os.uname().nodename
                and probe["exclusive_training_admission"] and len(probe["devices"]) == 1
                and jax.default_backend() == "gpu" and len(jax.devices()) == 1, "Single-GPU training admission")
        device = probe["devices"][0]["cuda_name"].upper()
        require(device in {"NVIDIA GEFORCE RTX 4090", "NVIDIA RPBZZZ6"} or "RTX PRO 6000" in device,
                f"Expected RTX 4090 or Baige Pro 6000D, found {device}")
        require(not (args.output / "predictions.json").exists(), "Attempt already exported")
    refs = check_sources(args.input_lock)
    source_lock_sha = sha(args.input_lock.read_bytes())
    _, plan = build_test_plan()
    archive = locked_archive_predictions(plan)
    panels = {panel: load_panel(panel, plan) for panel in PANELS}
    print("Verified 384 physical (panel,index) keys and passive NPZ digests.", flush=True)

    # The PR7 loader concatenates lexical shard order, then crops. train_model owns
    # JAX initialization, minibatch choice and relabel draws from its seed-local RNG.
    train_full, test, regimes = pr7.load_recorded(DIVERSE, LOCKED, CODE_OUT / CAMPAIGN)
    train = {k: np.ascontiguousarray(v) for k, v in pr7.take_chunks(train_full, pr7.SUPPORT_CHUNKS).items()}
    del train_full
    require(train["valid"].shape == (1466, 3) and train["valid"].all(), "PR7 training shape")
    locked = panels["gpu09test"][1]
    for k in pr7.CHUNK_KEYS:
        require(np.array_equal(locked[k], test[k]), f"PR7 locked input mismatch for {k}")
    require(regimes == [r["regime"] for r in plan], "PR7 regime order")
    for k in pr7.LABEL_KEYS:
        want = [getattr(get_embodiment_spec(r["embodiment"]), k) for r in plan]
        require(np.array_equal(test[k], np.asarray(want, dtype=test[k].dtype)), f"PR7 label order: {k}")
    spec, = (row for row in pr7.TRAINED if row[0] == TAG)
    _, head, cfg, families, lr, warmup, steps = spec
    require((head, cfg.max_chunks, families, lr, warmup, steps, pr7.SEEDS, pr7.SUPPORT_CHUNKS) ==
            ("factors", 24, ("family",), 3e-4, 200, 2500, SEEDS, 3), "PR7 recipe changed")

    if args.preflight:
        # An untrained template exercises the 24-position save/restore path.
        import cv2
        small = {k: v[:1] for k, v in pr7.take_chunks(panels["gpu09iiwa"][1], 3).items()}
        params = pr7.init_model(jax.random.key(41), head, cfg)
        with tempfile.TemporaryDirectory(prefix="export-") as tmp:
            restore(Path(tmp) / "export-template.msgpack", params, cfg, 41, 0, None, small, source_lock_sha)
        print("24-position template roundtrip exact on a cropped 24-slot IIWA tape.")
        for seed in SEEDS:
            p, c, meta = load_v2_checkpoint(paper_path(seed))
            require(meta["seed"] == seed and meta["tag"] == PAPER, paper_path(seed))
            pred = as_predictions(p, c, locked, per_case=True)
            comparison = compare_pr7(p, c, pred, "symmeta_paper+procrustes", seed, test, regimes, refs)
            print(json.dumps(dict(seed=seed, paper=comparison,
                                  archive=archive_residual(pred, archive[seed]))), flush=True)
            if seed == 41:
                for panel in PANELS[1:]:
                    batch = {k: v[:1] for k, v in panels[panel][1].items()}
                    print(panel, "native paper prediction, index 0", as_predictions(p, c, batch)[0])
        print(f"CPU preflight passed, JAX {jax.__version__}, OpenCV {cv2.__version__}; 0 refits.")
        return

    out = args.output
    ckpts = out / "checkpoints"
    ckpts.mkdir()
    rows, reports = [], []
    for seed in SEEDS:
        start = time.monotonic()
        p, loss = pr7.train_model(train, head, cfg, families, lr, min(warmup, steps // 4), steps, seed, 32)
        seconds = time.monotonic() - start
        require(np.isfinite(loss), f"Nonfinite training loss, seed {seed}")
        print(f"seed {seed} completed {steps} updates b32, loss={loss}, seconds={seconds:.1f}", flush=True)
        path = ckpts / f"{TAG}-passive_attempt1-seed{seed}.msgpack"
        tuned, tuned_cfg, restored = restore(path, p, cfg, seed, steps, loss, locked, source_lock_sha)
        paper, paper_cfg, meta = load_v2_checkpoint(paper_path(seed))
        require(meta["tag"] == PAPER and meta["seed"] == seed and meta["mode"] == "passive_attempt1", paper_path(seed))
        report = dict(seed=seed, steps=steps, batch_size=32, loss=loss, pr7_logged_loss=PR7_LOGGED_LOSS[seed],
                      fit_seconds=seconds,
                      checkpoint=path.relative_to(out).as_posix(), checkpoint_sha256=sha(path.read_bytes()),
                      restored_exact_cases=len(restored))
        for panel, (cases, batch) in panels.items():
            sym = as_predictions(paper, paper_cfg, batch, per_case=True)
            if panel == "gpu09test":
                pred = restored
                report["tuned"] = compare_pr7(tuned, tuned_cfg, pred, TAG, seed, test, regimes, refs)
                report["symmeta"] = compare_pr7(paper, paper_cfg, sym, "symmeta_paper+procrustes", seed, test, regimes, refs)
                report["symmeta_vs_rounded_campaign"] = archive_residual(sym, archive[seed])
            else:
                pred = as_predictions(tuned, tuned_cfg, pr7.take_chunks(batch, 3))
            rows.extend(dict(**r, seed=seed, tuned=t, symmeta=s) for r, t, s in zip(cases, pred, sym, strict=True))
        reports.append(report)
        write_json(out / "training-sanity.json", dict(pr7_sha256=REFERENCE_SHA, seeds=reports))
        print(json.dumps(report), flush=True)

    keys = {(r["panel"], r["index"], r["seed"]) for r in rows}
    require(len(rows) == 1152 and keys == {(p, i, s) for p in PANELS for i in range(96) for s in SEEDS},
            "Incomplete prediction matrix")
    # Exact tape identity, checkpoint roundtrip and coverage are validity checks.
    # Compare stochastic refits to the reported digits without hiding valid outputs
    # or converting hardware-dependent rounding into a scientific go/no-go rule.
    match = all(r[k]["same_published_ood_yaw"] for r in reports for k in ("tuned", "symmeta"))
    write_json(out / "predictions.json", dict(schema=1, rows=rows, pr7_ood_yaw_matches_published_precision=match))
    print("Published 1152 paired predictions.", flush=True)


if __name__ == "__main__":
    main()
