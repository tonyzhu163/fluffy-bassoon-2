"""Fit a dual-arm PaliGemma motor, select it on source development, and confirm once."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import platform
import random
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "libero-paligemma-crossarm-20261120-r1"
sys.path.insert(1, str(PRIOR))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import transformers

import motor as base
from motor_bank import BODIES, coverage, load_bank, partition, sample, validation_anchors
from motor_weights import sha256

MIX = "google/paligemma-3b-mix-224"


@contextmanager
def exclusive(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "motor.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def source_ids():
    return {p.name: sha256(p) for p in (
        HERE / "motor_train.py", HERE / "motor_bank.py", HERE / "motor_eval.py",
        PRIOR / "motor.py", PRIOR / "motor_data.py", PRIOR / "motor_weights.py",
        PRIOR / "adapt_eval.py", PRIOR / "sim.py", PRIOR / "pretrain_reader.py", PRIOR / "oracle_fallback.py",
        HERE.parent / "libero-crossarm-20261120-r1" / "arm_env.py",
        HERE.parent / "libero-crossarm-20261120-r1" / "sim_reset.py",
    )}


def runtime():
    prop = torch.cuda.get_device_properties(0)
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                transformers=transformers.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                gpu=prop.name, capability=[prop.major, prop.minor], gpu_bytes=prop.total_memory)


def save_pt(path, value):
    tmp = path.with_suffix(".pt.tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def read_pt(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def schedule(args):
    if args.steps < 1 or args.anchors_per_pose < 1 or not 0 <= args.seed < 2**32:
        raise ValueError("Require positive update/anchor counts and a uint32 training seed")
    steps = sorted(set(args.candidate_steps if args.candidate_steps is not None else
                       (max(1, args.steps // 4), max(1, args.steps // 2), args.steps)))
    if not steps or steps[0] < 1 or steps[-1] > args.steps or args.shortlist < 1:
        raise ValueError("Candidate steps must be in the fit and the shortlist must be positive")
    return steps


def update(motor, optim, params, trainable_lora, batch, microbatch, step):
    optim.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device="cuda")
    for i in range(0, len(batch), microbatch):
        obs, target, mask = base.collate(batch[i:i + microbatch])
        pred = motor(obs)
        errors = F.smooth_l1_loss(pred, target, beta=0.1, reduction="none").mean(-1)
        loss = ((errors * mask).sum(1) / mask.sum(1)).sum() / len(batch)
        loss.backward()
        loss_sum += loss.detach()
    metrics = {}
    if step == 1:
        lora_norm = torch.stack([p.grad.norm() for p in trainable_lora if p.grad is not None]).norm().item()
        if not np.isfinite(lora_norm) or lora_norm <= 0:
            raise RuntimeError("The supervised loss produced no finite signal at the pretrained Gemma tail")
        metrics["lora_grad_norm"] = lora_norm
    norm = nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    optim.step()
    metrics.update(step=step, train_huber=loss_sum.item(), grad_norm=norm.item())
    return metrics


def train(args):
    candidates = schedule(args)
    out = args.output
    manifest, restart = out / "manifest.json", out / "training.pt"
    if manifest.exists() and not args.resume:
        raise ValueError("Existing fit requires --resume")
    if args.resume and not manifest.exists():
        raise ValueError("--resume requires an existing manifest")
    if not manifest.exists() and any(p.name != "motor.lock" for p in out.iterdir()):
        raise ValueError("New fit requires an empty dedicated directory")
    base.require_gpu()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sampler = np.random.default_rng(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    scratch_root = args.scratch or out.parent
    scratch_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=scratch_root, prefix="motor-frames-") as temporary:
        episodes, records, indices = load_bank(args.demos, Path(temporary))
        groups, held = partition(episodes)
        anchors = validation_anchors(held, args.anchors_per_pose)
        motor = base.Motor(args.weights)
        if motor.identity["repo"] != MIX:
            raise ValueError(f"This run is pinned to {MIX}")
        unique = [r for r in records if "duplicate_of" not in r]
        episode_files = {id(e): [r["bank"], r["file"]] for e, r in zip(episodes, unique, strict=True)}
        states = np.concatenate([e.state for e in episodes if int(e.pose) < 28]).astype(np.float64)
        motor.state_mean.copy_(torch.as_tensor(states.mean(0)))
        motor.state_scale.copy_(torch.as_tensor(np.maximum(states.std(0), 0.01)))
        lora = [p for p in motor.vlm.parameters() if p.requires_grad]
        head = list(motor.head.parameters())
        params = lora + head
        optim = torch.optim.AdamW([dict(params=lora, lr=base.LORA_LR), dict(params=head, lr=base.HEAD_LR)],
                                  weight_decay=0.01)
        fit_recipe = base.recipe()
        fit_recipe.pop("validate_every")
        fit_recipe.update(candidate_steps=candidates, online_shortlist=args.shortlist,
                          anchors_per_offline_pose=args.anchors_per_pose, total_steps=args.steps,
                          sampler="equal arm; uniform successful pose, unique tape, control")
        contract = dict(protocol="dualarm-native-goal5-v2", revision=args.revision, seed=args.seed,
                        microbatch=args.microbatch, weights=motor.identity, source_bodies=list(BODIES),
                        episodes=records, coverage=coverage(records), collector_index_sha256=indices, recipe=fit_recipe,
                        holdout_poses=sorted({int(e.pose) for e in episodes if int(e.pose) >= 28}),
                        train_poses=sorted({int(e.pose) for e in episodes if int(e.pose) < 28}),
                        all_motor_poses=sorted({int(e.pose) for e in episodes}), sources=source_ids(),
                        validation_anchors=[[*episode_files[id(e)], t] for e, t in anchors],
                        runtime=runtime(), trainable=dict(lora=sum(p.numel() for p in lora),
                                                         head=sum(p.numel() for p in head)))
        if args.resume:
            if json.loads(manifest.read_text())["contract"] != contract:
                raise ValueError("Resume requires identical banks, procedure, code, weights, and runtime")
        else:
            base.atomic_json(manifest, dict(contract=contract, argv=sys.argv, demos=[str(p.resolve()) for p in args.demos],
                                            weights_path=str(args.weights.resolve())))
        fit_id = sha256(manifest)
        curve, snapshots, completed = [], [], 0
        if restart.exists():
            saved = read_pt(restart)
            if saved["fit_id"] != fit_id:
                raise ValueError("Training checkpoint and manifest differ")
            motor.restore_delta(saved["model"])
            optim.load_state_dict(saved["optimizer"])
            base.restore_rng(saved["rng"], sampler)
            completed, curve, snapshots = saved["step"], saved["curve"], saved["snapshots"]
            del saved
            for snap in snapshots:
                if sha256(out / snap["file"]) != snap["sha256"]:
                    raise ValueError("Retained candidate differs from the restart receipt")
        (out / "candidates").mkdir(exist_ok=True)
        torch.cuda.reset_peak_memory_stats()
        for step in range(completed + 1, args.steps + 1):
            torch.cuda.synchronize()
            tick = perf_counter()
            batch = sample(groups, sampler, base.GLOBAL_BATCH)
            row = update(motor, optim, params, lora, batch, args.microbatch, step)
            torch.cuda.synchronize()
            row["update_seconds"] = perf_counter() - tick
            row["frames_per_second"] = base.GLOBAL_BATCH / row["update_seconds"]
            row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            if step in candidates:
                vt = perf_counter()
                val = base.validate(motor, anchors, args.microbatch)
                row.update(validation=val, validation_seconds=perf_counter() - vt)
                file = f"candidates/step-{step:06d}.pt"
                save_pt(out / file, dict(schema=base.SCHEMA, contract=contract, step=step, fit_id=fit_id,
                                         best=dict(step=step, mae=val["mae"], model=motor.delta())))
                snapshots.append(dict(file=file, step=step, sha256=sha256(out / file), validation=val))
            curve.append(row)
            if step == 1 or step % base.CHECKPOINT_EVERY == 0 or step in candidates or step == args.steps:
                save_pt(restart, dict(fit_id=fit_id, step=step, model=motor.delta(), optimizer=optim.state_dict(),
                                      rng=base.rng_state(sampler), curve=curve, snapshots=snapshots))
                base.atomic_json(out / "curve.json", dict(updates=curve, snapshots=snapshots))
                print(json.dumps(row, allow_nan=False), flush=True)
        # Also repair a final index if the process stopped immediately after its last optimizer checkpoint.
        if [r["step"] for r in snapshots] != candidates:
            raise RuntimeError("Candidate schedule incomplete")
        shortlist = sorted(snapshots, key=lambda s: (s["validation"]["mae"], s["step"]))[:args.shortlist]
        base.atomic_json(out / "curve.json", dict(updates=curve, snapshots=snapshots))
        base.atomic_json(out / "candidates.json", dict(status="complete", fit_id=fit_id,
                                                       snapshots=snapshots, shortlist=[r["step"] for r in shortlist]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fit = stages.add_parser("train", help="fit both arms and retain candidate updates")
    dev = stages.add_parser("dev", help="score offline-shortlisted candidates and freeze one eligible winner")
    confirm = stages.add_parser("confirm", help="score the sealed development winner once on both sources")
    for p in (fit, dev, confirm):
        p.add_argument("--output", type=Path, required=True, help="same persistent fit directory for all stages")
        p.add_argument("--weights", type=Path, default=os.environ.get("DUAL_WEIGHTS"),
                       required=not bool(os.environ.get("DUAL_WEIGHTS")))
        p.add_argument("--revision", default=os.environ.get("MOTOR_CODE_REVISION"),
                       required=not bool(os.environ.get("MOTOR_CODE_REVISION")))
    fit.add_argument("--demos", type=Path, nargs="+", action="extend", required=True)
    fit.add_argument("--steps", type=int, default=800)
    fit.add_argument("--candidate-steps", type=int, nargs="+", help="default quarter, half and final update")
    fit.add_argument("--shortlist", type=int, default=3, help="maximum offline candidates for native development")
    fit.add_argument("--anchors-per-pose", type=int, default=32)
    fit.add_argument("--seed", type=int, default=20261120)
    fit.add_argument("--microbatch", type=int, choices=(1, 2, 4, 8), default=4)
    fit.add_argument("--scratch", type=Path, help="local or shared project NVMe for staged camera arrays")
    fit.add_argument("--resume", action="store_true")
    confirm.add_argument("--seal-dir", type=Path, required=True, help="one shared directory for this campaign")
    args = parser.parse_args()
    args.output = args.output.resolve()
    with exclusive(args.output):
        if args.stage == "train":
            train(args)
        else:
            from motor_eval import run
            return run(args, source_ids(), runtime(), exclusive)


if __name__ == "__main__":
    raise SystemExit(main())
