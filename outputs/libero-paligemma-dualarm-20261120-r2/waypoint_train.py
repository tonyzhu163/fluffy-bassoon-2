"""Fit or evaluate the visual waypoint motor on the registered dual-arm source partitions."""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "libero-paligemma-crossarm-20261120-r1"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import motor as base
import motor_eval as panel
import motor_train as common
import numpy as np
import torch
from motor_bank import BODIES, coverage, partition, sample, validation_anchors
from motor_weights import sha256
from torch.nn import functional as F
from waypoint_data import load_labeled
from waypoint_motor import ARCH, LOOKAHEAD, MOTION_SCALE_M, PLATE_SCALE_M, WaypointMotor


def sources():
    paths = [
        HERE / name
        for name in (
            "waypoint_train.py",
            "waypoint_motor.py",
            "waypoint_data.py",
            "waypoint_labels.py",
            "waypoint_capture.py",
            "paddle.py",
            "expert.py",
            "expert_sweep.py",
            "recover.py",
        )
    ]
    paths.append(common.PRIOR / "demos.py")
    return {**common.source_ids(), **{p.name: sha256(p) for p in paths}}


def collate(batch):
    observations, controls, masks, plates, motions = zip(*batch)
    arrays = [
        torch.from_numpy(np.stack(seq)).cuda()
        for seq in (controls, masks, plates, motions)
    ]
    return observations, *arrays


def per_anchor(values, mask):
    return (values.mean(-1) * mask).sum(1) / mask.sum(1)


def update(model, optimizer, params, lora, batch, micro, step):
    optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(3, device="cuda")
    for offset in range(0, len(batch), micro):
        obs, controls, mask, plates, motions = collate(batch[offset : offset + micro])
        plate, motion, actions = model.predict(obs)
        plate_loss = F.smooth_l1_loss(
            (plate - plates) / PLATE_SCALE_M,
            torch.zeros_like(plate),
            beta=0.1,
            reduction="none",
        ).mean(-1)
        motion_loss = per_anchor(
            F.smooth_l1_loss(
                motion / MOTION_SCALE_M,
                motions / MOTION_SCALE_M,
                beta=0.1,
                reduction="none",
            ),
            mask,
        )
        control_loss = per_anchor(
            F.smooth_l1_loss(actions, controls, beta=0.1, reduction="none"), mask
        )
        terms = torch.stack(
            (plate_loss.sum(), motion_loss.sum(), control_loss.sum())
        ) / len(batch)
        terms.sum().backward()
        totals += terms.detach()
    metrics = {}
    if step == 1:
        grads = [p.grad.norm() for p in lora if p.grad is not None]
        grad = torch.stack(grads).norm().item() if grads else 0.0
        if not np.isfinite(grad) or grad <= 0:
            raise RuntimeError("No finite supervised Gemma LoRA gradient")
        metrics["lora_grad_norm"] = grad
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    optimizer.step()
    metrics.update(
        step=step,
        train_loss=totals.sum().item(),
        grad_norm=norm.item(),
        loss_by_target=dict(zip(("plate", "motion", "native_action"), totals.tolist())),
    )
    return metrics


@torch.inference_mode()
def validate(model, anchors, micro, mean_plate):
    by_arm = defaultdict(list)
    mean_plate = torch.as_tensor(mean_plate, device="cuda", dtype=torch.float32)
    for i in range(0, len(anchors), micro):
        group = anchors[i : i + micro]
        obs, controls, mask, plates, motions = collate(
            [ep.sample(t) for ep, t in group]
        )
        plate, motion, actions = model.predict(obs)
        plate_error = (plate - plates).norm(dim=-1)
        prior_error = (mean_plate - plates).norm(dim=-1)
        motion_error = ((motion - motions).norm(dim=-1) * mask).sum(1) / mask.sum(1)
        zero_error = (motions.norm(dim=-1) * mask).sum(1) / mask.sum(1)
        axes = ((actions - controls).abs() * mask[..., None]).sum(1) / mask.sum(
            1, keepdim=True
        )
        rows = torch.cat(
            (
                axes,
                torch.stack((plate_error, prior_error, motion_error, zero_error), 1),
            ),
            1,
        )
        for (ep, _), row in zip(group, rows.cpu().numpy(), strict=True):
            by_arm[ep.body].append(row)
    means = {b: np.mean(by_arm[b], axis=0, dtype=np.float64) for b in BODIES}
    if any(not np.isfinite(r).all() for r in means.values()):
        raise FloatingPointError("Nonfinite offline metric")
    return {
        "mae": float(np.mean([r[:7].mean() for r in means.values()])),
        "anchors_by_body": {b: len(by_arm[b]) for b in BODIES},
        "axis_mae_by_body": {b: r[:7].tolist() for b, r in means.items()},
        "physical_errors_by_body": {
            b: dict(
                zip(
                    (
                        "plate_l2_m",
                        "train_mean_plate_l2_m",
                        "motion_l2_m",
                        "zero_motion_l2_m",
                    ),
                    r[7:].tolist(),
                )
            )
            for b, r in means.items()
        },
    }


def train(args):
    steps = common.schedule(args)
    out = args.output
    manifest, restart = out / "manifest.json", out / "training.pt"
    if manifest.exists() != args.resume or (
        not args.resume and any(p.name != "motor.lock" for p in out.iterdir())
    ):
        raise ValueError(
            "Use a fresh fit directory or --resume against the original fit"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    scratch = args.scratch or out.parent
    scratch.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=scratch, prefix="waypoint-frames-") as tmp:
        episodes, records, indices = load_labeled(args.demos, Path(tmp))
        groups, held = partition(episodes)
        anchors = validation_anchors(held, args.anchors_per_pose)
        training = [e for e in episodes if int(e.pose) < 28]
        states = np.concatenate([e.episode.state for e in training]).astype(np.float64)
        mean_plate = np.concatenate([e.plate for e in training]).mean(
            0, dtype=np.float64
        )
        model = WaypointMotor(args.weights)
        if model.identity["repo"] != common.MIX:
            raise ValueError("Expected the verified original PaliGemma MIX snapshot")
        model.state_mean.copy_(torch.as_tensor(states.mean(0)))
        model.state_scale.copy_(torch.as_tensor(states.std(0).clip(min=0.01)))
        lora = [p for p in model.vlm.parameters() if p.requires_grad]
        head = list(model.head.parameters())
        params = lora + head
        optimizer = torch.optim.AdamW(
            [
                {"params": lora, "lr": base.LORA_LR},
                {"params": head, "lr": base.HEAD_LR},
            ],
            weight_decay=0.01,
        )
        recipe = base.recipe()
        recipe.pop("validate_every")
        recipe.update(
            lookahead=LOOKAHEAD,
            motion_scale_m=MOTION_SCALE_M,
            plate_scale_m=PLATE_SCALE_M,
            loss_weights={"plate": 1.0, "motion": 1.0, "native_action": 1.0},
            total_steps=args.steps,
            candidate_steps=steps,
            online_shortlist=args.shortlist,
            anchors_per_offline_pose=args.anchors_per_pose,
            sampler="equal arm, uniform layout/tape/tick, inputs sorted by layout and trajectory digest",
        )
        contract = {
            "protocol": "dualarm-native-goal5-v2",
            "architecture": ARCH,
            "revision": args.revision,
            "seed": args.seed,
            "microbatch": args.microbatch,
            "weights": model.identity,
            "source_bodies": list(BODIES),
            "episodes": records,
            "coverage": coverage(records),
            "collector_index_sha256": indices,
            "recipe": recipe,
            "train_poses": sorted({int(e.pose) for e in training}),
            "holdout_poses": sorted({int(e.pose) for e, _ in anchors}),
            "all_motor_poses": sorted({int(e.pose) for e in episodes}),
            "sources": sources(),
            "validation_anchors": [
                [e.record["trajectory_sha256"], t] for e, t in anchors
            ],
            "runtime": common.runtime(),
            "trainable": {
                "lora": sum(p.numel() for p in lora),
                "head": sum(p.numel() for p in head),
            },
        }
        if args.resume:
            if panel.read(manifest)["contract"] != contract:
                raise ValueError(
                    "Registered banks, labels, settings, code, runtime or weights changed"
                )
        else:
            base.atomic_json(
                manifest,
                {
                    "contract": contract,
                    "argv": sys.argv,
                    "demos": [str(p.resolve()) for p in args.demos],
                    "weights_path": str(args.weights.resolve()),
                },
            )
        fit_id = sha256(manifest)
        completed, curve, snapshots = 0, [], []
        if restart.is_file():
            saved = common.read_pt(restart)
            if saved["fit_id"] != fit_id:
                raise ValueError("Restart checkpoint belongs to a different fit")
            model.restore_delta(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            base.restore_rng(saved["rng"], rng)
            completed, curve, snapshots = (
                saved["step"],
                saved["curve"],
                saved["snapshots"],
            )
            for item in snapshots:
                if sha256(out / item["file"]) != item["sha256"]:
                    raise ValueError(
                        "Retained candidate changed since the restart checkpoint"
                    )
            del saved
        (out / "candidates").mkdir(exist_ok=True)
        torch.cuda.reset_peak_memory_stats()
        for step in range(completed + 1, args.steps + 1):
            torch.cuda.synchronize()
            tic = perf_counter()
            row = update(
                model,
                optimizer,
                params,
                lora,
                sample(groups, rng, base.GLOBAL_BATCH),
                args.microbatch,
                step,
            )
            torch.cuda.synchronize()
            row.update(
                update_seconds=perf_counter() - tic,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
            if step in steps:
                val = validate(model, anchors, args.microbatch, mean_plate)
                row["validation"] = val
                file = f"candidates/step-{step:06d}.pt"
                common.save_pt(
                    out / file,
                    {
                        "schema": base.SCHEMA,
                        "contract": contract,
                        "fit_id": fit_id,
                        "step": step,
                        "best": {
                            "step": step,
                            "mae": val["mae"],
                            "model": model.delta(),
                        },
                    },
                )
                snapshots.append(
                    {
                        "step": step,
                        "file": file,
                        "sha256": sha256(out / file),
                        "validation": val,
                    }
                )
            curve.append(row)
            if (
                step == 1
                or step % base.CHECKPOINT_EVERY == 0
                or step in steps
                or step == args.steps
            ):
                common.save_pt(
                    restart,
                    {
                        "fit_id": fit_id,
                        "step": step,
                        "model": model.delta(),
                        "optimizer": optimizer.state_dict(),
                        "rng": base.rng_state(rng),
                        "curve": curve,
                        "snapshots": snapshots,
                    },
                )
                base.atomic_json(
                    out / "curve.json", {"updates": curve, "snapshots": snapshots}
                )
                print(json.dumps(row, allow_nan=False), flush=True)
        if [item["step"] for item in snapshots] != steps:
            raise ValueError("Incomplete registered candidate schedule")
        shortlist = sorted(
            snapshots, key=lambda s: (s["validation"]["mae"], s["step"])
        )[: args.shortlist]
        base.atomic_json(
            out / "candidates.json",
            {
                "status": "complete",
                "fit_id": fit_id,
                "snapshots": snapshots,
                "shortlist": [s["step"] for s in shortlist],
            },
        )


class FrozenWaypoint(panel.FrozenMotor):
    """Share the source gate's exact candidate-delta and update verification."""

    def __init__(self, weights, contract, fit_id):
        if contract["architecture"] != ARCH:
            raise ValueError("Expected a registered waypoint checkpoint")
        self.model = WaypointMotor(weights)
        if self.model.identity != contract["weights"]:
            raise ValueError("Candidate base snapshot changed")
        self.trained = [p for p in self.model.parameters() if p.requires_grad]
        self.contract, self.fit_id, self.current = contract, fit_id, None


def dev(args):
    from sim import Session

    out = args.output
    contract, fit_id, candidates, prompt = panel.audit_fit(out, sources())
    if contract["architecture"] != ARCH:
        raise ValueError("Expected a registered waypoint fit")
    stage = out / "dev"
    stage.mkdir(exist_ok=True)
    poses, threshold = panel.PANELS["dev"]
    binding = {
        "stage": "dev",
        "fit_id": fit_id,
        "sources": sources(),
        "revision": args.revision,
        "runtime": common.runtime(),
        "poses": list(poses),
        "source_bodies": list(BODIES),
        "required_per_body": threshold,
        "candidates": candidates,
        "prompt": prompt,
        "seed_base": panel.ae.seed(0),
        "horizon": panel.ae.HORIZON,
        "execute": panel.ae.EXECUTE,
        "base_weights": contract["weights"],
    }
    panel.bind(stage / "contract.json", binding)
    path = stage / "report.json"
    old = panel.read(path) if path.exists() else {}
    contract_sha = sha256(stage / "contract.json")
    if old and old["contract_sha256"] != contract_sha:
        raise ValueError(
            "Existing native result belongs to another development contract"
        )
    previous = {r["tape"]: r["tape_sha256"] for r in old.get("episodes", [])}
    report = {
        "stage": "dev",
        "fit_id": fit_id,
        "status": "running",
        "source_gate": False,
        "contract_sha256": contract_sha,
        "poses": list(poses),
        "source_bodies": list(BODIES),
        "required_per_body": threshold,
        "revision": args.revision,
        "scores": [],
        "episodes": [],
    }
    resets = {r["released_row_fingerprint"]: r["pose"] for r in contract["episodes"]}
    openings, panel_rows = {}, {}
    backend = None
    for candidate in candidates:
        dest = stage / f"step-{candidate['step']:06d}"
        dest.mkdir(exist_ok=True)
        counts = {}
        for body in BODIES:
            env = None
            wins = 0
            try:
                for pose in poses:
                    tape = dest / f"query-{body}-{pose}-raw.npz"
                    rel = str(tape.relative_to(stage))
                    if not tape.exists():
                        if rel in previous or old.get("status") == "complete":
                            raise ValueError(
                                f"Missing committed native evidence {tape}"
                            )
                        if backend is None:
                            backend = FrozenWaypoint(args.weights, contract, fit_id)
                        model = backend.select(out, candidate)
                        if env is None:
                            env = Session(body, task_id=panel.ae.TASK)
                        panel.rollout(
                            tape,
                            env,
                            model,
                            body,
                            pose,
                            prompt,
                            candidate["sha256"],
                            fit_id,
                        )
                    r = panel.check_episode(
                        tape,
                        stage,
                        body,
                        pose,
                        prompt,
                        candidate,
                        fit_id,
                        previous.get(rel),
                    )
                    opening = {
                        k: r["opening"][k]
                        for k in ("observation", "physics", "horizon")
                    }
                    if openings.setdefault((body, pose), opening) != opening:
                        raise ValueError(
                            "Candidates did not receive paired native development starts"
                        )
                    fp = opening["physics"]["released_row_fingerprint"]
                    if (
                        resets.setdefault(fp, pose) != pose
                        or panel_rows.setdefault(pose, fp) != fp
                    ):
                        raise ValueError("Overlapping or inconsistent released layout")
                    wins += int(r["success"])
                    report["episodes"].append(r)
                    if rel not in previous:
                        base.atomic_json(path, report)
                    print(
                        candidate["step"],
                        body,
                        pose,
                        r["success"],
                        r["steps"],
                        flush=True,
                    )
            finally:
                if env is not None:
                    env.close()
            counts[body] = wins
        report["scores"].append(
            {
                "step": candidate["step"],
                "sha256": candidate["sha256"],
                "native_successes": counts,
                "offline_mae": candidate["validation"]["mae"],
                "passed": min(counts.values()) >= threshold,
            }
        )
    winner = min(report["scores"], key=panel.rank)
    report.update(
        status="complete",
        source_gate=winner["passed"],
        best_dev_step=winner["step"],
        native_successes=winner["native_successes"],
        selected_step=None,
        motor_checkpoint_sha256=None,
    )
    if winner["passed"]:
        candidate = next(s for s in candidates if s["step"] == winner["step"])
        panel.promote(out, candidate)
        report.update(
            selected_step=candidate["step"], motor_checkpoint_sha256=candidate["sha256"]
        )
    if old.get("status") == "complete" and old != report:
        raise ValueError("Completed native development result changed")
    base.atomic_json(path, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "best_dev_step",
                    "selected_step",
                    "source_gate",
                    "native_successes",
                )
            }
        )
    )
    return 0 if winner["passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fit = stages.add_parser("train")
    development = stages.add_parser("dev")
    for p in (fit, development):
        p.add_argument("--output", type=Path, required=True)
        p.add_argument(
            "--weights",
            type=Path,
            default=os.environ.get("DUAL_WEIGHTS"),
            required=not bool(os.environ.get("DUAL_WEIGHTS")),
        )
        p.add_argument(
            "--revision",
            default=os.environ.get("MOTOR_CODE_REVISION"),
            required=not bool(os.environ.get("MOTOR_CODE_REVISION")),
        )
    fit.add_argument("--demos", type=Path, nargs="+", required=True)
    fit.add_argument("--steps", type=int, default=800)
    fit.add_argument("--candidate-steps", type=int, nargs="+")
    fit.add_argument("--shortlist", type=int, default=3)
    fit.add_argument("--anchors-per-pose", type=int, default=32)
    fit.add_argument("--seed", type=int, default=20261120)
    fit.add_argument("--microbatch", type=int, choices=(1, 2, 4, 8), default=4)
    fit.add_argument("--scratch", type=Path)
    fit.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.stage == "train":
        args.demos = [p.resolve(strict=True) for p in args.demos]
    base.require_gpu()
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    with common.exclusive(args.output):
        return train(args) if args.stage == "train" else dev(args)


if __name__ == "__main__":
    raise SystemExit(main())
