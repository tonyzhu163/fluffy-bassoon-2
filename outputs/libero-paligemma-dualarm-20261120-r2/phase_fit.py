"""Fit, check or roll out a phase-conditioned source skill. Use a fresh output directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "libero-paligemma-crossarm-20261120-r1"
sys.path.insert(1, str(PRIOR))
BODIES = ("Panda", "Sawyer")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("fit", "check", "native"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--weights", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True, help="Fresh directory")
        cmd.add_argument("--bodies", nargs="+", choices=BODIES, required=name == "fit", default=None)
        if name != "native":
            cmd.add_argument("--banks", type=Path, nargs="+", required=True)
            cmd.add_argument("--scratch", type=Path)
            cmd.add_argument("--microbatch", type=int, choices=(1, 2, 4, 8), default=4)
        if name != "fit":
            cmd.add_argument("--checkpoint", type=Path, required=True)
        if name == "fit":
            cmd.add_argument("--route", choices=("primary", "straight"), default="primary")
            cmd.add_argument("--seed", type=int, default=20261120)
            cmd.add_argument("--steps", type=int, default=1600)
            cmd.add_argument("--save-every", type=int, default=200)
            cmd.add_argument("--resume", action="store_true")
        elif name == "native":
            cmd.add_argument("--stride", type=int, choices=(1, 2, 3, 4, 5), default=1)
            cmd.add_argument("--seed", type=int, default=20261120)
    return p


def source_files():
    from motor_weights import sha256
    files = [*(HERE.glob("phase_*.py")), HERE / "recover.py", HERE / "generated_split.py",
             *(PRIOR / n for n in ("motor.py", "motor_data.py", "motor_weights.py"))]
    return {f.name: sha256(f) for f in files}


def load_episodes(args, split, route, scratch):
    from phase_data import iter_episodes
    return [ep for ep in iter_episodes(args.banks, split=split, route_scope=route, scratch=scratch)
            if args.bodies is None or ep.body in args.bodies]


def prepare(episodes):
    import numpy as np
    from collections import defaultdict
    pool = defaultdict(lambda: defaultdict(list))
    for ep in episodes:
        # Known pre-action phase interval bounds; sparse traces certify
        # monotone intervals when the adjacent endpoints have the same phase.
        ep.stage = ep.phase.copy()
        trusted = np.flatnonzero(ep.phase >= 0)
        for a, b in zip(trusted[:-1], trusted[1:]):
            if ep.phase[a] == ep.phase[b]:
                ep.stage[a:b] = ep.phase[a]
        ep.boundary = np.flatnonzero(ep.switch == 1)
        ep.landmarks = np.flatnonzero(np.isfinite(ep.error).all(1) & (ep.stage >= 0))
        pool[ep.body][ep.pose].append(ep)
    return pool


def ledger(episodes):
    import numpy as np
    return [dict(body=e.body, pose=e.pose, seed=e.seed, route=e.route, steps=len(e.actions),
                 trajectory=e.trajectory_sha256, labels=label_digest(e),
                 physical_labels=len(e.landmarks), known_phase=int((e.stage >= 0).sum()),
                 switches=len(e.boundary), orientation=bool(np.isfinite(e.hand_target).all()))
            for e in episodes]


def label_digest(ep):
    import numpy as np
    h = hashlib.sha256()
    for a in (ep.phase, ep.switch, ep.error, ep.plate, ep.hand_target):
        h.update(np.ascontiguousarray(a))
    return h.hexdigest()


def sample(pool, rng):
    import numpy as np
    body = rng.choice(sorted(pool))
    pose = rng.choice(sorted(pool[body]))
    tapes = pool[body][pose]
    ep = tapes[rng.integers(len(tapes))]
    mode = rng.random()
    if mode < .20 and len(ep.boundary):
        t = np.clip(rng.choice(ep.boundary) + rng.integers(-4, 5), 0, len(ep.actions)-1)
    elif mode < .60 and len(ep.landmarks):
        labels = ep.stage[ep.landmarks]
        stage = rng.choice(np.unique(labels))
        t = rng.choice(ep.landmarks[labels == stage])
    else:
        t = rng.integers(len(ep.actions))
    return ep, int(t)


def initialize(model, episodes):
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    states = np.concatenate([ep.state for ep in episodes])
    q = states[:, 3:7]
    q *= np.where(np.take_along_axis(q, abs(q).argmax(1)[:, None], axis=1) < 0, -1, 1)
    plates = np.concatenate([e.plate[np.isfinite(e.plate).all(1)] for e in episodes])
    hands = [e.hand_target for e in episodes if np.isfinite(e.hand_target).all()]
    if not len(plates) or not hands:
        raise ValueError("Positive tapes need aligned plate and orientation examples")
    head = model.head
    with torch.no_grad():
        model.state_mean.copy_(torch.tensor(states.mean(0)))
        model.state_scale.copy_(torch.tensor(states.std(0).clip(.01)))
        head.plate_origin.copy_(torch.tensor(np.median(plates, axis=0)))
        head.hand.copy_(torch.tensor(Rotation.from_quat(hands).mean().as_quat()))
        jaw = np.concatenate([e.actions[:, 6] for e in episodes]).mean()
        head.local.bias.view(3, 11)[:, 10].fill_(float(jaw))
        for phase in range(3):
            offsets = []
            for e in episodes:
                t = e.landmarks[e.stage[e.landmarks] == phase]
                if len(t):
                    targets = (e.state[t, :3] + e.error[t]).copy()
                    targets[:, :2] -= e.plate[t]
                    offsets.append(targets)
            if not offsets:
                # A short descent can fall entirely between legacy trace samples.
                # Seed the motor on an action-equivalent target; geometry losses
                # below still consume only observed teacher errors.
                for e in episodes:
                    ticks = np.flatnonzero(e.stage == phase)
                    measured = np.flatnonzero(np.isfinite(e.plate).all(1))
                    if len(ticks) and len(measured):
                        nearest = measured[abs(ticks[:, None] - measured).argmin(1)]
                        targets = e.state[ticks, :3] + .035 * e.actions[ticks, :3]
                        targets[:, :2] -= e.plate[nearest]
                        offsets.append(targets)
            if not offsets:
                raise ValueError(f"No ordered examples for phase {phase}; capture dense positives")
            head.motif[phase].copy_(torch.tensor(np.median(np.concatenate(offsets), axis=0)))
        head.log_angular_gain.copy_(torch.tensor([[np.log(1/.55)], [np.log(1/.55)], [0.]]))


def batch_loss(model, batch, rng):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from phase_policy import SLOTS, select
    ep, t = zip(*batch)
    ticks = np.asarray(t)
    # Keep the public clock-free inference path trained.
    clock = np.where(rng.random(len(t)) < .15, -1, ticks)
    pred = model.predict([e.observation(i) for e, i in batch], clock)
    device = pred["logits"].device

    def tensor(values):
        return torch.as_tensor(np.stack(values), device=device)

    idx = ticks[:, None] + np.arange(SLOTS)
    valid = idx < np.asarray([len(e.actions) for e in ep])[:, None]
    idx = np.minimum(idx, np.asarray([len(e.actions)-1 for e in ep])[:, None])
    valid = tensor(valid)
    stage = tensor([e.stage[ix] for e, ix in zip(ep, idx)]).long()
    labels = tensor([e.actions[ix] for e, ix in zip(ep, idx)])
    known = (stage >= 0) & valid
    logits = pred["logits"]
    chosen = torch.where(known, stage, logits.detach().argmax(-1))

    def mean_masked(values, mask):
        # Equal weight per anchor, invariant to how a global batch is sliced.
        per_anchor = (values * mask).reshape(len(batch), -1).sum(1)
        return (per_anchor / mask.reshape(len(batch), -1).sum(1).clamp_min(1)).mean()

    def huber(prediction, truth, scale):
        return F.smooth_l1_loss(prediction/scale, truth.nan_to_num()/scale,
                                beta=.12, reduction="none").mean(-1)

    actions = select(pred["actions"], chosen)
    switch = tensor([e.switch[ix] for e, ix in zip(ep, idx)])
    # At a switch Paddle mixes old-phase geometry and new-phase gains/bias.
    # Retain the geometry and stage label; omit this hybrid imitation target.
    action_loss = mean_masked(huber(actions, labels, 1.), valid & (switch != 1))
    stage_loss = mean_masked(F.cross_entropy(logits.transpose(1, 2), stage.clamp_min(0),
                                             reduction="none"), known)
    plate = tensor([e.plate[i] for e, i in batch]).float()
    plate_loss = mean_masked(huber(pred["plate"], plate, .05), plate.isfinite().all(-1))
    error = tensor([e.error[ix] for e, ix in zip(ep, idx)]).float()
    error_loss = mean_masked(huber(select(pred["error"], chosen), error, .035),
                             error.isfinite().all(-1) & known)
    hand = tensor([e.hand_target for e in ep])
    decoded = select(pred["hand"], chosen)[:, 0]
    align = (decoded * hand.nan_to_num()).sum(-1).abs().clamp(max=1)
    hand_loss = mean_masked(1 - align, hand.isfinite().all(-1))
    return torch.stack((action_loss, stage_loss * .5, plate_loss, error_loss, hand_loss * 4))


def train(args):
    import fcntl
    import numpy as np
    import random
    import subprocess
    import torch
    from phase_policy import ARCH, PhaseMotor, base
    from motor_weights import sha256
    if min(args.steps, args.save_every) < 1 or not 0 <= args.seed < 2**32:
        raise ValueError("Require positive updates, save interval and a uint32 seed")
    out = args.output
    out.mkdir(parents=True, exist_ok=args.resume)
    scratch = args.scratch or out
    scratch.mkdir(parents=True, exist_ok=True)
    with (out / "fit.lock").open("a") as lock, TemporaryDirectory(dir=scratch) as tmp:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved_path = out / "checkpoint.pt"
        if args.resume != saved_path.is_file():
            raise ValueError("Resume needs an existing checkpoint")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        rng = np.random.default_rng(args.seed)
        episodes = load_episodes(args, "fit", args.route, tmp)
        if not episodes or (args.bodies and set(args.bodies) != {e.body for e in episodes}):
            raise ValueError("Requested source arm has no fitting positives for this fixed route")
        groups = prepare(episodes)
        model = PhaseMotor(args.weights)
        initialize(model, episodes)
        indices = [sha256(b / "index.json") for b in args.banks]
        contract = dict(architecture=ARCH, route=args.route, train=ledger(episodes), indices=indices,
                        sources=source_files(), weights=model.identity, seed=args.seed,
                        revision=os.environ.get("MOTOR_CODE_REVISION") or subprocess.check_output(
                            ["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip(), microbatch=args.microbatch,
                        global_batch=base.GLOBAL_BATCH, bodies=sorted(groups),
                        lora_lr=base.LORA_LR, head_lr=base.HEAD_LR)
        adapter = [p for p in model.vlm.parameters() if p.requires_grad]
        head = list(model.head.parameters())
        opt = torch.optim.AdamW([dict(params=adapter, lr=base.LORA_LR),
                                 dict(params=head, lr=base.HEAD_LR)], weight_decay=.01)
        start = 0
        if args.resume:
            saved = torch.load(saved_path, weights_only=True, map_location="cpu")
            if saved["contract"] != contract or saved["step"] > args.steps:
                raise ValueError("Resume contract or horizon differs")
            model.restore_delta(saved["model"])
            opt.load_state_dict(saved["optimizer"])
            base.restore_rng(saved["rng"], rng)
            start = saved["step"]
            del saved
        else:
            base.atomic_json(out / "manifest.json", dict(contract=contract, argv=sys.argv,
                             gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                             parameter_counts=dict(lora=sum(p.numel() for p in adapter),
                                                   head=sum(p.numel() for p in head))))
        print(json.dumps(dict(route=args.route, fit_positives={b:sum(len(t) for t in poses.values())
                         for b, poses in groups.items()}, layouts={b:sorted(poses) for b, poses in groups.items()})),
              flush=True)
        torch.cuda.reset_peak_memory_stats()
        for step in range(start+1, args.steps+1):
            begin = perf_counter()
            batch = [sample(groups, rng) for _ in range(base.GLOBAL_BATCH)]
            opt.zero_grad(set_to_none=True)
            terms = torch.zeros(5, device="cuda")
            for i in range(0, len(batch), args.microbatch):
                loss = batch_loss(model, batch[i:i+args.microbatch], rng) * args.microbatch / len(batch)
                loss.sum().backward()
                terms += loss.detach()
            row = dict(step=step)
            if step == 1:
                for name, parameters in (("lora", adapter), ("decoder", model.head.local.parameters())):
                    grad = torch.stack([p.grad.norm() for p in parameters if p.grad is not None]).norm().item()
                    if not np.isfinite(grad) or grad <= 0:
                        raise RuntimeError(f"No finite nonzero supervised {name} gradient")
                    row[name + "_grad_norm"] = grad
            torch.nn.utils.clip_grad_norm_(adapter + head, 1., error_if_nonfinite=True)
            opt.step()
            if step == 1:
                # Both sets of weights start exactly at zero. This readback
                # distinguishes a gradient from an applied optimizer update.
                up = [p.detach().norm() for n, p in model.vlm.named_parameters() if n.endswith(".up.weight")]
                row["lora_updated_norm"] = torch.stack(up).norm().item()
                row["decoder_updated_norm"] = model.head.local.weight.detach().norm().item()
                if not all(np.isfinite(row[k]) and row[k] > 0
                           for k in ("lora_updated_norm", "decoder_updated_norm")):
                    raise RuntimeError("First update did not change both Gemma LoRA and the action readout")
            torch.cuda.synchronize()
            if step == 1 or step % 25 == 0 or step == args.steps:
                row.update(weighted_loss=dict(zip(("action", "phase", "plate", "error", "hand"), terms.tolist())),
                           seconds=perf_counter()-begin, peak_bytes=torch.cuda.max_memory_allocated())
                print(json.dumps(row, allow_nan=False), flush=True)
                with (out / "fit.jsonl").open("a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
            if step == 1 or step % args.save_every == 0 or step == args.steps:
                temporary = saved_path.with_suffix(".tmp")
                torch.save(dict(contract=contract, step=step, model=model.delta(),
                                optimizer=opt.state_dict(), rng=base.rng_state(rng)), temporary)
                temporary.replace(saved_path)


if __name__ == "__main__":
    args = parser().parse_args()
    if args.command == "fit":
        train(args)
    else:
        from phase_check import evaluate
        evaluate(args)
