"""Fit a history Transformer on hidden-rule sessions and score it against Bayes.

One fit per context condition. Training attempts come from a competent but
uninformed actor (`sessions.behave`); evaluation uses the learner's own
attempts, with every weight fixed.

    python train.py --cond attempts --out <dir>/attempts-s0.json

`--dense` supervises every scene of a session instead of one query, `--fourier`
adds sinusoidal attribute features, and `--d`/`--layers`/`--heads` set capacity.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import sessions as S


class Learner(nn.Module):
    def __init__(self, T: int, d: int = 128, layers: int = 4, heads: int = 4, fourier: int = 0):
        super().__init__()
        self.fourier = fourier
        self.inp = nn.Sequential(nn.Linear(S.F + 2 * fourier * S.A, d), nn.GELU(), nn.Linear(d, d))
        self.event = nn.Embedding(T + 1, d)  # scene index; the last slot marks the query
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, 1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.fourier:
            freq = 2 * math.pi * 2 ** torch.arange(self.fourier, device=x.device)
            a = x[..., : S.A, None] * freq
            x = torch.cat([x, a.sin().flatten(-2), a.cos().flatten(-2)], -1)
        return self.inp(x)

    def forward(self, x: torch.Tensor, pad: torch.Tensor, q: int) -> torch.Tensor:
        B, T, N, _ = x.shape
        idx = torch.arange(T, device=x.device)
        idx = torch.where(idx == q, torch.full_like(idx, T), idx)
        h = self.embed(x) + self.event(idx)[None, :, None, :]
        h = self.enc(h.flatten(1, 2), src_key_padding_mask=pad.repeat_interleave(N, 1))
        return self.out(h).view(B, T, N)[:, q]

    def dense(self, e: torch.Tensor, qx: torch.Tensor, q_time: torch.Tensor) -> torch.Tensor:
        """Logits [B, Tq, N] for several query scenes sharing one event history.

        `e` holds event scenes in order; query j sees the first `q_time[j]` of
        them and its own objects. Event t sees events up to t. No token sees a
        query other than its own, so one pass scores every prefix length.
        """
        B, Te, N, _ = e.shape
        Tq = qx.shape[1]
        e_time = torch.arange(Te, device=e.device)
        he = self.embed(e) + self.event(e_time)[None, :, None, :]
        hq = self.embed(qx) + self.event.weight[-1]
        te, tq = e_time.repeat_interleave(N), q_time.repeat_interleave(N)
        sq = torch.arange(Tq, device=e.device).repeat_interleave(N)
        ne = len(te)
        allow = torch.zeros(ne + len(tq), ne + len(tq), dtype=torch.bool, device=e.device)
        allow[:ne, :ne] = te[None, :] <= te[:, None]
        allow[ne:, :ne] = te[None, :] < tq[:, None]
        allow[ne:, ne:] = sq[None, :] == sq[:, None]
        h = self.enc(torch.cat([he.flatten(1, 2), hq.flatten(1, 2)], 1), mask=~allow)
        return self.out(h[:, ne:]).view(B, Tq, N)


def logits(model, device, s, cond, choice, outcome, k, q):
    x, pad = S.encode(s, cond, choice, outcome, k, q)
    return model(torch.from_numpy(x).to(device), torch.from_numpy(pad).to(device), q)


def dense_logits(model, device, s, cond, choice, outcome, k: int, scenes):
    """Logits [B, len(scenes), N]: each scene queried on the events before it, at most k."""
    T = s.true.shape[1]
    x, _ = S.encode(s, cond, choice, outcome, k, T - 1)
    ne = 1 if cond == "demo" else k + 1
    qx = np.zeros((len(s.rule), len(scenes), S.N, S.F), np.float32)
    qx[..., : S.A] = s.scenes[:, scenes]
    qx[..., S.A + 4] = 1
    q_time = torch.tensor([min(t, ne) for t in scenes], device=device)
    return model.dense(torch.from_numpy(x[:, :ne]).to(device), torch.from_numpy(qx).to(device), q_time)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cond", choices=S.CONDS, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dense", action="store_true", help="supervise every scene, not only the query")
    ap.add_argument("--fourier", type=int, default=0, help="sinusoidal octaves per attribute")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--eval-n", type=int, default=8192)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--eval-seed", type=int, default=10000, help="evaluation session RNG base, plus fit seed")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--stop-after", type=int, help="Save and exit at this step without changing the schedule")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if min(args.steps, args.batch, args.eval_n, args.eval_batch, args.save_every) <= 0:
        ap.error("step and batch counts must be positive")
    if min(args.d, args.layers, args.heads) <= 0 or args.d % args.heads or args.fourier < 0:
        ap.error("model dimensions must be positive, d divisible by heads, and fourier nonnegative")
    checkpoint = args.checkpoint or args.out.with_suffix(".pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() or (checkpoint.exists() and not args.resume):
        ap.error("output already exists; use a fresh identity or an explicit resume checkpoint")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    K, q = args.K, args.K + 1
    model = Learner(K + 2, args.d, args.layers, args.heads, args.fourier).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    first, elapsed = 0, 0.0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        # Checkpoints preceding the configurable architecture used these defaults.
        stored_args = dict(dense=False, fourier=0, d=128, layers=4, heads=4)
        stored_args.update(state["args"])
        for name in ("cond", "K", "steps", "batch", "lr", "seed", "dense", "fourier", "d", "layers", "heads"):
            if stored_args[name] != getattr(args, name):
                raise ValueError(f"Resume changes training contract: {name}")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        first, elapsed = state["step"], state["elapsed_seconds"]

    start = time.time()

    def save(step):
        state = dict(args=vars(args), step=step, model=model.state_dict(),
                     optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                     numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else None,
                     cuda_peak_memory_bytes=torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
                     elapsed_seconds=elapsed + time.time() - start)
        tmp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        torch.save(state, tmp)
        tmp.replace(checkpoint)

    for step in range(first + 1, args.steps + 1):
        s = S.sample(rng, args.batch, K, "train")
        choice, outcome = S.behave(rng, s, K)
        if args.dense:
            # Every scene is a query on the events before it: K + 2 labels per session.
            out = dense_logits(model, args.device, s, args.cond, choice, outcome, K, list(range(K + 2)))
            loss = nn.functional.cross_entropy(out.flatten(0, 1), torch.from_numpy(s.true).to(args.device).flatten())
        else:
            # A random prefix per session trains the whole learning curve.
            k = rng.integers(0, K + 1, (args.batch, 1))
            out = logits(model, args.device, s, args.cond, choice, outcome, k, q)
            loss = nn.functional.cross_entropy(out, torch.from_numpy(s.true[:, q]).to(args.device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            print(f"step {step} loss {loss.item():.4f} {elapsed + time.time() - start:.0f}s", flush=True)
        if step % args.save_every == 0 or step == args.steps or step == args.stop_after:
            save(step)
        if step == args.stop_after:
            return

    model.eval()
    result = {"args": {k: str(v) for k, v in vars(args).items()}, "rules": {"train": len(S.TRAIN), "heldout": len(S.HELDOUT)}}

    def learner_act(s):
        @torch.no_grad()
        def act(c, o, k, t):
            choices = []
            for lo in range(0, len(s.rule), args.eval_batch):
                sl = slice(lo, lo + args.eval_batch)
                chunk = S.Sessions(s.rule[sl], s.scenes[sl], s.tgt[sl], s.true[sl])
                if args.dense:
                    out = dense_logits(model, args.device, chunk, args.cond, c[sl], o[sl], k, [t])[:, 0]
                else:
                    out = logits(model, args.device, chunk, args.cond, c[sl], o[sl], k, t)
                choices.append(out.argmax(1).cpu().numpy())
            return np.concatenate(choices)

        return act

    for split in ("train", "heldout"):
        s = S.sample(np.random.default_rng(args.eval_seed + args.seed), args.eval_n, K, split)
        acc, bayes = S.rollout(s, K, args.cond, learner_act(s))
        oracle, _ = S.rollout(s, K, args.cond, S.oracle_act(s, args.cond))
        result[split] = {"learner": acc, "bayes_on_learner_history": bayes, "oracle_own_attempts": oracle}
        print(split, json.dumps(result[split]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
