"""Fit a history Transformer on hidden-rule sessions and score it against Bayes.

One fit per context condition. Training attempts come from a competent but
uninformed actor (`sessions.behave`); evaluation uses the learner's own
attempts, with every weight fixed.

    python train.py --cond attempts --out <dir>/attempts-s0.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import sessions as S


class Learner(nn.Module):
    def __init__(self, T: int, d: int = 128, layers: int = 4, heads: int = 4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(S.F, d), nn.GELU(), nn.Linear(d, d))
        self.event = nn.Embedding(T + 1, d)  # scene index; the last slot marks the query
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor, pad: torch.Tensor, q: int) -> torch.Tensor:
        B, T, N, _ = x.shape
        idx = torch.arange(T, device=x.device)
        idx = torch.where(idx == q, torch.full_like(idx, T), idx)
        h = self.inp(x) + self.event(idx)[None, :, None, :]
        h = self.enc(h.flatten(1, 2), src_key_padding_mask=pad.repeat_interleave(N, 1))
        return self.out(h).view(B, T, N)[:, q]


def logits(model, device, s, cond, choice, outcome, k, q):
    x, pad = S.encode(s, cond, choice, outcome, k, q)
    return model(torch.from_numpy(x).to(device), torch.from_numpy(pad).to(device), q)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cond", choices=S.CONDS, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-n", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    K, q = args.K, args.K + 1
    model = Learner(K + 2).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)

    start = time.time()
    for step in range(1, args.steps + 1):
        s = S.sample(rng, args.batch, K, "train")
        choice, outcome = S.behave(rng, s, K)
        # A random prefix per session trains the whole learning curve.
        k = rng.integers(0, K + 1, (args.batch, 1))
        out = logits(model, args.device, s, args.cond, choice, outcome, k, q)
        loss = nn.functional.cross_entropy(out, torch.from_numpy(s.true[:, q]).to(args.device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            print(f"step {step} loss {loss.item():.4f} {time.time() - start:.0f}s", flush=True)

    model.eval()
    result = {"args": {k: str(v) for k, v in vars(args).items()}, "rules": {"train": len(S.TRAIN), "heldout": len(S.HELDOUT)}}

    def learner_act(s):
        @torch.no_grad()
        def act(c, o, k, t):
            return logits(model, args.device, s, args.cond, c, o, k, t).argmax(1).cpu().numpy()

        return act

    for split in ("train", "heldout"):
        s = S.sample(np.random.default_rng(10_000 + args.seed), args.eval_n, K, split)
        acc, bayes = S.rollout(s, K, args.cond, learner_act(s))
        oracle, _ = S.rollout(s, K, args.cond, S.oracle_act(s, args.cond))
        result[split] = {"learner": acc, "bayes_on_learner_history": bayes, "oracle_own_attempts": oracle}
        print(split, json.dumps(result[split]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
