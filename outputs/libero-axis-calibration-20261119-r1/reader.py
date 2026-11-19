"""Predict the corrected first-command XY signs from two probes and a nominal query.

predictions.jsonl records development targets, predicted signs and learned logits
by source trial, convention and support condition. summary.json reports fractions
correct on both axes (joint) and on each axis (x, y). Each source trial has equal
weight. Its four convention relabels are paired observations of that trial.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from fluffy_bassoon.meta_rl.causal_decoder import CausalDecoderCore
from panel import load_panel


NAMES = ("nom", "x", "y", "xy")
SIGNS = torch.tensor([[1, 1], [-1, 1], [1, -1], [-1, -1]])
SEEDS = (41, 42, 43)
STEPS, SOURCE_BATCH, EMPTY_SOURCES = 400, 40, 4
LR = 0.001
CORE = dict(input_dim=48, width=48, layers=2, heads=4, memory_length=2)


def read(episodes, split):
    panel = load_panel(episodes, split)
    keys = list(panel["keys"])
    support, query = (torch.as_tensor(panel[k], dtype=torch.float32, device="cpu")
                      for k in ("support", "query"))
    if not keys or not all(isinstance(k, str) for k in keys) or len(set(keys)) != len(keys):
        raise ValueError(f"{split} needs a unique string key per source trial")
    if support.shape != (len(keys), 2, 4) or query.shape != (len(keys), 2):
        raise ValueError(f"Invalid {split} support or query shape")
    if not support.isfinite().all() or not query.isfinite().all() or (query == 0).any():
        raise ValueError(f"{split} needs finite inputs and nonzero commands on each query axis")
    return keys, support, query, panel["audit"]


def relabel(support, conventions):
    # Axes are [convention, trial, probe, feature]. Physical motion is archived.
    issued = support[None, ..., :2] * conventions[:, None, None, :]
    motion = support[None, ..., 2:].expand_as(issued)
    return torch.cat((issued, motion), dim=-1)


class Reader(nn.Module):
    def __init__(self, fit_support, fit_query):
        super().__init__()
        for name, values in (("support", fit_support.reshape(-1, 4)), ("query", fit_query)):
            scale = values.std(dim=0, unbiased=False)
            self.register_buffer(name + "_mean", values.mean(dim=0))
            self.register_buffer(name + "_scale", torch.where(scale > 1e-8, scale, 1.0))
        width = CORE["input_dim"]
        self.support_token = nn.Linear(4, width)
        self.query_token = nn.Linear(2, width)
        self.core = CausalDecoderCore(**CORE)
        self.head = nn.Linear(width, 2)

    def forward(self, support, query, cleared):
        history = self.support_token((support - self.support_mean) / self.support_scale)
        current = self.query_token((query - self.query_mean) / self.query_scale)
        tokens = torch.cat((history, current[:, None]), dim=1).transpose(0, 1)
        valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        reset = torch.zeros_like(valid)
        reset[-1] = cleared  # The query starts a new attention segment in cleared lanes.
        return self.head(self.core.sequence(tokens, reset=reset, valid=valid)[-1])


def reflect(probes, signs):
    """Change directions of probe motion and command together, preserving calibration."""
    return probes * signs.repeat(1, 2)[:, None, :]


def fit(fitting, seed, output, balanced):
    keys, raw, query, _ = fitting
    torch.manual_seed(seed)
    support = relabel(raw, SIGNS[:3])
    targets = (query[None] * SIGNS[:3, None] > 0).float()
    fitting_inputs = support
    if balanced:
        fitting_inputs = torch.cat([support * signs.repeat(2) for signs in SIGNS])
    model = Reader(fitting_inputs, query)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # Every sampled source appears under all three fitting conventions, including
    # the randomly drawn sources assigned to empty support on this update.
    cleared = (torch.arange(SOURCE_BATCH) < EMPTY_SOURCES).repeat(3)
    for step in range(1, STEPS + 1):
        ids = torch.randint(len(keys), (SOURCE_BATCH,))
        probes = support[:, ids].flatten(0, 1)
        if balanced:
            signs = SIGNS[(torch.arange(SOURCE_BATCH) + step) % len(SIGNS)].repeat(3, 1)
            probes = reflect(probes, signs)
        logits = model(probes, query[ids].repeat(3, 1), cleared)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets[:, ids].flatten(0, 1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), step=step, seed=seed, core=CORE,
                    source_batch=SOURCE_BATCH, empty_sources=EMPTY_SOURCES, lr=LR,
                    fitting_keys=keys, fitting_conventions=NAMES[:3], balanced=balanced, final_loss=loss.item()),
               output / f"seed-{seed}.pt")
    return model.eval()


def accuracy(correct):
    x, y = correct.reshape(-1, 2).float().mean(dim=0).tolist()
    return dict(joint=correct.all(dim=-1).float().mean().item(), x=x, y=y)


def record(stream, keys, method, condition, direction, target, prediction, logits=None):
    correct = prediction == target
    for i, name in enumerate(NAMES):
        for j, key in enumerate(keys):
            row = dict(key=key, convention=name, support=condition, direction=direction, method=method,
                       target=target[i, j].tolist(), prediction=prediction[i, j].tolist())
            if logits is not None:
                row["logits"] = logits[i, j].tolist()
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return dict(overall=accuracy(correct),
                by_convention={name: accuracy(correct[i]) for i, name in enumerate(NAMES)},
                by_source_trial={key: accuracy(correct[:, j]) for j, key in enumerate(keys)})


@torch.inference_mode()
def evaluate(development, models, stream, direction):
    keys, raw, query, _ = development
    raw = raw * SIGNS[NAMES.index(direction)].repeat(2)
    nominal = query.sign().long()
    target = nominal[None] * SIGNS[:, None]
    # Closest available candidate given the true target. Ties use NAMES order.
    # With both axes flipped, no fitting convention can get the pair right.
    candidates = target[:3]
    nearest = (target[:, None] == candidates[None]).sum(-1).argmax(dim=1)
    ceiling = candidates[nearest, torch.arange(len(keys))]
    matched = relabel(raw, SIGNS)
    report = {}
    for condition, probes in (("matched", matched), ("cleared", torch.zeros_like(matched)),
                              ("opposite", relabel(raw, -SIGNS))):
        def save(method, prediction, logits=None):
            report.setdefault(method, {})[condition] = record(
                stream, keys, method, condition, direction, target, prediction, logits)

        correlation = (probes[..., :2] * probes[..., 2:]).sum(dim=2)
        save("sign_correlation", nominal[None] * torch.where(correlation < 0, -1, 1))
        save("true_sign_oracle", target)
        save("best_of_three_ceiling", ceiling)
        for seed, model in models.items():
            batch = probes.flatten(0, 1)
            cleared = torch.full((len(batch),), condition == "cleared", dtype=torch.bool)
            logits = model(batch, query.repeat(len(NAMES), 1), cleared).reshape_as(target)
            save(f"seed-{seed}", torch.where(logits >= 0, 1, -1), logits)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    fitting = read(args.episodes, "fitting")
    model_groups = {}
    for arm in ("archived", "balanced"):
        folder = args.output / arm
        folder.mkdir()
        model_groups[arm] = {}
        for seed in SEEDS:
            model_groups[arm][seed] = fit(fitting, seed, folder, balanced=(arm == "balanced"))
            print(f"Finished {arm}, seed {seed}, {STEPS} updates", flush=True)
    development = read(args.episodes, "development")
    for arm, models in model_groups.items():
        folder = args.output / arm
        with (folder / "predictions.jsonl").open("w") as stream:
            metrics = {direction: evaluate(development, models, stream, direction) for direction in NAMES}
        (folder / "metrics.json").write_text(json.dumps(metrics, allow_nan=False) + "\n")
    report = dict(conventions=dict(zip(NAMES, SIGNS.tolist())), held_out="xy",
                  fitting=fitting[3], development=development[3], seeds=SEEDS, steps=STEPS,
                  correlation_tie="A zero command-displacement dot product uses nominal sign.",
                  ceiling_tie="Maximize correct axes among nom, x, y; ties take the first.")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
