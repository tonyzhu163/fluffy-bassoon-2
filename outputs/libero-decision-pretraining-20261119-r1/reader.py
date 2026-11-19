"""Train and evaluate the categorical reader without simulator or VLA training."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fluffy_bassoon.libero import estimate_command_lag
from fluffy_bassoon.meta_rl.causal_decoder import CausalDecoderCore

PREFIX = 80
CONTROLS_PER_TOKEN = 5
TRAJECTORY_CHANNELS = 14
SEEDS = (41, 42, 43)
UPDATES = 120
LEARNING_RATE = 3e-4


@dataclass
class Panel:
    keys: list
    timing: np.ndarray
    history: torch.Tensor
    lengths: torch.Tensor
    query: torch.Tensor
    outcomes: torch.Tensor
    delayed: np.ndarray


def read_trials(directory, modes, expected_tasks):
    rows = json.loads((directory / "progress.json").read_text())
    complete = json.loads((directory / "complete.json").read_text())
    assert complete == dict(trials=len(rows), episodes=len(rows) * (1 + len(modes)))
    expected = {(suite, task, start, label) for suite, tasks in expected_tasks.items()
                for task in tasks for start in (0, 3) for label in ("BASE", "S20")}
    keys, timing, histories, lengths, queries, outcomes, delayed = [], [], [], [], [], [], []
    seen = set()
    for row in rows:
        folder = directory / row["trial"]
        observed = row["support"]["case"]
        suite, task, start, label = (observed[k] for k in ("suite", "task_id", "init_state_id", "variant"))
        key = suite, task, start, label
        assert key in expected and key not in seen
        seen.add(key)
        branches = row["queries"]
        assert set(branches) == set(modes)
        first = branches[modes[0]]
        assert first["case"] == dict(observed, init_state_id=start + 1, seed=observed["seed"] + 1)
        assert all((r["case"], r["identity"], r["opening_sha256"]) ==
                   (first["case"], first["identity"], first["opening_sha256"]) for r in branches.values())
        for mode, result in [("support", row["support"]), *branches.items()]:
            assert json.loads((folder / mode / "result.json").read_text()) == result
            with np.load(folder / mode / "public.npz") as tape:
                count = result["controls"]
                assert 0 < count <= result["identity"]["horizon"]
                assert tape["issued"].shape == (count, 7)
                assert tape["positions"].shape == (count, 3) and tape["gripper"].shape == (count, 2)
                assert len(tape["rewards"]) == len(tape["waits"]) == count
                assert bool(np.any(tape["rewards"] > 0)) == result["success"]
                assert int(tape["waits"].sum()) == result["waits"]
                assert not np.any(tape["issued"][tape["waits"], :6])
                assert all(np.isfinite(tape[k]).all() for k in tape.files)
                if mode == "support":
                    issued, positions = tape["issued"][:PREFIX], tape["positions"][:PREFIX]
                    motion = np.diff(np.concatenate((tape["initial_position"][None], positions)), axis=0)
                    packed = np.column_stack((issued, motion, tape["gripper"][:PREFIX],
                                              tape["rewards"][:PREFIX], tape["waits"][:PREFIX]))
                    length = len(packed)
                    history = np.zeros((PREFIX, TRAJECTORY_CHANNELS), dtype=np.float32)
                    history[:length] = packed
                    lag = estimate_command_lag(issued, positions, positions_timing="after",
                                               max_lag=21, prefix=PREFIX) if length == PREFIX else None
        feature = np.load(folder / "query.npy", allow_pickle=False)
        assert feature.ndim == 1 and len(feature) > 8 and np.isfinite(feature).all()
        keys.append(key)
        timing.append(label)
        histories.append(history)
        lengths.append(length)
        queries.append(feature)
        outcomes.append([branches[mode]["success"] for mode in modes])
        # Reproduce the reference decision boundary under the post-step convention.
        delayed.append(lag is None or lag >= 8)
    assert seen == expected
    return Panel(keys, np.array(timing), torch.from_numpy(np.stack(histories)),
                 torch.tensor(lengths), torch.from_numpy(np.stack(queries)), torch.tensor(outcomes),
                 np.array(delayed))


class Reader(nn.Module):
    def __init__(self, fitting, mode_count):
        super().__init__()
        width = 128
        self.support_projection = nn.Linear(CONTROLS_PER_TOKEN * (TRAJECTORY_CHANNELS + 1), width)
        self.boundary = nn.Parameter(torch.zeros(width))
        self.query_projection = nn.Linear(fitting.query.shape[-1], width)
        self.core = CausalDecoderCore(width, width=width, layers=2, heads=4)
        self.head = nn.Linear(width, mode_count)
        supports = fitting.history[torch.arange(PREFIX)[None] < fitting.lengths[:, None]]
        self.set_stats("support", supports)
        self.set_stats("state", fitting.query[:, :8])

    def set_stats(self, prefix, x):
        scale = x.std(dim=0, unbiased=False)
        self.register_buffer(f"{prefix}_mean", x.mean(dim=0))
        self.register_buffer(f"{prefix}_scale", torch.where(scale > 1e-6, scale, 1.0))

    def forward(self, history, lengths, query, *, retain):
        example = torch.arange(len(query))
        control_valid = torch.arange(PREFIX)[None] < lengths[:, None]
        history = (history - self.support_mean) / self.support_scale
        history = torch.where(control_valid[..., None], history, 0.0)
        inputs = torch.cat((history, control_valid[..., None]), dim=-1)
        blocks = self.support_projection(inputs.reshape(len(example), PREFIX // CONTROLS_PER_TOKEN, -1))
        blocks = torch.cat((blocks[:, :1] + self.boundary, blocks[:, 1:]), dim=1)
        positions = (lengths + CONTROLS_PER_TOKEN - 1) // CONTROLS_PER_TOKEN if retain else torch.zeros_like(lengths)
        state = (query[:, :8] - self.state_mean) / self.state_scale
        features = nn.functional.layer_norm(query[:, 8:], (query.shape[1] - 8,))
        query_embedding = self.query_projection(torch.cat((state, features), dim=-1))
        tokens = torch.cat((blocks, torch.zeros_like(query_embedding[:, None])), dim=1)
        tokens[example, positions] = query_embedding
        valid = torch.arange(tokens.shape[1])[:, None] <= positions[None]
        output = self.core.sequence(tokens.transpose(0, 1), valid=valid, reset=torch.zeros_like(valid))
        return self.head(output[positions, example])


def train(panel, mode_count, *, seed, current_only, checkpoint):
    torch.manual_seed(seed)
    model = Reader(panel, mode_count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    eligible = panel.outcomes.any(-1)
    if not eligible.any():
        raise ValueError("Every fitting query failed every candidate")
    h, lengths, q = (x[eligible] for x in (panel.history, panel.lengths, panel.query))
    targets = panel.outcomes[eligible].float()
    targets /= targets.sum(-1, keepdim=True)
    losses = []
    for _ in range(UPDATES):
        optimizer.zero_grad(set_to_none=True)
        cleared = nn.functional.cross_entropy(model(h, lengths, q, retain=False), targets)
        if current_only:
            loss = cleared
        else:
            full = nn.functional.cross_entropy(model(h, lengths, q, retain=True), targets)
            loss = 0.5 * (cleared + full)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), updates=UPDATES,
                    seed=seed, torch_rng=torch.random.get_rng_state(), current_only=current_only), checkpoint)
    return model.eval(), [losses[0], losses[-1]]


def timing_indices(panel, fitting):
    fits = fitting.outcomes.numpy()
    fixed = int(fits.sum(axis=0).argmax())
    by_timing = np.array([fits[fitting.delayed == label].sum(axis=0).argmax()
                          if np.any(fitting.delayed == label) else fixed for label in (False, True)])
    return dict(fitting_fixed=np.full(len(panel.keys), fixed),
                lag=by_timing[panel.delayed.astype(int)],
                legacy_lag=panel.delayed.astype(int))


def readout(panel, mode_names, predictions):
    successes = panel.outcomes.numpy()
    chosen = {k: successes[np.arange(len(panel.keys)), v] for k, v in predictions.items()}
    chosen.update({mode: successes[:, idx] for idx, mode in enumerate(mode_names)})
    chosen["oracle"] = successes.any(axis=1)
    choices = {k: [mode_names[i] for i in v] for k, v in predictions.items()}
    timing = {t: {k: int(v[panel.timing == t].sum()) for k, v in chosen.items()} for t in ("BASE", "S20")}
    task = {}
    for suite, task_id in sorted({key[:2] for key in panel.keys}):
        selected = np.array([key[:2] == (suite, task_id) for key in panel.keys])
        task[f"{suite}/{task_id}"] = {k: int(v[selected].sum()) for k, v in chosen.items()}
    paired = {}
    if "retained" in chosen:
        for k, v in chosen.items():
            paired[k] = dict(wins=int((chosen["retained"] & ~v).sum()),
                             losses=int((~chosen["retained"] & v).sum()))
    detail = [dict(case=key, success={k: bool(v[i]) for k, v in chosen.items()},
                   choices={k: v[i] for k, v in choices.items()}) for i, key in enumerate(panel.keys)]
    return dict(trials=len(panel.keys), overall={k: int(v.sum()) for k, v in chosen.items()},
                timing=timing, task=task, paired_retained=paired, detail=detail)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episodes", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir()
    design = json.loads((args.episodes / "design.json").read_text())
    modes = design["modes"]
    splits_path = Path(__file__).resolve().parents[1] / "libero-meta-learning-20261117-r1" / "SPLITS.json"
    splits = json.loads(splits_path.read_text())
    assert set(splits["development"]) == set(splits["train"])
    fitting_tasks = splits["development_scaling"]["task_subsets"]["8"]
    for suite, ids in fitting_tasks.items():
        assert set(ids) <= set(splits["train"][suite])
        assert not set(ids) & set(splits["development"][suite] + splits["test"][suite])
    fitting = read_trials(args.episodes / "fitting", modes, fitting_tasks)
    fitted = []
    for seed in SEEDS:
        models = [train(fitting, len(modes), seed=seed, current_only=is_current,
                        checkpoint=args.output / f"seed-{seed}-{name}.pt")
                  for name, is_current in (("history", False), ("current", True))]
        fitted.append(models)
        print(f"Fitted both heads for seed {seed}", flush=True)
    dev = read_trials(args.episodes / "development", modes, splits["development"])
    index = {key: idx for idx, key in enumerate(dev.keys)}
    other = [index[(*key[:3], "S20" if key[-1] == "BASE" else "BASE")] for key in dev.keys]
    baselines = timing_indices(dev, fitting)
    results = {}
    for seed, ((model, loss), (current, current_loss)) in zip(SEEDS, fitted):
        with torch.no_grad():
            choices = dict(
                retained=model(dev.history, dev.lengths, dev.query, retain=True).argmax(-1).numpy(),
                cleared=model(dev.history, dev.lengths, dev.query, retain=False).argmax(-1).numpy(),
                swapped=model(dev.history[other], dev.lengths[other], dev.query, retain=True).argmax(-1).numpy(),
                current_only=current(dev.history, dev.lengths, dev.query, retain=False).argmax(-1).numpy(),
            )
        results[str(seed)] = dict(loss=loss, current_loss=current_loss,
                                  **readout(dev, modes, {**baselines, **choices}))
    report = dict(modes=modes, updates=UPDATES, seed_results=results,
                  fitting=readout(fitting, modes, timing_indices(fitting, fitting)),
                  dev_without_model=readout(dev, modes, baselines))
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
