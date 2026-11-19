"""Fit both full-action readers on randomized fitting trajectories using Linux CPU."""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from model import (ActionReader, FORMAT, PREFIX, PROPOSAL_SHAPE, Practice, RECIPES,
                   SEEDS, STATE_DIM, load_scorer, numeric, pack_practice)

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "libero90-meta-retry-20261119-r1" / "cases.json"
LR = 3e-4
HORIZON = 400
CADENCE = 5


@dataclass
class Trajectory:
    # task and support_id are sampling/lookup keys only.
    task: int
    support_id: int
    query: Tensor
    proposals: Tensor
    index: Tensor
    success: bool


@dataclass
class Archive:
    supports: Practice
    trajectories: list[Trajectory]
    digest: str

    @property
    def embedding_dim(self):
        return self.trajectories[0].query.shape[1] - STATE_DIM


def stack_practices(practices):
    return Practice(*(torch.stack(fields) for fields in zip(*practices)))


def read_outcome(folder, tape):
    result = json.loads((folder / "result.json").read_text())
    controls, success = result["controls"], result["success"]
    if (type(controls) is not int or not 0 < controls <= HORIZON
            or type(success) is not bool or result["truncated"] is not (not success)
            or (not success and controls != HORIZON)
            or np.asarray(tape["controls"]).shape != () or tape["controls"].item() != controls
            or np.asarray(tape["terminal"]).shape != () or tape["terminal"].item() != success):
        raise ValueError(f"Inconsistent terminal or duration in {folder}")
    rewards = numeric(tape["rewards"], (controls,), "rewards")
    if ((rewards[:-1] > 0).any() or bool(rewards[-1] > 0) != success
            or not np.isclose(rewards.sum(dtype=np.float64), result["score"])):
        raise ValueError(f"Native rewards disagree with terminal result in {folder}")
    return controls, success


def integer_vector(tape, key, size):
    values = tape[key]
    if values.shape != (size,) or values.dtype.kind not in "iu":
        raise ValueError(f"Expected integer {key}[{size}]")
    return values.astype(np.int64)


def read_query(folder, task, support_id):
    with np.load(folder / "public.npz", allow_pickle=False) as public:
        controls, success = read_outcome(folder, public)
        issued = numeric(public["issued"], (controls, 7), "issued")
    n = (controls + CADENCE - 1) // CADENCE
    with np.load(folder / "decisions.npz", allow_pickle=False) as decisions:
        offsets = integer_vector(decisions, "offset", n)
        lengths = integer_vector(decisions, "length", n)
        index = integer_vector(decisions, "index", n)
        if (not np.array_equal(offsets, np.arange(0, controls, CADENCE))
                or not np.array_equal(lengths, np.minimum(CADENCE, controls - offsets))
                or not np.isin(index, (0, 1)).all()
                or numeric(decisions["scores"], (n, 2), "scores").any()):
            raise ValueError(f"Invalid randomized collection decisions in {folder}")
        proposals = numeric(decisions["proposals"], (n, *PROPOSAL_SHAPE), "proposals")
        chosen = proposals[np.arange(n), index, :CADENCE].reshape(-1, 7)
        if not np.array_equal(chosen[:controls], issued):
            raise ValueError(f"Executed controls differ from the logged choice in {folder}")
        state = numeric(decisions["state"], (n, STATE_DIM), "state")
        embedding = decisions["embedding"]
        if embedding.ndim != 2 or embedding.shape[1] == 0:
            raise ValueError(f"Expected a pooled embedding per decision in {folder}")
        embedding = numeric(embedding, (n, embedding.shape[1]), "embedding")
        query = np.concatenate((state, embedding), axis=1)
    return Trajectory(task, support_id, torch.from_numpy(query), torch.from_numpy(proposals),
                      torch.from_numpy(index), success)


def hash_tensor(digest, tensor):
    array = tensor.contiguous().numpy()
    digest.update(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())


def read_archive(root, manifest):
    catalogue = json.loads(manifest.read_text())
    rows = sorted((r for r in catalogue if r["split"] == "fitting"),
                  key=lambda r: (r["task_id"], r["repeat"]))
    ids = {(r["task_id"], r["repeat"]) for r in rows}
    other_tasks = {r["task_id"] for r in catalogue if r["split"] != "fitting"}
    if not rows or len(ids) != len(rows) or any(t in other_tasks for t, _ in ids):
        raise ValueError("Expected unique fitting cases on disjoint fitting tasks")
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode())
    supports, trajectories = [], []
    for i, row in enumerate(rows):
        folder = root / "fitting" / f"task{row['task_id']:02d}-r{row['repeat']:02d}"
        trial = json.loads((folder / "trial.json").read_text())
        if trial["manifest_row"] != row:
            raise ValueError(f"Published trial disagrees with the manifest in {folder}")
        with np.load(folder / "support" / "public.npz", allow_pickle=False) as tape:
            read_outcome(folder / "support", tape)
            practice = pack_practice(tape)
        supports.append(practice)
        for field in practice:
            hash_tensor(digest, field)
        for name in ("query0", "query1"):
            episode = read_query(folder / name, row["task_id"], i)
            if trajectories and episode.query.shape[1] != trajectories[0].query.shape[1]:
                raise ValueError(f"Frozen feature dimension changed in {folder / name}")
            trajectories.append(episode)
            for field in (episode.query, episode.proposals, episode.index):
                hash_tensor(digest, field)
            digest.update(bytes([episode.success]))
    return Archive(stack_practices(supports), trajectories, digest.hexdigest())


def initialize(data, seed):
    torch.manual_seed(seed)
    model = ActionReader(data.embedding_dim)
    valid = torch.arange(PREFIX)[None] < data.supports.length[:, None]
    state = torch.cat([e.query[:, :STATE_DIM] for e in data.trajectories])
    for key, values in (("history", data.supports.history[valid]), ("state", state)):
        scale, mean = torch.std_mean(values, dim=0, unbiased=False)
        getattr(model, key + "_mean").copy_(mean)
        getattr(model, key + "_scale").copy_(torch.where(scale > 1e-6, scale, 1.0))
    return model


class Sampler:
    """Uniform task, trajectory within task, and visited decision within trajectory.

    Stratification guarantees successful examples for CE at every shared update.
    BCE recombines the strata using their mass under the original sampling law.
    """

    def __init__(self, episodes, seed):
        self.rng = np.random.default_rng(seed)
        task_counts = Counter(e.task for e in episodes)
        weights = np.array([1 / (len(task_counts) * task_counts[e.task]) for e in episodes])
        success = np.array([e.success for e in episodes], dtype=bool)
        if not success.any():
            raise ValueError("No successful fitting query for success-conditioned imitation")
        self.success_mass = float(weights[success].sum()) if not success.all() else 1.0
        self.strata = []
        for mask in (success, ~success):
            ids = np.flatnonzero(mask)
            if len(ids):
                self.strata.append((ids, weights[ids] / weights[ids].sum()))
        self.lengths = np.array([len(e.index) for e in episodes])

    def draw(self, batch_size):
        if len(self.strata) == 1:
            counts = (batch_size,)
        else:
            n = min(batch_size - 1, max(1, round(batch_size * self.success_mass)))
            counts = (n, batch_size - n)
        episodes = np.concatenate([self.rng.choice(ids, n, p=probs)
                                   for (ids, probs), n in zip(self.strata, counts)])
        decisions = self.rng.integers(self.lengths[episodes])
        return episodes, decisions


def batch_at(data, episodes, decisions):
    sampled = [data.trajectories[i] for i in episodes]
    ids = [e.support_id for e in sampled]
    practice = Practice(*(field[ids] for field in data.supports))
    query = torch.stack([e.query[j] for e, j in zip(sampled, decisions)])
    proposals = torch.stack([e.proposals[j] for e, j in zip(sampled, decisions)])
    selected = torch.stack([e.index[j] for e, j in zip(sampled, decisions)])
    success = torch.tensor([e.success for e in sampled], dtype=torch.bool)
    return practice, query, proposals, selected, success


def objective(recipe, logits, selected, success, success_mass):
    if recipe == "imitation":
        return F.cross_entropy(logits[success], selected[success])
    if recipe != "meta":
        raise ValueError(f"Unknown recipe {recipe}")
    chosen = logits.gather(1, selected[:, None]).squeeze(1)
    losses = F.binary_cross_entropy_with_logits(chosen, success.float(), reduction="none")
    if success_mass == 1.0:
        return losses.mean()
    return success_mass * losses[success].mean() + (1 - success_mass) * losses[~success].mean()


def update(recipe, model, optimizer, batch, success_mass):
    practice, query, proposals, selected, success = batch
    optimizer.zero_grad(set_to_none=True)
    # Exact 50:50 expectation over complete practice and fully cleared practice.
    losses = [objective(recipe, model(view, query, proposals), selected, success, success_mass)
              for view in (practice, None)]
    loss = (losses[0] + losses[1]) * 0.5
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Nonfinite {recipe} loss")
    loss.backward()
    optimizer.step()
    return loss.item()


def save_atomic(payload, path):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def fit_seed(data, output, seed, config, every, resume):
    sampler = Sampler(data.trajectories, seed)
    first = initialize(data, seed)
    models = {"meta": first, "imitation": deepcopy(first)}
    optimizers = {r: torch.optim.AdamW(m.parameters(), lr=LR) for r, m in models.items()}
    path = output / f"seed-{seed}.pt"
    step = 0
    if path.exists():
        if not resume:
            raise FileExistsError(f"Use --resume for {path}")
        saved = torch.load(path, weights_only=True, map_location="cpu")
        if saved["config"] != config or saved["seed"] != seed:
            raise ValueError(f"Checkpoint inputs or configuration changed in {path}")
        for r in RECIPES:
            models[r].load_state_dict(saved["models"][r])
            optimizers[r].load_state_dict(saved["optimizers"][r])
        sampler.rng.bit_generator.state = saved["sampler_rng"]
        torch.set_rng_state(saved["torch_rng"])
        step = saved["step"]
    while step < config["steps"]:
        batch = batch_at(data, *sampler.draw(config["batch_size"]))
        losses = {r: update(r, models[r], optimizers[r], batch, sampler.success_mass) for r in RECIPES}
        step += 1
        if step % every == 0 or step == config["steps"]:
            save_atomic(dict(config=config, seed=seed, step=step, losses=losses,
                             models={r: m.state_dict() for r, m in models.items()},
                             optimizers={r: opt.state_dict() for r, opt in optimizers.items()},
                             torch_rng=torch.get_rng_state(), sampler_rng=sampler.rng.bit_generator.state),
                        path)
            print(json.dumps(dict(seed=seed, step=step, loss=losses)), flush=True)
    return {r: m.state_dict() for r, m in models.items()}


def source_digest():
    digest = hashlib.sha256()
    core = HERE.parents[1] / "src/fluffy_bassoon/meta_rl/causal_decoder.py"
    for path in (Path(__file__), HERE / "model.py", core):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def fit(args):
    data = read_archive(args.root, args.manifest)
    config = dict(format=FORMAT, data=data.digest, code=source_digest(),
                  embedding_dim=data.embedding_dim, steps=args.steps, batch_size=args.batch_size,
                  threads=args.threads, learning_rate=LR, torch_version=str(torch.__version__),
                  numpy_version=str(np.__version__))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    # One writer owns the checkpoint set through ensemble publication.
    lock = os.open(output, os.O_RDONLY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = output / "frozen"
        if frozen.exists():
            if args.resume and json.loads((frozen / "config.json").read_text()) == config:
                return frozen
            raise FileExistsError(f"Frozen fit already exists at {frozen}")
        ensembles = [fit_seed(data, output, seed, config, args.checkpoint_every, args.resume)
                     for seed in SEEDS]
        with tempfile.TemporaryDirectory(dir=output) as temp:
            stage = Path(temp) / "frozen"
            stage.mkdir()
            for recipe in RECIPES:
                torch.save(dict(format=FORMAT, recipe=recipe, embedding_dim=data.embedding_dim,
                                seeds=SEEDS, models=[m[recipe] for m in ensembles]), stage / f"{recipe}.pt")
            (stage / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            stage.rename(frozen)
        return frozen
    finally:
        os.close(lock)


def check(directory):
    """A deterministic CPU regression through the archive, checkpoint and collector APIs."""
    from unittest.mock import patch

    rng = np.random.default_rng(12)
    root = directory / "episodes"
    row = dict(split="fitting", task_id=0, repeat=0, support_id=1, query_id=2, seed=17)
    manifest = directory / "cases.json"
    manifest.write_text(json.dumps([row]))
    trial = root / "fitting/task00-r00"
    trial.mkdir(parents=True)
    (trial / "trial.json").write_text(json.dumps(dict(manifest_row=row)))
    for name, controls, success in (("support", HORIZON, False), ("query0", 7, True),
                                    ("query1", HORIZON, False)):
        folder = trial / name
        folder.mkdir()
        offsets = np.arange(0, controls, CADENCE)
        n = len(offsets)
        proposals = rng.standard_normal((n, *PROPOSAL_SHAPE), dtype=np.float32)
        index = rng.integers(2, size=n)
        issued = proposals[np.arange(n), index, :CADENCE].reshape(-1, 7)[:controls]
        rewards = np.zeros(controls, np.float32)
        rewards[-1] = float(success)
        tape = dict(controls=controls, terminal=success, rewards=rewards, issued=issued,
                    positions=rng.normal(size=(controls, 3)), initial_position=np.zeros(3),
                    gripper=rng.normal(size=(controls, 2)))
        np.savez(folder / "public.npz", **tape)
        result = dict(controls=controls, success=success, score=float(success), truncated=not success)
        (folder / "result.json").write_text(json.dumps(result))
        if name == "support":
            support = tape
        else:
            state = rng.standard_normal((n, STATE_DIM), dtype=np.float32)
            embedding = rng.standard_normal((n, 16), dtype=np.float32)
            np.savez(folder / "decisions.npz", state=state, embedding=embedding, index=index,
                     proposals=proposals, scores=np.zeros((n, 2), np.float32), offset=offsets,
                     length=np.minimum(CADENCE, controls - offsets))
    data = read_archive(root, manifest)

    # Detect an unselected BCE target, a failed-query CE contribution or a missing
    # alternate in the CE denominator. The two successes have different choices.
    selected = torch.tensor([1, 0, 0])
    success = torch.tensor([True, False, True])
    logits = torch.tensor([[0.1, 1.2], [3.0, -0.2], [0.7, -0.5]], requires_grad=True)
    meta = objective("meta", logits, selected, success, 0.4)
    bc = objective("imitation", logits, selected, success, 0.4)
    target_meta = (0.2 * F.softplus(-logits[0, 1]) + 0.6 * F.softplus(logits[1, 0])
                   + 0.2 * F.softplus(-logits[2, 0]))
    target_bc = (F.softplus(logits[0, 0] - logits[0, 1]) + F.softplus(logits[2, 1] - logits[2, 0])) / 2
    for actual, reference in ((meta, target_meta), (bc, target_bc)):
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(torch.autograd.grad(actual, logits)[0], torch.autograd.grad(reference, logits)[0])

    options = argparse.Namespace(root=root, manifest=manifest, output=directory / "fit",
                                 steps=2, batch_size=4, threads=1, checkpoint_every=1, resume=False)
    frozen = fit(options)
    # Replay the second optimizer step from an interrupted joint checkpoint.
    replay = directory / "replay"
    replay.mkdir()
    config = json.loads((frozen / "config.json").read_text())
    original_save = save_atomic

    def interrupt(payload, path):
        original_save(payload, path)
        raise InterruptedError

    with patch.dict(fit_seed.__globals__, save_atomic=interrupt):
        try:
            fit_seed(data, replay, SEEDS[0], config, 1, False)
        except InterruptedError:
            pass
    resumed = fit_seed(data, replay, SEEDS[0], config, 1, True)
    example = data.trajectories[0]
    query, proposals = example.query[0], example.proposals[0]
    practice = stack_practices([pack_practice(support)])
    for recipe in RECIPES:
        path = frozen / f"{recipe}.pt"
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        for key, value in resumed[recipe].items():
            torch.testing.assert_close(value, bundle["models"][0][key], rtol=0, atol=0)
        scorer = load_scorer(path, recipe)
        def predict(tape, actions):
            return scorer(tape, query[:STATE_DIM].numpy(), query[STATE_DIM:].numpy(), actions.numpy())

        for tape, view in ((support, practice), (None, None)):
            logits = predict(tape, proposals)
            with torch.inference_mode():
                members = []
                for weights in bundle["models"]:
                    model = ActionReader(data.embedding_dim).eval()
                    model.load_state_dict(weights)
                    members.append(model(view, query[None], proposals[None])[0])
                np.testing.assert_allclose(logits, torch.stack(members).mean(0).numpy(), rtol=0, atol=0)
            np.testing.assert_allclose(predict(tape, proposals.flip(0)), logits[::-1], atol=1e-6)
            ties = predict(tape, proposals[[0, 0]])
            if ties[0] != ties[1] or ties.argmax() != 0:
                raise AssertionError("Identical proposals must tie in collector order")
        changed = support["issued"].copy()
        changed[PREFIX:] += 10
        np.testing.assert_array_equal(predict(dict(support, issued=changed), proposals), predict(support, proposals))
    print("PASS: logged objectives, checkpoint replay, retained/cleared ensemble logits, proposal order and ties")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="fit both recipes and publish frozen/{meta,imitation}.pt")
    train.add_argument("--root", type=Path, required=True, help="episode root containing fitting/")
    train.add_argument("--output", type=Path, required=True, help="directory for checkpoints and frozen/")
    train.add_argument("--manifest", type=Path, default=MANIFEST)
    train.add_argument("--steps", type=int, default=450, help="optimizer updates per seed per recipe")
    train.add_argument("--batch-size", type=int, default=64, help="shared decisions per update")
    train.add_argument("--threads", type=int, default=2)
    train.add_argument("--checkpoint-every", type=int, default=50)
    train.add_argument("--resume", action="store_true", help="recover a fit with identical inputs and settings")
    sub.add_parser("check", help="run constructed CPU regression in a temporary directory")
    args = parser.parse_args()
    if args.command == "check":
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            check(Path(temp))
    else:
        if min(args.steps, args.threads, args.checkpoint_every) < 1 or args.batch_size < 2:
            parser.error("Require positive steps, threads, checkpoint interval, and batch size >= 2")
        torch.set_num_threads(args.threads)
        print(fit(args))


if __name__ == "__main__":
    main()
