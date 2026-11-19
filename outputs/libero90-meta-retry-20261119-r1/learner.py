"""Frozen Reader comparison for nominal LIBERO-90. Run 'format' for the collector contract."""
import argparse
from collections import defaultdict
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
READER_DIR = HERE.parent / "libero-decision-pretraining-20261119-r1"
LOGGED_DIR = HERE.parent / "libero-logged-retry-20261119-r1"
ARMS = ("default", "alternate")
SEEDS = (41, 42, 43)
BATCH, UPDATES = 64, 450
LOG_SEED, SUBSET_SEED, DONOR_SEED = 101, 102, 103
COUNTS = {"fitting": (48, 20), "development": (12, 12), "test": (30, 24)}
REGIMES = ("full", "single", "full_equal_cost")
FIELDS = ("split", "task_id", "repeat", "support_id", "query_id", "seed")

FORMAT_MD = """# Nominal LIBERO-90 retry corpus

Use the registered `cases.json` list. Each row contains `split`, `task_id`,
`repeat`, `support_id`, `query_id` and `seed`. Splits are `fitting`,
`development` and `test`. Each trial directory is keyed by task and repeat.
For task 7, repeat 3 the directory is `root/<split>/task07-r03/`.

    query.npy
    support/public.npz      support/result.json
    default/public.npz      default/result.json
    alternate/public.npz    alternate/result.json

Save the single public query vector before executing either retry. Use the
existing decision-pretraining query representation, a finite float vector with
eight public state values followed by the pooled frozen visual/language prefix.
Each public tape contains chronological arrays `issued[T,7]`, `positions[T,3]`
(post-control wrist positions), `initial_position[3]`, `gripper[T,2]`,
`rewards[T]` (native), and `waits[T]` (binary). Save a complete tape for every
success and failure. Every result records native binary `success` and a positive
integer `controls` equal to T. Include integer `policy_calls` for measured
inference cost and `scene` or the standard LIBERO `task_name` for scene totals.
Unavailable policy-call counts are reported as null.

Execute the default support on support_id. Start both retries at query_id with
identical simulator state, public opening, policy noise and horizon. The
collector records the manifest seed as case.trial_seed and the environment's
scene seed as case.seed. It writes behavior.json before executing the retries.
Its arm is a fair coin from NumPy default_rng(101) in task_id, repeat order. Keep
evaluator metadata in result JSON. The earlier collector's optional `case`,
`identity` and `opening_sha256` fields are checked across the two branches.
Use one settled opening and retain all native results irrespective of support
success. No split, task ID, reset ID, seed, branch trajectory or retry outcome
is a decision feature.

Use a fresh output directory separate from the episode root. Commands:

    python learner.py format > format.md
    python learner.py fit --root EPISODES --output RUN
    python learner.py score --split development --root EPISODES --output RUN
    python learner.py score --split test --root EPISODES --output RUN

Fit reads fitting episodes only. RUN/frozen/identity.json marks completion of
all fits and fixes every weight file, manifest, seed, subset and decision rule.
Collect held-out episodes after that freeze. Score the registered fits as-is.
Development and test reports are separate files under RUN.
"""


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("format", help="print format.md for the collector")
    for name in ("fit", "score"):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True, type=Path, help="episode root")
        command.add_argument("--output", required=True, type=Path, help="learner run directory")
        command.add_argument("--manifest", type=Path, default=HERE / "cases.json")
        if name == "score":
            command.add_argument("--split", required=True, choices=("development", "test"))
    return parser.parse_args()


# The collector can obtain format.md or CLI help with a standard-library Python.
if __name__ == "__main__":
    args = arguments()
    if args.command == "format":
        print(FORMAT_MD, end="")
        sys.exit(0)

import numpy as np
import torch
from torch.nn import functional as F

sys.path[:0] = [str(LOGGED_DIR), str(READER_DIR), str(REPO / "src")]
from fit import Public, forward
from reader import LEARNING_RATE, PREFIX, TRAJECTORY_CHANNELS, Reader


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, allow_nan=False, indent=2) + "\n")


def read_json(path):
    return json.loads(path.read_text())


def read_cases(data):
    rows = json.loads(data)
    groups = defaultdict(list)
    for row in rows:
        split = row["split"]
        if split not in COUNTS or any(type(row[k]) is not int for k in FIELDS[1:]):
            raise ValueError(f"Invalid manifest identity {row}")
        groups[split, row["task_id"]].append(row)
    used = set()
    for split, (tasks, repeats) in COUNTS.items():
        ids = {task for group, task in groups if group == split}
        if len(ids) != tasks or ids & used:
            raise ValueError(f"Task count or overlap in {split}")
        used.update(ids)
        for task in ids:
            cases = groups[split, task]
            poses = [r[k] for r in cases for k in ("support_id", "query_id")]
            if (sorted(r["repeat"] for r in cases) != list(range(repeats))
                    or len(set(poses)) != 2 * repeats or not set(poses) <= set(range(1, 50))):
                raise ValueError(f"Invalid repeats or reused reset for {split} task {task}")
    if used != set(range(90)) or len({r["seed"] for r in rows}) != len(rows):
        raise ValueError("Expected all 90 nominal task IDs and distinct trial seeds")
    return sorted(rows, key=lambda r: (r["split"], r["task_id"], r["repeat"]))


def trial_dir(root, row):
    return root / row["split"] / f"task{row['task_id']:02d}-r{row['repeat']:02d}"


def episode(folder, arm, row, *, pack=False):
    directory = folder / arm
    result = read_json(directory / "result.json")
    count, success = result["controls"], result["success"]
    if type(count) is not int or count < 1 or success not in (0, 1):
        raise ValueError(f"Invalid native result in {directory}")
    calls = result.get("policy_calls")
    if calls is not None and (type(calls) is not int or calls < 1):
        raise ValueError(f"Invalid policy_calls in {directory}")
    case = result.get("case", {})
    reset_id = row["support_id" if arm == "support" else "query_id"]
    for key, expected in (("task_id", row["task_id"]), ("init_state_id", reset_id),
                          ("trial_seed", row["seed"]), ("suite", "libero_90"), ("variant", "BASE")):
        if key in case and case[key] != expected:
            raise ValueError(f"Wrong {key} in {directory}")
    for key in FIELDS[:-1]:
        if key in result and result[key] != row[key]:
            raise ValueError(f"Manifest mismatch for {key} in {directory}")
    history = None
    with np.load(directory / "public.npz", allow_pickle=False) as tape:
        rewards = tape["rewards"]
        if (rewards.shape != (count,) or not np.isfinite(rewards).all()
                or bool((rewards > 0).any()) != bool(success)):
            raise ValueError(f"Native reward or duration mismatch in {directory}")
        if pack:
            n = min(count, PREFIX)
            parts = []
            for name, shape in (("issued", (count, 7)), ("positions", (count, 3)),
                                ("gripper", (count, 2)), ("waits", (count,))):
                values = tape[name]
                if values.shape != shape or not np.isfinite(values).all():
                    raise ValueError(f"Invalid {name} in {directory}")
                parts.append(values[:n])
            issued, positions, gripper, waits = parts
            initial = tape["initial_position"]
            if (initial.shape != (3,) or not np.isfinite(initial).all()
                    or not np.isin(waits, (0, 1)).all()):
                raise ValueError(f"Invalid initial position or waits in {directory}")
            motion = np.diff(np.vstack((initial, positions)), axis=0)
            history = np.zeros((PREFIX, TRAJECTORY_CHANNELS), dtype=np.float32)
            history[:n] = np.column_stack((issued, motion, gripper, rewards[:n], waits))
    return result, history


def public_inputs(root, rows):
    histories, lengths, queries, supports = [], [], [], []
    for row in rows:
        folder = trial_dir(root, row)
        result, history = episode(folder, "support", row, pack=True)
        query = np.load(folder / "query.npy", allow_pickle=False)
        if (query.ndim != 1 or len(query) <= 8 or query.dtype.kind != "f"
                or not np.isfinite(query).all()):
            raise ValueError(f"Invalid opening features in {folder}")
        histories.append(history)
        lengths.append(min(result["controls"], PREFIX))
        queries.append(query.astype(np.float32))
        supports.append(result)
    x = Public(torch.from_numpy(np.stack(histories)), torch.tensor(lengths),
               torch.from_numpy(np.stack(queries)),
               torch.tensor([[float(r["success"]), 1.] for r in supports]))
    return x, supports


def native_branches(root, rows):
    branches = []
    for row in rows:
        folder = trial_dir(root, row)
        pair = [episode(folder, arm, row)[0] for arm in ARMS]
        for key in ("case", "identity", "opening_sha256", "noise_bank_sha256"):
            if pair[0].get(key) != pair[1].get(key):
                raise ValueError(f"Unpaired retry {key} in {folder}")
        branches.append(pair)
    return branches


def groups_by_task(rows):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["task_id"]].append(i)
    return groups


def designs(rows):
    # These draws depend only on sorted manifest identities, before outcomes.
    all_ids = np.arange(len(rows))
    logged = np.random.default_rng(LOG_SEED).integers(2, size=(len(rows), 1))
    both = np.tile(np.arange(2), (len(rows), 1))
    rng = np.random.default_rng(SUBSET_SEED)
    groups = groups_by_task(rows)
    # 16 tasks x 14 plus 32 x 13 = 640, nested in the same 960 histories.
    extra = set(rng.choice(sorted(groups), 16, replace=False))
    nested = sorted(i for task, ids in groups.items()
                    for i in rng.permutation(ids)[:13 + int(task in extra)])
    return {"full": (all_ids, both), "single": (all_ids, logged),
            "full_equal_cost": (np.array(nested), both[nested])}


def verify_behavior(root, rows, arms, queries):
    for row, (action,), query in zip(rows, arms, queries):
        record_path = trial_dir(root, row) / "behavior.json"
        if record_path.exists():
            recorded = read_json(record_path)
            value = query.numpy()
            fingerprint = hashlib.sha256(str((value.shape, value.dtype.str)).encode()
                                         + value.tobytes()).hexdigest()
            expected = dict(arm=ARMS[action], probability=0.5, log_seed=LOG_SEED,
                            query_sha256=fingerprint, control_offset=0)
            if any(recorded[k] != v for k, v in expected.items()):
                raise ValueError(f"Logged action or opening differs from registration: {record_path}")


def reassociate(rows):
    # A random cycle per task gives one distinct donor per case, without labels.
    rng = np.random.default_rng(DONOR_SEED)
    other = np.empty(len(rows), dtype=int)
    for ids in groups_by_task(rows).values():
        cycle = rng.permutation(ids)
        other[cycle] = np.roll(cycle, 1)
    return other


def total_field(records, field):
    values = [r.get(field) for r in records]
    return sum(values) if None not in values else None


def acquisition(supports, branches, ids, arms):
    practice = [supports[i] for i in ids]
    retries = [branches[i][a] for i, actions in zip(ids, arms) for a in actions]
    records = practice + retries
    return dict(histories=len(practice), retry_labels=len(retries), episodes=len(records),
                support_controls=sum(r["controls"] for r in practice),
                retry_controls=sum(r["controls"] for r in retries),
                controls=sum(r["controls"] for r in records),
                settling_controls=total_field(records, "settling_controls"),
                policy_calls=total_field(records, "policy_calls"))


def minibatches(count, seed):
    sampler = torch.Generator().manual_seed(seed)
    while True:
        yield from torch.randperm(count, generator=sampler).split(BATCH)


def binary_loss(model, batch, actions, labels, retain):
    prediction = forward(model, batch, retain).gather(1, actions)
    # Sum independent binary labels per history and average histories.
    return F.binary_cross_entropy_with_logits(prediction, labels, reduction="sum") / len(labels)


def fit_model(x, actions, labels, seed, current, checkpoint):
    # Only the revealed (branch, label) pairs cross this fitting interface.
    start = time.perf_counter()
    torch.manual_seed(seed)
    stats = SimpleNamespace(history=x.history, lengths=x.lengths,
                            query=torch.cat((x.query, x.reward), dim=1))
    model = Reader(stats, len(ARMS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    seen = 0
    for step, ids in enumerate(islice(minibatches(len(labels), seed), UPDATES)):
        batch = x.subset(ids)
        seen += len(ids)
        revealed = actions[ids], labels[ids]
        objective = binary_loss(model, batch, *revealed, retain=False)
        if not current:
            # Exact expectation of a 50% complete-history dropout.
            objective = 0.5 * (objective + binary_loss(model, batch, *revealed, retain=True))
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        if step == 0:
            first = objective.item()
    final = objective.item()
    if not np.isfinite([first, final]).all():
        raise ValueError(f"Nonfinite fitting objective for {checkpoint.name}")
    torch.save(model.state_dict(), checkpoint)
    return dict(seed=seed, current_only=current, updates=UPDATES, batch=BATCH,
                histories_seen=seen, labels_seen=seen * actions.shape[1],
                initial_loss=first, final_loss=final, elapsed_seconds=time.perf_counter() - start,
                sha256=digest(checkpoint))


def checkpoint_name(regime, seed, current):
    return f"{regime}-{seed}-{'current' if current else 'history'}.pt"


def code_identity():
    files = [Path(__file__), LOGGED_DIR / "fit.py", READER_DIR / "reader.py",
             REPO / "src/fluffy_bassoon/meta_rl/causal_decoder.py"]
    return {str(path.resolve().relative_to(REPO)): digest(path) for path in files}


def fit(root, output, rows, manifest):
    if output.exists():
        raise FileExistsError(f"Fit requires a fresh run directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".retry-fit-") as tmp:
        stage = Path(tmp) / "run"
        frozen = stage / "frozen"
        frozen.mkdir(parents=True)
        (stage / "format.md").write_text(FORMAT_MD)
        masks = designs(rows)
        x, supports = public_inputs(root, rows)
        verify_behavior(root, rows, masks["single"][1], x.query)
        branches = native_branches(root, rows)
        y = np.array([[float(r["success"]) for r in pair] for pair in branches], dtype=np.float32)
        costs, checkpoints = {}, {}
        for regime, (ids, arms) in masks.items():
            costs[regime] = acquisition(supports, branches, ids, arms)
            train_x = x.subset(ids)
            observed = torch.from_numpy(y[ids[:, None], arms])
            for seed in SEEDS:
                for current in (False, True):
                    name = checkpoint_name(regime, seed, current)
                    checkpoints[name] = fit_model(train_x, torch.from_numpy(arms), observed,
                                                  seed, current, frozen / name)
                    print(f"Fitted {name}", flush=True)
        design = dict(fitting_rows=rows, logged_arm=masks["single"][1].ravel().tolist(),
                      nested_full_indices=masks["full_equal_cost"][0].tolist(),
                      fitting_native_outcomes=y.astype(int).tolist(),
                      public_tensor_sha256={
                          key: hashlib.sha256(value.numpy().tobytes()).hexdigest()
                          for key, value in vars(x).items()})
        write_json(frozen / "design.json", design)
        identity = dict(
            suite="libero_90", manifest_sha256=manifest, source_sha256=code_identity(),
            git_revision=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                                 text=True).strip(),
            torch_version=str(torch.__version__), numpy_version=np.__version__,
            query_width=x.query.shape[1], branches=ARMS, seeds=SEEDS, log_seed=LOG_SEED,
            subset_seed=SUBSET_SEED, donor_seed=DONOR_SEED, batch=BATCH, updates=UPDATES,
            learning_rate=LEARNING_RATE, history_dropout=0.5, optimizer="AdamW",
            decision="mean raw logits across seeds, argmax, exact ties choose default",
            design_sha256=digest(frozen / "design.json"), charged_collection=costs,
            physical_fitting_archive=dict(
                costs["full"], fresh_policy_calls=total_field(
                    supports + [r for pair in branches for r in pair], "fresh_policy_calls")),
            primary_budget="full and single share all 960 histories with different episode costs",
            secondary_budget="full_equal_cost and single each charge 1920 episodes with measured controls",
            checkpoints=checkpoints)
        write_json(frozen / "identity.json", identity)
        stage.rename(output)


def verify_freeze(frozen, manifest):
    identity = read_json(frozen / "identity.json")
    if (identity["manifest_sha256"] != manifest or identity["source_sha256"] != code_identity()
            or digest(frozen / "design.json") != identity["design_sha256"]):
        raise ValueError("Manifest, fitting design or learner source differs from the freeze")
    for regime in REGIMES:
        for seed in SEEDS:
            for current in (False, True):
                name = checkpoint_name(regime, seed, current)
                if digest(frozen / name) != identity["checkpoints"][name]["sha256"]:
                    raise ValueError(f"Checkpoint differs from the freeze: {name}")
    return identity


@torch.inference_mode()
def predict(frozen, x, width, other):
    # Reader requires fitting statistics at construction. All buffers in this
    # placeholder are replaced by the fitted state before the first prediction.
    shape = SimpleNamespace(history=torch.zeros(1, PREFIX, TRAJECTORY_CHANNELS),
                            lengths=torch.ones(1, dtype=torch.long), query=torch.zeros(1, width + 2))
    model = Reader(shape, len(ARMS)).eval()
    predictions = {str(seed): {} for seed in SEEDS}
    for regime in REGIMES:
        for seed in SEEDS:
            for current in (False, True):
                path = frozen / checkpoint_name(regime, seed, current)
                model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                modes = ("current",) if current else ("own", "cleared", "mismatched")
                for mode in modes:
                    blocks = []
                    for start in range(0, len(other), BATCH):
                        ids = slice(start, start + BATCH)
                        batch = x.subset(ids)
                        if mode == "mismatched":
                            donor = other[ids]
                            batch = Public(x.history[donor], x.lengths[donor], batch.query,
                                           x.reward[donor])
                        blocks.append(forward(model, batch, mode in ("own", "mismatched")).numpy())
                    logits = np.concatenate(blocks)
                    if not np.isfinite(logits).all():
                        raise ValueError(f"Nonfinite logits in {path.name}, {mode}")
                    predictions[str(seed)][f"{regime}_{mode}"] = logits
    predictions["ensemble"] = {
        name: np.mean([predictions[str(s)][name] for s in SEEDS], axis=0, dtype=np.float64)
        for name in predictions[str(SEEDS[0])]
    }
    return predictions


def paired_contrasts(hits):
    deltas = {}
    for regime in REGIMES:
        own = hits[f"{regime}_own"]
        for control in ("base", "fixed_alternate", "cleared", "mismatched", "current"):
            key = control if control in ("base", "fixed_alternate") else f"{regime}_{control}"
            deltas[f"{regime}_own_minus_{control}"] = own - hits[key]
    for method in ("full", "full_equal_cost"):
        deltas[f"{method}_minus_single_history_gain"] = (
            deltas[f"{method}_own_minus_cleared"] - deltas["single_own_minus_cleared"])
        deltas[f"{method}_own_minus_single_own"] = hits[f"{method}_own"] - hits["single_own"]
    return deltas


def summarize_group(ids, hits, deltas):
    return dict(trials=len(ids), successes={m: int(v[ids].sum()) for m, v in hits.items()},
                paired_net={m: int(v[ids].sum()) for m, v in deltas.items()},
                paired_gains={m: int((v[ids] > 0).sum()) for m, v in deltas.items()},
                paired_losses={m: int((v[ids] < 0).sum()) for m, v in deltas.items()})


def contrast_summary(delta, tasks):
    # Two-sided Student intervals across equally weighted task means. The
    # registered panels have 12 or 30 tasks (11 or 29 degrees of freedom).
    means = np.array([delta[ids].mean() for ids in tasks.values()]) * 100
    df = len(means) - 1
    critical = {11: 2.200985160082949, 29: 2.045229642132704}[df]
    mean = float(means.mean())
    margin = critical * means.std(ddof=1) / np.sqrt(len(means))
    return dict(net=int(delta.sum()), gains=int((delta > 0).sum()), losses=int((delta < 0).sum()),
                effect_pp=mean, ci95_pp=[mean - margin, mean + margin], df=df,
                tasks_improved=int((means > 0).sum()), tasks_tied=int((means == 0).sum()),
                tasks_harmed=int((means < 0).sum()))


def scene_name(row, record):
    for source in (row, record, record.get("case", {}), record.get("identity", {})):
        if source.get("scene"):
            return source["scene"]
        match = re.search(r"([A-Z_]+_SCENE\d+)", str(source.get("task_name", "")))
        if match:
            return match.group(1)
    return "unrecorded"


def readout(logits, decisions, supports, branches, tasks, scenes):
    n = len(branches)
    ids = np.arange(n)
    y = np.array([[r["success"] for r in pair] for pair in branches], dtype=int)
    practice = np.array([r["success"] for r in supports], dtype=int)
    choice = dict(decisions, base=np.zeros(n, dtype=int), fixed_alternate=np.ones(n, dtype=int))
    hits = {name: y[ids, arm] for name, arm in choice.items()}
    hits["branch_ceiling"] = y.max(axis=1)
    deltas = paired_contrasts(hits)
    deployment = {}
    for name, chosen_arms in choice.items():
        selected = [pair[a] for pair, a in zip(branches, chosen_arms)]
        deployment[name] = dict(
            successes_in_either_attempt=int(np.maximum(practice, hits[name]).sum()),
            episodes=2 * n, controls=sum(r["controls"] for r in supports + selected),
            settling_controls=total_field(supports + selected, "settling_controls"),
            policy_calls=total_field(supports + selected, "policy_calls"))
    return dict(
        successes={k: int(v.sum()) for k, v in hits.items()},
        task_mean_success_pct={k: float(np.mean([v[idx].mean() for idx in tasks.values()]) * 100)
                               for k, v in hits.items()},
        contrasts={k: contrast_summary(v, tasks) for k, v in deltas.items()},
        by_task={str(t): summarize_group(idx, hits, deltas) for t, idx in tasks.items()},
        by_scene={s: summarize_group(idx, hits, deltas) for s, idx in scenes.items()},
        deployment=deployment,
        # Every array below follows the same top-level cases list.
        branch_indices={k: v.tolist() for k, v in choice.items()},
        native_outcomes={k: v.tolist() for k, v in hits.items()},
        full_minus_single_history_gain=deltas["full_minus_single_history_gain"].tolist(),
        logits={k: v.tolist() for k, v in logits.items()})


def score(root, output, rows, manifest, split):
    target = output / f"{split}.json"
    if target.exists():
        raise FileExistsError(f"Score already published: {target}")
    frozen = output / "frozen"
    identity = verify_freeze(frozen, manifest)
    x, supports = public_inputs(root, rows)
    if x.query.shape[1] != identity["query_width"]:
        raise ValueError("Opening feature width differs from fitting")
    donors = reassociate(rows)
    predictions = predict(frozen, x, identity["query_width"], donors)
    decisions = {run: {m: logits.argmax(axis=1) for m, logits in values.items()}
                 for run, values in predictions.items()}
    branches = native_branches(root, rows)
    scenes = defaultdict(list)
    cases = []
    for i, (row, support, pair) in enumerate(zip(rows, supports, branches)):
        scene = scene_name(row, support)
        scenes[scene].append(i)
        costs = [{key: r.get(key) for key in ("success", "controls", "policy_calls")}
                 for r in (support, *pair)]
        cases.append(dict(**row, donor_repeat=rows[donors[i]]["repeat"],
                          donor_support_id=rows[donors[i]]["support_id"], scene_group=scene,
                          observed=dict(zip(("support", *ARMS), costs))))
    tasks = groups_by_task(rows)
    results = {name: readout(values, decisions[name], supports, branches, tasks, scenes)
               for name, values in predictions.items()}
    report = dict(
        split=split, manifest_sha256=manifest, freeze_sha256=digest(frozen / "identity.json"),
        cases=cases, branch_order=ARMS, primary_method="ensemble/full_own",
        fitting_collection=identity["charged_collection"],
        evaluation_collection=acquisition(supports, branches, np.arange(len(rows)),
                                          np.tile(np.arange(2), (len(rows), 1))),
        one_shot_support=dict(successes=sum(int(r["success"]) for r in supports),
                              episodes=len(rows), controls=sum(r["controls"] for r in supports),
                              policy_calls=total_field(supports, "policy_calls")),
        ensemble=results.pop("ensemble"), per_seed=results)
    if split == "test":
        contrasts = report["ensemble"]["contrasts"]
        base_positive = contrasts["full_own_minus_base"]["ci95_pp"][0] > 0
        history_positive = contrasts["full_own_minus_cleared"]["ci95_pp"][0] > 0
        report["fixed_sequence"] = dict(controller_gain_supported=bool(base_positive),
                                        history_test_reached=bool(base_positive),
                                        history_gain_supported=bool(base_positive and history_positive))
    with tempfile.TemporaryDirectory(dir=output, prefix=f".{split}-") as tmp:
        staged = Path(tmp) / "report.json"
        write_json(staged, report)
        os.link(staged, target)  # Publish once, including concurrent score invocations.
    return target


def main(options):
    root, output = options.root.resolve(), options.output.resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Use separate corpus and learner output directories")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data = options.manifest.read_bytes()
    rows = read_cases(data)
    manifest = hashlib.sha256(data).hexdigest()
    registered = HERE / "cases.json"
    if options.manifest.resolve() != registered and rows != read_cases(registered.read_bytes()):
        raise ValueError("Manifest rows differ from the registered cases.json")
    split = "fitting" if options.command == "fit" else options.split
    rows = [r for r in rows if r["split"] == split]
    if split == "fitting":
        fit(root, output, rows, manifest)
        print(output / "frozen" / "identity.json")
    else:
        print(score(root, output, rows, manifest, split))


if __name__ == "__main__":
    main(args)
