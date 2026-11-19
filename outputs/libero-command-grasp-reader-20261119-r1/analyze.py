"""Score the frozen readers against measured native queries. Usage: python analyze.py OUT."""
import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

import readers


SEEDS = (41, 42, 43)
CELLS = ((1, "BASE"), (-1, "BASE"), (1, "D17"), (-1, "D17"))
HELD_OUT = (-1, "D17")
CHOICES = tuple(product((1, -1), (0, 1)))
PULSES = np.zeros((10, 7), dtype=np.float32)
PULSES[:5, 0], PULSES[5:, 0], PULSES[:, -1] = 0.45, -0.45, -1
COMPARATORS = ("analytic_probe", "factored_probe", "current", "swap_grip", "swap_wire")


def case_name(pose, wire, condition):
    return f"p{pose:02d}_w{wire:+d}_{condition}"


def branch_name(decode, grip):
    return f"d{decode:+d}_g{grip}"


def read_json(path):
    return json.loads(path.read_text())


def require(ok, where, reason):
    if not ok:
        raise ValueError(f"{where}: {reason}")


@dataclass
class Episode:
    row: dict
    commands: np.ndarray
    states: np.ndarray
    rewards: np.ndarray

    @property
    def success(self):
        return bool((self.rewards > 0).any())

    @property
    def controls(self):
        return len(self.rewards)

    def outcome(self):
        result = dict(success=self.success, controls=self.controls,
                      native_return=float(self.rewards.sum(dtype=np.float64)))
        for k in ("choice_after", "d17_triggered", "regrasp_controls"):
            if k in self.row:
                result[k] = self.row[k]
        return result


def read_episode(path, pose, wire, condition, decode, grip):
    r = read_json(path / "result.json")
    require((r["pose"], r["seed"], r["wire"], r["condition"]) ==
            (pose, 19_260_000 + pose, wire, condition), path, "wrong pose, seed, wire or condition")
    with np.load(path / "tape.npz", allow_pickle=False) as tape:
        cmd, state, reward = (tape[k] for k in ("commands", "states", "rewards"))
    n = r["controls"]
    require(type(n) is int and 10 <= n <= 300 and
            (cmd.shape, state.shape, reward.shape) == ((n, 7), (n + 1, 9), (n,)),
            path, "invalid control count or tape dimensions")
    require(all(np.isfinite(a).all() for a in (cmd, state, reward)), path, "nonfinite tape")
    require(np.allclose(cmd[:10], PULSES, atol=1e-7, rtol=0), path, "wrong public wire pulses")
    ep = Episode(r, cmd, state, reward)
    require(type(r["success"]) is bool and r["success"] == ep.success and
            not (reward[:-1] > 0).any(), path, "native reward disagrees with outcome or stopping time")
    if "native_return" in r:
        require(np.isclose(reward.sum(dtype=np.float64), r["native_return"]), path, "wrong native return")
    require(all(r[k] for k in ("initial_physics", "initial_public", "first_motor_input", "first_proposal")),
            path, "missing pairing evidence")
    sign = 1 if state[5, 0] >= state[0, 0] else -1
    require(r["probe_sign"] == sign, path, "probe_sign disagrees with wrist tape")
    applied = sign if decode is None else decode
    require(r["decode"] == decode and r.get("applied_decode", applied) == applied, path,
            "wrong assigned or applied decoder")
    require(r["grip"] == grip, path, "wrong assigned grip")
    if path.name != "current":
        require(r["chosen_grip"] == grip, path, "static branch applied a different grip")
    if "reactive" in r:
        require(r["reactive"] == (path.name == "current"), path, "incorrect reactive arm")
    t = r["choice_after"]
    require(t is None or (type(t) is int and 10 <= t < n), path, "choice outside executed prefix")
    return ep


class Pairing:
    def __init__(self):
        self.reset_by_pose = {}
        self.pose_by_physics = {}
        self.opening_by_case = {}
        self.opening_differences = []

    def check(self, ep, path):
        r = ep.row
        pose = r["pose"]
        physics = r["initial_physics"]
        require(self.pose_by_physics.setdefault(physics, pose) == pose, path,
                "two distinct scheduled poses share a physical start")
        reset = (physics, r["initial_public"], ep.states[0].tolist(), r.get("noise_bank"))
        require(self.reset_by_pose.setdefault(pose, reset) == reset, path,
                "different reset physics, public inputs or motor noise at the same pose")
        key = (pose, r["wire"], r["condition"])
        opening = dict(first_motor_input=r["first_motor_input"], first_proposal=r["first_proposal"],
                       probe_states=ep.states[:11].tolist())
        reference, baseline = self.opening_by_case.setdefault(key, (path.name, opening))
        changed = [field for field in opening if opening[field] != baseline[field]]
        if changed:
            self.opening_differences.append(dict(episode="/".join(path.parts[-3:]),
                                                 relative_to=reference, fields=changed))


def read_stage(out, fitting, pairing):
    folder = out / ("fit" if fitting else "eval")
    poses, cells, offset = (range(8), CELLS[:3], 8) if fitting else (range(16, 24), CELLS, 20)
    expected = {case_name(p, w, c) for p, (w, c) in product(poses, cells)}
    require({p.name for p in folder.iterdir() if p.is_dir()} == expected, folder,
            "case directories differ from the registered split")
    panel = {}
    for pose, (wire, condition) in product(poses, cells):
        case = folder / case_name(pose, wire, condition)
        branches = {"support": (pose, None, 0)}
        for decode, grip in CHOICES:
            if not fitting or decode == wire:
                branches[branch_name(decode, grip)] = (pose + offset, decode, grip)
        if not fitting:
            branches["current"] = (pose + offset, None, 0)
        require({p.name for p in case.iterdir() if p.is_dir()} == branches.keys(), case,
                "missing or extra episode directories")
        episodes = {}
        for name, (physical_pose, decode, grip) in branches.items():
            ep = read_episode(case / name, physical_pose, wire, condition, decode, grip)
            pairing.check(ep, case / name)
            episodes[name] = ep
        panel[pose, wire, condition] = episodes
    return panel


def fitting_summary(panel, fixed_grip):
    by_cell, totals = {}, [0, 0]
    for wire, condition in CELLS[:3]:
        cases = [v for (_, w, c), v in panel.items() if (w, c) == (wire, condition)]
        successes = [sum(case[branch_name(wire, g)].success for case in cases) for g in (0, 1)]
        for g in (0, 1):
            totals[g] += successes[g]
        by_cell[f"w{wire:+d}_{condition}"] = dict(cases=len(cases), query_success_by_grip=successes)
    require(type(fixed_grip) is int and fixed_grip == int(totals[1] > totals[0]),
            "frozen.json", "fixed_grip disagrees with fitting query returns (tie selects 0)")
    return dict(cases=len(panel), independent_pose_pairs=8, cells=by_cell,
                query_success_by_grip=totals, fixed_grip=fixed_grip)


def current_grip(ep):
    if ep.row["choice_after"] is None:
        return None
    # This is the live callback's decision, distinct from the initial grip argument.
    grip = ep.row["chosen_grip"]
    require(grip in (0, 1), "current", "invalid recorded live choice")
    return int(grip)


def decisions(model, seed, key, panel, fixed_grip):
    pose, wire, condition = key
    case = panel[key]
    support = case["support"]
    swapped_grip = panel[pose, wire, "D17" if condition == "BASE" else "BASE"]["support"]
    swapped_wire = panel[pose, -wire, condition]["support"]

    def predict(kind, commands, states):
        sign, grip = model.predict(commands, states, kind, seed)
        require(sign in (-1, 1) and grip in (0, 1), (key, kind, seed), "invalid reader prediction")
        return int(sign), int(grip)

    current = case["current"]
    return {
        "factored": predict("factored", support.commands, support.states),
        "joint": predict("joint", support.commands, support.states),
        "cleared": predict("factored", support.commands[:0], support.states[:0]),
        "factored_probe": predict("factored", support.commands[:10], support.states[:11]),
        "analytic_probe": (support.row["probe_sign"], fixed_grip),
        "swap_grip": predict("factored", swapped_grip.commands, swapped_grip.states),
        "swap_wire": predict("factored", swapped_wire.commands, swapped_wire.states),
        "supplied": (wire, int(condition == "D17")),
        "current": (current.row["probe_sign"], current_grip(current)),
    }


def counts(prior_success, prior_controls, query, choice, wire, target_grip):
    sign, grip = choice
    return dict(query_success=int(query.success), support_success=int(prior_success),
                support_or_query_success=int(prior_success or query.success),
                support_controls=prior_controls, query_controls=query.controls,
                total_controls=prior_controls + query.controls, sign_correct=int(sign == wire),
                grip_correct=int(grip == target_grip), grip_decisions=int(grip is not None))


def score_case(model, seed, key, panel, fixed_grip):
    _, wire, condition = key
    selected = decisions(model, seed, key, panel, fixed_grip)
    case = panel[key]
    support = case["support"]
    branches, metrics = {}, {}
    for arm, choice in selected.items():
        name = "current" if arm == "current" else branch_name(*choice)
        branches[arm] = name
        query = case[name]
        target = int(condition == "D17")
        metrics[arm] = counts(support.success, support.controls, query, choice, wire, target)
        if arm in ("analytic_probe", "current"):
            n = 10 if arm == "analytic_probe" else 0
            metrics[arm + "_solo"] = counts(bool((support.rewards[:n] > 0).any()), n,
                                            query, choice, wire, target)
    return branches, metrics, selected


def summarize(keys, results):
    arms, paired = {}, {}
    for arm in results[keys[0]][1]:
        rows = [results[k][1][arm] for k in keys]
        sums = {field: sum(r[field] for r in rows) for field in rows[0]}
        sums["sign_accuracy"] = sums["sign_correct"] / len(rows)
        n_grip = sums["grip_decisions"]
        sums["grip_accuracy"] = sums["grip_correct"] / n_grip if n_grip else None
        arms[arm] = sums
    for other in COMPARATORS:
        table = dict(wins=0, losses=0, both_success=0, both_fail=0, sign_changed=0, grip_changed=0,
                     comparable_grips=0)
        for key in keys:
            _, metrics, selected = results[key]
            left, right = (metrics[a]["query_success"] for a in ("factored", other))
            outcome = (("both_fail", "losses"), ("wins", "both_success"))[left][right]
            table[outcome] += 1
            sign, grip = selected[other]
            table["sign_changed"] += int(sign != selected["factored"][0])
            if grip is not None:
                table["comparable_grips"] += 1
                table["grip_changed"] += int(grip != selected["factored"][1])
        paired[other] = table
    return dict(cases=len(keys), independent_pose_pairs=len({k[0] for k in keys}),
                arms=arms, paired_factored_vs=paired)


def analyze(out):
    pairing = Pairing()
    fit = read_stage(out, True, pairing)
    frozen = read_json(out / "frozen.json")
    fixed_grip = frozen["fixed_grip"]
    fitting = fitting_summary(fit, fixed_grip)
    panel = read_stage(out, False, pairing)
    weights = out / "weights"
    if "weights" in frozen:
        hashes = {str(p.relative_to(weights)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in weights.rglob("*") if p.is_file()}
        require(hashes and hashes == frozen["weights"], weights, "weights differ from frozen manifest")
    model = readers.load(weights)
    audit = {}
    for (p, w, c), episodes in panel.items():
        current = episodes["current"]
        audit[case_name(p, w, c)] = dict(
            support_pose=p, query_pose=p + 20, wire=w, condition=c,
            support=episodes["support"].outcome(),
            queries={name: ep.outcome() for name, ep in episodes.items() if name != "support"},
            current_choice=dict(sign=current.row["probe_sign"], grip=current_grip(current),
                                after_controls=current.row["choice_after"]),
            swap_grip_donor=case_name(p, w, "D17" if c == "BASE" else "BASE"),
            swap_wire_donor=case_name(p, -w, c), branches_by_seed={})
    by_seed = {}
    for seed in SEEDS:
        scored = {k: score_case(model, seed, k, panel, fixed_grip) for k in panel}
        for key, (branches, _, _) in scored.items():
            audit[case_name(*key)]["branches_by_seed"][seed] = branches
        by_seed[seed] = dict(held_out=summarize([k for k in panel if k[1:] == HELD_OUT], scored),
                             balanced_all_cells=summarize(list(panel), scored))
    return dict(fit=fitting, fit_episodes=sum(map(len, fit.values())),
                eval_episodes=sum(map(len, panel.values())), distinct_physical_starts=len(pairing.pose_by_physics),
                post_probe_differences=pairing.opening_differences,
                by_seed=by_seed, cases=audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", metavar="OUT", type=Path)
    report = analyze(parser.parse_args().out)
    print(json.dumps(report, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
