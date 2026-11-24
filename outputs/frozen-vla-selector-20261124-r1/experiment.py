"""Collect paired native interventions, fit source heads, and evaluate on Kinova3."""
import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np

from learner import fit
from rollout import MENU, collect_support, rollout
from ur5e_motor import load_frozen_pi05, write_json
from ur5e_sim_worker import UR5eSimulator


SOURCE = ("Panda", "UR5e")
TARGET = "Kinova3"
HEADS = ("factorized", "generic")
BOUNDARIES = (45, 60, 75, 90)


def case(arm, layout):
    return dict(arm=arm, suite="libero_goal", task_id=3, variant="BASE",
                init_state_id=layout, seed=42710300 + layout)


def verify(record, reference):
    tape = record["native_rewards"]
    if (not tape or len(tape) != record["controls"] or any(r not in (0, 1) for r in tape)
            or sum(tape) != record["native_return"] or bool(max(tape)) != record["success"]
            or not (record["terminated"] or record["truncated"])):
        raise ValueError("Incomplete native outcome")
    if record["reset"] != reference["reset"] or record["opening_proposal_sha256"] != reference["opening_proposal_sha256"]:
        raise ValueError("Branches disagree on reset, noise bank or opening proposal")


def paired_case(sim, policy, support, query, boundary, models):
    branches = {}
    modes = [("continue", "continue", False), ("repeat", "continue", False),
             ("correct", "correct", False), ("cue", "cue", False)]
    if models:
        modes.append(("correct_online", "correct", True))
    for head in models:
        modes.extend([(head, head, False), (head + "_online", head, True)])
    for name, mode, repeated in modes:
        trial = rollout(sim, policy, support, query, boundary, mode, models.get(mode), repeated)
        reference = branches.get("continue", trial)
        record = trial["record"]
        verify(record, reference["record"])
        if (trial["prefix"] != reference["prefix"] or trial["fork"] != reference["fork"]
                or record["native_rewards"][:boundary] != reference["record"]["native_rewards"][:boundary]):
            raise ValueError("Intervention starts from an unmatched prefix, state or proposal")
        if trial["fork"] is None and (record["controls"] > boundary
                                      or record["native_rewards"] != reference["record"]["native_rewards"]):
            raise ValueError("Missing fork or unmatched pre-fork termination")
        branches[name] = trial
    return dict(case=query, boundary=boundary, branches=branches)


def training_row(cell):
    trials = cell["branches"]
    fork = trials["continue"]["fork"]
    if fork is None:
        return None  # keep early terminals in evaluation, omit nonexistent choices from fitting
    return dict(case=cell["case"], boundary=cell["boundary"], motion=fork["motion"],
                progress=fork["progress"], outcomes={k: trials[k]["record"]["success"] for k in MENU})


def summarize(cells):
    # One independent layout per row, all branches paired. Report the small panel in full.
    outcomes = {name: np.array([c["branches"][name]["record"]["success"] for c in cells])
                for name in cells[0]["branches"]}
    outcomes["one_choice_hindsight"] = np.any([outcomes[k] for k in MENU], axis=0)
    base = outcomes["continue"]
    bootstrap = np.random.default_rng(42710300).integers(len(base), size=(10000, len(base)))
    summary = {}
    for name, wins in outcomes.items():
        delta = wins.astype(int) - base.astype(int)
        summary[name] = dict(successes=int(wins.sum()), total=len(base),
                             rescues=int((wins & ~base).sum()), losses=int((~wins & base).sum()),
                             delta=float(delta.mean()),
                             layout_interval95=np.quantile(delta[bootstrap].mean(axis=1), [.025, .975]).tolist())
    if "factorized" in outcomes:
        for suffix in ("", "_online"):
            difference = outcomes["factorized" + suffix].astype(int) - outcomes["generic" + suffix].astype(int)
            summary["factorized_minus_generic" + suffix] = dict(delta=float(difference.mean()),
                layout_interval95=np.quantile(difference[bootstrap].mean(axis=1), [.025, .975]).tolist())
    return summary


def run(sim, policy, out):
    rows, models, reports = [], {}, {}
    for arm in (*SOURCE, TARGET):
        if arm == TARGET:
            # Both heads are frozen and persisted before creating the first target reset.
            for head in HEADS:
                models[head] = fit(rows, head)
            write_json(out / "selectors.json", models)
        support = collect_support(sim, case(arm, 11))
        write_json(out / f"{arm}-support.json", support)
        cells = []
        layouts = range(36, 40) if arm == TARGET else range(10)
        for index, layout in enumerate(layouts):
            query = case(arm, layout)
            cell = paired_case(sim, policy, support, query, BOUNDARIES[index % len(BOUNDARIES)], models)
            write_json(out / f"{arm}-{layout}.json", cell)
            cells.append(cell)
            if arm in SOURCE:
                row = training_row(cell)
                if row is not None:
                    rows.append(row)
        reports[arm] = summarize(cells)
        write_json(out / "summary.json", reports)
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    policy = load_frozen_pi05()
    # Renderer subprocess uses the qualified EGL device; the policy process retains CUDA.
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        sim = UR5eSimulator()
    with sim:
        print(json.dumps(run(sim, policy, args.out), indent=2))


if __name__ == "__main__":
    main()
