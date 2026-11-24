"""Run registered starts x adapters on the frozen policy and compare them with the untreated branch."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

import numpy as np

from harness import Adapter, collect_support, rollout
from ur5e_motor import write_json


OUTPUTS = Path(__file__).resolve().parents[1]
LANES = "frozen-shift-*-20261124-r1"
RAW, REPEAT = "raw", "repeat"
CALIBRATION = 11
FAILURES = (14, 15, 17, 36, 37, 38)  # frozen pi0.5 timed out here before this program
STARTS = dict(
    development=[("Kinova3", layout) for layout in (*FAILURES, 16, 31, 34, 35, 39)],
    locked=[("Kinova3", layout) for layout in range(40, 48)],
    panda=[("Panda", layout) for layout in range(4)],
)
HOOKS = ("reset", "observation", "prompt", "noises", "choose", "command", "observe")


def case(arm, layout):
    return dict(arm=arm, suite="libero_goal", task_id=3, variant="BASE",
                init_state_id=layout, seed=42710300 + layout)


def select_starts(text):
    """Comma-separated group names or registered layout numbers, in the order given."""
    registered = {layout: (arm, layout) for group in STARTS.values() for arm, layout in group}
    chosen = []
    for token in text.split(","):
        if token in STARTS:
            members = STARTS[token]
        elif token.isdigit() and int(token) in registered:
            members = [registered[int(token)]]
        else:
            raise ValueError(f"Unregistered start {token!r}")
        chosen.extend(start for start in members if start not in chosen)
    return chosen


def group_of(arm, layout):
    if (arm, layout) in STARTS["development"]:
        return "development_failures" if layout in FAILURES else "development_successes"
    return next(name for name, members in STARTS.items() if (arm, layout) in members)


def discover(root=OUTPUTS, names=None):
    """Load `ADAPTERS` from every lane directory that has an adapter.py.

    An entry is an adapter instance, reused across episodes and cleared by its own
    `reset`, or a zero-argument callable that builds one per episode. Returns the
    factories by adapter name and a manifest of what each lane directory supplied.
    """
    factories, manifest = {}, []
    for folder in sorted(Path(root).glob(LANES)):
        path = folder / "adapter.py"
        entry = dict(lane=folder.name, adapters=[])
        manifest.append(entry)
        if not path.is_file():
            entry["skipped"] = "no adapter.py"
            continue
        if str(folder) not in sys.path:
            sys.path.append(str(folder))  # lane helpers stay importable inside hooks
        try:
            spec = importlib.util.spec_from_file_location("shift_" + folder.name.replace("-", "_"), path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            found = {}
            for item in module.ADAPTERS:
                built = not isinstance(item, type) and all(hasattr(item, hook) for hook in HOOKS)
                make = (lambda item=item: item) if built else item
                probe = item if built or isinstance(item, type) else make()
                name = getattr(probe, "name", None)
                missing = [hook for hook in HOOKS if not callable(getattr(probe, hook, None))]
                if missing or not isinstance(name, str):
                    raise TypeError(f"{name!r} lacks hooks {missing}")
                if name in (RAW, REPEAT) or name in factories or name in found:
                    raise ValueError(f"Adapter name {name!r} is reserved or already registered")
                found[name] = make
        except Exception:
            entry["skipped"] = traceback.format_exc()
            continue
        entry["adapters"] = list(found)
        factories.update(found)
    if names is not None:
        unknown = [name for name in names if name not in factories]
        if unknown:
            raise ValueError(f"Unknown adapters {unknown}; available {list(factories)}. Lanes: {manifest}")
        factories = {name: factories[name] for name in names}
    return factories, manifest


def verify(record, reference):
    tape = record["native_rewards"]
    if (not tape or len(tape) != record["controls"] or any(r not in (0, 1) for r in tape)
            or sum(tape) != record["native_return"] or bool(max(tape)) != record["success"]
            or not (record["terminated"] or record["truncated"])):
        raise ValueError("Incomplete native outcome")
    if record["reset"] != reference["reset"] or (
        record["untreated_opening_proposal_sha256"] != reference["untreated_opening_proposal_sha256"]
    ):
        raise ValueError("Branches disagree on reset, noise bank or opening proposal")


def run_start(sim, policy, support, query, factories, path=None):
    """Untreated, untreated repeat, then every adapter from one reset of one start."""
    cell = dict(case=query, group=group_of(query["arm"], query["init_state_id"]), branches={}, errors={})
    for name, make in ((RAW, Adapter), (REPEAT, Adapter), *factories.items()):
        try:
            record = rollout(sim, policy, support, query, make())
        except Exception:
            if name in (RAW, REPEAT):
                raise
            # A lane's fault is recorded and excluded. It is not a task failure.
            cell["errors"][name] = traceback.format_exc()
        else:
            verify(record, cell["branches"].get(RAW, record))
            cell["branches"][name] = dict(record, adapter=name)
        if path is not None:
            write_json(path, cell)
    return cell


def divergence(record, reference):
    """First control at which the consumed commands differ from the untreated branch."""
    if record["commands_sha256"] == reference["commands_sha256"]:
        return None
    control = 0
    for mine, theirs in zip(record["replans"], reference["replans"]):
        for a, b in zip(mine["executed_commands"], theirs["executed_commands"]):
            if a != b:
                return control
            control += 1
    return control


def interval(delta, draws):
    return np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()


def compare(cells):
    """Paired by layout. An adapter is scored on the starts where its episode completed."""
    names = list(dict.fromkeys(name for cell in cells for name in (*cell["branches"], *cell["errors"])))
    report = {}
    for name in names:
        paired = [cell for cell in cells if name in cell["branches"]]
        row = dict(successes=0, total=len(paired), layouts={},
                   errors=[cell["case"]["init_state_id"] for cell in cells if name in cell["errors"]])
        report[name] = row
        if not paired:
            continue
        outcome = {key: np.array([cell["branches"][key]["success"] for cell in paired]) for key in (name, RAW, REPEAT)}
        wins, base = outcome[name], outcome[RAW]
        draws = np.random.default_rng(42710300).integers(len(paired), size=(10000, len(paired)))
        delta = wins.astype(int) - base.astype(int)
        beyond = wins.astype(int) - outcome[REPEAT].astype(int)
        row.update(
            successes=int(wins.sum()), rescues=int((wins & ~base).sum()), losses=int((~wins & base).sum()),
            delta=float(delta.mean()), layout_interval95=interval(delta, draws),
            delta_from_repeat=float(beyond.mean()), repeat_layout_interval95=interval(beyond, draws),
        )
        for cell in paired:
            record = cell["branches"][name]
            row["layouts"][str(cell["case"]["init_state_id"])] = dict(
                success=record["success"], controls=record["controls"], candidates=record["candidates"],
                changed_replans=record["changed_replans"], policy_calls=record["policy_calls"],
                first_divergence=divergence(record, cell["branches"][RAW]),
            )
    return report


def summarize(cells):
    groups = {}
    for cell in cells:
        groups.setdefault(cell["group"], []).append(cell)
        if cell["group"].startswith("development"):
            groups.setdefault("development", []).append(cell)
    return {name: compare(members) for name, members in groups.items()}


def run(sim, policy, out, starts, factories):
    supports, cells, report = {}, [], {}
    for arm, layout in starts:
        if arm not in supports:
            supports[arm] = collect_support(sim, case(arm, CALIBRATION))
            write_json(out / f"{arm}-support.json", supports[arm])
        cells.append(run_start(sim, policy, supports[arm], case(arm, layout), factories, out / f"{arm}-{layout}.json"))
        report = summarize(cells)
        write_json(out / "summary.json", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--starts", default="development",
                   help="comma-separated groups (development, locked, panda) or registered layouts")
    p.add_argument("--adapters", help="comma-separated adapter names; empty runs only raw and repeat")
    p.add_argument("--list", action="store_true", help="resolve starts and adapters, then exit without a rollout")
    args = p.parse_args(argv)
    starts = select_starts(args.starts)
    names = None if args.adapters is None else [name for name in args.adapters.split(",") if name]
    if names is None and any(start in STARTS["locked"] for start in starts):
        raise ValueError("Locked starts run once for adapters named with --adapters")
    factories, manifest = discover(names=names)
    plan = dict(starts=starts, adapters=list(factories), lanes=manifest)
    print(json.dumps(plan, indent=2))
    if args.list:
        return
    from ur5e_motor import load_frozen_pi05
    from ur5e_sim_worker import UR5eSimulator

    write_json(args.out / "panel.json", plan)
    policy = load_frozen_pi05()
    # Renderer subprocess uses the qualified EGL device; the policy process retains CUDA.
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        sim = UR5eSimulator()
    with sim:
        print(json.dumps(run(sim, policy, args.out, starts, factories), indent=2))


if __name__ == "__main__":
    main()
