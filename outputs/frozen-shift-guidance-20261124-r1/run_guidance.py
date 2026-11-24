"""Development starts x {raw, repeat, guidance, time-aligned, random, same-layout} with layout-aware donors."""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "frozen-vla-proposal-shift-20261124-r1")]

import adapter
import donors
import panel
from harness import collect_support
from ur5e_motor import write_json


def run(sim, policy, out, starts, tapes=None):
    tapes = donors.library() if tapes is None else tapes
    supports, cells, report = {}, [], {}
    for arm, layout in starts:
        if arm not in supports:
            supports[arm] = collect_support(sim, panel.case(arm, panel.CALIBRATION))
            write_json(out / f"{arm}-support.json", supports[arm])
        built = {}

        def keep(name, make):
            def factory():
                built[name] = make()
                return built[name]
            return factory

        factories = {name: keep(name, make) for name, make in adapter.for_layout(layout, tapes).items()}
        cells.append(panel.run_start(sim, policy, supports[arm], panel.case(arm, layout), factories,
                                     out / f"{arm}-{layout}.json"))
        write_json(out / f"{arm}-{layout}-choices.json", {name: item.log for name, item in built.items()})
        report = panel.summarize(cells)
        write_json(out / "summary.json", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--starts", default="development")
    p.add_argument("--list", action="store_true", help="resolve starts and donors, then exit without a rollout")
    args = p.parse_args(argv)
    starts = panel.select_starts(args.starts)
    if any(start in panel.STARTS["locked"] for start in starts):
        raise ValueError("Locked starts run once through the shared panel with a locked adapter")
    tapes = donors.library()
    plan = dict(starts=starts, donors=[tape.name for tape in tapes],
                adapters={str(layout): list(adapter.for_layout(layout, tapes)) for _, layout in starts})
    print(json.dumps(plan, indent=2))
    if args.list:
        return
    from ur5e_motor import load_frozen_pi05
    from ur5e_sim_worker import UR5eSimulator

    write_json(args.out / "panel.json", plan)
    policy = load_frozen_pi05()
    # As in the shared panel: the renderer subprocess uses the qualified EGL device; the policy keeps CUDA.
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        sim = UR5eSimulator()
    with sim:
        print(json.dumps(run(sim, policy, args.out, starts, tapes), indent=2))


if __name__ == "__main__":
    main()
