"""Prepare the shared development panel, then score 50 layouts on each source arm."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from large_panel_protocol import read, require


PANEL = Path(__file__).resolve().with_name("large_panel.py")


def run_stage(python, stage, *args):
    command = [python, "-Bu", str(PANEL), stage, *map(str, args)]
    print("Starting", stage, flush=True)
    subprocess.run(command, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Run one job per fit. To resume, use a fresh launcher attempt with "
               "the same pin, panel root, fit and step. Example: bash run.sh "
               '"$PIN" bigbase03 vlm large_panel_campaign.py --fit sft01/fit --step 1600',
    )
    parser.add_argument("--fit", type=Path, required=True,
                        help="completed motor fit, absolute or relative to the campaign directory")
    parser.add_argument("--step", type=int, required=True, help="registered candidate update")
    parser.add_argument("--root", type=Path, default=Path("large-panel-v3"),
                        help="shared panel directory, relative to the campaign if not absolute (default: %(default)s)")
    parser.add_argument("--sim", default=os.environ.get("CROSSARM_SIM_PYTHON"),
                        help="native simulator Python (defaults to CROSSARM_SIM_PYTHON)")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8,
                        help="simulator processes per scoring job (default: %(default)s)")
    args = parser.parse_args(argv)
    if args.step < 1:
        parser.error("--step must be positive")
    if not args.sim:
        parser.error("native preparation needs --sim or CROSSARM_SIM_PYTHON")
    os.environ["CROSSARM_SIM_PYTHON"] = args.sim
    for key in ("DUAL_OUTPUT", "DUAL_WEIGHTS", "MOTOR_CODE_REVISION"):
        if not os.environ.get(key):
            parser.error(f"missing {key}. Launch with run.sh PIN ATTEMPT vlm large_panel_campaign.py")

    campaign = Path(os.environ["DUAL_OUTPUT"]).resolve().parent
    root = (campaign / args.root).resolve()
    fit = (campaign / args.fit).resolve(strict=True)
    pin = os.environ["MOTOR_CODE_REVISION"]
    manifest = root / "dev" / "manifest.json"
    common = ["--root", root, "--phase", "dev"]
    # prepare owns .prepare.lock and validates or resumes the shared panel under it.
    run_stage(os.environ["CROSSARM_SIM_PYTHON"], "prepare", *common, "--fit", campaign / "sft01/fit")
    panel = read(manifest)
    require(panel["revision"] == pin
            and panel["phase"] == "dev" and panel["size"] == 50 and len(panel["layouts"]) == 50,
            "Expected 50 prepared development layouts at the launch revision")
    score = [*common, "--fit", fit, "--step", args.step, "--stride", 5]
    run_stage(sys.executable, "source", *score, "--arm", "both", "--workers", args.workers)
    run_stage(sys.executable, "aggregate", *score)


if __name__ == "__main__":
    main()
