"""Apply the previously verified archive workflow to the eight approved roots."""

import argparse
import os
from pathlib import Path

import archive_payloads as worker

worker.RUN = "outputs-retention-20261113-r1"
worker.WORK = Path("/tmp") / worker.RUN
worker.COLD = worker.BOS / "strong-plant-archives" / worker.RUN
worker.COMPACT_LIMIT = 0


def select(root):
    checkpoint = root.name == "global_step_1"
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        for name in sorted(files):
            if checkpoint or name.endswith((".pt", ".pt.gz")):
                yield Path(parent) / name


worker.select = select
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("phase", choices=("prepare", "release"))
args = parser.parse_args()
worker.require(os.path.ismount(worker.BOS), f"BOS is not mounted: {worker.BOS}")
(worker.prepare if args.phase == "prepare" else worker.release)("PLAN.json")
