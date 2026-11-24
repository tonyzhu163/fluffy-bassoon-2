"""Install the immutable analysis and archived helper paths after native cpu.sh sets its environment."""
import os
from pathlib import Path
import runpy
import sys

here = Path(__file__).resolve().parent
sys.path[:0] = [str(here), str(here.parent / "frozen-vla-reachability-20261123-r1"), os.environ["POSE_PANEL"]]
mode = sys.argv.pop(1)
assert mode in ("replay", "solve", "readback", "treatment")
runpy.run_path(str(here / f"{mode}.py"), run_name="__main__")
