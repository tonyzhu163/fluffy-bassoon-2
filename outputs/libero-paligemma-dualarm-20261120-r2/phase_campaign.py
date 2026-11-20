"""Collect fixed-route native teacher labels and fit one dual-arm source policy."""
import os
from pathlib import Path
import subprocess
import sys


def main():
    code = Path(__file__).resolve().parent
    out = Path.cwd()
    parent = out.parent
    fixed = out / "dense"
    subprocess.run([os.environ["CROSSARM_SIM_PYTHON"], "-Bu", str(code / "phase_capture.py"),
                    "--route", "straight", "--poses", "0-27", "--bodies", "Panda", "Sawyer",
                    "--out", str(fixed)], check=True)
    banks = [str(parent / bank / "bank") for bank in ("panda02", "sawyer01", "sawyer02", "panda03", "sawyer03")]
    common = ["--weights", os.environ["DUAL_WEIGHTS"]]
    entry = [sys.executable, "-Bu", str(code / "phase_fit.py")]
    subprocess.run([*entry, "fit", "--route", "straight", "--bodies", "Panda", "Sawyer",
                    "--banks", *banks, str(fixed), *common, "--output", str(out / "fit"),
                    "--scratch", str(out / "scratch"), "--steps", "2200", "--microbatch", "8"], check=True)
    subprocess.run([*entry, "check", "--banks", *banks, str(fixed), *common, "--checkpoint",
                    str(out / "fit/checkpoint.pt"), "--output", str(out / "check"),
                    "--scratch", str(out / "scratch-check")], check=True)


if __name__ == "__main__":
    main()
