"""Fit PaliGemma to measured source plate locations, wrist waypoints and native controls."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank",required=True,type=Path)
    p.add_argument("--official",required=True,type=Path)
    p.add_argument("--steps",default=5200,type=int)
    args=p.parse_args()
    here=Path(__file__).resolve().parent
    output=Path.cwd()
    extra=output/"fixed-panda"
    subprocess.run([os.environ["CROSSARM_SIM_PYTHON"],"-Bu",str(here/"waypoint_extra.py"),
                    "--out",str(extra),"--bank",str(args.official)],check=True)
    entry=[sys.executable,"-Bu",str(here/"waypoint_train.py")]
    common=["--output",str(output/"fit")]
    subprocess.run([*entry,"train",*common,"--demos",str(args.bank),str(extra),"--steps",str(args.steps),
                    "--microbatch","8"],check=True)
    dev=subprocess.run([*entry,"dev",*common])
    if dev.returncode not in (0,2):
        dev.check_returncode()


if __name__=="__main__":
    main()
