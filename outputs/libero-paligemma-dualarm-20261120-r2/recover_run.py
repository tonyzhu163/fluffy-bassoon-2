"""Capture a frozen source motor and collect camera-verified teacher recoveries in one job."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--sha",required=True)
    parser.add_argument("--banks",type=Path,nargs="+",required=True)
    parser.add_argument("--count",type=int,default=24)
    args=parser.parse_args()
    code=Path(__file__).resolve().with_name("recover.py")
    root=Path.cwd()
    capture=root/"prefixes"
    subprocess.run([sys.executable,"-Bu",str(code),"capture","--checkpoint",str(args.checkpoint),
                    "--checkpoint-sha256",args.sha,"--count",str(args.count),"--out",str(capture),
                    "--bank",*map(str,args.banks)],check=True)
    receipt=capture/"capture.json"
    digest=hashlib.sha256(receipt.read_bytes()).hexdigest()
    subprocess.run([os.environ["CROSSARM_SIM_PYTHON"],"-Bu",str(code),"teach","--capture",str(receipt),
                    "--capture-sha256",digest,"--out",str(root/"bank")],check=True)


if __name__=="__main__":
    main()
