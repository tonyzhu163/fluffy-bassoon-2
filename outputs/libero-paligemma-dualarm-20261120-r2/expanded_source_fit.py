"""Continue a source PaliGemma fit with completed generated banks and corrected native suffixes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


PARENTS = {
    "sft03": ("spatial_bins_v1", "824c695f68dc5b8edaa37dae4e79c9f24756de39f5fddb420deca1c44cb1cf1a"),
    "sft02": ("chunk_huber_v1", "77d18c386b2cc176252cfb68449e4415748382f554046ca50367ed4592676d9f"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", choices=PARENTS, required=True)
    p.add_argument("--banks", nargs="+", choices=("gendemos01", "gendemos02", "gendemos03"), required=True)
    p.add_argument("--steps", type=int, required=True)
    args = p.parse_args()
    if args.steps <= 0 or len(args.banks) != len(set(args.banks)):
        p.error("Use positive updates and distinct collection campaigns")
    campaign = Path.cwd().parent
    arch, digest = PARENTS[args.parent]
    inputs = [campaign / "recover01/bank"]
    for name in args.banks:
        root = campaign / name / "banks"
        spec = json.loads((root / "campaign.json").read_text())
        if spec["status"] not in ("complete", "shortfall") or len(spec["workers"]) != 4:
            raise RuntimeError(f"Collection is incomplete at {root}")
        for i, worker in enumerate(spec["workers"]):
            src = root / f"bank{i}"
            index = json.loads((src / "index.json").read_text())
            if (worker["worker"] != i or worker["returncode"] not in (0, 2)
                    or Path(worker["bank"]).resolve() != src.resolve()
                    or index["status"] not in ("complete", "shortfall")
                    or index["seed_base"] != worker["seed_start"]):
                raise ValueError(f"Unexpected bank assignment at {src}")
            inputs.append(src)
    entry = Path(__file__).with_name("fit_campaign.py")
    subprocess.run([sys.executable, "-Bu", str(entry), "--train-only", "--base-fit",
                    str(campaign / args.parent / "fit"), "--base-step", "6400", "--base-sha256", digest,
                    "--banks", *map(str, inputs), "--architecture", arch, "--steps", str(args.steps),
                    "--microbatch", "8", "--shortlist", "3"], check=True)


if __name__ == "__main__":
    main()
