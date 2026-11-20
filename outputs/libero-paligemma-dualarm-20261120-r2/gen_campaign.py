"""Launch four independent native source collectors on one allocated GPU."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
WORKERS = 4
TRAIN_SEEDS = range(300000, 400000)
EVAL_SEEDS = (range(740000, 750000), range(840000, 850000))


def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def allocations(total):
    base, extra = divmod(total, WORKERS)
    return [base + (i < extra) for i in range(WORKERS)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=TRAIN_SEEDS.start)
    parser.add_argument("--layouts", type=int, required=True, help="Total requested draws across four workers")
    parser.add_argument("--max-seeds", type=int, help="Total consecutive seed budget (default: --layouts)")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--bodies", nargs="+", choices=("Panda", "Sawyer"), default=["Panda", "Sawyer"])
    args = parser.parse_args()
    budget = args.layouts if args.max_seeds is None else args.max_seeds
    if not WORKERS <= args.layouts <= budget or args.candidates < 1 or not 1 <= args.max_steps <= 300:
        parser.error("Require 4 <= layouts <= max-seeds, positive candidates and 1–300 steps")
    if len(set(args.bodies)) != len(args.bodies):
        parser.error("Repeated source body")
    first, stop = args.seed_start, args.seed_start + budget
    if first not in TRAIN_SEEDS or stop - 1 not in TRAIN_SEEDS:
        parser.error("The complete seed budget must lie inside [300000,400000)")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    workers = []
    for i, (count, span) in enumerate(zip(allocations(args.layouts), allocations(budget), strict=True)):
        bank = out / f"bank{i}"
        command = [sys.executable, "-Bu", str(HERE / "generated_collect.py"), "--out", str(bank),
                   "--seed-start", str(first), "--layouts", str(count), "--max-seeds", str(span),
                   "--candidates", str(args.candidates), "--max-steps", str(args.max_steps), "--bodies", *args.bodies]
        workers.append(dict(worker=i, bank=str(bank), log=f"worker{i}.log", command=command,
                            seed_start=first, seed_stop=first + span, requested_layouts=count))
        first += span
    path = out / "campaign.json"
    plan = dict(status="planned", training_seeds=[TRAIN_SEEDS.start, TRAIN_SEEDS.stop],
                eval_exclusions=[[r.start, r.stop] for r in EVAL_SEEDS], workers=workers)
    write(path, plan)
    processes = []
    try:
        with ExitStack() as files:
            for row in workers:
                log = files.enter_context((out / row["log"]).open("x"))
                processes.append(subprocess.Popen(row["command"], stdout=log, stderr=subprocess.STDOUT))
            plan["status"] = "running"
            write(path, plan)
            for row, process in zip(workers, processes, strict=True):
                row["returncode"] = process.wait()
                index = Path(row["bank"]) / "index.json"
                if index.is_file():
                    raw = index.read_bytes()
                    row.update(index=str(index), index_sha256=hashlib.sha256(raw).hexdigest())
                    receipt = json.loads(raw)
                    for key in ("status", "admitted_layouts", "accepted_by_body", "paired_positive_groups"):
                        row[key] = receipt.get(key)
                write(path, plan)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
    complete = all(r["returncode"] == 0 and r.get("status") == "complete" for r in workers)
    shortfall = all(r["returncode"] in (0, 2) and r.get("status") in ("complete", "shortfall") for r in workers)
    plan["status"] = "complete" if complete else "shortfall" if shortfall else "failed"
    write(path, plan)
    return 0 if complete else 2 if shortfall else 1


if __name__ == "__main__":
    raise SystemExit(main())
