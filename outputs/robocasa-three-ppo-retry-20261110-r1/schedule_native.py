"""Compare training durations on development resets and evaluate the locked choice."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from readout import TASKS, curve_readout
from retry import save

STEPS = (144, 288, 576, 1152, 2304)
BY_WORKER = ((144, 288), (1152,), (2304,))
ARMS = ("memoryless", "own", "donor")


def wait(path):
    while not path.exists():
        time.sleep(5)


def call_retry(args, output, bank, steps, arms):
    output.mkdir()
    cmd = [sys.executable, "-Bu", str(Path(__file__).with_name("retry.py")), str(args.code),
           str(output), args.task, "--first-bank", str(bank), "--end-bank", str(bank+1),
           "--reuse", str(args.support), "--curves", str(args.curves),
           "--memoryless-step", str(steps["memoryless"]), "--dpt-step", str(steps["own"]), "--arms", *arms]
    subprocess.run(cmd, check=True)
    return output / f"bank-{bank:02d}"


def hits(folder, arm):
    panel = json.loads((folder / (arm+".json")).read_text())
    assert len(panel["episodes"]) == 8 and panel["reset"]["physical_state_exact"]
    return sum(r["success_once"] for r in panel["episodes"])


def choose(args):
    study = args.out.parent
    records = []
    for t in TASKS:
        for rank in range(3):
            path = study / f"{t}-{rank}" / "development.json"
            wait(path)
            records.extend(json.loads(path.read_text()))
    for seed in (991,1481,2029):
        wait(args.curves / f"seed-{seed}/held-out.json")
    curves = curve_readout(args.curves)["mean_by_epoch"]
    counts = {}
    for step in STEPS:
        subset = [r for r in records if r["step"] == step]
        assert {r["task"] for r in subset} == set(TASKS) and len(subset) == 3
        counts[step] = {arm:sum(r[arm] for r in subset) for arm in ARMS}
    losses = {int(r["epoch"]*288):r for r in curves}
    memoryless = max(STEPS, key=lambda s:(counts[s]["memoryless"],-losses[s]["memoryless"],-s))
    own = max(STEPS, key=lambda s:(counts[s]["own"], counts[s]["own"]-counts[s]["donor"],-losses[s]["own"],-s))
    return {"counts":counts, "memoryless":memoryless, "own":own}


def work(args):
    baserun = args.fixed / f"{args.task}-0" / "bank-00"
    prior = []
    if args.worker == 0:
        wait(baserun / "complete.json")
        prior.append({"task":args.task,"step":576,**{a:hits(baserun,a) for a in ARMS}})
    for step in BY_WORKER[args.worker]:
        wait(args.curves / "seed-991" / f"step-{step:04d}/held-out.json")
        panel = call_retry(args, args.out / f"step-{step:04d}", 0, {"memoryless":step, "own":step}, ARMS)
        prior.append({"task":args.task, "step":step, **{a:hits(panel,a) for a in ARMS}})
    save(args.out/"development.json", prior)
    lock = args.out.parent/"selected.json"
    if args.worker == 0 and args.task == TASKS[0]:
        save(lock, choose(args))
    wait(lock)
    decision = json.loads(lock.read_text())
    bank = args.worker+1
    reference = args.fixed / f"{args.task}-{bank}" / f"bank-{bank:02d}"
    wait(reference/"complete.json")
    arms = [a for a in ARMS if decision["own" if a=="donor" else a] != 576]
    panel = call_retry(args, args.out/"confirmation", bank, decision, arms) if arms else reference
    paths = {a:str((panel if a in arms else reference)/(a+".json")) for a in ARMS}
    for a in ("support","frozen"):
        paths[a] = str(reference/(a+".json"))
    save(args.out/"complete.json", {"task":args.task, "bank":bank, "sources":paths, "selection":decision})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("code",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("task",choices=TASKS)
    p.add_argument("worker",type=int,choices=range(3))
    p.add_argument("support",type=Path)
    p.add_argument("fixed",type=Path)
    p.add_argument("curves",type=Path)
    work(p.parse_args())
