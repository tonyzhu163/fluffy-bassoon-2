"""Freeze a point-in-time source data bank, fit the motor, and run each eligible stage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def snapshot(path, out, fixed_offline):
    raw = (path / "index.json").read_bytes()
    data = json.loads(raw)
    if not set(data["bodies"]) <= {"Panda", "Sawyer"} or data["task_id"] != 5:
        raise ValueError("Expected native Goal 5 source episodes only")
    out.mkdir()
    rows = [dict(row) for row in data["attempts"] if row["accepted"] and
            not (fixed_offline and data.get("source_constructor") and row["pose"] >= 28
                 and row.get("candidate_number") != 0)]
    for row in rows:
        f = Path(row["file"])
        if f.is_absolute() or not (path / f).resolve().is_relative_to(path.resolve()):
            raise ValueError("Snapshot path outside source bank")
        dest = out / f
        dest.parent.mkdir(exist_ok=True, parents=True)
        os.link(path / f, dest)
    data.update(attempts=rows, original_bank=str(path), original_index_sha256=hashlib.sha256(raw).hexdigest())
    (out / "index.json").write_text(json.dumps(data, indent=2))
    print("immutable source snapshot",path,len(rows),"accepted native episodes", flush=True)


def run(file, *args, allow_gate=False):
    result = subprocess.run([sys.executable,"-Bu",str(HERE/file),*map(str,args)])
    if result.returncode and not (allow_gate and result.returncode == 2):
        result.check_returncode()
    return result.returncode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--banks",nargs="+",required=True,type=Path)
    p.add_argument("--steps",type=int,default=1200)
    p.add_argument("--shortlist",type=int,default=3)
    p.add_argument("--microbatch",type=int,default=4)
    p.add_argument("--stage1",type=Path)
    p.add_argument("--architecture",choices=("chunk_huber_v1","spatial_bins_v1"),default="chunk_huber_v1")
    p.add_argument("--fixed-offline",action="store_true")
    p.add_argument("--dev-only",action="store_true")
    args=p.parse_args()
    out=Path.cwd()
    parent=out/"data"
    parent.mkdir()
    banks=[]
    for i,path in enumerate(args.banks):
        bank=parent/f"bank{i}"
        snapshot(path,bank,args.fixed_offline)
        banks.append(bank)
    fit=out/"fit"
    run("motor_train.py","train","--output",fit,"--demos",*banks,"--steps",args.steps,
        "--shortlist",args.shortlist,"--microbatch",args.microbatch,"--architecture",args.architecture)
    run("motor_train.py","dev","--output",fit,allow_gate=True)
    dev=json.loads((fit/"dev/report.json").read_text())
    if args.dev_only:
        print("Source development completed",dev.get("scores"),flush=True)
        return
    if not dev.get("source_gate"):
        print("Source development stopped before confirmation",dev.get("scores"),flush=True)
        return
    run("motor_train.py","confirm","--output",fit,"--seal-dir",out.parent/"source-confirm-seal",allow_gate=True)
    report=json.loads((fit/"confirm/report.json").read_text())
    if not report.get("source_gate"):
        print("Confirmation stopped before UR5e",report.get("scores"),flush=True)
        return
    common=["--motor-output",fit/"selected","--confirmation",fit/"confirm"]
    stage1=["--stage1",args.stage1] if args.stage1 else []
    run("transfer.py","fit",*common,*stage1,"--demos",*banks,"--out",out/"joint")
    run("transfer.py","evaluate",*common,"--fit",out/"joint","--out",out/"ur5e")


if __name__=="__main__":
    main()
