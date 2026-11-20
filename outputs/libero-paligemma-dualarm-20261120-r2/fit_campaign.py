"""Freeze a point-in-time source data bank, fit the motor, and run each eligible stage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def link_bank_file(root, snapshot_root, name):
    relative = Path(name)
    source = (root / relative).resolve(strict=True)
    if relative.is_absolute() or ".." in relative.parts or not source.is_relative_to(root.resolve()):
        raise ValueError(f"Snapshot path outside source bank: {name}")
    dest = snapshot_root / relative
    dest.parent.mkdir(exist_ok=True, parents=True)
    os.link(source, dest)


def snapshot(path, out, fixed_offline, expected_sha=None):
    raw = (path / "index.json").read_bytes()
    if expected_sha and hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f"Collector index changed since campaign completion: {path}")
    data = json.loads(raw)
    if not set(data["bodies"]) <= {"Panda", "Sawyer"} or data["task_id"] != 5:
        raise ValueError("Expected native Goal 5 source episodes only")
    out.mkdir()
    if data.get("source_kind") == "native_bddl_sample" or "capture_sha256" in data:
        # The generated reader binds camera successes to headless search rows,
        # saved command drafts and layout files. Keep the collector index verbatim.
        if data["status"] not in ("complete", "shortfall"):
            raise ValueError("Generated and recovery banks must have a terminal index")
        names = {layout["state_file"] for layout in data.get("layouts", [])}
        for row in data["attempts"]:
            names.update(row[key] for key in ("search_file", "receipt") if key in row)
            if row["accepted"]:
                names.add(row["file"])
        for name in sorted(names):
            link_bank_file(path, out, name)
        (out / "index.json").write_bytes(raw)
        return
    rows = [dict(row) for row in data["attempts"] if row["accepted"] and
            not (fixed_offline and data.get("source_constructor") and row["pose"] in range(28, 32)
                 and row.get("candidate_number") != 0)]
    for row in rows:
        link_bank_file(path, out, row["file"])
    data.update(attempts=rows, original_bank=str(path), original_index_sha256=hashlib.sha256(raw).hexdigest())
    (out / "index.json").write_text(json.dumps(data, indent=2))
    print("immutable source snapshot",path,len(rows),"accepted native episodes", flush=True)


def frozen_inputs(fit):
    """Reuse the exact ordered bank paths and index bytes supplied to the parent fit."""
    manifest = fit / "manifest.json"
    raw = manifest.read_bytes()
    data = json.loads(raw)
    candidates = json.loads((fit / "candidates.json").read_bytes())
    if candidates["status"] != "complete" or candidates["fit_id"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("Parent snapshot requires a completed fit bound to its manifest")
    banks = [Path(p).resolve(strict=True) for p in data["demos"]]
    for bank, expected in zip(banks, data["contract"]["collector_index_sha256"], strict=True):
        if hashlib.sha256((bank / "index.json").read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed input snapshot from parent fit: {bank}")
    return banks


def campaign_inputs(path):
    from gen_campaign import TRAIN_SEEDS, WORKERS
    plan = json.loads(path.read_bytes())
    workers = plan["workers"]
    if plan["status"] not in ("complete", "shortfall") or len(workers) != WORKERS:
        raise ValueError("Expected the four terminal generated collectors")
    inputs = []
    for i, row in enumerate(workers):
        root = path.parent.resolve() / f"bank{i}"
        if (row["worker"] != i or row["bank"] != str(root) or row["index"] != str(root / "index.json")
                or row["status"] not in ("complete", "shortfall") or row["returncode"] not in (0, 2)
                or row["seed_start"] not in TRAIN_SEEDS or row["seed_stop"] - 1 not in TRAIN_SEEDS
                or not 0 < row["requested_layouts"] <= row["seed_stop"] - row["seed_start"]
                or (i and workers[i-1]["seed_stop"] != row["seed_start"])):
            raise ValueError("Invalid fixed worker assignment or collector receipt")
        index = json.loads((root / "index.json").read_bytes())
        if (index["seed_base"] != row["seed_start"] or index["max_seeds"] != row["seed_stop"] - row["seed_start"]
                or index["requested_layouts"] != row["requested_layouts"] or index["status"] != row["status"]
                or index["accepted_by_body"] != row["accepted_by_body"]):
            raise ValueError("Generated bank differs from its assigned seed budget or terminal counts")
        inputs.append((root, row["index_sha256"]))
    if not any(sum(row["accepted_by_body"].values()) for row in workers):
        raise ValueError("The generated campaign supplied zero successful demonstrations")
    return inputs


def run(file, *args, allow_gate=False):
    result = subprocess.run([sys.executable,"-Bu",str(HERE/file),*map(str,args)])
    if result.returncode and not (allow_gate and result.returncode == 2):
        result.check_returncode()
    return result.returncode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--banks",nargs="+",default=[],type=Path)
    p.add_argument("--generated-campaign",type=Path,help="terminal campaign.json from gen_campaign.py")
    p.add_argument("--base-fit",type=Path,help="prepend the immutable bank selections of this completed fit")
    p.add_argument("--base-step",type=int,help="initialize weights and normalization from this parent update")
    p.add_argument("--base-sha256",help="preselected SHA-256 for --base-step")
    p.add_argument("--steps",type=int,default=1200)
    p.add_argument("--shortlist",type=int,default=3)
    p.add_argument("--microbatch",type=int,default=4)
    p.add_argument("--stage1",type=Path)
    p.add_argument("--architecture",choices=("chunk_huber_v1","spatial_bins_v1"),default="chunk_huber_v1")
    p.add_argument("--fixed-offline",action="store_true")
    p.add_argument("--dev-only",action="store_true")
    p.add_argument("--train-only",action="store_true",help="finish after SFT and candidate publication")
    args=p.parse_args()
    if not args.banks and not args.generated_campaign:
        p.error("Supply --banks or --generated-campaign")
    if (args.base_step is None) != (args.base_sha256 is None) or (args.base_step is not None and not args.base_fit):
        p.error("--base-step requires --base-fit and --base-sha256")
    out=Path.cwd()
    parent=out/"data"
    parent.mkdir()
    banks=frozen_inputs(args.base_fit) if args.base_fit else []
    additional = [(path, None) for path in args.banks]
    if args.generated_campaign:
        additional.extend(campaign_inputs(args.generated_campaign))
    for i,(path, digest) in enumerate(additional):
        bank=parent/f"bank{i}"
        snapshot(path,bank,args.fixed_offline,digest)
        banks.append(bank)
    fit=out/"fit"
    init = []
    if args.base_step is not None:
        candidate = args.base_fit / "candidates" / f"step-{args.base_step:06d}.pt"
        init = ["--init-candidate", candidate, "--init-sha256", args.base_sha256]
    run("motor_train.py","train","--output",fit,"--demos",*banks,"--steps",args.steps,
        "--shortlist",args.shortlist,"--microbatch",args.microbatch,"--architecture",args.architecture,*init)
    if args.train_only:
        return
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
