"""Read-only observer of one scheduled qualification; receipts live next to its exports."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time


def read(path):
    try:
        return json.loads(path.read_bytes())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def snapshot(root, arms=("dpt","dit")):
    probe = read(root / "GPU-PROBE.json")
    record = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  job=root.name, allocation_probe=probe.get("status"), methods={})
    for method in arms:
        progress = read(root / f"{method}-PROGRESS.json") or read(root / f"{method}-progress.json")
        final = read(root / f"{method}-RESULT.json")
        journal = read(root / f"{method}-current-journal.json")
        checkpoint = read(root / f"{method}-OPERATIONAL-CHECKPOINT.json")
        exit_path = root / f"{method}-exit.txt"
        rows = (final or progress).get("results",[])
        record["methods"][method] = dict(
            started=(root / f"{method}-start.txt").is_file(),
            latest_bank_journal=journal.get("attempt"),
            latest_journal_resolved_policy_controls=sum(row["executed_or_conservative"] for row in
                journal.get("calls",[]) if row["resolved"] and row["phase"]=="policy_controls"),
            complete_waves=[dict(task=r["task"], mode=r["mode"], cases=r["denominators"],
                simulator_episodes=r["denominators"]*6, frozen=r["native_parameters_frozen"],
                edited_layers=len(r["fitted_deployment_check"]["edited_KV_layers"]),
                fitted_command_delta=r["fitted_deployment_check"]["fitted_vs_native_command_max_deltas"])
                for r in rows],
            completed_entire_qualification=bool(final.get("completed")),
            checkpoint_next_bank=checkpoint.get("next_bank"), exported_exit=exit_path.read_text().strip() if exit_path.exists() else None)
    return record


def verify_exports(root, record):
    for method, info in record["methods"].items():
        receipt = root / (method+".sha256")
        if not receipt.is_file() or not info["exported_exit"]:
            continue
        expected, name = receipt.read_text().split()
        assert name == method+".tar"
        h=hashlib.sha256()
        with (root / name).open("rb") as f:
            for buf in iter(lambda:f.read(8*1024*1024),b""):
                h.update(buf)
        info["independent_archive_hash_matches"] = h.hexdigest()==expected


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path)
    p.add_argument("--until",type=int,default=0,help="UTC epoch for bounded observation; zero prints once")
    p.add_argument("--pilot", action="store_true", help="Observe DIT full, DPT reset, DIT reset")
    a=p.parse_args()
    arms=("dit-full","dpt-reset","dit-reset") if a.pilot else ("dpt","dit")
    if not a.until:
        print(json.dumps(snapshot(a.root,arms),indent=2))
    else:
        out=a.root/"cpu-observations";out.mkdir(exist_ok=False)
        while True:
            observation=snapshot(a.root,arms)
            done=all(v["exported_exit"] for v in observation["methods"].values())
            if done or time.time()>=a.until:
                verify_exports(a.root,observation)
                (out/"FINAL.json").write_text(json.dumps(observation,indent=2))
                break
            (out/f"{int(time.time())}.json").write_text(json.dumps(observation,indent=2))
            time.sleep(min(60,max(1,a.until-time.time())))
