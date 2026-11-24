"""Lock an outcome-blind selection and conservatively charge scheduler reservations."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import time


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def write(path, data):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def selection(path):
    raw = path.read_bytes()
    plan = json.loads(raw)
    require(set(plan) == {"batches", "remaining_wall_seconds", "remaining_gpu_seconds",
            "startup_and_scheduler_seconds", "score_seconds_per_batch", "pair_seconds",
            "ordinary_continuation", "budget_evidence"},
            "Unexpected selection fields")
    batches = plan["batches"]
    require(isinstance(batches, list) and batches and all(type(b) is int and 54 <= b < 70 for b in batches)
            and sorted(set(batches)) == batches, "Use explicit ascending unique enrolled batch IDs 54–69")
    for field in ("remaining_wall_seconds", "remaining_gpu_seconds", "startup_and_scheduler_seconds",
                  "score_seconds_per_batch", "pair_seconds"):
        require(type(plan[field]) is int and plan[field] > 0, f"Invalid positive seconds: {field}")
    for field in ("ordinary_continuation", "budget_evidence"):
        require(isinstance(plan[field], str) and plan[field].strip(), f"Missing {field} record")
    limit = min(plan["remaining_wall_seconds"], plan["remaining_gpu_seconds"])
    reserve = plan["startup_and_scheduler_seconds"] + len(batches) * plan["score_seconds_per_batch"]
    cap = (limit - reserve) // (len(batches) * plan["pair_seconds"])
    require(cap >= 1, "Budget cannot cover even one reset pair per assigned batch and baseline scoring")
    return raw, dict(plan, sha256=sha256(raw).hexdigest(), total_seconds=limit, pair_cap=cap)


def census(store):
    base = store / "batches"
    require(base.is_dir() and not base.is_symlink(), "Missing original scored batch store")
    paths = list(base.iterdir())
    require(all(re.fullmatch(r"batch[0-9]{3}", p.name) and int(p.name[5:]) < 70 and
                p.is_dir() and not p.is_symlink() and (p / "producer.json").is_file() for p in paths),
            "Unexpected sealed batch directory or missing producer")
    sealed = {int(path.name[5:]) for path in paths}
    require(set(range(54)) <= sealed, "Original 54 sealed batch paths differ")
    return sealed


def reserve(store, ledger, requested, attempt_id, allocation, payload, pin):
    """The caller holds the original scored/active.lock. No partial outcome files are read."""
    raw, plan = selection(requested)
    require(0 < payload < allocation and allocation - payload >= plan["startup_and_scheduler_seconds"],
            "The launch needs the locked scheduler/startup headroom")
    require(re.fullmatch(r"retry[0-9]+", attempt_id), "Use a fresh retryNN attempt ID")
    attempt = store / "attempts" / attempt_id
    require(not os.path.lexists(attempt), "Attempt ID already used")
    sealed = census(store)
    ledger.mkdir(parents=True, exist_ok=True)
    pinned = ledger / "selection.json"
    reservations = sorted(ledger.glob("retry*.json")) if pinned.is_file() else []
    if pinned.is_file():
        require(pinned.read_bytes() == raw, "The fallback selection is already locked")
    receipts = [read(path) for path in reservations]
    require(all(row["pin"] == pin for row in receipts), "The locked fallback operational pin differs")
    require(not any((store / "attempts" / row["attempt"] / "paired-reset-error.json").exists()
                    for row in receipts), "An earlier structural halt requires parent review")
    spent = sum(row["allocation_seconds"] for row in receipts)
    pending = set(plan["batches"]) - sealed
    if not pending:
        return "sealed"
    if all(any((store / "attempts" / row["attempt"] / f"batch{b:03d}").exists()
               for row in receipts) for b in pending):
        print(json.dumps(dict(quarantined_batches=sorted(pending), quarantined_starts=4 * len(pending))))
        return "quarantined"
    require(spent + allocation <= plan["total_seconds"], "Cumulative scheduler reservations exceed the locked budget")
    if not pinned.exists():
        with pinned.open("xb") as stream:
            stream.write(raw)
    record = dict(plan, pin=pin, attempt=attempt_id, allocation_seconds=allocation, payload_seconds=payload,
                  prior_attempts=[row["attempt"] for row in receipts], prior_reserved_seconds=spent,
                  payload_deadline_monotonic=time.monotonic() + payload)
    # Reserve before starting work. A crash here may spend conservatively, but cannot admit duplicate work.
    write(ledger / f"{attempt_id}.json", record)
    attempt.mkdir()
    write(attempt / "paired-reset-selection.json", record)
    return record


def statuses(store, plan):
    sealed = census(store)
    attempts = store / "attempts"
    return {
        b: ("sealed_parked" if b in sealed else "prior_attempt_quarantine"
            if any((attempts / a / f"batch{b:03d}").exists() for a in plan["prior_attempts"]) else "unstarted")
        for b in plan["batches"]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("attempt")
    parser.add_argument("allocation", type=int)
    parser.add_argument("payload", type=int)
    parser.add_argument("pin")
    args = parser.parse_args()
    record = reserve(args.store, args.ledger, args.selection, args.attempt, args.allocation, args.payload, args.pin)
    raise SystemExit(10 if record == "sealed" else 3 if record == "quarantined" else 0)
