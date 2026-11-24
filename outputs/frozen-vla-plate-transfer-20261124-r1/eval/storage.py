"""Reserve fixed PFS inode slots and durably copy immutable native cases from the job's local disk."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

MODES = ("stock", "panda", "mixed")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def slot(path):
    return path.with_name(f".{path.name}.slot")


def install(source, target, expected=None):
    """Single writer. Read the durable slot back before publishing its final name."""
    expected = expected or sha(source)
    if target.exists():
        need(sha(target) == expected, f"existing receipt differs at {target}")
        return
    reserved = slot(target)
    with source.open("rb") as src, reserved.open("r+b") as dst:
        dst.seek(0)
        for block in iter(lambda: src.read(1 << 20), b""):
            dst.write(block)
        dst.truncate()
        dst.flush()
        os.fsync(dst.fileno())
    need(sha(reserved) == expected, f"durable readback mismatch at {target}")
    reserved.rename(target)


def case_files(modes):
    return [f"{m}.{suffix}" for m in modes for suffix in ("h5", "json")] + ["complete.json"]


def reserve(base, pin, start, stop, lane, modes=MODES):
    base.mkdir()
    before = os.statvfs(base)
    for k in ("console.log", "launch.txt", "storage.txt"):
        (base / k).touch(exist_ok=False)
    for k in ("exit.txt", "gpu.json"):
        slot(base / k).touch(exist_ok=False)
    for shard in (range(4) if lane == "all" else [int(lane)]):
        root = base / f"shard{shard}"
        root.mkdir()
        for k in ("console.log", "run.txt"):
            (root / k).touch(exist_ok=False)
        for k in ("before.json", "after.json"):
            slot(root / k).touch(exist_ok=False)
        for i in range(start + shard, stop, 4):
            case = root / f"case-{i:05}"
            case.mkdir()
            for k in case_files(modes):
                slot(case / k).touch(exist_ok=False)
    audit = base / "audit"
    audit.mkdir()
    for k in (f"look-{stop}.json", f"readback-{stop}.json"):
        slot(audit / k).touch(exist_ok=False)
    after = os.statvfs(base)
    facts = dict(pin=pin, start=start, stop=stop, lane=lane, slots=sum(1 for _ in base.rglob("*.slot")),
                 entries=sum(1 for _ in base.rglob("*"))+2, host=os.uname().nodename,
                 inode_free_before=before.f_favail, inode_free_after=after.f_favail,
                 available_bytes=after.f_bavail*after.f_frsize)
    with (base / "reservation.json").open("x") as stream:
        json.dump(facts, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    need(json.loads((base / "reservation.json").read_bytes()) == facts, "reservation readback")
    print(json.dumps(facts), flush=True)


def publish(src, dst):
    report = json.loads((src / "complete.json").read_bytes())
    for m, fingerprint in report["receipts"].items():
        need(m in MODES and sha(src / f"{m}.json") == fingerprint, "invalid local case receipt")
        episode = json.loads((src / f"{m}.json").read_bytes())
        install(src / f"{m}.h5", dst / f"{m}.h5", episode["tape_sha256"])
        install(src / f"{m}.json", dst / f"{m}.json", fingerprint)
    install(src / "complete.json", dst / "complete.json")


def run(local, remote, cmd):
    local.mkdir()
    # Identical metadata allows the pinned panel's exact-contract resume check.
    (local / "run.txt").write_bytes((remote / "run.txt").read_bytes())
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")

    def forward(signum, frame):
        if p.poll() is None:
            p.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
            if line.startswith("sealed case "):
                i = int(line.split()[2])
                name = f"case-{i:05}"
                install(local / "before.json", remote / "before.json")
                publish(local / name, remote / name)
                print("durable case", i, flush=True)
        rc = p.wait()
        # Admission may reuse a prior triplet; it prints no new rollout timing.
        for path in sorted(local.glob("case-*/complete.json")):
            name = path.parent.name
            if not (remote / name / "complete.json").exists():
                publish(path.parent, remote / name)
        for k in ("before.json", "after.json"):
            if (local / k).is_file():
                install(local / k, remote / k)
        return rc
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="op", required=True)
    q = sub.add_parser("reserve")
    q.add_argument("base", type=Path)
    q.add_argument("pin")
    q.add_argument("start", type=int)
    q.add_argument("stop", type=int)
    q.add_argument("lane", choices=["all", "0", "1", "2", "3"])
    q = sub.add_parser("run")
    q.add_argument("local", type=Path)
    q.add_argument("remote", type=Path)
    q.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.op == "reserve":
        reserve(args.base, args.pin, args.start, args.stop, args.lane)
    else:
        sys.exit(run(args.local, args.remote, args.cmd))
