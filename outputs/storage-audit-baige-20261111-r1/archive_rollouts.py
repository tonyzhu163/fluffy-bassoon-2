"""Archive completed RoboTwin rollout payloads, with verification before release."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

BASE = Path("/mowenjie/fluffy-bassoon-2")
RUN = BASE / "outputs/robotwin-sidney-matched-baselines-20261106-r1"
WORK = Path("/tmp/storage-audit-baige-20261111-r1")
BOS = Path("/mnt/bos/adgfyl")
COLD = BOS / "strong-plant-archives" / WORK.name / "sidney-training-rollouts"
ARMS = ("ppo02", "grpo01")
BUFFER = 8 * 1024**2


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_bytes())


def write(path, data):
    temp = path.with_name(path.name + ".writing")
    with temp.open("w") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def stamp(path):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, f"Expected a singly linked regular file: {path}")
    keys = ("st_dev", "st_ino", "st_size", "st_blocks", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_uid", "st_gid")
    return {key: getattr(info, key) for key in keys}


def digest(path, copy_to=None):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        if copy_to is None:
            for block in iter(lambda: source.read(BUFFER), b""):
                checksum.update(block)
        else:
            with copy_to.open("xb") as target:
                for block in iter(lambda: source.read(BUFFER), b""):
                    target.write(block)
                    checksum.update(block)
                target.flush()
                os.fsync(target.fileno())
    return checksum.hexdigest()


def has_readers(paths):
    targets = {str(path) for path in paths}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal() or int(proc.name) == os.getpid():
            continue
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    if os.readlink(fd) in targets:
                        return True
                except OSError:
                    continue
            with (proc / "maps").open(errors="replace") as maps:
                for row in maps:
                    fields = row.rstrip("\n").split(maxsplit=5)
                    if len(fields) == 6 and fields[-1] in targets:
                        return True
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return False


def sources():
    for arm in ARMS:
        run = RUN / arm
        require(read(run / "EXIT.json")["exit_code"] == 0, f"{arm} did not exit successfully")
        require(read(run / "RESULT.json")["status"] == "complete", f"{arm} is not complete")
        root = run / "training-rollouts"
        require(root.resolve() == root, f"Unexpected source directory: {root}")
        for iteration in (1, 2):
            for rank in range(4):
                yield root / f"iteration-{iteration:04d}-policy-{iteration - 1:04d}-rank-{rank:02d}.pt"


def archive(source):
    before = stamp(source)
    metadata = source.with_suffix(".json")
    raw_metadata = metadata.read_bytes()
    ref = json.loads(raw_metadata)
    require(ref["completed"] and ref["path"] == str(source) and ref["bytes"] == before["st_size"], f"Invalid original receipt: {metadata}")
    arm = source.parent.parent.name
    target = COLD / arm / source.name
    if target.exists():
        require(digest(source) == ref["sha256"], f"Source hash mismatch: {source}")
    else:
        require(digest(source, target) == ref["sha256"], f"Source hash mismatch: {source}")
    require(target.stat().st_size == ref["bytes"], f"Cold object length mismatch: {target}")
    staged = WORK / "readback" / arm / source.name
    staged.unlink(missing_ok=True)
    try:
        require(digest(target, staged) == ref["sha256"], f"Cold SHA256 mismatch: {target}")
        require(digest(staged) == ref["sha256"], f"Restored SHA256 mismatch: {staged}")
    finally:
        staged.unlink(missing_ok=True)
    require(stamp(source) == before and metadata.read_bytes() == raw_metadata, f"Source changed: {source}")
    cold_metadata = target.with_suffix(".json")
    if not cold_metadata.exists():
        cold_metadata.write_bytes(raw_metadata)
    require(cold_metadata.read_bytes() == raw_metadata, f"Cold metadata mismatch: {cold_metadata}")
    receipt = {
        "source": str(source), "archive": str(target), "sha256": ref["sha256"],
        "bytes": ref["bytes"], "source_stat": before, "allocated_bytes": before["st_blocks"] * 512,
        "metadata_sha256": hashlib.sha256(raw_metadata).hexdigest(),
        "cold_readback_verified": True, "restored_file_verified": True, "verified_unix": time.time(),
    }
    write(target.with_suffix(".receipt.json"), receipt)
    return receipt


def prepare():
    paths = list(sources())
    require(not has_readers(paths), "A visible process is reading the selected files")
    for arm in ARMS:
        (COLD / arm).mkdir(parents=True, exist_ok=True)
        (WORK / "readback" / arm).mkdir(parents=True, exist_ok=True)
    report = {"state": "copying", "start_unix": time.time(), "verified": [], "errors": {}}
    write(WORK / "ARCHIVE.json", report)
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(archive, source): source for source in paths}
        for job in as_completed(jobs):
            try:
                report["verified"].append(job.result())
            except Exception as error:
                report["errors"][str(jobs[job])] = str(error)
            write(WORK / "ARCHIVE.json", report)
    report["state"] = "verified" if not report["errors"] else "copy_errors"
    write(WORK / "ARCHIVE.json", report)
    require(not report["errors"], "One or more files failed verification. See ARCHIVE.json.")


def release():
    scheduler = read(WORK / "SCHEDULER.json")
    require(scheduler["selected_consumers"] == [] and time.time() - scheduler["observed_unix"] < 600,
            "Refresh the scheduler and confirm there are no live consumers")
    report = read(WORK / "ARCHIVE.json")
    paths = set(sources())
    receipts = sorted(report["verified"], key=lambda item: item["source"])
    require(report["state"] == "verified" and {Path(row["source"]) for row in receipts} == paths,
            "The batch must be completely verified before release")
    require(not has_readers(paths), "A visible process is reading the selected files")
    free_before = shutil.disk_usage(BASE).free
    released = []
    for receipt in receipts:
        source = Path(receipt["source"])
        cold = Path(receipt["archive"])
        require(read(cold.with_suffix(".receipt.json")) == receipt and cold.stat().st_size == receipt["bytes"],
                f"Missing verified cold copy: {cold}")
        pointer = source.with_name(source.name + ".cold.json")
        write(pointer, receipt)
        require(stamp(source) == receipt["source_stat"], f"Source changed: {source}")
        source.unlink()
        released.append(receipt)
        write(WORK / "RELEASE.json", {
            "state": "complete" if len(released) == len(receipts) else "releasing",
            "released": released, "released_allocated_bytes": sum(row["allocated_bytes"] for row in released),
            "free_bytes_before_release": free_before, "free_bytes_after_release": shutil.disk_usage(BASE).free,
            "updated_unix": time.time(), "scheduler": scheduler,
        })
    require(all(not path.exists() and path.with_suffix(".json").is_file() for path in paths),
            "Post-release check failed")
    write(COLD / "RELEASE.json", read(WORK / "RELEASE.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "release"))
    args = parser.parse_args()
    require(os.path.ismount(BOS), f"BOS is not mounted: {BOS}")
    (prepare if args.phase == "prepare" else release)()
