"""Archive selected, closed-run payloads, then release verified PFS copies."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import time

from archive_outputs import copy_archive
from archive_rollouts import BOS, digest, has_readers, read, require, stamp, write

BASE = Path("/mowenjie")
RUN = "outputs-retention-20261111-r2"
WORK = Path("/tmp") / RUN
COLD = BOS / "strong-plant-archives" / RUN
GIB = 2**30
PAYLOAD = (".pt", ".pt.gz", ".pth", ".pth.gz", ".npz", ".npy", ".distcp", ".safetensors",
           ".pkl", ".pkl.gz", ".bin", ".mp4", ".mkv", ".webm", ".avi", ".h5", ".hdf5", ".tar.zst")
COMPACT_LIMIT = 2**20


def select(root):
    if root.is_file():
        yield root
        return
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(name for name in dirs if not name.startswith("."))
        for name in sorted(files):
            if name.lower().endswith(PAYLOAD):
                yield Path(parent) / name


def batch(plan):
    for item in plan:
        key = item["id"]
        group = COLD / key
        group.mkdir(exist_ok=True)
        piece, total, number = [], 0, 0
        skipped = []
        for entry in item["roots"]:
            root = BASE / entry
            require(root.exists() and root.resolve() == root and root.parts[3] == "outputs",
                    f"Unexpected source {root}")
            for source in select(root):
                info = source.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_size < COMPACT_LIMIT:
                    continue
                if info.st_nlink != 1:
                    skipped.append(str(source))
                    continue
                size = info.st_size
                require(size <= 25 * GIB, f"Single payload too large: {source}")
                if piece and (size >= 2 * GIB or size + total > 12 * GIB):
                    number += 1
                    yield group, number, piece
                    piece, total = [], 0
                if size >= 2 * GIB:
                    number += 1
                    yield group, number, [source]
                else:
                    piece.append(source)
                    total += size
        if piece:
            number += 1
            yield group, number, piece
        if skipped:
            write(WORK / f"{key}.linked.json", skipped)


def archive(group, number, sources):
    part = f"{number:03d}"
    before = [stamp(source) for source in sources]
    rows = [{"source": str(source), "source_stat": item, "bytes": item["st_size"],
             "allocated_bytes": item["st_blocks"] * 512} for source, item in zip(sources, before)]
    with tempfile.TemporaryDirectory(dir=WORK, prefix=f"{group.name}-{part}-") as temp:
        stage = Path(temp)
        if len(sources) == 1:
            source = sources[0]
            target = group / f"{part}-{source.name}"
            checksum, _ = copy_archive(source, target)
            rows[0]["sha256"] = checksum
            restored = stage / source.name
            require(digest(target, restored) == checksum and digest(restored) == checksum,
                    f"Object restore failed: {target}")
            archive_type = "file"
        else:
            for source, row in zip(sources, rows):
                row["sha256"] = digest(source)
            listing = stage / "members"
            listing.write_bytes(b"".join(os.fsencode(source.relative_to(BASE)) + b"\0" for source in sources))
            local = stage / "payload.tar.zst"
            target = group / f"{part}.tar.zst"
            subprocess.run(["tar", "--format=posix", "--acls", "--xattrs", "--sparse",
                            "--no-recursion", "-I", "zstd --fast=1 -T1", "-cf", str(local),
                            "-C", str(BASE), "--null", "-T", str(listing)], check=True)
            checksum, _ = copy_archive(local, target)
            local.unlink()
            require(digest(target, local) == checksum, f"BOS readback failed: {target}")
            restored = stage / "restore"
            restored.mkdir()
            subprocess.run(["tar", "--acls", "--xattrs", "--same-owner", "-I", "zstd",
                            "-xf", str(local), "-C", str(restored)], check=True)
            expected = {str(source.relative_to(BASE)) for source in sources}
            actual = {str(path.relative_to(restored)) for path in restored.rglob("*") if path.is_file()}
            require(expected == actual, f"Wrong archive members: {target}")
            for source, row in zip(sources, rows):
                path = restored / source.relative_to(BASE)
                check = stamp(path)
                keys = ("st_size", "st_mtime_ns", "st_mode", "st_uid", "st_gid")
                require(all(check[key] == row["source_stat"][key] for key in keys)
                        and digest(path) == row["sha256"], f"Restored member changed: {source}")
            archive_type = "tar.zst"
        require(before == [stamp(source) for source in sources], "Source changed while archiving")
        receipt = dict(archive=str(target), archive_type=archive_type, archive_sha256=checksum,
                       archive_bytes=target.stat().st_size, rows=rows, verified_unix=time.time())
        write(group / f"{part}.receipt.json", receipt)
        write(WORK / "verified" / f"{group.name}__{part}.json", receipt)
    return group.name, part, sum(row["allocated_bytes"] for row in rows)


def prepare(plan_name):
    plan = read(WORK / plan_name)
    COLD.mkdir(exist_ok=True)
    (WORK / "verified").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(archive, group, part, paths): (group.name, part)
                for group, part, paths in batch(plan["archive"])}
        errors = []
        for job in as_completed(jobs):
            try:
                name, part, amount = job.result()
                print(f"VERIFIED {name} {part} {amount/GIB:.2f} GiB", flush=True)
            except Exception as error:
                errors.append(dict(part=jobs[job], error=str(error)))
                print(f"FAILED {jobs[job]} {error}", flush=True)
        write(WORK / f"{Path(plan_name).stem}-DONE.json", dict(parts=len(jobs), errors=errors))
        require(not errors, "Some archive parts did not verify")


def release(plan_name):
    scheduler = read(WORK / "SCHEDULER.json")
    require(not scheduler["selected_consumers"] and time.time() - scheduler["observed_unix"] < 900,
            "Refresh active task dependencies")
    releases = WORK / "released"
    releases.mkdir(exist_ok=True)
    for item in read(WORK / plan_name)["archive"]:
        key = item["id"]
        for receipt_path in sorted((WORK / "verified").glob(f"{key}__*.json")):
            release_path = releases / receipt_path.name
            if release_path.exists():
                continue
            receipt = read(receipt_path)
            target = Path(receipt["archive"])
            part = receipt_path.stem.split("__")[-1]
            require(read(target.parent / f"{part}.receipt.json") == receipt
                    and target.stat().st_size == receipt["archive_bytes"], f"Cold part missing: {target}")
            paths = [Path(row["source"]) for row in receipt["rows"]]
            require(not has_readers(paths), f"Visible readers in {receipt_path.name}")
            for path, row in zip(paths, receipt["rows"]):
                require(stamp(path) == row["source_stat"], f"Changed source: {path}")
            for root in item["roots"]:
                root = BASE / root
                folder = root if root.is_dir() else root.parent
                write(folder / f"COLD-ARCHIVE-{RUN}-{key}.json",
                      dict(cold_directory=str(target.parent), receipt_glob="*.receipt.json", group=key))
            for path in paths:
                path.unlink()
            require(all(not path.exists() for path in paths), "Some paths remain hot")
            receipt["released_unix"] = time.time()
            write(release_path, receipt)
            print(f"RELEASED {key} {part}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "release"))
    parser.add_argument("--plan", choices=("PLAN.json", "ADDITIONAL.json", "RECOVERY.json"), default="PLAN.json")
    args = parser.parse_args()
    require(os.path.ismount(BOS), f"BOS is not mounted: {BOS}")
    (prepare if args.phase == "prepare" else release)(args.plan)
