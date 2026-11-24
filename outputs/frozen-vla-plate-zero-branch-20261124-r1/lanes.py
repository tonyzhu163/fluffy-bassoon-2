"""Preallocate and publish new-study files with the qualified storage protocol."""
import argparse
import errno
import json
import os
from pathlib import Path
import sys

import storage

MODES = ("stock", "mixed_on", "mixed_zero")
GRAPH_FILES = ("stock.hlo.gz", "history.hlo.gz")


def reserve(out, pin, start, stop, lane):
    storage.reserve(out, pin, start, stop, lane, modes=MODES)
    shard = out / f"shard{lane}"
    for name in GRAPH_FILES:
        storage.slot(shard / name).touch(exist_ok=False)
    planned_bytes = 48 * 2**20
    slots = sorted(shard.glob("case-*/.*.h5.slot"))
    state = os.statvfs(out)
    storage.need(state.f_bavail * state.f_frsize > len(slots) * planned_bytes, "insufficient shared tape capacity")
    allocation_status = "supported"
    for slot in slots:
        try:
            with slot.open("r+b") as file:
                os.posix_fallocate(file.fileno(), 0, planned_bytes)
                os.fsync(file.fileno())
        except OSError as error:
            if error.errno not in (errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL):
                raise
            allocation_status = f"unsupported errno {error.errno}"
            break
    allocated = sum(slot.stat().st_blocks * 512 for slot in slots)
    state = os.statvfs(out)
    if allocated < planned_bytes * len(slots) and allocation_status == "supported":
        allocation_status = "shared filesystem did not report physical blocks"
    receipt = dict(status=allocation_status, hdf5_slots=len(slots), planned_bytes=planned_bytes * len(slots),
                   reported_allocated_bytes=allocated, available_bytes=state.f_bavail * state.f_frsize,
                   available_inodes=state.f_favail)
    with (out / "hdf5-space.json").open("x") as file:
        json.dump(receipt, file)
        file.flush()
        os.fsync(file.fileno())
    print(json.dumps(receipt), flush=True)


def run(local, remote, command):
    storage.MODES = MODES
    publish = storage.publish

    def publish_with_graphs(src, dst):
        for name in GRAPH_FILES:
            src_graph = src.parent / name
            if src_graph.is_file() and not (dst.parent / name).exists():
                storage.install(src_graph, dst.parent / name)
        publish(src, dst)

    storage.publish = publish_with_graphs
    return storage.run(local, remote, command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("reserve")
    r.add_argument("out", type=Path)
    r.add_argument("pin")
    r.add_argument("start", type=int)
    r.add_argument("stop", type=int)
    r.add_argument("lane", choices=["0", "1", "2", "3"])
    r = sub.add_parser("run")
    r.add_argument("local", type=Path)
    r.add_argument("remote", type=Path)
    r.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "reserve":
        reserve(args.out, args.pin, args.start, args.stop, args.lane)
    else:
        sys.exit(run(args.local, args.remote, args.argv))
