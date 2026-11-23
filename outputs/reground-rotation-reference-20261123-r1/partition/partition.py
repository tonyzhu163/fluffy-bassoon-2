"""Launcher payload for the pinned ReGround partition transport."""
import argparse
import contextlib
import errno
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

import reference as ref

CANONICAL = ref.ROOT / "reference-native"
NATIVE_RUNNER = "23a6641562a4d833837f27ab0535a20ad94e8ec8"
MAX_REFERENCE_GPUH = 40  # 40 reference + 8.5 streaming, see the registered run ledgers.


def segment(value):
    ref.require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}", value), "Invalid name")
    return value


def store_for(namespace):
    segment(namespace)
    if namespace == "canonical":
        return CANONICAL
    return ref.ROOT / "partition-namespaces" / namespace


def validate(args):
    ref.require(args.attempt.parent == CANONICAL / "attempts", "All namespaces share the reference budget")
    segment(args.attempt.name)
    store_for(args.namespace)
    ref.require(len(set(args.seeds)) == len(args.seeds) and all(s in ref.SEEDS for s in args.seeds),
                "Supply distinct registered seeds")
    ref.require(0 < args.cap <= MAX_REFERENCE_GPUH and 20 < args.wall <= 999999, "Invalid wall time or cap")
    if args.phase == "fit":
        ref.require(len(args.seeds) == args.gpus == 1 and args.shards is None and not args.descending,
                    "Fit requires exactly one card and one explicit eligible seed")
    else:
        limit = {"collect": 25, "babble": 50}[args.phase]
        ref.require(args.shards and 0 <= args.shards[0] < args.shards[1] <= limit,
                    f"{args.phase} requires an explicit half-open range in [0,{limit})")


def contract(args):
    return dict(phase=args.phase, seeds=args.seeds, shards=args.shards, descending=args.descending,
                namespace=args.namespace, gpus=args.gpus, wall_seconds=args.wall, cap_gpuh=args.cap,
                transport=os.environ["REFERENCE_CODE_PIN"], reference=NATIVE_RUNNER, upstream=ref.PIN,
                policy=ref.POLICY)


def reserve(args):
    # Publish a nonempty owner directory in a single rename. A killed reservation
    # transaction fails closed; never infer owner death from PID or wall time.
    proposal = args.attempt / "budget-owner"
    proposal.mkdir()
    ref.write(proposal / "owner.json", dict(attempt=args.attempt.name, host=socket.gethostname(),
                                            pid=os.getpid(), utc=time.time()))
    gate = args.attempt.parent / ".partition-reserving"
    try:
        proposal.rename(gate)
    except OSError as exc:
        if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
            raise
        raise RuntimeError(f"Reference reservation in progress. Inspect {gate}/owner.json") from exc
    try:
        ref.write(args.attempt / "partition.json", contract(args))
        ref.reserve(args)  # original .budget.lock and accounts, including foreign unfinished reservations
    finally:
        gate.rename(proposal)


def read_contract(args):
    ref.require(contract(args) == ref.read(args.attempt / "partition.json"), "Partition contract changed")
    row = ref.read(args.attempt / "reservation.json")
    ref.require(row["code"] == os.environ["REFERENCE_CODE_PIN"] and row["gpus"] == args.gpus
                and row["phase"] == args.phase and row["seeds"] == args.seeds
                and row["max_gpu_seconds"] == args.gpus * args.wall and row["cap_gpuh"] == args.cap
                and row["host"] == socket.gethostname(), "Reservation mismatch")


def configure(namespace):
    ref.STORE = store_for(namespace)
    ref.NATIVE = ref.STORE / "runs"
    ref.configure()  # sets upstream utils.RUNS before consumers import it


def dispatch(args):
    """Give each lane and its simulator descendants one process group for teardown."""
    def terminated(signum, frame):
        raise SystemExit(128 + signum)

    children = []
    previous = signal.signal(signal.SIGTERM, terminated)
    with contextlib.ExitStack() as logs:
        try:
            for rank in range(args.gpus):
                log = logs.enter_context((args.attempt / f"{args.phase}-gpu{rank}.log").open("x"))
                children.append(subprocess.Popen([sys.executable, "-Bu", __file__, "lane",
                                                  *sys.argv[2:], "--rank", str(rank)], start_new_session=True,
                                                 stdout=log, stderr=subprocess.STDOUT))
            while None in [p.poll() for p in children]:
                if any(p.returncode not in (None, 0, 75) for p in children):
                    raise RuntimeError("A lane failed. Inspect its attempt log.")
                time.sleep(1)
            ref.require(all(p.returncode in (0, 75) for p in children), "Lane failed")
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            for sig in (signal.SIGTERM, signal.SIGKILL):
                for p in children:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(p.pid, sig)
                if sig == signal.SIGTERM:
                    time.sleep(1)
            for p in children:
                p.wait()
            signal.signal(signal.SIGTERM, previous)
    if args.phase == "fit":
        return children[0].returncode
    import collect
    return collect.summary(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["reserve", "dispatch", "lane"])
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=["babble", "collect", "fit"])
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--shards", type=int, nargs=2, metavar=("FIRST", "STOP"))
    parser.add_argument("--descending", action="store_true")
    parser.add_argument("--gpus", required=True, choices=[1, 2, 3, 4, 5], type=int)
    parser.add_argument("--wall", required=True, type=int)
    parser.add_argument("--cap", required=True, type=float)
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()
    validate(args)
    if args.action == "reserve":
        reserve(args)
        return 0
    read_contract(args)
    configure(args.namespace)
    if args.action == "dispatch":
        return dispatch(args)
    ref.require(args.rank is not None and 0 <= args.rank < args.gpus, "Invalid GPU rank")
    import collect
    if args.phase == "fit":
        import fit
        return fit.run_lane(args)
    collect.run_lane(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
