"""Prepare, score and audit generated native source evaluation panels."""

import argparse
import os
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("prepare", "source", "aggregate", "accept"):
        q = sub.add_parser(name)
        q.add_argument("--root", type=Path, required=True)
        q.add_argument("--fit", type=Path, required=True)
        q.add_argument("--phase", choices=("dev", "test"), default="dev")
        if name != "prepare":
            q.add_argument("--step", type=int, default=1600)
            q.add_argument("--stride", type=int, choices=(1, 2, 5), default=5)
        if name == "accept":
            q.add_argument("--extended-fits", nargs=2, type=Path, required=True,
                           metavar=("CONVENTIONAL_FIT", "SERVO_FIT"),
                           help="Registered 6400-update fits, following the original 1600-update --fit")
        if name == "source":
            q.add_argument("--weights", type=Path, default=os.environ.get("DUAL_WEIGHTS"))
            q.add_argument("--workers", type=int, default=8)
            q.add_argument("--arm", choices=("both", "Panda", "Sawyer"), default="both")
            q.add_argument("--shard-index", type=int, default=0)
            q.add_argument("--shard-count", type=int, default=1)
    args = p.parse_args(argv)
    args.root = args.root.resolve()
    args.fit = args.fit.resolve(strict=True)
    if args.command == "prepare":
        from large_panel_sim import prepare
        prepare(args)
    elif args.command == "source":
        if not args.weights or args.workers < 1 or not 0 <= args.shard_index < args.shard_count:
            p.error("source requires --weights or DUAL_WEIGHTS, positive workers and a valid zero-based shard")
        from large_panel_rollout import source
        source(args)
    else:
        from large_panel_protocol import summarize
        summarize(args, accept=args.command == "accept")


if __name__ == "__main__":
    main()
