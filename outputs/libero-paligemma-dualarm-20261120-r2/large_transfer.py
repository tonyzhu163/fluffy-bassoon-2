#!/usr/bin/env python3
"""Fit on sources, verify both native source panels, and evaluate the frozen transfer plan."""

import argparse
import json
import os
import sys
from pathlib import Path

from large_transfer_gate import GateClosed
from large_transfer_stats import aggregate, sha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fit = stages.add_parser(
        "fit",
        help="source-only dynamics pretraining and motor-bound joint residual fit",
    )
    fit.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="exact candidate .pt from the source fit",
    )
    fit.add_argument(
        "--demos",
        nargs="+",
        type=Path,
        required=True,
        help="immutable registered source expert banks",
    )
    fit.add_argument(
        "--stage1",
        type=Path,
        help="verified Stage 1 directory with all ten native source tapes",
    )
    review = stages.add_parser(
        "qualify", aliases=["review"],
        help="recount native source tapes, publish an immutable receipt and print its SHA-256",
    )
    evaluate = stages.add_parser(
        "evaluate",
        help="verify the pinned source evidence and execute the sealed UR5e panel",
    )
    for cmd in (review, evaluate):
        cmd.add_argument(
            "--root", type=Path, required=True, help="producer's large panel directory"
        )
        cmd.add_argument(
            "--fit",
            type=Path,
            required=True,
            help="selected motor's original SFT fit directory",
        )
        cmd.add_argument(
            "--joint",
            type=Path,
            required=True,
            help="completed large_transfer.py fit directory",
        )
    for cmd in (fit, evaluate):
        cmd.add_argument(
            "--weights",
            type=Path,
            default=os.environ.get("DUAL_WEIGHTS"),
            required=not os.environ.get("DUAL_WEIGHTS"),
            help="pinned PaliGemma base snapshot",
        )
        cmd.add_argument(
            "--revision",
            default=os.environ.get("MOTOR_CODE_REVISION"),
            required=not os.environ.get("MOTOR_CODE_REVISION"),
        )
        cmd.add_argument(
            "--out", type=Path, required=True, help="fresh stage directory"
        )
    review.add_argument(
        "--out", type=Path, required=True, help="immutable source receipt JSON path"
    )
    evaluate.add_argument(
        "--reuse",
        type=Path,
        help="reuse verified native episodes from an identical partial run",
    )
    evaluate.add_argument(
        "--receipt", type=Path, required=True, help="source qualification JSON"
    )
    evaluate.add_argument(
        "--receipt-sha256", required=True, help="exact digest printed by qualify"
    )
    evaluate.add_argument("--shard-index", type=int, default=0, help="zero-based layout shard")
    evaluate.add_argument("--shard-count", type=int, default=1, help="number of interleaved layout shards")
    stats = stages.add_parser(
        "aggregate",
        help="CPU-only readback of 100 paired assignments and their native tapes",
    )
    stats.add_argument("--run", type=Path, required=True)
    stats.add_argument("--reuse", type=Path, nargs="+", default=[], help="additional completed shard directories")
    args = parser.parse_args(argv)
    if args.stage == "fit":
        from large_transfer_fit import fit as source_fit

        source_fit(args)
    elif args.stage in ("qualify", "review"):
        from large_panel_protocol import publish
        from large_transfer_gate import receipt

        plan = receipt(args)
        encoded = (json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        with publish(args.out) as temporary:
            temporary.write_bytes(encoded)
        print(sha(args.out))
    elif args.stage == "evaluate":
        from large_transfer_query import evaluate as run

        summary = run(args)
        print(summary.get("prior_history_answer") or json.dumps(summary))
    else:
        print(
            json.dumps(aggregate(args.run, args.reuse), indent=2, sort_keys=True, allow_nan=False)
        )


if __name__ == "__main__":
    try:
        main()
    except GateClosed as exc:
        print(f"Source gate closed. {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
