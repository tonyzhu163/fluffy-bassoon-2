"""Capture native frames and pre-action plate labels together on a prespecified fixed Paddle route."""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from expert import parse_layouts, sha, write_json
from paddle import Route
from waypoint_labels import BODIES, LOOKAHEAD, Diverged, camera_capture, publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bodies", choices=BODIES, nargs="+", default=list(BODIES))
    parser.add_argument("--poses", type=parse_layouts, default=list(range(28, 32)))
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--out", type=Path, default=Path("bank"))
    args = parser.parse_args()
    bodies, poses = sorted(set(args.bodies)), sorted(args.poses)
    if min(poses) + args.seed < 0 or max(poses) + args.seed >= 2**32:
        parser.error("Seeds must fit uint32")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in (*bodies, "aux"):
        (out / name).mkdir()
    route = asdict(Route("rim", 28, 180, -35))
    index = {
        "suite": "libero_goal",
        "case": "BASE",
        "task_id": 5,
        "bodies": bodies,
        "poses": poses,
        "seed_base": args.seed,
        "fixed_route": route,
        "attempts": [],
        "status": "running",
        "label_schema": 1,
        "lookahead": LOOKAHEAD,
        "revision": os.environ.get("MOTOR_CODE_REVISION"),
        "label_code_sha256": sha(Path(__file__).with_name("waypoint_labels.py")),
        "capture_code_sha256": sha(Path(__file__)),
    }
    write_json(out / "index.json", index)
    for body in bodies:
        for pose in poses:
            job = {
                "body": body,
                "pose": pose,
                "seed": args.seed + pose,
                "params": route,
            }
            key = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
            with TemporaryDirectory(dir=out) as directory:
                try:
                    row, tape, data, plate = camera_capture(
                        Path(directory), job, None, "route"
                    )
                except Diverged as exc:
                    result = dict(**job, accepted=False, reason=str(exc))
                    # Preserve the collector's actual native failure and reset when available.
                    receipt = (
                        Path(directory) / "records" / f"{pose}-{job['seed']}-0.json"
                    )
                    if receipt.exists():
                        result.update(json.loads(receipt.read_text()))
                else:
                    provenance = {
                        "mode": "fixed_direct_camera_route",
                        "fixed_spec": job,
                        "skips": [],
                    }
                    result = publish(out, row, tape, data[0], plate, key, provenance)
            index["attempts"].append(result)
            write_json(out / "index.json", index)
            print(body, pose, result["accepted"], result.get("reason", ""), flush=True)
    index.update(
        status="complete", accepted=sum(row["accepted"] for row in index["attempts"])
    )
    write_json(out / "index.json", index)
    return 0 if index["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
