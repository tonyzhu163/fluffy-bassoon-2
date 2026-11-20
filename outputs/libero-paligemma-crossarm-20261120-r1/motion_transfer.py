"""Apply the frozen source reader to existing command-matched native UR5e probes."""

import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import tarfile

import numpy as np
import torch

from adapt_eval import K, ReaderAdapter, digest_file, write_json


@torch.inference_mode()
def measure(archive, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["stage"] != "source_motion" or saved["steps"] != 100:
        raise ValueError("Expected the complete source motion reader")
    net = ReaderAdapter().eval()
    net.load_state_dict(saved["state_dict"])
    files = {}

    def load_probe(bundle, pool, pose, body):
        path = f"gpu01/pilot/{pool}/libero_goal-5-i{pose}-s{20261620+pose}/{body}/prescribed/tape.npz"
        data = bundle.extractfile(path).read()
        files[path] = sha256(data).hexdigest()
        with np.load(BytesIO(data), allow_pickle=False) as tape:
            a, y = (torch.tensor(tape[k], dtype=torch.float32) for k in ("commands", "motion"))
        if a.shape != y.shape or a.shape[1] != 7 or len(a) < K or not (a.isfinite().all() and y.isfinite().all()):
            raise ValueError(path)
        return a, y

    panel = []
    with tarfile.open(archive) as bundle:
        x, y = load_probe(bundle, "calibration_check", 9, "UR5e")
        for pose in (0, 1):
            commands = None
            scores = {}
            for body, pool in (("UR5e", "oracle_calibration"), ("Panda", "calibration"), ("Sawyer", "calibration")):
                a, m = load_probe(bundle, pool, pose, body)
                if commands is not None and not torch.equal(commands, a):
                    raise ValueError("Supports are not command matched")
                commands = a
                z = net.encode(torch.cat((a[:K], m[:K]), dim=-1).unsqueeze(0))
                for label, context in ((body, z), ("zero", torch.zeros_like(z))):
                    prediction = net.affine(net.response, context.expand(len(x), -1), x)
                    scores[label] = float((prediction - y / net.motion_scale).square().mean())
            panel.append({"support_pose": pose, "normalized_mse": scores})
    return {"scope": "Frozen response prediction on a simulated development arm excluded from fitting",
            "checkpoint_sha256": digest_file(checkpoint), "archive_sha256": digest_file(archive),
            "support_length": K, "query_body": "UR5e", "query_pose": 9, "query_controls": len(x), "rows": panel,
            "mean": {k: sum(r["normalized_mse"][k] for r in panel)/len(panel) for k in scores}, "tapes": files}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("archive", "checkpoint", "out"):
        p.add_argument("--" + flag, type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "report.json", measure(args.archive, args.checkpoint))
