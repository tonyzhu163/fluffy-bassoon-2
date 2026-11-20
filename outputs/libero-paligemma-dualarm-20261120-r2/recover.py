"""Capture frozen Goal 5 policy prefixes and collect successful native teacher suffixes."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from multiprocessing import Pipe
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R1 = HERE.parent / "libero-paligemma-crossarm-20261120-r1"
ARM = HERE.parent / "libero-crossarm-20261120-r1"
sys.path.insert(1, str(R1))

from motor_weights import sha256

BODIES = ("Panda", "Sawyer")
PROMPT = "push the plate to the front of the stove"
PREFIXES = (5, 10, 20, 30, 45)
# pretrain_reader.NATIVE_RESET, kept usable without PyTorch in the simulator venv.
RELEASE = {
    "source_pin": "e5500c690c1498eca1b7050350654964190cdf70",
    "manifest_sha256": "71a6e4e25a2a4a162844d77136e7bff8e6f69c839d2d8194620c4afeabde2bba",
    "bddl_sha256": "c2db20071f58fd43fe029e5bf9176f3c7371ee62cadc311f2b8682c089174c0d",
}


def code_ids():
    paths = [
        HERE / name
        for name in (
            "recover.py",
            "recover_sim.py",
            "recover_teacher.py",
            "paddle.py",
            "expert.py",
            "expert_sweep.py",
        )
    ]
    paths += [
        R1 / name
        for name in (
            "motor.py",
            "motor_data.py",
            "motor_weights.py",
            "sim.py",
            "demos.py",
            "oracle_fallback.py",
        )
    ]
    paths += [ARM / name for name in ("arm_env.py", "sim_reset.py")]
    return {p.name: sha256(p) for p in paths}


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(json.dumps([a.dtype.str, a.shape]).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def public_hashes(obs):
    return {
        "state": array_hash(obs["state"]),
        "rgb": array_hash(obs["rgb"]),
        "wrist": array_hash(obs["wrist"]),
        "prompt": obs["prompt"],
    }


def check_source(body, pose):
    if body not in BODIES or type(pose) is not int or pose not in range(28):
        raise ValueError(f"Recovery requires a source body and training layout 0..27: {body}/{pose}")


def reset_identity(reset, body, pose, seed):
    check_source(body, pose)
    p = reset["environment_reproducibility"]
    expected = dict(**RELEASE, suite="libero_goal", task_id=5, init_state_id=pose, seed=seed)
    if (
        type(seed) is not int
        or seed not in range(2**32)
        or reset["model_identity"]["arm"] != body
        or any(p.get(k) != v for k, v in expected.items())
        or (p["perturbation_plan"]["case_id"], p["perturbation_plan"]["seed"]) != ("BASE", seed)
        or not all(p[k] for k in ("initial_state_fingerprint", "released_row_fingerprint"))
    ):
        raise ValueError(f"Unexpected native Goal 5 reset for {body}/{pose}/{seed}")
    return p


def action(value):
    a = np.asarray(value, dtype=np.float32)
    if a.shape != (7,) or not np.isfinite(a).all() or np.any(abs(a) > 1):
        raise ValueError("Invalid native 7D control")
    return a


def checked_prefixes(values):
    if (
        not values
        or len(set(values)) != len(values)
        or any(type(n) is not int or (n not in (5, 10) and n not in range(20, 46)) for n in values)
    ):
        raise ValueError("Use distinct prefixes of 5, 10, or 20..45 native controls")
    return sorted(values)


def model_and_receipt(args):
    import motor
    import torch

    motor.require_gpu()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    ckpt = args.checkpoint.resolve(strict=True)
    if sha256(ckpt) != args.checkpoint_sha256:
        raise ValueError("Frozen checkpoint SHA-256 mismatch")
    saved = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
    contract = saved["contract"]
    rows = contract["episodes"]
    if (
        saved["schema"] != motor.SCHEMA
        or not rows
        or not saved["best"]
        or contract["source_bodies"] not in (["Panda"], list(BODIES))
        or contract["weights"]["repo"] != "google/paligemma-3b-mix-224"
    ):
        raise ValueError("Expected a validated general PaliGemma source motor or candidate")
    for row in rows:
        pose = int(row["pose"])
        if (
            row["body"] not in contract["source_bodies"]
            or row["prompt"] != PROMPT
            or row["split"] not in ("train", "validation")
            or pose not in range(32)
            or (row["split"] == "train" and pose >= 28)
        ):
            raise ValueError("Checkpoint provenance outside the source fit and offline selection split")
    model = motor.Motor(args.weights)
    if model.identity != contract["weights"]:
        raise ValueError("Base snapshot differs from the frozen checkpoint")
    model.restore_delta(saved["best"]["model"])
    model.requires_grad_(False).eval()
    receipt = {
        "file": str(ckpt),
        "sha256": args.checkpoint_sha256,
        "step": saved["best"]["step"],
        "fit_id": saved.get("fit_id"),
        "source_bodies": contract["source_bodies"],
        "base": model.identity,
        "contract_sha256": digest(contract),
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0),
    }
    return model, receipt


def capture(args):
    from sim import _unpack

    prefixes = checked_prefixes(args.prefixes)
    if args.rolls and (len(set(args.rolls)) != len(args.rolls) or not 1 <= len(args.rolls) <= 2 * 28):
        raise ValueError("Specify distinct source body/layout pairs")
    for body, pose in args.rolls or ():
        check_source(body, pose)
    banks = [str(p.resolve(strict=True)) for p in args.bank]
    sim_python = os.environ["CROSSARM_SIM_PYTHON"]
    if not args.weights:
        raise ValueError("Supply --weights or DUAL_WEIGHTS")
    args.out.mkdir(parents=True, exist_ok=False)
    model, receipt = model_and_receipt(args)
    config = args.out / "config.json"
    write(
        config,
        {
            "version": 1,
            "motor": receipt,
            "banks": banks,
            "bank_hashes": [sha256(Path(b) / "index.json") for b in banks],
            "prefixes": prefixes,
            "rolls": args.rolls,
            "count": len(args.rolls) if args.rolls else args.count,
            "code": code_ids(),
            "revision": os.environ.get("MOTOR_CODE_REVISION"),
            "argv": sys.argv,
            "policy_python": sys.executable,
            "simulator_python": sim_python,
        },
    )
    parent, child = Pipe()
    cmd = [sim_python, "-Bu", str(HERE / "recover_sim.py"), str(child.fileno()), str(config.resolve())]
    with parent, child, subprocess.Popen(cmd, pass_fds=(child.fileno(),)) as proc:
        child.close()
        try:
            while True:
                try:
                    packet = parent.recv()
                except EOFError:
                    break
                obs = _unpack(packet)
                if obs["prompt"] != PROMPT:
                    raise ValueError("Unexpected public motor instruction")
                chunk = model.act(obs)
                if chunk.dtype != np.float32 or chunk.shape != (5, 7):
                    raise ValueError("Expected a five-control float32 motor chunk")
                parent.send((public_hashes(obs), [action(a).tolist() for a in chunk]))
        finally:
            parent.close()
        if proc.wait() != 0:
            raise RuntimeError("Native prefix capture failed; see simulator traceback")
    manifest = args.out / "capture.json"
    print("capture_sha256", sha256(manifest), flush=True)


def source_arg(text):
    try:
        body, pose = text.split(":")
        pose = int(pose)
        check_source(body, pose)
        return body, pose
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use Panda:0 through Sawyer:27 on a source arm") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    cap = stages.add_parser("capture", help="GPU motor and native simulator subprocess")
    cap.add_argument("--checkpoint", type=Path, required=True)
    cap.add_argument("--checkpoint-sha256", required=True)
    cap.add_argument("--weights", type=Path, default=os.environ.get("DUAL_WEIGHTS"))
    cap.add_argument("--bank", nargs="+", type=Path, required=True)
    which = cap.add_mutually_exclusive_group()
    which.add_argument("--rolls", nargs="+", type=source_arg, metavar="BODY:POSE")
    which.add_argument("--count", type=int, choices=range(1, 57), default=4)
    cap.add_argument("--prefixes", type=int, nargs="+", default=PREFIXES)
    cap.add_argument("--out", type=Path, required=True)
    teach = stages.add_parser("collect", help="simulator-only replay and corrective labels")
    teach.add_argument("--capture", type=Path, required=True, help="capture.json")
    teach.add_argument("--capture-sha256", required=True)
    teach.add_argument("--archive", type=Path, default=os.environ.get("DUAL_ARCHIVE"))
    teach.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "capture":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        capture(args)
    else:
        from recover_sim import collect

        collect(args)


if __name__ == "__main__":
    main()
