"""Fit the source motion reader independently of the motor. See STAGE1.md."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from adapt_eval import (
    FIT_SEED, K, MOTION_QUERY, SOURCES, SUPPORT, TASK, History, Query, ReaderAdapter,
    digest_file, practice, probe, real_array, seed, train_stage, write_json,
)


STEPS = 100
BATCH_SIZE = 32
CHECK_POSE = 28
# Native reset identity for released BASE libero_goal/5.
NATIVE_RESET = {
    "source_pin": "e5500c690c1498eca1b7050350654964190cdf70",
    "manifest_sha256": "71a6e4e25a2a4a162844d77136e7bff8e6f69c839d2d8194620c4afeabde2bba",
    "bddl_sha256": "c2db20071f58fd43fe029e5bf9176f3c7371ee62cadc311f2b8682c089174c0d",
}


def read_tape(directory, body, pose):
    if body not in SOURCES or pose not in (*SUPPORT, *MOTION_QUERY, CHECK_POSE):
        raise ValueError(f"Unexpected source tape {body}/{pose}")
    name = f"practice-{body}-{pose}.npz"
    with np.load(directory / name, allow_pickle=False) as tape:
        labels = {"body": body, "pose": pose, "seed": seed(pose)}
        if any(tape[k].item() != v for k, v in labels.items()):
            raise ValueError(f"Source reset labels differ in {name}")
        reset = json.loads(tape["reset_json"].item())
        if any(reset.get(k) != v for k, v in NATIVE_RESET.items()):
            raise ValueError(f"Native Goal 5 reset identity differs in {name}")
        commands = tape["commands"]
        if commands.dtype != np.float32 or not np.array_equal(commands, probe(pose)):
            raise ValueError(f"Expected the matched random probe in {name}")
        motion = real_array(tape["motion"], (K, 7), "measured motion")
        rewards = real_array(tape["rewards"], (K,), "native rewards")
        done, success = tape["done"], tape["success"]
        if (done.shape != (K,) or success.shape != (K,) or done.dtype != np.bool_
                or success.dtype != np.bool_ or (done[:-1] | success[:-1]).any()
                or not np.array_equal(rewards > 0, success)):
            raise ValueError(f"Invalid native transition sequence in {name}")
    return History(commands, motion, name)


def source_tape(out, from_stage, body, pose, evidence):
    name = f"practice-{body}-{pose}.npz"
    replay = from_stage is not None and (from_stage / name).is_file()
    if replay:
        shutil.copyfile(from_stage / name, out / name)
    else:
        practice(out, body, pose)
    history = read_tape(out, body, pose)
    evidence[name] = {"origin": "replayed" if replay else "collected", "sha256": digest_file(out / name)}
    return history


@torch.inference_mode()
def motion_check(net, supports, checks):
    rows = []
    per_arm = {}
    for body, other in zip(SOURCES, reversed(SOURCES), strict=True):
        query = checks[body]
        commands = torch.from_numpy(query.commands)
        observed = torch.from_numpy(query.motion) / net.motion_scale
        scores = {"correct": [], "wrong": [], "zero": []}
        for i, pose in enumerate(SUPPORT):
            history = supports[body][i]
            donor = supports[other][i]
            correct = net.encode(history.tensor().unsqueeze(0))
            wrong = net.encode(donor.tensor().unsqueeze(0))
            contexts = {"correct": correct, "wrong": wrong, "zero": torch.zeros_like(correct)}
            row = {}
            for condition, context in contexts.items():
                prediction = net.affine(net.response, context.expand(len(commands), -1), commands)
                row[condition] = (prediction - observed).square().mean().item()
                if not np.isfinite(row[condition]):
                    raise FloatingPointError(f"Nonfinite pose {CHECK_POSE} score for {body}/{pose}/{condition}")
                scores[condition].append(row[condition])
            rows.append({"body": body, "support_pose": pose, "correct": history.path,
                         "wrong": donor.path, "query": query.path, "normalized_mse": row})
        per_arm[body] = {condition: float(np.mean(values)) for condition, values in scores.items()}
    aggregate = {c: float(np.mean([arm[c] for arm in per_arm.values()])) for c in ("correct", "wrong", "zero")}
    return {"pose": CHECK_POSE, "rows": rows, "per_arm": per_arm, "aggregate": aggregate}


def run(args, report):
    torch.manual_seed(FIT_SEED)
    torch.set_num_threads(1)
    rng = np.random.default_rng(FIT_SEED)
    net = ReaderAdapter().train()
    supports, queries = {}, {}

    def tape(body, pose):
        return source_tape(args.out, args.from_stage, body, pose, report["tapes"])

    for body in SOURCES:
        supports[body] = [tape(body, pose) for pose in SUPPORT]
        histories = [tape(body, pose) for pose in MOTION_QUERY]
        queries[body] = [Query(h.commands, h.motion, h.path) for h in histories]
    report["source_histories"] = {
        b: {"supports": [h.path for h in supports[b]], "motion_queries": [q.path for q in queries[b]]}
        for b in SOURCES
    }
    motions = np.concatenate([h.motion for hs in supports.values() for h in hs])
    scale = np.maximum(np.sqrt((motions.astype(np.float64) ** 2).mean(0)), 1e-6)
    net.motion_scale.copy_(torch.as_tensor(scale))
    report["pretrain"] = train_stage(net, supports, queries, STEPS, BATCH_SIZE, rng, motion=True)
    if not all(torch.isfinite(t).all() for t in net.state_dict().values()):
        raise FloatingPointError("Nonfinite source checkpoint")
    checkpoint = args.out / "reader-stage1.pt"
    torch.save({"version": 1, "stage": "source_motion", "revision": args.revision,
                "steps": STEPS, "batch_size": BATCH_SIZE, "fit_seed": FIT_SEED,
                "state_dict": net.state_dict(), "rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state()}, checkpoint)
    report.update(checkpoint=checkpoint.name, checkpoint_sha256=digest_file(checkpoint),
                  motion_scale=net.motion_scale.tolist())
    write_json(args.out / "report.json", report)
    net.eval().requires_grad_(False)
    checks = {body: tape(body, CHECK_POSE) for body in SOURCES}
    report["motion_check"] = motion_check(net, supports, checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    parser.add_argument("--revision", required=True, help="deployed code revision")
    parser.add_argument("--from-stage", type=Path, help="directory containing native source practice tapes")
    args = parser.parse_args()
    if args.from_stage is not None and not args.from_stage.is_dir():
        parser.error("--from-stage must be an existing directory")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"stage": "source_motion", "status": "running", "revision": args.revision,
              "argv": sys.argv, "torch": str(torch.__version__), "numpy": np.__version__,
              "task_id": TASK, "train_bodies": SOURCES, "support_poses": SUPPORT,
              "motion_query_poses": MOTION_QUERY, "validation_pose": CHECK_POSE,
              "fit_seed": FIT_SEED, "batch_size": BATCH_SIZE, "steps": STEPS,
              "practice_transitions": K, "from_stage": str(args.from_stage.resolve())
              if args.from_stage is not None else None, "tapes": {}}
    path = args.out / "report.json"
    write_json(path, report)
    try:
        run(args, report)
        report["status"] = "complete"
        write_json(path, report)
    except Exception as exc:
        report.update(status="error", reason=str(exc))
        write_json(path, report)
        raise
    print(path, flush=True)


if __name__ == "__main__":
    main()
