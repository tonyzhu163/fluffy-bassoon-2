"""Validate the selected checkpoint and its native two-source confirmation before transfer."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path

import torch

import adapt_eval as ev
from motor_eval import check_episode, rank
from pretrain_reader import NATIVE_RESET


CONFIRM = tuple(range(40, 48))
MIN_SUCCESSES = 6


def check_source_contract(saved):
    contract = saved["contract"]
    if (contract["protocol"] != "dualarm-native-goal5-v2"
            or contract["weights"]["repo"] != "google/paligemma-3b-mix-224"
            or contract["source_bodies"] != list(ev.SOURCES)):
        raise ValueError("Expected a two-source R2 PaliGemma motor")
    prompts = set()
    layouts = {k: {b: set() for b in ev.SOURCES} for k in ("train", "validation")}
    for row in contract["episodes"]:
        body, pose, split = row["body"], int(row["pose"]), row["split"]
        allowed = range(28) if split == "train" else range(28, 32)
        if (body not in ev.SOURCES or split not in layouts or pose not in allowed
                or row["frames"] < 1 or not row["native_terminal_reward"] > 0):
            raise ValueError("Motor fit contains an undeclared body, layout or non-expert episode")
        if row.get("reference_pose") is not None and int(row["reference_pose"]) not in (
                range(28) if split == "train" else range(32)):
            raise ValueError("Motor expert reference violates the source split")
        if "duplicate_of" not in row:
            layouts[split][body].add(pose)
        prompts.add(row["prompt"])
    if len(prompts) != 1 or not all(poses for arms in layouts.values() for poses in arms.values()):
        raise ValueError("Both arms need successful training and validation episodes for one instruction")
    return prompts.pop(), {k: {b: sorted(ps) for b, ps in arms.items()} for k, arms in layouts.items()}


class GateClosed(Exception):
    """A completed source confirmation did not clear the registered threshold."""


def confirm_receipt(directory, motor_sha, saved, code_ids):
    path = directory / "report.json"
    report_bytes = path.read_bytes()
    result = json.loads(report_bytes)
    panel_path = directory / "contract.json"
    panel = json.loads(panel_path.read_bytes())
    step, fit_id = saved["best"]["step"], saved["fit_id"]
    prompt, layouts = check_source_contract(saved)
    expected = {"stage": "confirm", "fit_id": fit_id, "source_bodies": list(ev.SOURCES),
                "poses": list(CONFIRM), "required_per_body": MIN_SUCCESSES}
    if (any(result.get(k) != v or panel.get(k) != v for k, v in expected.items())
            or result["status"] != "complete" or result["selected_step"] != step
            or result["motor_checkpoint_sha256"] != motor_sha
            or result["contract_sha256"] != ev.digest_file(panel_path)):
        raise ValueError("Need the complete registered two-arm confirmation of the selected motor")
    candidate, = panel["candidates"]
    if ((candidate["sha256"], candidate["step"], candidate["file"]) != (motor_sha, step, "selected/checkpoint.pt")
            or panel["prompt"] != prompt or panel["base_weights"] != saved["contract"]["weights"]
            or (panel["execute"], panel["horizon"], panel["seed_base"]) != (ev.EXECUTE, ev.HORIZON, ev.seed(0))):
        raise ValueError("Source confirmation contract differs from selected checkpoint or native protocol")
    if panel["sources"] != saved["contract"]["sources"] or any(
            code_ids.get(k) != v for k, v in panel["sources"].items()):
        raise ValueError("Source motor, simulator or confirmation implementation changed")

    # This file was bound before the producer's first confirmation reset.
    seal_dir = Path(panel["seal_dir"])
    seal = json.loads((seal_dir / "selection.json").read_bytes())
    if (seal["confirmation"] != panel or seal["motor_checkpoint_sha256"] != motor_sha
            or seal["fit_id"] != fit_id or seal["selected_step"] != step):
        raise ValueError("Confirmation seal differs from the preselected source winner")
    dev_path = directory.parent / "dev" / "report.json"
    if ev.digest_file(dev_path) != panel["development_report_sha256"]:
        raise ValueError("Source development report changed after confirmation was sealed")
    dev = json.loads(dev_path.read_bytes())
    winner = min(dev["scores"], key=rank)
    if (dev["stage"] != "dev" or dev["status"] != "complete" or dev["source_gate"] is not True
            or dev["fit_id"] != fit_id or dev["selected_step"] != step or dev["motor_checkpoint_sha256"] != motor_sha
            or dev["poses"] != list(range(32, 36)) or winner["sha256"] != motor_sha
            or set(winner["native_successes"]) != set(ev.SOURCES) or min(winner["native_successes"].values()) < 3):
        raise ValueError("Confirmation checkpoint was not selected on the registered source development panel")

    counts = dict.fromkeys(ev.SOURCES, 0)
    seen, fingerprints, tapes = set(), {}, {}
    for row in result["episodes"]:
        body, pose = key = row["body"], row["pose"]
        if body not in ev.SOURCES or pose not in CONFIRM or key in seen:
            raise ValueError(f"Unexpected or repeated confirmation episode {key}")
        seen.add(key)
        tape = directory / f"step-{step:06d}" / f"query-{body}-{pose}-raw.npz"
        # Read the producer's exact NPZ schema and compare its embedded model, seed,
        # raw actions, native rewards, terminal flags and complete receipt to this row.
        verified = check_episode(tape, directory, body, pose, prompt, candidate, fit_id, row["tape_sha256"])
        if verified != row:
            raise ValueError(f"Confirmation report and native tape disagree: {tape}")
        physics = row["opening"]["physics"]
        if any(physics.get(k) != v for k, v in NATIVE_RESET.items()):
            raise ValueError("Source confirmation differs from the registered native BASE task")
        released = physics["released_row_fingerprint"]
        if fingerprints.setdefault(str(pose), released) != released:
            raise ValueError("Source confirmation arms did not share a released layout")
        tapes[row["tape"]] = row["tape_sha256"]
        counts[body] += int(row["success"])
    if seen != {(b, p) for b in ev.SOURCES for p in CONFIRM} or counts != result["native_successes"]:
        raise ValueError("Incomplete confirmation panel or inconsistent native counts")
    exposed = {r["released_row_fingerprint"] for r in saved["contract"]["episodes"]}
    exposed.update(r["opening"]["physics"]["released_row_fingerprint"] for r in dev["episodes"])
    reserved = set(fingerprints.values())
    if len(reserved) != len(CONFIRM) or exposed & reserved:
        raise ValueError("Confirmation needs eight distinct released layouts outside all motor selection data")
    if min(counts.values()) < MIN_SUCCESSES:
        raise GateClosed(f"Confirmation requires >=6/8 per arm, observed {counts}")
    if result["source_gate"] is not True:
        raise ValueError("Source producer did not release this motor")
    return {"motor_sha256": motor_sha, "selected_step": step, "fit_id": fit_id,
            "weights": saved["contract"]["weights"], "prompt": prompt, "motor_layouts": layouts,
            "report_sha256": hashlib.sha256(report_bytes).hexdigest(), "native_successes": counts,
            "released_rows": fingerprints, "tape_sha256": tapes, "seal_dir": str(seal_dir)}


def inspect_motor(motor, saved):
    trainable = sum(p.numel() for p in motor.parameters() if p.requires_grad)
    grads = sum(p.grad is not None for p in motor.parameters())
    if motor.training or trainable or grads or motor.identity != saved["contract"]["weights"]:
        raise ValueError("Expected a frozen confirmed motor and its verified base weights")
    state = motor.state_dict()
    for name, tensor in saved["best"]["model"].items():
        actual = state[name].detach().cpu()
        if actual.dtype != tensor.dtype or not torch.equal(actual, tensor):
            raise ValueError(f"Loaded motor differs from the confirmed checkpoint at {name}")
    return {"trainable_parameters": trainable, "parameter_gradients": grads, "eval_mode": True,
            "matched_checkpoint_tensors": len(saved["best"]["model"])}


@contextmanager
def confirmed_motor(args, report):
    path = args.motor_output / "checkpoint.pt"
    with (args.motor_output / "motor.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        digest = ev.digest_file(path)
        saved = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
        report["confirmation"] = confirm_receipt(args.confirmation, digest, saved, report["code_sha256"])
        ev.write_json(args.out / "report.json", report)
        from servo_motor import load_selected
        motor = load_selected(saved, args.weights)
        report["motor_before"] = inspect_motor(motor, saved)
        yield motor
        report["motor_after"] = inspect_motor(motor, saved)
        report["motor_sha256_after"] = ev.digest_file(path)
        if report["motor_sha256_after"] != digest:
            raise ValueError("Motor checkpoint changed during transfer stage")
