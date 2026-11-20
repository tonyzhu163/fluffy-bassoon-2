"""Fit a source-only reader/residual and evaluate a confirmed dual-arm motor on UR5e."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
R1 = HERE.parent / "libero-paligemma-crossarm-20261120-r1"
sys.path.insert(1, str(R1))

import adapt_eval as ev
from motor_train import source_ids
from pretrain_reader import NATIVE_RESET
import transfer_fit as trainer
from transfer_gate import CONFIRM, GateClosed, confirmed_motor


TARGET = "UR5e"
PRACTICE = tuple(range(8, 16))
CALIBRATION = (48, 49)
CALIBRATION_CHECK = 39
CONDITIONS = ("raw", "correct", "empty", "wrong", "calibration")


def implementations():
    own = ("transfer.py", "transfer_fit.py", "transfer_gate.py")
    return {**source_ids(), **{name: ev.digest_file(HERE / name) for name in own}}


def fit(args, report, motor):
    gate = report["confirmation"]
    directory = args.stage1
    if directory is None:
        directory = args.out / "stage1"
        subprocess.run([sys.executable, "-Bu", str(R1 / "pretrain_reader.py"), "--out", str(directory),
                        "--revision", args.revision], check=True)
    reserved = set(gate["released_rows"].values())
    net, histories, report["stage1"] = trainer.load_stage1(directory, reserved)
    banks, report["expert_data"] = trainer.cache_experts(args.out, args.demos, motor, gate["prompt"], reserved)
    report["train_bodies"] = list(ev.SOURCES)
    report["fit_layouts"] = {b: sorted(poses) for b, poses in banks["train"].items()}
    report["validation_layouts"] = {b: sorted(poses) for b, poses in banks["validation"].items()}
    ev.write_json(args.out / "report.json", report)
    net.train()
    report["joint"] = trainer.joint_fit(net, histories, banks)
    net.eval().requires_grad_(False)
    report["crossed_validation"] = trainer.crossed_scores(net, histories, banks["validation"])
    checkpoint = args.out / "reader-adapter.pt"
    binding = {
        "version": 2, "stage": "dual_source_expert", "train_bodies": list(ev.SOURCES),
        "motor_sha256": gate["motor_sha256"], "confirmation_sha256": gate["report_sha256"],
        "stage1_sha256": report["stage1"]["checkpoint_sha256"],
    }
    torch.save({**binding, "state_dict": net.state_dict()}, checkpoint)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if any(not torch.equal(v, restored["state_dict"][k]) for k, v in net.state_dict().items()):
        raise ValueError("Reader/residual checkpoint differs from the final joint update")
    report["checkpoint_sha256"] = ev.digest_file(checkpoint)


def fitted_reader(directory, gate, report):
    path = directory / "report.json"
    training = json.loads(path.read_bytes())
    checkpoint = directory / "reader-adapter.pt"
    digest = ev.digest_file(checkpoint)
    if (training["stage"] != "fit" or training["status"] != "complete"
            or training["train_bodies"] != list(ev.SOURCES) or training["code_sha256"] != report["code_sha256"]
            or training["confirmation"] != gate or training["checkpoint_sha256"] != digest
            or training["motor_sha256_after"] != gate["motor_sha256"]):
        raise ValueError("Completed source fit, confirmed motor and current implementations must match")
    actual = training["expert_data"]["episodes"]
    if any(r["body"] not in ev.SOURCES or r["split"] not in ("train", "validation") for r in actual):
        raise ValueError("Unregistered body or split in the source joint fit")
    for split, allowed in (("train", trainer.TRAIN), ("validation", trainer.VALID)):
        used = [r for r in actual if r["split"] == split]
        if ({r["body"] for r in used} != set(ev.SOURCES)
                or any(r["pose"] not in allowed or not r["native_terminal_reward"] > 0 for r in used)):
            raise ValueError("Stage 2 fit lacks the required two-body native expert data")
    curve = training["joint"]["curve"]
    if (len(curve) != trainer.STEPS or any(set(row["action_mse"]) != set(ev.SOURCES) for row in curve)
            or training["joint"]["per_body_per_update"] != trainer.PER_BODY
            or not all(training["joint"]["parameter_change_l2"][k] > 0 for k in ("reader", "adapter"))):
        raise ValueError("Missing evidence of balanced updates to both reader and residual")
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    expected = {
        "version": 2, "stage": "dual_source_expert", "train_bodies": list(ev.SOURCES),
        "motor_sha256": gate["motor_sha256"], "confirmation_sha256": gate["report_sha256"],
        "stage1_sha256": training["stage1"]["checkpoint_sha256"],
    }
    if any(saved.get(k) != v for k, v in expected.items()):
        raise ValueError("Reader checkpoint is bound to a different source fit")
    net = ev.ReaderAdapter()
    net.load_state_dict(saved["state_dict"])
    if not all(t.isfinite().all() for t in net.state_dict().values()):
        raise ValueError("Nonfinite fitted reader or residual")
    report.update(fit_report_sha256=ev.digest_file(path), checkpoint_sha256=digest,
                  train_bodies=training["train_bodies"])
    return net.eval().requires_grad_(False)


def checked_practice(out, body, pose, prompt, reserved, *, pulses=False):
    history = ev.practice(out, body, pose, calibration=pulses)
    tape_path = out / history.path
    with np.load(tape_path, allow_pickle=False) as tape:
        physics = json.loads(tape["reset_json"].item())
        if (any(physics.get(k) != v for k, v in NATIVE_RESET.items()) or tape["prompt"].item() != prompt
                or physics["released_row_fingerprint"] in reserved
                or tape["body"].item() != body or tape["pose"].item() != pose
                or tape["seed"].item() != ev.seed(pose)
                or not np.array_equal(history.commands, ev.probe(pose, calibration=pulses))):
            raise ValueError(f"Unmatched native practice on {body}/{pose}")
    return history, {"body": body, "pose": pose, "seed": ev.seed(pose),
                     "file": history.path, "sha256": ev.digest_file(tape_path),
                     "released_row": physics["released_row_fingerprint"], "scene": physics["scene_fingerprint"]}


def bind_target(report):
    # Share the campaign's existing source lock. Replays can recover from an
    # interrupted panel in a fresh output without admitting a second reader.
    seal_dir = Path(report["confirmation"]["seal_dir"])
    path = seal_dir / "ur5e-transfer.json"
    binding = {k: report[k] for k in ("target", "practice", "queries", "conditions", "code_sha256",
                                       "fit_report_sha256", "checkpoint_sha256")}
    binding.update(motor_sha256=report["confirmation"]["motor_sha256"],
                   confirmation_sha256=report["confirmation"]["report_sha256"],
                   calibration_poses=list(CALIBRATION), calibration_check=CALIBRATION_CHECK)
    with (seal_dir / "motor.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        replay = path.exists()
        if replay:
            if json.loads(path.read_bytes()) != binding:
                raise ValueError("A different source fit or query panel was already bound before target access")
        else:
            ev.write_json(path, binding)
    return {"file": str(path), "sha256": ev.digest_file(path), "replay": replay}


@torch.inference_mode()
def evaluate(args, report, motor):
    gate = report["confirmation"]
    net = fitted_reader(args.fit, gate, report)
    prompt = gate["prompt"]
    reserved = set(gate["released_rows"].values())
    path = args.out / "report.json"
    report.update(target=TARGET, practice=list(PRACTICE), queries=list(CONFIRM),
                  conditions=list(CONDITIONS), calibration={}, pairs=[])
    report["target_binding"] = bind_target(report)
    ev.write_json(path, report)
    responses = {}
    for body in ("Panda", TARGET):
        pulses = [checked_practice(args.out, body, p, prompt, reserved, pulses=True) for p in CALIBRATION]
        response = ev.measured_response([h for h, _ in pulses])
        check, check_file = checked_practice(args.out, body, CALIBRATION_CHECK, prompt, reserved)
        report["calibration"][body] = {**response.report(check), "pulse_tapes": [r for _, r in pulses],
                                       "check_tape": check_file}
        responses[body] = response
    panda = report["calibration"]["Panda"]
    target = report["calibration"][TARGET]
    for a, b in zip([*panda["pulse_tapes"], panda["check_tape"]],
                    [*target["pulse_tapes"], target["check_tape"]], strict=True):
        if any(a[k] != b[k] for k in ("released_row", "scene", "seed")):
            raise ValueError("Independent calibration resets used different released layouts")
    calibration = ev.privileged(responses["Panda"], responses[TARGET])
    for i, (pose, support) in enumerate(zip(CONFIRM, PRACTICE, strict=True)):
        donor = ev.SOURCES[i % len(ev.SOURCES)]
        correct, correct_file = checked_practice(args.out, TARGET, support, prompt, reserved)
        wrong, wrong_file = checked_practice(args.out, donor, support, prompt, reserved)
        if (not np.array_equal(correct.commands, wrong.commands)
                or any(correct_file[k] != wrong_file[k] for k in ("released_row", "scene"))):
            raise ValueError(f"Wrong-arm practice differs in commands or initial scene on layout {support}")
        transforms = {"raw": None, "correct": ev.learned(net, correct), "empty": ev.learned(net, None),
                      "wrong": ev.learned(net, wrong), "calibration": calibration}
        results = {}
        report["pairs"].append({"pose": pose, "practice_pose": support, "donor": donor,
                                "correct": correct_file, "wrong": wrong_file, "conditions": results})
        for condition in CONDITIONS:
            opening = None if condition == "raw" else results["raw"]["opening"]
            row = ev.episode(args.out, motor, TARGET, pose, condition, transforms[condition], opening)
            observed = row["opening"]
            if (observed["observation"]["prompt"] != prompt or observed["horizon"] != ev.HORIZON
                    or any(observed["physics"].get(k) != v for k, v in NATIVE_RESET.items())
                    or observed["physics"]["released_row_fingerprint"] != gate["released_rows"][str(pose)]):
                raise ValueError(f"Native target query mismatch on layout {pose}")
            row["tape_sha256"] = ev.digest_file(args.out / row["tape"])
            results[condition] = row
            ev.write_json(path, report)
    paired = [pair["conditions"] for pair in report["pairs"]]
    report["native_successes"] = {k: sum(row[k]["success"] for row in paired) for k in CONDITIONS}
    report["terminal_rewards"] = {k: [row[k]["terminal_reward"] for row in paired] for k in CONDITIONS}
    report["versus_raw"] = {k: ev.paired_change(paired, k, "raw") for k in CONDITIONS if k != "raw"}
    report["correct_versus"] = {k: ev.paired_change(paired, "correct", k) for k in CONDITIONS if k != "correct"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("fit", "evaluate"):
        cmd = sub.add_parser(stage)
        cmd.add_argument("--motor-output", type=Path, required=True, help="immutable FIT/selected")
        cmd.add_argument("--confirmation", type=Path, required=True, help="FIT/confirm directory")
        cmd.add_argument("--weights", type=Path, default=os.environ.get("DUAL_WEIGHTS"),
                         required=not os.environ.get("DUAL_WEIGHTS"), help="base snapshot verified by Motor.load")
        cmd.add_argument("--revision", default=os.environ.get("MOTOR_CODE_REVISION"),
                         required=not os.environ.get("MOTOR_CODE_REVISION"))
        cmd.add_argument("--out", required=True, type=Path, help="fresh stage output directory")
        if stage == "fit":
            cmd.add_argument("--stage1", type=Path, help="complete source motion fit with native tapes "
                             "(omit to retrain on the registered source probes)")
            cmd.add_argument("--demos", type=Path, nargs="+", action="extend", required=True,
                             help="native source bank directories")
        else:
            cmd.add_argument("--fit", type=Path, required=True, help="completed source joint fit directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    path = args.out / "report.json"
    report = {"stage": args.stage, "status": "running", "task": "BASE libero_goal/5", "revision": args.revision,
              "argv": sys.argv, "torch": str(torch.__version__), "numpy": np.__version__,
              "execute_prefix": ev.EXECUTE, "practice_transitions": ev.K, "horizon": ev.HORIZON,
              "reset_seed_base": ev.seed(0), "plan_seed_rule": "900000 + 1000 * pose + elapsed_controls"}
    ev.write_json(path, report)
    try:
        report["code_sha256"] = implementations()
        torch.set_num_threads(1)
        torch.manual_seed(ev.FIT_SEED)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        with confirmed_motor(args, report) as motor:
            if args.stage == "fit":
                fit(args, report, motor)
            else:
                evaluate(args, report, motor)
        report["status"] = "complete"
    except GateClosed as exc:
        report.update(status="gated", reason=str(exc))
        print(f"{path}: {exc}", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        report.update(status="error", error_type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        ev.write_json(path, report)
    print(path, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
