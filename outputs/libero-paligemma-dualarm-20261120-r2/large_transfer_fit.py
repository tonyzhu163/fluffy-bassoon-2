"""Source-only dynamics pretraining and frozen-motor expert residual fit."""

import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from large_transfer_stats import ARMS, read, require, sha

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "libero-paligemma-crossarm-20261120-r1"
sys.path.insert(1, str(PRIOR))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def source_code():
    paths = [
        HERE / f
        for f in (
            "large_transfer_fit.py",
            "large_transfer_bank.py",
            "transfer_fit.py",
            "transfer_gate.py",
            "generated_split.py",
            "recover.py",
            "motor_bank.py",
            "servo_motor.py",
            "waypoint_motor.py",
            "large_panel_protocol.py",
            "large_panel_rollout.py",
            "large_panel_sim.py",
        )
    ]
    paths.extend(
        PRIOR / f
        for f in (
            "adapt_eval.py",
            "pretrain_reader.py",
            "sim.py",
            "motor.py",
            "motor_data.py",
            "motor_weights.py",
        )
    )
    return {p.name: sha(p) for p in sorted(paths)}


@contextmanager
def frozen(path, weights):
    import motor
    import torch
    from motor_bank import validate_source_records
    from servo_motor import load_selected
    from transfer_gate import inspect_motor

    motor.require_gpu()
    torch.manual_seed(41)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    digest = sha(path)
    saved = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
    contract = saved["contract"]
    require(
        saved["schema"] == motor.SCHEMA
        and saved["step"] == saved["best"]["step"]
        and contract["protocol"] == "dualarm-native-goal5-v2"
        and contract["source_bodies"] == list(ARMS)
        and contract["weights"]["repo"] == "google/paligemma-3b-mix-224"
        and {r["body"] for r in contract["episodes"]} == set(ARMS),
        "Expected exact dual-source PaliGemma candidate and training rows",
    )
    validate_source_records(contract["episodes"], contract.get("public_row_fingerprints"))
    prompts = {r["prompt"] for r in contract["episodes"]}
    require(len(prompts) == 1, "Multiple source tasks")
    if contract.get("architecture") == "plate_waypoint_v1":
        from waypoint_motor import WaypointMotor

        model = WaypointMotor(weights)
        model.restore_delta(saved["best"]["model"])
        model.requires_grad_(False).eval()
    else:
        model = load_selected(saved, weights)
    identity = {
        "motor_sha256": digest,
        "fit_id": saved["fit_id"],
        "step": saved["step"],
        "prompt": prompts.pop(),
        "weights": contract["weights"],
        "frozen": inspect_motor(model, saved),
    }
    try:
        yield model, identity
    finally:
        require(
            sha(path) == digest and inspect_motor(model, saved) == identity["frozen"],
            "Frozen motor changed during the stage",
        )


class PanelPolicy:
    """Use the same batch-shaped frozen inference as the qualified source panel."""

    def __init__(self, model):
        self.model = model

    def act(self, obs):
        from large_panel_rollout import infer

        return infer(self.model, [obs])[0]


def fit(args):
    import adapt_eval as ev
    import torch
    import transfer_fit as trainer
    from large_transfer_bank import cache_experts

    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "stage": "large-source-joint-v1",
        "status": "running",
        "revision": args.revision,
        "code_sha256": source_code(),
        "source_bodies": list(ARMS),
    }
    path = args.out / "report.json"
    ev.write_json(path, report)
    with frozen(args.checkpoint, args.weights) as (motor, identity):
        directory = args.stage1
        if directory is None:
            directory = args.out / "stage1"
            subprocess.run(
                [
                    sys.executable,
                    "-Bu",
                    str(PRIOR / "pretrain_reader.py"),
                    "--out",
                    str(directory),
                    "--revision",
                    args.revision,
                ],
                check=True,
            )
        net, histories, report["stage1"] = trainer.load_stage1(directory, set())
        # The loader numbers banks in order and attributes duplicates to the first tape.
        directories = [p.resolve(strict=True) for p in args.demos]
        require(
            len(set(directories)) == len(directories), "Repeated source bank directory"
        )
        banks, sources = cache_experts(
            args.out, directories, PanelPolicy(motor), identity["prompt"]
        )
        report.update(motor=identity, expert_data=sources)
        ev.write_json(path, report)
        net.train()
        report["joint"] = trainer.joint_fit(net, histories, banks)
        net.eval().requires_grad_(False)
        report["crossed_validation"] = trainer.crossed_scores(
            net, histories, banks["validation"]
        )
        state = args.out / "reader-adapter.pt"
        torch.save(
            {
                "stage": report["stage"],
                "motor_sha256": identity["motor_sha256"],
                "stage1_sha256": report["stage1"]["checkpoint_sha256"],
                "state_dict": net.state_dict(),
            },
            state,
        )
        restored = torch.load(state, weights_only=True, map_location="cpu")
        require(
            all(
                torch.equal(v, restored["state_dict"][k])
                for k, v in net.state_dict().items()
            ),
            "Reader checkpoint differs from joint optimization",
        )
        report["checkpoint_sha256"] = sha(state)
    report["status"] = "complete"
    ev.write_json(path, report)


def restore(directory, motor, reserved):
    """Reopen source evidence and match the motor-specific joint fit before target access."""
    import adapt_eval as ev
    import torch
    import transfer_fit as trainer
    from large_transfer_bank import verify_expert_data

    report = read(directory / "report.json")
    require(
        report["status"] == "complete"
        and report["stage"] == "large-source-joint-v1"
        and report["code_sha256"] == source_code()
        and report["motor"] == motor,
        "Incompatible source-only joint fit",
    )
    _, _, stage1 = trainer.load_stage1(Path(report["stage1"]["directory"]), reserved)
    require(stage1 == report["stage1"], "Changed source dynamics evidence")
    verify_expert_data(
        directory, report["expert_data"], motor["prompt"], reserved
    )
    joint = report["joint"]
    require(
        len(joint["curve"]) == trainer.STEPS
        and joint["per_body_per_update"] == trainer.PER_BODY
        and all(
            row["step"] == i and set(row["action_mse"]) == set(ARMS)
            for i, row in enumerate(joint["curve"], 1)
        )
        and all(joint["parameter_change_l2"][k] > 0 for k in ("reader", "adapter")),
        "Missing evidence of joint reader and residual updates",
    )
    path = directory / "reader-adapter.pt"
    require(sha(path) == report["checkpoint_sha256"], "Changed source reader")
    saved = torch.load(path, weights_only=True, map_location="cpu")
    require(
        saved["stage"] == report["stage"]
        and saved["motor_sha256"] == motor["motor_sha256"]
        and saved["stage1_sha256"] == stage1["checkpoint_sha256"],
        "Incompatible reader checkpoint",
    )
    net = ev.ReaderAdapter()
    net.load_state_dict(saved["state_dict"])
    require(
        all(v.isfinite().all() for v in net.state_dict().values()),
        "Nonfinite source reader",
    )
    return net.eval().requires_grad_(False)
