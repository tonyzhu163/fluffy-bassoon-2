"""Source-only Stage 2 motion matching on frozen Panda SFT proposals."""

import argparse
import fcntl
import json
from pathlib import Path
import sys

import numpy as np
import torch

import adapt_eval as ae
from pretrain_reader import NATIVE_RESET, read_tape
from source_control import audit


BODIES = ("Panda", "Sawyer")
POSES = (24, 25)
TIMES = (0, 5, 10, 15)
STAGE1_SHA = "f2add8aa19135a6841d0970bbeb496951714a6e03bd255579ed4664a95ac07e3"
STEPS = 150
RIDGE = 0.01
LR = 0.003
RANGE_WEIGHT = 1.0
ACTION_WEIGHT = 0.01
CLIP_NORM = 5.0


def read_source(stage, report):
    receipt = json.loads((stage / "report.json").read_text())
    checkpoint = stage / "reader-stage1.pt"
    digest = ae.digest_file(checkpoint)
    if (receipt["status"] != "complete" or receipt["stage"] != "source_motion"
            or receipt["train_bodies"] != list(BODIES) or receipt["task_id"] != ae.TASK
            or receipt["checkpoint"] != checkpoint.name or receipt["steps"] != 100
            or digest != receipt["checkpoint_sha256"] or digest != STAGE1_SHA):
        raise ValueError("Expected the completed reader01 Stage 1 report and matching checkpoint")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (saved["version"], saved["stage"], saved["steps"], saved["revision"]) != (
            1, "source_motion", 100, receipt["revision"]):
        raise ValueError("Stage 1 checkpoint metadata differs from the report")
    net = ae.ReaderAdapter()
    net.load_state_dict(saved["state_dict"], strict=True)
    net.response.requires_grad_(False)
    support, bases = [], []
    scale = net.motion_scale.numpy().astype(np.float64)
    if not np.array_equal(scale, receipt["motion_scale"]) or not np.all(scale > 0):
        raise ValueError("Stage 1 motion scale differs from the receipt")
    report["stage1"] = {"directory": str(stage.resolve()), "revision": saved["revision"],
                        "report_sha256": ae.digest_file(stage / "report.json"),
                        "checkpoint_sha256": digest, "motion_scale": scale.tolist(), "tapes": {}}
    report["bases"] = {}
    for body in BODIES:
        histories = {}
        for pose in (*ae.SUPPORT, *ae.MOTION_QUERY):
            name = f"practice-{body}-{pose}.npz"
            tape_digest = ae.digest_file(stage / name)
            if tape_digest != receipt["tapes"][name]["sha256"]:
                raise ValueError(f"Source tape fingerprint mismatch: {name}")
            histories[pose] = read_tape(stage, body, pose)
            report["stage1"]["tapes"][name] = tape_digest
        support.append(torch.stack([histories[p].tensor() for p in ae.SUPPORT]))
        x = np.concatenate([histories[p].commands for p in ae.MOTION_QUERY]).astype(np.float64)
        y = np.concatenate([histories[p].motion for p in ae.MOTION_QUERY]) / scale
        # Centering gives an unpenalized intercept in the ridge solution.
        xc, yc = x - x.mean(0), y - y.mean(0)
        gain = np.linalg.solve(xc.T @ xc + RIDGE * np.eye(7), xc.T @ yc)
        intercept = y.mean(0) - x.mean(0) @ gain
        residual = x @ gain + intercept - y
        report["bases"][body] = {
            "files": [histories[p].path for p in ae.MOTION_QUERY], "n": len(x),
            "gain": gain.tolist(), "intercept": intercept.tolist(),
            "fit_mse_by_axis": np.mean(residual ** 2, axis=0).tolist(),
            "intercept_only_mse_by_axis": np.mean(yc ** 2, axis=0).tolist(),
            "command_min": x.min(0).tolist(), "command_max": x.max(0).tolist(),
        }
        bases.append(torch.from_numpy(np.vstack([gain, intercept])).float())
    return net, torch.stack(support), torch.stack(bases)


def cache_practice(out, motor, body, pose, prompt):
    if body not in BODIES or pose not in POSES:
        raise ValueError("Unexpected Stage 2 source reset")
    commands = ae.probe(pose)
    chunks, transitions = [], []
    frames = {k: [] for k in ("rgb", "wrist", "state")}
    with ae.session(body) as env:
        obs, reset, horizon = ae.reset(env, body, pose)
        if horizon < len(commands) or any(reset.get(k) != v for k, v in NATIVE_RESET.items()):
            raise ValueError("Wrong native practice reset")
        for t, command in enumerate(commands):
            if t in TIMES:
                public = ae.public_obs(obs)
                if public["prompt"] != prompt:
                    raise ValueError("Practice instruction differs from Panda motor training")
                for key in frames:
                    frames[key].append(public[key].copy())
                chunk = ae.proposal(motor, public, ae.plan_seed(pose, t))
                if chunk.shape != (ae.EXECUTE, 7):
                    raise ValueError(f"Unexpected motor horizon at {body}/{pose}/{t}")
                chunks.append(chunk)
            obs, motion, reward, done, success = ae.step(env, command)
            transitions.append((motion, reward, done, success))
            if (done or success) and t + 1 < len(commands):
                raise RuntimeError(f"Native practice terminated early at {body}/{pose}/{t}")
    motion, rewards, dones, successes = zip(*transitions, strict=True)
    chunks = np.stack(chunks)
    path = out / f"proposals-{body}-{pose}.npz"
    np.savez_compressed(path, **{k: np.stack(v) for k, v in frames.items()}, chunks=chunks,
                        times=TIMES, plan_seeds=[ae.plan_seed(pose, t) for t in TIMES],
                        body=body, pose=pose, seed=ae.seed(pose), prompt=prompt,
                        commands=commands, motion=np.stack(motion), rewards=rewards,
                        done=dones, success=successes, reset_json=json.dumps(reset, sort_keys=True))
    return torch.from_numpy(chunks.reshape(-1, 7)), {
        "file": path.name, "sha256": ae.digest_file(path), "chunk_sha256": ae.digest_array(chunks),
        "body": body, "pose": pose, "seed": ae.seed(pose), "shape": list(chunks.shape),
    }


def frozen_counts(motor):
    counts = {"requires_grad": sum(p.requires_grad for p in motor.parameters()),
              "has_grad": sum(p.grad is not None for p in motor.parameters())}
    if motor.training or any(counts.values()):
        raise RuntimeError("Expected a frozen eval-mode motor without parameter gradients")
    return counts


def fit(net, supports, bases, bank, report):
    # One full batch crosses each arm's two histories with every row of its chunks.
    n_bodies, n_support = supports.shape[:2]
    raw = bank[:, None].expand(-1, n_support, -1, -1)
    panda = BODIES.index("Panda")
    target = raw @ bases[panda, :7] + bases[panda, 7]

    def objective(context="correct"):
        z = net.encode(supports.flatten(0, 1)).reshape(n_bodies, n_support, -1)
        if context == "wrong":
            z = z.flip([0])
        elif context == "cleared":
            z = torch.zeros_like(z)
        z = z.unsqueeze(2).expand(-1, -1, raw.size(2), -1)
        unbounded = net.adapt(z.reshape(-1, z.size(-1)), raw.reshape(-1, 7)).reshape_as(raw)
        candidate = unbounded.clamp(-1, 1)
        pred = candidate @ bases[:, None, :7] + bases[:, None, None, 7]
        mse = (pred - target).square().mean(dim=(1, 2, 3))
        overflow = (unbounded - candidate).square().mean()
        change = (candidate - raw).square().mean()
        loss = mse.mean() + RANGE_WEIGHT * overflow + ACTION_WEIGHT * change
        return loss, {"motion_mse": dict(zip(BODIES, mse.detach().tolist(), strict=True)),
                      "range_mse": overflow.item(), "action_change_mse": change.item(),
                      "loss": loss.item()}

    @torch.no_grad()
    def scores():
        return {k: objective(k)[1] for k in ("correct", "cleared", "wrong")}

    modules = {"reader": net.reader, "adapter": net.adapter}
    initial = {k: [p.detach().clone() for p in mod.parameters()] for k, mod in modules.items()}
    params = [p for mod in modules.values() for p in mod.parameters()]
    opt = torch.optim.Adam(params, lr=LR)
    report["pre"] = scores()
    report["curve"] = []
    for update in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        loss, row = objective()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, CLIP_NORM, error_if_nonfinite=True)
        opt.step()
        report["curve"].append({"update": update, "grad_norm": norm.item(), **row})
    report["post"] = scores()
    changes = {k: sum((p.detach().double() - start.double()).square().sum()
                      for p, start in zip(mod.parameters(), initial[k], strict=True)).sqrt().item()
               for k, mod in modules.items()}
    report["parameter_change_l2"] = changes
    if not all(np.isfinite(v) and v > 0 for v in changes.values()):
        raise RuntimeError("Joint fit must change both reader and adapter parameters")
    if not all(v.isfinite().all() for v in net.state_dict().values()):
        raise FloatingPointError("Nonfinite reader-adapter state")


def run(args, report):
    torch.manual_seed(ae.FIT_SEED)
    torch.set_num_threads(1)
    net, supports, bases = read_source(args.pretrain, report)
    ckpt = args.motor_output / "checkpoint.pt"
    with (args.motor_output / "motor.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        audited, prompt = audit(ckpt, open_loop=False)
        motor_id = audited["checkpoint"]
        if motor_id["selected_step"] != 150:
            raise ValueError("Expected panda01's validation-selected update 150")
        report["motor"] = motor_id
        report["motor_sha256"] = motor_id["sha256"]
        motor = ae.load_motor(args)
        report["weights"] = motor.identity
        report["motor_before"] = frozen_counts(motor)
        banks = []
        report["proposals"] = []
        for body in BODIES:
            chunks = []
            for pose in POSES:
                rows, receipt = cache_practice(args.out, motor, body, pose, prompt)
                chunks.append(rows)
                report["proposals"].append(receipt)
            banks.append(torch.cat(chunks))
        fit(net, supports, bases, torch.stack(banks), report)
        report["motor_after"] = frozen_counts(motor)
        report["motor_sha256_after"] = ae.digest_file(ckpt)
        if report["motor_sha256_after"] != motor_id["sha256"]:
            raise RuntimeError("Frozen motor checkpoint changed during Stage 2")
    checkpoint = args.out / "reader-adapter.pt"
    torch.save({"version": 1, "state_dict": net.state_dict()}, checkpoint)
    restored = ae.load_reader(args.out)
    if any(not torch.equal(v, restored.state_dict()[k]) for k, v in net.state_dict().items()):
        raise RuntimeError("Saved reader-adapter differs on evaluator readback")
    report.update(checkpoint=checkpoint.name, checkpoint_sha256=ae.digest_file(checkpoint))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("motor-output", "weights", "pretrain", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    prior = here.parent / "libero-crossarm-20261120-r1"
    files = [here / n for n in ("motion_joint.py", "adapt_eval.py", "pretrain_reader.py",
                                "source_control.py", "sim.py", "motor.py", "motor_data.py", "motor_weights.py")]
    files.extend(prior / n for n in ("arm_env.py", "sim_reset.py"))
    report = {
        "stage": "source_motion_joint_diagnostic", "train_bodies": list(BODIES), "status": "running", "revision": args.revision,
        "argv": sys.argv, "torch": str(torch.__version__), "numpy": np.__version__,
        "code_sha256": {p.name: ae.digest_file(p) for p in files},
        "config": {"task": ae.TASK, "bodies": BODIES, "support": ae.SUPPORT, "basis": ae.MOTION_QUERY,
                   "proposal_poses": POSES, "proposal_times": TIMES, "practice_length": ae.K,
                   "seed": ae.FIT_SEED, "chunk_length": ae.EXECUTE, "reference": "Panda",
                   "optimizer": "Adam", "lr": LR, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0,
                   "steps": STEPS, "sampling": "full Cartesian batch per arm", "dropout": 0,
                   "grad_clip_l2": CLIP_NORM, "ridge_sum_squares": RIDGE, "range_weight": RANGE_WEIGHT,
                   "action_weight": ACTION_WEIGHT, "controller_bounds": [-1, 1]},
    }
    path = args.out / "report.json"
    ae.write_json(path, report)
    try:
        run(args, report)
    except Exception as exc:
        report.update(status="error", reason=str(exc))
        ae.write_json(path, report)
        raise
    report["status"] = "complete"
    ae.write_json(path, report)
    print(path, flush=True)


if __name__ == "__main__":
    main()
