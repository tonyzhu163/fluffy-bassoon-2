"""Source-only history learning and gated native Goal 5 evaluation. See EVAL.md."""

import argparse
from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
import traceback

import numpy as np
import torch
from torch import nn


SOURCES = ("Panda", "Sawyer")
TASK = 5
K = 16
EXECUTE = 5
HORIZON = 300
FIT_SEED = 41
SUPPORT = (20, 21)
MOTION_QUERY = (22, 23)
QUALIFY = (30, 31)
PANELS = {
    "UR5e": {"calibration": (32, 33), "check": 34, "practice": (35, 36), "query": (37, 38)},
    "IIWA": {"calibration": (39, 40), "check": 41, "practice": (42, 43), "query": (44, 45)},
}
CONDITIONS = ("raw", "correct", "cleared", "wrong", "privileged")
PUBLIC = ("rgb", "wrist", "state", "prompt")


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest_array(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(f"{a.dtype.str}/{a.shape}/".encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def poses(text):
    result = []
    for item in text.split(","):
        bounds = item.split("-")
        if len(bounds) > 2 or not all(x.isdigit() for x in bounds):
            raise argparse.ArgumentTypeError("poses must be comma-separated IDs or inclusive ranges")
        first, last = int(bounds[0]), int(bounds[-1])
        if not 0 <= first <= last < 50:
            raise argparse.ArgumentTypeError("expected ordered pose IDs in [0,49]")
        result.extend(range(first, last + 1))
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("duplicate pose")
    return tuple(result)


def fit_partition(values):
    # Reserve all IDs from 30 onward for independent native resets on every arm.
    if any(not 0 <= p < 30 for p in values):
        raise ValueError("a motor or learner fitting pose overlaps the evaluation partition")


def seed(pose):
    return 20261120 + pose


def real_array(value, shape, name, *, bounded=False):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite numeric {shape}, got {a.dtype} {a.shape}")
    if bounded and np.any(np.abs(a.astype(np.float64)) > 1):
        raise ValueError(f"{name} outside native action bounds")
    a = a.astype(np.float32, copy=False)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} exceeds float32 range")
    return a


def prompt_text(value):
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected a nonempty instruction")
    return str(value)


def public_obs(obs):
    public = {k: obs[k] for k in PUBLIC}
    for key in ("rgb", "wrist"):
        image = np.asarray(public[key])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"invalid {key} camera frame")
        public[key] = image
    public["state"] = real_array(public["state"], (8,), "proprioception")
    public["prompt"] = prompt_text(public["prompt"])
    return public


def reset(env, body, pose):
    obs = env.reset(pose, seed(pose))
    info = getattr(env, "reset_info", None) or obs.get("metadata", {})
    physics = info.get("environment_reproducibility", {})
    expected = {"suite": "libero_goal", "task_id": TASK, "init_state_id": pose, "seed": seed(pose)}
    if any(physics[k] != v for k, v in expected.items() if k in physics):
        raise ValueError(f"simulator reset differs from requested {body}/{pose}")
    if info.get("model_identity", {}).get("arm", body) != body:
        raise ValueError(f"simulator instantiated the wrong arm for {body}/{pose}")
    keys = ("source_pin", "manifest_sha256", "bddl_sha256", "released_row_fingerprint",
            "scene_fingerprint", "post_settle_scene_fingerprint", "initial_state_fingerprint")
    return public_obs(obs), {k: physics[k] for k in keys if k in physics}, int(info.get("max_steps", HORIZON))


def session(body):
    from sim import Session

    return closing(Session(body, task_id=TASK))


def step(env, action):
    obs, motion, reward, done, success = env.step(action.copy())
    motion = real_array(motion, (7,), "measured motion").copy()
    if not np.isfinite(float(reward)):
        raise ValueError("nonfinite native reward")
    if bool(success) != (float(reward) > 0):
        raise ValueError("native Goal 5 success flag and sparse reward disagree")
    return obs, motion, float(reward), bool(done), bool(success)


@torch.inference_mode()
def proposal(motor, obs, plan_seed):
    # A direct simulator session may share this process. Preserve its RNG streams.
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            random.seed(plan_seed)
            np.random.seed(plan_seed)
            torch.manual_seed(plan_seed)
            value = motor.act(obs)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    a = np.asarray(value)
    if a.ndim != 2 or a.shape[0] < 1:
        raise ValueError(f"motor.act must return a nonempty H x 7 chunk, got {a.shape}")
    return real_array(a, (len(a), 7), "motor proposal", bounded=True).copy()


def plan_seed(pose, t):
    return 900_000 + pose * 1000 + t


def probe(pose, calibration=False):
    # Commands depend only on pose, never on body, images or simulator feedback.
    rng = np.random.default_rng(seed(pose) + 1000)
    if calibration:
        axial = np.vstack((np.eye(6), -np.eye(6), np.zeros((4, 6))))
        cart = 0.4 * axial[rng.permutation(K)]
        jaw = np.full((K, 1), -1)
    else:
        cart = rng.uniform(-0.4, 0.4, (K, 6))
        jaw = rng.choice([-1, 1], (K, 1))
    return np.column_stack((cart, jaw)).astype(np.float32)


@dataclass
class History:
    commands: np.ndarray
    motion: np.ndarray
    path: str

    def tensor(self):
        return torch.from_numpy(np.column_stack((self.commands, self.motion)))


def practice(out, body, pose, *, calibration=False):
    commands = probe(pose, calibration)
    motion, rewards, dones, successes = [], [], [], []
    name = f"practice-{body}-{pose}.npz"
    with session(body) as env:
        obs, physics, _ = reset(env, body, pose)
        for idx, cmd in enumerate(commands):
            _, delta, reward, done, success = step(env, cmd)
            motion.append(delta)
            rewards.append(reward)
            dones.append(done)
            successes.append(success)
            if (done or success) and idx + 1 != K:
                raise RuntimeError(f"native episode {body}/{pose} ended after {idx + 1}/{K} transitions")
    motion = np.stack(motion)
    np.savez_compressed(out / name, commands=commands, motion=motion, rewards=rewards, done=dones,
                        success=successes, pose=pose, seed=seed(pose), body=body, prompt=obs["prompt"],
                        reset_json=json.dumps(physics, sort_keys=True))
    return History(commands, motion, name)


@torch.inference_mode()
def episode(out, motor, body, pose, condition, transform=None, expected=None):
    """Check the entire opening chunk before issuing the first action."""
    actions, motions, proposed, rewards, dones, successes = [], [], [], [], [], []
    with session(body) as env:
        obs, physics, horizon = reset(env, body, pose)
        initial = {k: (obs[k] if k == "prompt" else digest_array(obs[k])) for k in PUBLIC}
        if horizon < 1:
            raise ValueError("native query horizon must be positive")
        while len(actions) < horizon:
            t = len(actions)
            chunk = proposal(motor, obs, plan_seed(pose, t))
            if t == 0:
                first = {"observation": initial, "physics": physics, "horizon": horizon,
                         "chunk_sha256": digest_array(chunk), "chunk": chunk.tolist()}
                if expected is not None and first != expected:
                    write_json(out / "opening-mismatch.json", {"body": body, "pose": pose,
                               "condition": condition, "expected": expected, "observed": first})
                    raise RuntimeError(f"unpaired opening for {body}/{pose}/{condition}")
            chunk = chunk[:min(EXECUTE, horizon - t)]
            issued = chunk.copy() if transform is None else transform(chunk.copy())
            issued = real_array(issued, chunk.shape, "issued actions", bounded=True)
            for raw, cmd in zip(chunk, issued, strict=True):
                next_obs, delta, reward, done, success = step(env, cmd)
                actions.append(cmd.copy())
                proposed.append(raw.copy())
                motions.append(delta)
                rewards.append(reward)
                dones.append(done)
                successes.append(success)
                if done or success:
                    break
            if dones[-1] or successes[-1] or len(actions) == horizon:
                break
            obs = public_obs(next_obs)
    name = f"query-{body}-{pose}-{condition}.npz"
    np.savez_compressed(out / name, proposals=proposed, actions=actions, motion=motions,
                        rewards=rewards, done=dones, success=successes)
    return {"body": body, "pose": pose, "seed": seed(pose), "condition": condition,
            "opening": first, "first_action": actions[0].tolist(), "steps": len(actions),
            "native_return": sum(rewards), "terminal_reward": rewards[-1], "success": any(successes),
            "done": dones[-1], "tape": name}


class ReaderAdapter(nn.Module):
    """Pool command-motion interactions, then condition a residual on the frozen proposal."""

    def __init__(self):
        super().__init__()
        self.register_buffer("motion_scale", torch.ones(7))
        self.reader = nn.Sequential(nn.Linear(63, 48), nn.Tanh(), nn.Linear(48, 16), nn.Tanh())
        self.response = nn.Linear(16, 8 * 7)
        self.adapter = nn.Linear(16, 8 * 7)
        nn.init.zeros_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)

    def encode(self, history):
        action, motion = history.split(7, dim=-1)
        motion = motion / self.motion_scale
        products = (action.unsqueeze(-1) * motion.unsqueeze(-2)).flatten(-2)
        return self.reader(torch.cat((action, motion, products), dim=-1)).mean(dim=1)

    @staticmethod
    def affine(head, context, x):
        coeff = head(context).reshape(-1, 8, 7)
        return torch.bmm(x.unsqueeze(1), coeff[:, :7]).squeeze(1) + coeff[:, 7]

    def adapt(self, context, raw):
        return raw + self.affine(self.adapter, context, raw)


@dataclass
class Query:
    x: np.ndarray
    y: np.ndarray
    path: str


def demo_query(args, motor, body, pose):
    path = args.demos / body / f"{pose}.npz"
    with np.load(path, allow_pickle=False) as archive:
        for key, expected in (("body", body), ("pose", pose), ("native_success", True)):
            if key in archive and archive[key].item() != expected:
                raise ValueError(f"{path} disagrees with expected {key}={expected}")
        a = archive["actions"]
        actions = real_array(a, (len(a), 7), "demo actions", bounded=True)
        n = len(actions)
        rewards = real_array(archive["rewards"], (n,), "demo rewards")
        if n == 0 or rewards[-1] <= 0 or (rewards[:-1] != 0).any():
            raise ValueError(f"{path} must end at its first native success reward")
        real_array(archive["motion"], (n, 7), "demo motion")
        streams = {k: archive[k] for k in ("rgb", "wrist", "state")}
        for key, v in streams.items():
            if len(v) not in (n, n + 1):
                raise ValueError(f"{path} pre-action {key} missing or misaligned")
        prompts = archive["prompt"].reshape(-1)
        if len(prompts) not in (1, n, n + 1):
            raise ValueError(f"{path} has misaligned prompt")
        instruction = prompt_text(prompts[0])
        if not (prompts == prompts[0]).all():
            raise ValueError(f"{path} instruction changed during the demonstration")
        bank = []
        t = 0
        while t < n:
            obs = {k: v[t] for k, v in streams.items()}
            obs["prompt"] = instruction
            p = proposal(motor, public_obs(obs), plan_seed(pose, t))[:min(EXECUTE, n - t)]
            bank.append(p)
            t += len(p)
    raw = np.concatenate(bank)
    name = f"demonstration-{body}-{pose}.npz"
    np.savez_compressed(args.out / name, proposals=raw, demonstrated=actions, rewards=rewards)
    return Query(raw, actions, name), {"body": body, "pose": pose, "source": str(path),
                                       "sha256": digest_file(path), "transitions": n,
                                       "terminal_native_reward": float(rewards[-1]), "bank": name,
                                       "prompt": instruction}


def train_stage(net, practices, queries, steps, batch_size, rng, *, motion):
    head = net.response if motion else net.adapter
    params = [*net.reader.parameters(), *head.parameters()]
    opt = torch.optim.Adam(params, lr=0.003)
    before = [p.detach().clone() for p in net.reader.parameters()]

    def batch():
        histories, inputs, targets = [], [], []
        for body_id in rng.integers(len(SOURCES), size=batch_size):
            body = SOURCES[body_id]
            hs, qs = practices[body], queries[body]
            hist = hs[rng.integers(len(hs))]
            query = qs[rng.integers(len(qs))]
            t = rng.integers(len(query.x))
            histories.append(hist.tensor())
            inputs.append(query.x[t])
            targets.append(query.y[t])
        return torch.stack(histories), torch.from_numpy(np.stack(inputs)), torch.from_numpy(np.stack(targets))

    def loss(data, *, training=False):
        history, x, y = data
        z = net.encode(history)
        if motion:
            return (net.affine(head, z, x) - y / net.motion_scale).square().mean()
        if training:
            keep = torch.from_numpy((rng.random(len(x)) >= 0.125).astype(np.float32))
            z = z * keep[:, None]
        return (net.adapt(z, x) - y).square().mean()

    audit = batch()
    with torch.no_grad():
        first = loss(audit).item()
    curve = []
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        value = loss(batch(), training=True)
        value.backward()
        nn.utils.clip_grad_norm_(params, 5.0, error_if_nonfinite=True)
        opt.step()
        if i in (0, steps - 1) or (i + 1) % 50 == 0:
            curve.append([i + 1, value.item()])
    with torch.no_grad():
        last = loss(audit).item()
        delta = sum((p - b).square().sum() for p, b in zip(net.reader.parameters(), before)).sqrt().item()
    if not np.isfinite([first, last, delta]).all():
        raise FloatingPointError("nonfinite source fit diagnostics")
    return {"steps": steps, "sampled_training_mse_before": first, "sampled_training_mse_after": last,
            "reader_change_l2": delta, "minibatch_mse": curve}


@dataclass
class Response:
    gain: np.ndarray   # Row vector: action @ gain + drift -> metres / radians.
    drift: np.ndarray

    def report(self, validation):
        observed = validation.motion[:, :6]
        predicted = validation.commands[:, :6] @ self.gain + self.drift
        return {"gain": self.gain.tolist(), "drift": self.drift.tolist(),
                "rank": int(np.linalg.matrix_rank(self.gain)), "validation": validation.path,
                "per_axis_rmse": np.sqrt(((observed - predicted) ** 2).mean(0)).tolist(),
                "drift_only_rmse": np.sqrt(((observed - self.drift) ** 2).mean(0)).tolist()}


def measured_response(histories):
    cmd = np.concatenate([h.commands[:, :6] for h in histories]).astype(np.float64)
    motion = np.concatenate([h.motion[:, :6] for h in histories]).astype(np.float64)
    zero = (cmd == 0).all(1)
    gains = []
    for axis in cmd.T:
        plus, minus = axis > 0, axis < 0
        # Calibration contains one positive and one negative isolated pulse per axis/reset.
        gains.append((motion[plus].mean(0) - motion[minus].mean(0)) /
                     (axis[plus].mean() - axis[minus].mean()))
    return Response(np.stack(gains), motion[zero].mean(0))


def privileged(reference, target):
    scale = np.linalg.norm(reference.gain, axis=0)
    if np.any(scale <= np.finfo(np.float64).eps):
        raise ValueError("raw Panda calibration has an unobserved motion coordinate")
    gain = target.gain / scale
    inverse = np.linalg.solve(gain @ gain.T + 0.01 * np.eye(6), gain).T
    delta = ((reference.gain - target.gain) / scale) @ inverse
    bias = ((reference.drift - target.drift) / scale) @ inverse

    def transform(raw):
        result = raw.copy()
        result[:, :6] = np.clip(raw[:, :6] + raw[:, :6] @ delta + bias, -1, 1)
        return result

    return transform


@torch.inference_mode()
def learned(net, history):
    context = torch.zeros(1, 16) if history is None else net.encode(history.tensor().unsqueeze(0))

    def transform(raw):
        return net.adapt(context.expand(len(raw), -1), torch.from_numpy(raw)).clamp(-1, 1).numpy()

    return transform


def load_motor(args):
    from motor import Motor

    with torch.inference_mode():
        motor = Motor.load(args.motor_output, args.weights)
        if isinstance(motor, nn.Module):
            motor.eval().requires_grad_(False)
    return motor


def identity(args):
    here = Path(__file__).resolve().parent
    prior = here.parent / "libero-crossarm-20261120-r1"
    # Motor.load separately validates the original VLM shard bytes against the trained revision.
    return {"motor_checkpoint": digest_file(args.motor_output / "checkpoint.pt"),
            "base_config": digest_file(args.weights / "config.json"),
            "code": {p.name: digest_file(p) for p in
                     (here / "adapt_eval.py", here / "motor.py", here / "sim.py",
                      here / "motor_data.py", here / "motor_weights.py",
                      prior / "arm_env.py", prior / "sim_reset.py") if p.is_file()}}


def read_stage(path, name, run):
    record = json.loads((path / "report.json").read_text())
    if record["stage"] != name or record["identity"] != run["identity"]:
        raise ValueError(f"{path} has a different stage, implementation or motor")
    return record


def stop(report, message):
    report.update(status="stopped", reason=message)


def qualify(args, report):
    # Audit the exact checkpoint selected by Motor.load, including model-selection exposure.
    ckpt = torch.load(args.motor_output / "checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
    contract = ckpt["contract"]
    records = contract["episodes"]
    splits = {}
    for split in ("train", "validation"):
        grouped = {b: sorted({int(r["pose"]) for r in records if r["split"] == split and r["body"] == b})
                   for b in SOURCES}
        if any(not ids for ids in grouped.values()):
            raise ValueError(f"motor {split} requires recorded demonstrations on each source")
        for ids in grouped.values():
            fit_partition(ids)
        splits[split] = grouped
    if any(r["body"] not in SOURCES or r["split"] not in splits for r in records):
        raise ValueError("motor checkpoint includes an undeclared arm or split")
    if ({p for ps in splits["train"].values() for p in ps} &
            {p for ps in splits["validation"].values() for p in ps}):
        raise ValueError("motor fit and validation share a pose")
    instructions = {r["prompt"] for r in records}
    if len(instructions) != 1:
        raise ValueError("motor checkpoint must study one instruction")
    report.update(motor_poses=splits, prompt=instructions.pop(), base_model=contract["weights"],
                  source_motor_step=ckpt["best"]["step"])
    del ckpt
    report["query_poses"] = QUALIFY
    report["episodes"] = []
    motor = load_motor(args)
    for body in SOURCES:
        for p in QUALIFY:
            row = episode(args.out, motor, body, p, "raw")
            if row["opening"]["observation"]["prompt"] != report["prompt"]:
                raise ValueError("source motor demonstration instruction differs from the native query")
            report["episodes"].append(row)
            write_json(args.out / "report.json", report)
    counts = {b: sum(e["success"] for e in report["episodes"] if e["body"] == b) for b in SOURCES}
    report["native_successes"] = counts
    report["source_gate"] = all(counts.values())
    if not report["source_gate"]:
        stop(report, "at least one source motor had zero independent native successes")


def fit(args, report):
    q = read_stage(args.qualification, "qualify", report)
    if q["status"] != "complete" or not q.get("source_gate"):
        return stop(report, "source motor qualification did not pass")
    fit_partition(args.demo_poses)
    if set(args.demo_poses) & set(SUPPORT + MOTION_QUERY):
        raise ValueError("demonstration queries must use different poses from source motion episodes")
    if args.joint_steps < 2 or args.pretrain_steps < 1 or args.batch_size < 1:
        raise ValueError("need pretrain_steps >= 1, joint_steps >= 2 and batch_size >= 1")
    report.update(source_gate=True, qualification_sha256=digest_file(args.qualification / "report.json"),
                  source_successes=q["native_successes"], motor_poses=q["motor_poses"], prompt=q["prompt"],
                  support_poses=SUPPORT, motion_query_poses=MOTION_QUERY, demo_poses=args.demo_poses,
                  fit_seed=FIT_SEED, batch_size=args.batch_size, train_bodies=SOURCES)
    torch.manual_seed(FIT_SEED)
    torch.set_num_threads(1)
    rng = np.random.default_rng(FIT_SEED)
    net = ReaderAdapter().train()
    practices, motion_queries = {}, {}
    for body in SOURCES:
        practices[body] = [practice(args.out, body, p) for p in SUPPORT]
        hs = [practice(args.out, body, p) for p in MOTION_QUERY]
        motion_queries[body] = [Query(h.commands, h.motion, h.path) for h in hs]
    report["source_histories"] = {b: {"supports": [h.path for h in practices[b]],
                                      "motion_queries": [q.path for q in motion_queries[b]]} for b in SOURCES}
    motions = np.concatenate([h.motion for hs in practices.values() for h in hs])
    scale = np.maximum(np.sqrt((motions.astype(np.float64) ** 2).mean(0)), 1e-6)
    net.motion_scale.copy_(torch.as_tensor(scale))
    report["pretrain"] = train_stage(net, practices, motion_queries, args.pretrain_steps,
                                     args.batch_size, rng, motion=True)
    write_json(args.out / "report.json", report)
    motor = load_motor(args)
    demos = {b: [] for b in SOURCES}
    report["demonstrations"] = []
    for body in SOURCES:
        for p in args.demo_poses:
            query, row = demo_query(args, motor, body, p)
            if row["prompt"] != q["prompt"]:
                raise ValueError("joint demonstration instruction differs from the qualified native task")
            demos[body].append(query)
            report["demonstrations"].append(row)
    report["joint"] = train_stage(net, practices, demos, args.joint_steps, args.batch_size, rng, motion=False)
    net.eval().requires_grad_(False)
    for param in net.parameters():
        param.grad = None
    if not all(torch.isfinite(t).all() for t in net.state_dict().values()):
        raise FloatingPointError("nonfinite source checkpoint")
    checkpoint = args.out / "reader-adapter.pt"
    torch.save({"version": 1, "state_dict": net.state_dict()}, checkpoint)
    report["checkpoint_sha256"] = digest_file(checkpoint)


def load_reader(fit_dir):
    checkpoint = torch.load(fit_dir / "reader-adapter.pt", map_location="cpu", weights_only=True)
    if checkpoint["version"] != 1:
        raise ValueError("unknown reader checkpoint")
    net = ReaderAdapter()
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    return net.eval().requires_grad_(False)


def paired_change(paired, condition, baseline):
    return {"wins": [r[condition]["pose"] for r in paired if r[condition]["success"] and not r[baseline]["success"]],
            "losses": [r[condition]["pose"] for r in paired if r[baseline]["success"] and not r[condition]["success"]]}


@torch.inference_mode()
def evaluate(args, report):
    training = read_stage(args.fit, "fit", report)
    if training["status"] != "complete" or not training.get("source_gate"):
        return stop(report, "a completed source-only fit with passing motor qualification is required")
    ckpt = digest_file(args.fit / "reader-adapter.pt")
    if ckpt != training["checkpoint_sha256"]:
        raise ValueError("source reader checkpoint differs from the completed fit")
    report.update(fit_report_sha256=digest_file(args.fit / "report.json"), checkpoint_sha256=ckpt,
                  source_gate=True, source_successes=training["source_successes"])
    body = "UR5e" if args.stage == "dev" else "IIWA"
    if args.stage == "reserved":
        dev = read_stage(args.dev, "dev", report)
        if (dev["status"] != "complete" or not dev.get("reserved_gate")
                or dev.get("fit_report_sha256") != report["fit_report_sha256"]
                or dev.get("checkpoint_sha256") != ckpt):
            return stop(report, "sealed IIWA requires two paired UR5e calibration wins from this source fit")
        report["dev_report_sha256"] = digest_file(args.dev / "report.json")
    panel = PANELS[body]
    report.update(body=body, panel=panel, calibration={}, pairs=[])
    net = load_reader(args.fit)
    motor = load_motor(args)
    responses = {}
    for b in ("Panda", body):
        h = [practice(args.out, b, p, calibration=True) for p in panel["calibration"]]
        ref = measured_response(h)
        check = practice(args.out, b, panel["check"])
        report["calibration"][b] = {**ref.report(check), "supports": [s.path for s in h]}
        responses[b] = ref
    transform = privileged(responses["Panda"], responses[body])
    write_json(args.out / "report.json", report)
    for i, (support, query) in enumerate(zip(panel["practice"], panel["query"], strict=True)):
        donor = SOURCES[i % len(SOURCES)]
        correct = practice(args.out, body, support)
        wrong = practice(args.out, donor, support)
        if not np.array_equal(correct.commands, wrong.commands):
            raise ValueError("donor and correct practice commands differ")
        transforms = {"raw": None, "correct": learned(net, correct), "cleared": learned(net, None),
                      "wrong": learned(net, wrong), "privileged": transform}
        records = {}
        report["pairs"].append({"pose": query, "practice_pose": support, "correct": correct.path,
                                "wrong": wrong.path, "donor": donor, "conditions": records})
        for condition in CONDITIONS:
            expected = records["raw"]["opening"] if records else None
            records[condition] = episode(args.out, motor, body, query, condition,
                                         transforms[condition], expected)
            write_json(args.out / "report.json", report)
    paired = [pair["conditions"] for pair in report["pairs"]]
    report["native_successes"] = {c: sum(row[c]["success"] for row in paired) for c in CONDITIONS}
    report["versus_raw"] = {c: paired_change(paired, c, "raw") for c in CONDITIONS if c != "raw"}
    report["correct_versus"] = {c: paired_change(paired, "correct", c) for c in ("cleared", "wrong")}
    if body == "UR5e":
        change = report["versus_raw"]["privileged"]
        report["reserved_gate"] = len(change["wins"]) >= 2 and not change["losses"]
        if not report["reserved_gate"]:
            stop(report, "calibration did not produce two paired native rescue wins on development")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name in ("qualify", "fit", "dev", "reserved"):
        cmd = stages.add_parser(name)
        cmd.add_argument("--out", type=Path, required=True, help="fresh stage directory")
        cmd.add_argument("--motor-output", type=Path, required=True, help="trained source motor directory")
        cmd.add_argument("--weights", type=Path, required=True, help="weights_path for Motor.load")
        cmd.add_argument("--revision", required=True, help="deployed code revision, supplied by the launcher")
        if name == "fit":
            cmd.add_argument("--qualification", type=Path, required=True)
            cmd.add_argument("--demos", type=Path, required=True)
            cmd.add_argument("--demo-poses", type=poses, required=True)
            cmd.add_argument("--pretrain-steps", type=int, default=100)
            cmd.add_argument("--joint-steps", type=int, default=150)
            cmd.add_argument("--batch-size", type=int, default=32)
        elif name in ("dev", "reserved"):
            cmd.add_argument("--fit", type=Path, required=True)
            if name == "reserved":
                cmd.add_argument("--dev", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    report = {"stage": args.stage, "status": "running", "task_id": TASK, "practice_transitions": K,
              "horizon": HORIZON, "execute_prefix": EXECUTE, "argv": sys.argv,
              "revision": args.revision, "torch": str(torch.__version__), "numpy": np.__version__}
    path = args.out / "report.json"
    write_json(path, report)
    try:
        report["identity"] = identity(args)
        if args.stage == "qualify":
            qualify(args, report)
        elif args.stage == "fit":
            fit(args, report)
        else:
            evaluate(args, report)
        if report["status"] == "running":
            report["status"] = "complete"
    except Exception as exc:
        report.update(status="error", error_type=type(exc).__name__, reason=str(exc))
        traceback.print_exc()
    write_json(path, report)
    print(json.dumps({"stage": args.stage, "status": report["status"], "report": str(path),
                      "reason": report.get("reason")}), flush=True)
    return {"complete": 0, "error": 1, "stopped": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
