"""Persistent panel identities, frozen selections and native tape readback."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

ARMS = ("Panda", "Sawyer")
# Inherit the registered 3/4 development and 6/8 confirmation gates.
PHASES = {"dev": (740000, 50, 38), "test": (840000, 100, 75)}
HASH_FIELDS = ("initial_state_sha256", "scene_hash", "movable_xy_hash")
PUBLIC = {"rgb", "wrist", "state", "prompt"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


@contextmanager
def publish(path):
    """Atomically install a complete file, preserving previously published originals."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    os.close(fd)
    tmp = Path(tmp)
    try:
        yield tmp
        try:
            os.link(tmp, path)
        except FileExistsError:
            require(sha(tmp) == sha(path), f"Existing evidence differs at {path}")
    finally:
        tmp.unlink(missing_ok=True)


def bind(path, value):
    if path.exists():
        require(read(path) == value, f"Existing record differs at {path}")
        return
    with publish(path) as tmp:
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def training(fit):
    contract = read(fit / "manifest.json")["contract"]
    require(contract["protocol"] == "dualarm-native-goal5-v2" and
            set(contract["source_bodies"]) == set(ARMS) and
            contract["weights"]["repo"] == "google/paligemma-3b-mix-224", "Unexpected fit or base model")
    rows = contract["episodes"]
    require({r["body"] for r in rows} == set(ARMS) and
            len({r["prompt"] for r in rows}) == 1, "Incomplete source fit or inconsistent task prompts")
    seeds, hashes = set(), set()
    for r in rows:
        require(r["split"] in ("train", "validation") and type(r["seed"]) is int, "Invalid fit episode")
        seeds.add(r["seed"])
        origin = r.get("layout_origin")
        if origin is not None:
            require(r["split"] == "train" and origin["layout_seed"] == r["seed"] and
                    r["pose"] == 1_000_000 + r["seed"], "Inconsistent generated fitting group")
            hashes.update(origin[k] for k in HASH_FIELDS)
        else:
            require(r["pose"] in (range(28) if r["split"] == "train" else range(28, 32)) and
                    bool(r["released_row_fingerprint"]), "Unexpected released fitting group")
            hashes.add(r["released_row_fingerprint"])
    for first, _, _ in PHASES.values():
        require(not any(first <= s < first + 10000 for s in seeds), "Fit used a reserved panel seed")
    return contract, seeds, hashes


def candidate(fit, step):
    contract, _, _ = training(fit)
    fit_id = sha(fit / "manifest.json")
    idx = read(fit / "candidates.json")
    require(idx["fit_id"] == fit_id and idx["status"] == "complete", "Incomplete candidate index")
    row, = [r for r in idx["snapshots"] if r["step"] == step]
    file = f"candidates/step-{step:06d}.pt"
    require(row["file"] == file and sha(fit / file) == row["sha256"], "Candidate file changed")
    return dict(fit_id=fit_id, step=step, checkpoint_sha256=row["sha256"],
                architecture=contract.get("architecture", "chunk_huber_v1"),
                prompt=contract["episodes"][0]["prompt"]), fit / file


def manifest(root, phase):
    path = root / phase / "manifest.json"
    data = read(path)
    require(read(path.with_suffix(".sha256.json")) == sha(path), "Uncommitted or changed panel manifest")
    start, count, threshold = PHASES[phase]
    rows = data["layouts"]
    require((data["phase"], data["size"], data["required"], data["seed_range"]) ==
            (phase, count, threshold, [start, start + 9999]) and len(rows) == count, "Wrong panel")
    seeds = [r["layout_seed"] for r in rows]
    rejected = [r["layout_seed"] for r in data["rejected"]]
    require(seeds == sorted(set(seeds)) and start <= seeds[0] <= seeds[-1] < start + 10000 and
            sorted(seeds + rejected) == list(range(start, seeds[-1] + 1)), "Panel seed order or coverage changed")
    for field in HASH_FIELDS:
        require(len({r[field] for r in rows}) == count, f"Duplicate {field}")
    for r in rows:
        seed = r["layout_seed"]
        require(r["layout_id"] == 1_000_000 + seed and r["split"] == phase and
                r["state_file"] == f"rows/{seed}.npy" and set(r["preflight"]) == set(ARMS) and
                all(r["preflight"][b]["scene_fingerprint"] == r["scene_hash"] for b in ARMS) and
                sha(path.parent / r["state_file"]) == r["file_sha256"], f"Bad native row {seed}")
    return data


def selection(root):
    seal = read(root / "accepted" / "selection.json")
    require(sha(root / "accepted" / "checkpoint.pt") == seal["checkpoint_sha256"], "Selected bytes changed")
    stage = directory(root, "dev", seal, seal["stride"])
    report = read(stage / "report.json")
    require(sha(stage / "report.json") == seal["report_sha256"] and report["passed"] and
            report["phase"] == "dev" and report["panel_sha256"] == sha(root / "dev/manifest.json") and
            all(report[k] == seal[k] for k in ("stride", "fit_id", "step", "checkpoint_sha256")) and
            all(report["arms"][arm]["total"] == 50 and report["arms"][arm]["successes"] >= PHASES["dev"][2]
                for arm in ARMS), "Sealed test requires completed accepted development evidence")
    return seal


def directory(root, phase, c, stride):
    return root / phase / "scores" / c["checkpoint_sha256"] / f"stride-{stride}"


def inputs(args):
    c, file = candidate(args.fit, args.step)
    if args.phase == "test":
        seal = selection(args.root)
        require(all(c[k] == seal[k] for k in c) and args.stride == seal["stride"],
                "Confirmation is restricted to the development-accepted motor and stride")
        file = args.root / "accepted/checkpoint.pt"
    panel = manifest(args.root, args.phase)
    _, seeds, hashes = training(args.fit)
    require(not any(r["layout_seed"] in seeds or any(r[k] in hashes for k in HASH_FIELDS)
                    for r in panel["layouts"]), "Evaluation overlaps fit exposure")
    contract = dict(**c, phase=args.phase, stride=args.stride,
                    panel_sha256=sha(args.root / args.phase / "manifest.json"),
                    evaluator=implementation(), horizon=300)
    if args.phase == "test":
        require(panel["development_selection_sha256"] == sha(args.root / "accepted/selection.json") and
                selection(args.root)["evaluator"] == contract["evaluator"],
                "Sealed evaluator or selection differs from development")
    return panel, file, contract, directory(args.root, args.phase, c, args.stride)


def implementation():
    here = Path(__file__).resolve().parent
    prior = here.parent / "libero-paligemma-crossarm-20261120-r1"
    native = here.parent / "libero-crossarm-20261120-r1"
    paths = [*here.glob("large_panel*.py"), here / "generated_layouts.py", here / "generated_split.py",
             here / "expert.py", here / "servo_motor.py", here / "waypoint_motor.py",
             prior / "sim.py", prior / "motor.py", prior / "motor_weights.py", native / "arm_env.py", native / "sim_reset.py"]
    return {p.name: sha(p) for p in paths}


def evidence(stage, arm, row):
    base = stage / arm / str(row["layout_seed"])
    return base.with_suffix(".npz"), base.with_suffix(".json")


def check_tape(stage, arm, row, contract, recover=False):
    import numpy as np
    path, receipt = evidence(stage, arm, row)
    checksum = sha(path)
    rec = read(receipt) if receipt.exists() or not recover else None
    if rec is not None:
        require(rec["tape_sha256"] == checksum, f"Tape hash differs at {path}")
    with np.load(path, allow_pickle=False) as tape:
        header = json.loads(tape["header"].item())
        chunks, actions, motion = (tape[k] for k in ("chunks", "actions", "motion"))
        rewards, done, success = (tape[k] for k in ("rewards", "done", "success"))
        public = json.loads(tape["observations"].item())
    stride = contract["stride"]
    n = len(rewards)
    plans = math.ceil(n / stride)
    reset = header["reset"]
    require(header["contract_sha256"] == digest(contract) and header["arm"] == arm and
            header["layout_id"] == row["layout_id"] and header["layout_seed"] == row["layout_seed"] and
            reset["admission"] == row["preflight"][arm] and reset["model_identity"]["arm"] == arm and
            "NVIDIA" in reset["renderer"][0].upper() and bool(reset["renderer"][1]), "Unpaired native tape")
    require(0 < n <= 300 and chunks.shape == (plans, 5, 7) and len(public) == plans and
            all(set(obs) == PUBLIC and obs["prompt"] == contract["prompt"] and
                all(isinstance(obs[k], str) and len(obs[k]) == 64 for k in PUBLIC - {"prompt"}) for obs in public) and
            np.isfinite(chunks).all() and np.abs(chunks).max() <= 1 and actions.shape == motion.shape == (n, 7) and
            np.array_equal(actions, chunks[:, :stride].reshape(-1, 7)[:n]) and np.isfinite(motion).all(),
            "Noncausal schedule, changed proposal prefix or invalid native action")
    require(rewards.shape == done.shape == success.shape == (n,) and done.dtype == success.dtype == np.bool_ and
            not done[:-1].any() and done[-1] and not rewards[:-1].any() and rewards[-1] in (0, 1) and
            np.array_equal(success, rewards == 1) and (n == 300 or success[-1]), "Invalid native termination")
    expected = dict(layout_id=row["layout_id"], layout_seed=row["layout_seed"], arm=arm,
                    scene_hash=row["scene_hash"], movable_xy_hash=row["movable_xy_hash"],
                    steps=n, success=bool(success[-1]), native_return=float(rewards.sum()),
                    tape_sha256=checksum, tape=str(path.relative_to(stage)))
    if rec is None:
        bind(receipt, expected)
    else:
        require(rec == expected, "Receipt differs from native evidence")
    return expected


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0., center-radius), min(1., center+radius)]


def report_for(panel, contract, stage):
    require(read(stage / "contract.json") == contract, "Source rollout contract differs")
    arms = {}
    for b in ARMS:
        records = [check_tape(stage, b, r, contract) for r in panel["layouts"]]
        winners = [r["layout_id"] for r in records if r["success"]]
        k, n = len(winners), len(records)
        arms[b] = dict(successes=k, total=n, rate=k/n, wilson95=wilson(k, n),
                       successful_layout_ids=winners, episodes=records)
    return dict(**contract, status="complete", arms=arms, required_per_arm=panel["required"],
                passed=all(a["successes"] >= panel["required"] for a in arms.values()))


def summarize(args, accept=False):
    panel, file, contract, stage = inputs(args)
    report = report_for(panel, contract, stage)
    bind(stage / "report.json", report)
    print(json.dumps({b: {k: v[k] for k in ("successes", "total", "wilson95")}
                      for b, v in report["arms"].items()}), flush=True)
    if accept:
        require(args.phase == "dev" and report["passed"], "Acceptance requires a passing full dev panel")
        selected = args.root / "accepted"
        with lock(selected / ".lock"):
            with publish(selected / "checkpoint.pt") as tmp:
                shutil.copyfile(file, tmp)
                require(sha(tmp) == contract["checkpoint_sha256"], "Checkpoint changed during promotion")
            bind(selected / "selection.json", dict(**contract, report_sha256=sha(stage / "report.json")))
            selection(args.root)
