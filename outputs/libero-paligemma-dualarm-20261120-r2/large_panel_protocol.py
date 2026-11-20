"""Persistent panel identities, frozen selections and native tape readback."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile

ARMS = ("Panda", "Sawyer")
# Inherit the registered 3/4 development and 6/8 confirmation gates.
PHASES = {"dev": (740000, 50, 38), "test": (840000, 100, 75)}
HASH_FIELDS = ("initial_state_sha256", "scene_hash", "movable_xy_hash")
GEOMETRY_FIELDS = (*HASH_FIELDS, "mobile_joint_xy_hash")
PUBLIC = {"rgb", "wrist", "state", "prompt"}
SFT01 = "0d8353ace1d15b7c6fa6a10b2dd3022b8637f6d71bac5748cdd72b69924dccdd"


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


def mobile_joints(scene):
    """Translation of every free movable joint, independent of robot qpos, time and fixture seeds."""
    free = {k: v["qpos"][:2] for k, v in scene["joints"].items() if v["type"] == 0}
    require(len(free) == 4, "Expected all four free Goal 5 movable joints")
    return digest(free)


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
            from generated_split import validate_origin
            validate_origin(origin)
            hashes.update(origin[k] for k in HASH_FIELDS)
            hashes.add(mobile_joints(origin["sampled_scene"]))
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
    if step == 1600 and fit.parent.name == "sft01":
        require(row["sha256"] == SFT01, "sft01 update 1600 differs from the preregistered checkpoint")
    return dict(fit_id=fit_id, step=step, checkpoint_sha256=row["sha256"],
                architecture=contract.get("architecture", "chunk_huber_v1"),
                prompt=contract["episodes"][0]["prompt"]), fit / file


def check_prepared(panel, seeds, hashes):
    # Candidates may train on different source banks. Every realized start
    # must be disjoint from both the preparation fit and the candidate fit.
    blocked_seeds = set(panel["fitting_seeds"]) | seeds
    blocked_hashes = set(panel["fitting_hashes"]) | hashes
    require(all(r["layout_seed"] not in blocked_seeds and
                all(r[k] not in blocked_hashes for k in HASH_FIELDS) for r in panel["layouts"]),
            "Prepared layout overlaps training or offline validation")
    require(panel["implementation"] == implementation(), "Prepared panel uses a different implementation")


def check_preflight(row):
    from large_panel_audit import angle, audit_pair

    for arm, admission in row["preflight"].items():
        require(admission["model_identity"]["arm"] == arm and admission["renderer"] == [],
                "Preflight used a different arm or renderer")
        audit_pair(admission, admission)
    panda = row["preflight"]["Panda"]["poses"]
    sawyer = row["preflight"]["Sawyer"]["poses"]
    require(panda == row["panda_settled_poses"] and panda.keys() == sawyer.keys(),
            "Source preflights disagree on the paired body identities or Panda reference")
    saved = row["source_pair_errors"]
    for field, unit, maximum, body, distance, limit in (
            ("xyz", "m", "max_position_m", "position_body", math.dist, 0.001),
            ("quat_xyzw", "rad", "max_rotation_rad", "rotation_body", angle, 0.01)):
        errors = {name: distance(pose[field], sawyer[name][field]) for name, pose in panda.items()}
        worst = max(errors.values())
        require(worst <= limit and math.isclose(saved[maximum], worst, abs_tol=1e-12) and
                math.isclose(errors[saved[body]], worst, abs_tol=1e-12),
                f"Inconsistent source preflight error ({unit})")


def manifest(root, phase, fit=None):
    import numpy as np
    from large_panel_audit import audit_pair
    from generated_split import BANK_SHA, TASK, expect, public_denylist, validate_origin

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
    expect(data, TASK)
    require(read(path.parent / "prepare-contract.json") == {
            k: data[k] for k in ("phase", "revision", "implementation", "fitting_seeds", "fitting_hashes",
                                 "development_selection_sha256")}, "Panel differs from preparation contract")
    if fit is not None:
        _, fit_seeds, fit_hashes = training(fit)
        check_prepared(data, fit_seeds, fit_hashes)
    else:
        require(data["implementation"] == implementation(), "Prepared panel uses a different implementation")
    public = data["released"]
    require(data["released_bank_sha256"] == BANK_SHA, "Unexpected released source bank")
    public_denylist([r["initial_state_sha256"] for r in public])
    require([r["released_id"] for r in public] == list(range(50)), "Incomplete released arrangement denylist")
    require(all(isinstance(r.get(k), str) and len(r[k]) == 64 and
                all(c in "0123456789abcdef" for c in r[k]) for r in public
                for k in ("movable_xy_hash", "mobile_joint_xy_hash")), "Missing released geometry fingerprints")
    for r in rows:
        seed = r["layout_seed"]
        # The sampler's origin schema uses train. Check the actual panel split separately below.
        validate_origin(dict(r, split="train"))
        require(r["layout_id"] == 1_000_000 + seed and r["split"] == phase and
                r["state_file"] == f"rows/{seed}.npy" and set(r["preflight"]) == set(ARMS) and
                r["scene_hash"] == digest(r["sampled_scene"]) and
                r["mobile_joint_xy_hash"] == mobile_joints(r["sampled_scene"]) and
                r["movable_xy_hash"] == digest({k: v[:2] for k, v in r["movable_xyz_m"].items()}) and
                all((a["scene_fingerprint"], a["generated_row_fingerprint"], a["seed"], a["init_state_id"]) ==
                    (r["scene_hash"], r["initial_state_sha256"], seed, r["layout_id"]) for a in r["preflight"].values()) and
                sha(path.parent / r["state_file"]) == r["file_sha256"], f"Bad native row {seed}")
        state = np.load(path.parent / r["state_file"], allow_pickle=False)
        require(state.dtype == np.dtype("<f8") and state.shape == (79,) and np.isfinite(state).all() and
                hashlib.sha256(b'{"dtype":"<f8","shape":[79]}\0' + state.tobytes()).hexdigest()
                == r["initial_state_sha256"], f"Donor array differs from sampled state at {seed}")
        check_preflight(r)
        require(read(path.parent / "preflight" / f"{seed}.json") == r, "Saved preflight row changed")
        if seed == seeds[0]:
            require(set(r["camera_ab"]) == set(ARMS), "Missing paired camera qualification")
            for arm, ab in r["camera_ab"].items():
                rendered = ab["rendered"]
                vendor, renderer = rendered["renderer"]
                require("NVIDIA" in vendor.upper() and renderer and
                        audit_pair(r["preflight"][arm], rendered) == ab["comparison"],
                        "Camera qualification differs from paired numerical readback")
    previous = manifest(root, "dev", fit=fit)["layouts"] if phase == "test" else []
    fitting = set(data["fitting_hashes"])
    for reserved, _, _ in PHASES.values():
        require(not any(reserved <= s < reserved + 10000 for s in data["fitting_seeds"]), "Reserved seed in fit")
    seen = {k: fitting | {r[k] for r in (*public, *previous) if k in r} for k in GEOMETRY_FIELDS}
    by_seed = {r["layout_seed"]: r for r in rows}
    exclusions = {r["layout_seed"]: r["matching_hashes"] for r in data["rejected"]}
    for seed in range(start, seeds[-1] + 1):
        if seed in exclusions:
            matched = exclusions[seed]
            require(matched and all(k in seen and v in seen[k] for k, v in matched.items()),
                    "Seed was skipped without a prior duplicate fingerprint")
        else:
            r = by_seed[seed]
            for k in GEOMETRY_FIELDS:
                require(r[k] not in seen[k], f"Duplicate {k} at seed {seed}")
                seen[k].add(r[k])
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
    comparison = read(root / "dev/comparison.json")
    winner = comparison["candidates"][comparison["winner"]]
    require(sha(root / "dev/comparison.json") == seal["comparison_sha256"] and comparison["passed"] and
            all(winner[k] == seal[k] for k in ("fit_id", "step", "checkpoint_sha256", "report_sha256")),
            "Saved development comparison differs from accepted checkpoint")
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
    panel = manifest(args.root, args.phase, fit=args.fit)
    versions = implementation()
    contract = dict(**c, phase=args.phase, stride=args.stride,
                    panel_sha256=sha(args.root / args.phase / "manifest.json"),
                    evaluator=versions, horizon=300, inference_batch_size=8)
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
             prior / "sim.py", prior / "motor.py", prior / "motor_data.py", prior / "motor_weights.py", native / "arm_env.py", native / "sim_reset.py"]
    return {p.name: sha(p) for p in paths}


def evidence(stage, arm, row):
    base = stage / arm / str(row["layout_seed"])
    return base.with_suffix(".npz"), base.with_suffix(".json")


def check_opening(stage, arm, row, contract, observation, reset, *, create=True):
    """Pin the first rendered opening for an arm/layout and verify every candidate against it."""
    from large_panel_audit import audit_pair, check_admission
    import struct

    check_admission(row, arm, reset)
    require(arm in ARMS and set(observation) == PUBLIC and observation["prompt"] == contract["prompt"] and
            all(isinstance(observation[k], str) and len(observation[k]) == 64 and
                all(c in "0123456789abcdef" for c in observation[k]) for k in ("rgb", "wrist", "state")),
            "Invalid opening observation")
    state = struct.pack("<8f", *reset["admission"]["public_state"])
    require(observation["state"] == hashlib.sha256(b"<f4/(8,)/" + state).hexdigest(),
            "Opening public state differs from the audited wrist and jaw")
    identity = dict(panel_sha256=contract["panel_sha256"], arm=arm, layout_id=row["layout_id"],
                    layout_seed=row["layout_seed"], rgb=observation["rgb"], wrist=observation["wrist"],
                    prompt=observation["prompt"])
    path = stage.parents[2] / "openings" / arm / f"{row['layout_seed']}.json"
    if create:
        with lock(path.with_suffix(".lock")):
            if not path.exists():
                bind(path, dict(identity=identity, admission=reset["admission"]))
    opening = read(path)
    require(opening["identity"] == identity, "Rendered opening differs across candidates")
    audit_pair(row["preflight"][arm], opening["admission"])
    audit_pair(opening["admission"], reset["admission"])


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
        native_steps = tape["native_steps"]
        public = json.loads(tape["observations"].item())
    stride = contract["stride"]
    n = len(rewards)
    plans = math.ceil(n / stride)
    reset = header["reset"]
    phys = reset["environment_reproducibility"]
    require(reset["max_steps"] == 300 and
            phys["perturbation_plan"]["case_id"] == "BASE" and phys["perturbation_plan"]["seed"] == row["layout_seed"]
            and (phys["init_state_id"], phys["bank_row_index"], phys["seed"], phys["suite"], phys["task_id"]) ==
            (row["layout_id"], 0, row["layout_seed"], "libero_goal", 5) and
            all(phys[k] == row[k] for k in ("source_pin", "bddl_sha256", "manifest_sha256")), "Bad native receipt")
    require(header["contract_sha256"] == digest(contract) and header["arm"] == arm and
            header["layout_id"] == row["layout_id"] and header["layout_seed"] == row["layout_seed"] and
            reset["model_identity"]["arm"] == arm and
            "NVIDIA" in reset["renderer"][0].upper() and bool(reset["renderer"][1]), "Unpaired native tape")
    require(native_steps.dtype.kind in "iu" and np.array_equal(native_steps, np.arange(1, n + 1)),
            "Missing or duplicated native step")
    require(0 < n <= 300 and chunks.shape == (plans, 5, 7) and len(public) == plans and
            all(set(obs) == PUBLIC and obs["prompt"] == contract["prompt"] and
                all(isinstance(obs[k], str) and len(obs[k]) == 64 for k in PUBLIC - {"prompt"}) for obs in public) and
            np.isfinite(chunks).all() and np.abs(chunks).max() <= 1 and actions.shape == motion.shape == (n, 7) and
            np.array_equal(actions, chunks[:, :stride].reshape(-1, 7)[:n]) and np.isfinite(motion).all(),
            "Noncausal schedule, changed proposal prefix or invalid native action")
    check_opening(stage, arm, row, contract, public[0], reset, create=False)
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


def audit_development(root, seal):
    stage = directory(root, "dev", seal, seal["stride"])
    contract = read(stage / "contract.json")
    require(all(contract[k] == seal[k] for k in contract) and
            report_for(manifest(root, "dev"), contract, stage) == read(stage / "report.json"),
            "Development checkpoint or native tapes changed after acceptance")


def summarize(args, accept=False):
    panel, file, contract, stage = inputs(args)
    report = report_for(panel, contract, stage)
    bind(stage / "report.json", report)
    print(json.dumps({b: {k: v[k] for k in ("successes", "total", "wilson95")}
                      for b, v in report["arms"].items()}), flush=True)
    if accept:
        require((args.phase, args.stride, args.step, contract["checkpoint_sha256"]) == ("dev", 5, 1600, SFT01),
                "Acceptance starts with the registered sft01 update 1600, stride five")
        comparison = []
        options = [(contract, file, stage, report)]
        for fit, arch in zip(args.extended_fits, ("chunk_huber_v1", "spatial_bins_v1")):
            extended = SimpleNamespace(root=args.root, fit=fit.resolve(strict=True), phase="dev", step=6400, stride=5)
            data, ckpt, c, path = inputs(extended)
            require(c["architecture"] == arch and c["panel_sha256"] == contract["panel_sha256"],
                    "Extended candidate or shared panel differs from registration")
            result = report_for(data, c, path)
            bind(path / "report.json", result)
            options.append((c, ckpt, path, result))
        source_fits = [args.fit, *args.extended_fits]
        for source_fit, (c, _, path, result) in zip(source_fits, options):
            counts = [result["arms"][b]["successes"] for b in ARMS]
            comparison.append(dict(fit=str(source_fit.resolve()), fit_id=c["fit_id"], step=c["step"],
                                   checkpoint_sha256=c["checkpoint_sha256"],
                                   architecture=c["architecture"], counts=dict(zip(ARMS, counts)),
                                   report_sha256=sha(path / "report.json")))
        best = max(range(len(options)), key=lambda i: (min(comparison[i]["counts"].values()),
                                                       sum(comparison[i]["counts"].values()), -i))
        contract, file, stage, report = options[best]
        bind(args.root / "dev/comparison.json", dict(candidates=comparison, winner=best, passed=report["passed"]))
        require(report["passed"], "No candidate reached the complete development gate on both arms")
        selected = args.root / "accepted"
        with lock(selected / ".lock"):
            with publish(selected / "checkpoint.pt") as tmp:
                shutil.copyfile(file, tmp)
                require(sha(tmp) == contract["checkpoint_sha256"], "Checkpoint changed during promotion")
            bind(selected / "selection.json", dict(**contract, report_sha256=sha(stage / "report.json"),
                                                   comparison_sha256=sha(args.root / "dev/comparison.json")))
            selection(args.root)
