r"""Replay the registered 50 Sawyer dev tapes in the R2 Linux simulator interpreter.

    python -Bu replay.py --checkout /path/to/fc61bf2a --panel /path/to/large-panel-v3 \
        --out /fresh/result [--workers 4] [--traces]

Use the R2 CPU runtime environment (CROSSARM_RELEASE, LIBERO_PRO_ROOT, LIBERO_CONFIG_PATH).
--panel is the archive root containing dev/. The original sft02/fit is its sibling.
One fresh native process per episode. Geometry(env, opening_obs) samples reset,
sample(obs) samples each returned control, and result() supplies metrics and trace.
report.json lists all 50 seeds. rejected/ and exceptions/ retain individual diagnostics.
Exit 0 means all 50 passed, 1 means at least one exception, 2 means validation rejection.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import traceback


HERE = Path(__file__).resolve().parent
R2 = "outputs/libero-paligemma-dualarm-20261120-r2"
PIN = "fc61bf2afb48d4639664db93dae64ae9cb04e2df"
RELEASE = "e5500c690c1498eca1b7050350654964190cdf70"
# Published in MOTOR-LIVE.md's evidence/motor/pfs-native.json, under large-panel-v3.
CHECKPOINT = "77d18c386b2cc176252cfb68449e4415748382f554046ca50367ed4592676d9f"
FIT = "307e81088eb5b67506e8b30b3348f564c1c142f7e293e6db34f3149776929e80"
INDEX = "286daa677ea27648a302d960c81e42864452140f9eca22f0cb1107a050a8975c"
REPORT = "7e2122efbd976638680c8215c3830eeb974179c5944f5185b38771988a3d464a"
PANEL = "663077ec7aad3621173d725d2cf1550e3859a7ca46a0de50727e25aec8c6bfc0"
LAYOUTS = "3b7e9e274586ae622a37d9b5fa33381833d37d1e0e673965096653dd5fa71798"
SEEDS = range(740000, 740050)
BODY = "Sawyer"
# The R2 motion tape stores float32 world translation, rotation vectors and jaw deltas.
# Allow roundoff in composing 300 deltas. These caps are 1/10 of the original
# same-arm opening allowances, applied to EACH increment AND each absolute pose.
POSITION_M = 1e-6
ANGLE_RAD = 1e-5


class Rejected(Exception):
    """An explicit identity, archive or numerical comparison failed."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


def require(ok, message, details=None):
    if not ok:
        raise Rejected(message, details)


@contextmanager
def archived_check():
    """Only used around the pinned routines whose purpose is validating evidence."""
    try:
        yield
    except (ValueError, KeyError, TypeError) as exc:
        raise Rejected(str(exc), getattr(exc, "report", None)) from exc


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verified(path, expected):
    actual = sha(path)
    require(actual == expected, f"SHA-256 differs at {path}", dict(expected=expected, actual=actual))
    return actual


def read(path):
    return json.loads(path.read_bytes())


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args])


def stage_for(panel):
    return panel / "dev" / "scores" / CHECKPOINT / "stride-5"


def prepare(args, source):
    """Authenticate the cohort and common inputs once. Individual files fail per seed."""
    stage = stage_for(args.panel)
    dev = args.panel / "dev"
    source["report_sha256"] = verified(stage / "report.json", REPORT)
    source["panel_sha256"] = verified(dev / "manifest.json", PANEL)
    report, panel, contract = read(stage / "report.json"), read(dev / "manifest.json"), read(stage / "contract.json")
    require(read(dev / "manifest.sha256.json") == PANEL, "Panel digest receipt differs")
    require(contract == {k: v for k, v in report.items()
                         if k not in ("arms", "passed", "status", "required_per_arm")}, "Score contract differs")
    require((panel["revision"], panel["phase"], panel["size"], report["phase"], report["status"],
             contract["checkpoint_sha256"], contract["fit_id"], contract["step"], contract["stride"],
             contract["architecture"], contract["horizon"], contract["panel_sha256"]) ==
            (PIN, "dev", 50, "dev", "complete", CHECKPOINT, FIT, 6400, 5, "chunk_huber_v1", 300, PANEL),
            "Wrong score or panel identity")
    require(git(args.checkout, "rev-parse", "HEAD").decode().strip() == PIN, "Use the original evaluator checkout")
    release = Path(os.environ["CROSSARM_RELEASE"]).resolve(strict=True)
    require(git(release, "rev-parse", "HEAD").decode().strip() == RELEASE, "Wrong LIBERO-Pro2.0 release")
    source.update(checkout=str(args.checkout), release=str(release), panel_root=str(args.panel),
                  evaluator_commit=PIN, release_commit=RELEASE, replay_sha256=sha(Path(__file__)),
                  geometry_sha256=sha(HERE / "geometry.py"), contract_file_sha256=sha(stage / "contract.json"))
    # Use the evaluator's own source manifest. Check its protocol before importing it.
    orig = args.checkout / R2
    require((orig / "large_panel_protocol.py").read_bytes() ==
            git(args.checkout, "show", f"{PIN}:{R2}/large_panel_protocol.py"), "Modified score protocol")
    sys.path[:0] = [str(orig), str(args.checkout / "src"), str(release), os.environ["LIBERO_PRO_ROOT"]]
    import large_panel_protocol as protocol

    source["evaluator"] = protocol.implementation()
    require(source["evaluator"] == panel["implementation"] == contract["evaluator"],
            "Runtime source hashes differ from the panel or the scorer")
    source["contract_sha256"] = protocol.digest(contract)
    fit = args.panel.parent / "sft02/fit"
    source["fit_sha256"] = verified(fit / "manifest.json", FIT)
    source["candidate_index_sha256"] = verified(fit / "candidates.json", INDEX)
    index = read(fit / "candidates.json")
    candidate, = [r for r in index["snapshots"] if r["step"] == 6400]
    require(index["fit_id"] == FIT and index["status"] == "complete" and candidate["sha256"] == CHECKPOINT
            and candidate["file"] == "candidates/step-006400.pt", "Wrong archived checkpoint receipt")
    source["checkpoint_sha256"] = verified(fit / candidate["file"], CHECKPOINT)

    layouts = panel["layouts"]
    fields = ("layout_seed", "layout_id", "initial_state_sha256", "scene_hash")
    source["source50_identity"] = protocol.digest([[r[k] for k in fields] for r in layouts])
    require(source["source50_identity"] == LAYOUTS and [r["layout_seed"] for r in layouts] == list(SEEDS),
            "Expected the original ordered 50 development layouts")
    arm = report["arms"][BODY]
    receipts = {r["layout_seed"]: r for r in arm["episodes"]}
    require(set(receipts) == set(SEEDS) and len(arm["episodes"]) == arm["total"] == 50
            and sum(r["success"] for r in receipts.values()) == arm["successes"] == 20, "Wrong Sawyer cohort")
    for row in layouts:
        seed = row["layout_seed"]
        require((row["split"], row["layout_id"], row["state_file"], receipts[seed]["arm"],
                 receipts[seed]["layout_id"]) == ("dev", seed + 1000000, f"rows/{seed}.npy", BODY, row["layout_id"]),
                f"Wrong case association at {seed}")
    return contract, layouts, receipts


def runtime_identity(args, env, layout):
    """Check imports actually used, including the hydrated LIBERO Python source."""
    release = Path(os.environ["CROSSARM_RELEASE"]).resolve()
    hydrated = Path(os.environ["LIBERO_PRO_ROOT"]).resolve()
    orig = args.checkout / R2
    from large_panel_protocol import digest
    from sim import PRIOR

    for name, root in (("sim", orig.parent / "libero-paligemma-crossarm-20261120-r1"),
                       ("arm_env", PRIOR), ("sim_reset", PRIOR),
                       *((k, orig) for k in ("expert", "generated_layouts", "generated_split",
                                             "large_panel_protocol", "large_panel_sim", "large_panel_audit")),
                       ("geometry", HERE)):
        require(Path(sys.modules[name].__file__).resolve() == root / (name + ".py"), f"Wrong live module {name}")
    # Match actual imported Python files to Git blobs at the release, including
    # the setup's byte-for-byte copies under LIBERO_PRO_ROOT (see bootstrap.py).
    tree = git(release, "ls-tree", "-r", RELEASE, "--", "libero_remote", "simulator-source/libero").decode()
    blobs = {p: meta.split()[2] for line in tree.splitlines() for meta, p in [line.split("\t", 1)]}
    loaded = {}
    for name, module in tuple(sys.modules.items()):
        package = name.split(".")[0]
        if package not in ("libero", "libero_remote"):
            continue
        file = getattr(module, "__file__", None)
        if not file:
            continue
        file = Path(file).resolve()
        if package == "libero":
            require(file.is_relative_to(hydrated), f"Unexpected LIBERO import {file}")
            relative = str(Path("simulator-source") / file.relative_to(hydrated))
        else:
            require(file.is_relative_to(release), f"Unexpected libero_remote import {file}")
            relative = str(file.relative_to(release))
        raw = file.read_bytes()
        blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        require(blobs.get(relative) == blob, f"Modified runtime module {relative}")
        loaded[relative] = hashlib.sha256(raw).hexdigest()
    require(any(k.startswith("libero_remote/") for k in loaded) and
            any(k.startswith("simulator-source/libero/") for k in loaded), "Missing live simulator source")
    return dict(release=RELEASE, python=sys.version.split()[0], executable=sys.executable,
                versions={p: version(p) for p in ("numpy", "scipy", "mujoco", "robosuite")},
                loaded_release_modules=len(loaded), loaded_release_modules_sha256=digest(loaded),
                bddl_sha256=verified(Path(env.arm.bddl), layout["bddl_sha256"]),
                task_manifest_sha256=verified(release / "assets/task_manifest.json", layout["manifest_sha256"]))


def load_episode(args, contract, layout, receipt, validation):
    import numpy as np
    from large_panel_protocol import check_tape, digest, evidence

    dev = args.panel / "dev"
    tape, sidecar = evidence(stage_for(args.panel), BODY, layout)
    seed = layout["layout_seed"]
    require(read(sidecar) == receipt, "Tape receipt differs from authenticated score report")
    validation.update(receipt_sha256=sha(sidecar), receipt_canonical_sha256=digest(receipt),
                      tape_sha256=verified(tape, receipt["tape_sha256"]))
    preflight = dev / "preflight" / f"{seed}.json"
    require(read(preflight) == layout, "Preflight differs from authenticated manifest")
    state_file = dev / layout["state_file"]
    validation.update(state_file_sha256=verified(state_file, layout["file_sha256"]), preflight_sha256=sha(preflight))
    state = np.load(state_file, allow_pickle=False)
    require(state.dtype == np.dtype("<f8") and state.shape == (79,) and np.isfinite(state).all(), "Bad donor array")
    validation["donor_state_sha256"] = hashlib.sha256(b'{"dtype":"<f8","shape":[79]}\0' + state.tobytes()).hexdigest()
    require(validation["donor_state_sha256"] == layout["initial_state_sha256"], "Wrong numerical donor state")
    # recover=False verifies the shared opening, rendered admission, plan hashes,
    # complete issued prefixes, native indices, rewards, done, success and receipt.
    # The original validator opens no locks and publishes nothing in this mode.
    with archived_check():
        require(check_tape(stage_for(args.panel), BODY, layout, contract, recover=False) == receipt,
                "Tape readback differs from the report")
    validation["archived_opening_file_sha256"] = sha(dev / "openings" / BODY / f"{seed}.json")
    with np.load(tape, allow_pickle=False) as z:
        data = {k: z[k] for k in ("actions", "chunks", "motion", "native_steps", "done", "success", "rewards")}
        header = json.loads(z["header"].item())
        plans = json.loads(z["observations"].item())
    validation["archived_actions_reconstructed"] = True
    return state, data, header["reset"], plans


def check_opening(obs, live, archived, plan, validation):
    from large_panel_audit import audit_pair
    from large_panel_protocol import digest

    a, b = archived["admission"], live["admission"]
    p, q = archived["environment_reproducibility"], live["environment_reproducibility"]
    # Compare to this motor's actual rendered tape; panel preflight is a separate reference.
    with archived_check():
        comparison = audit_pair(a, b)
    state_hash = state_sha(obs["state"])
    validation["opening"] = dict(archived_observation=plan, live_state_sha256=state_hash,
                                 exact_public_state=state_hash == plan["state"], archived_renderer=a["renderer"],
                                 live_renderer=b["renderer"], max_errors=comparison["max_errors"],
                                 limits=comparison["limits"], physics_hashes=comparison["hashes"],
                                 archived_admission_sha256=digest(a), live_admission_sha256=digest(b),
                                 exact_physics=a["raw_state_sha256"] == b["raw_state_sha256"],
                                 exact_poses=a["poses"] == b["poses"], model_identity=live["model_identity"])
    identity = ("source_pin", "manifest_sha256", "bddl_sha256", "suite", "task_id", "init_state_id",
                "seed", "bank_row_index", "source_kind", "native_sampler_seed", "layout_seed",
                "native_api_origin", "source_index", "scene_hash", "scene_fingerprint",
                "generated_row_fingerprint", "perturbation_plan", "perturbation_plan_fingerprint",
                "settled_collision_check")
    require(all(p[k] == q[k] for k in identity) and p["source_pin"] == RELEASE
            and archived["max_steps"] == live["max_steps"] == 300 and a["model_identity"] == live["model_identity"]
            and not live["renderer"] and obs["prompt"] == plan["prompt"]
            and b["public_state"] == obs["state"].tolist(), "Replay opening identity differs")


def state_sha(state):
    # large_panel_rollout.observation_hashes framing, for the only non-image array.
    return hashlib.sha256(f"{state.dtype.str}/{state.shape}/".encode() + state.tobytes(order="C")).hexdigest()


def transitions(env, obs, geometry, data, archived, plans, v):
    import numpy as np
    from scipy.spatial.transform import Rotation

    actions = data["actions"]
    initial = np.array(archived["admission"]["public_state"], dtype=np.float64)
    xyz, rot, jaw = initial[:3].copy(), Rotation.from_quat(initial[3:7]), initial[7]
    v.update(steps=0, native_return=0., terminal_reward=None, done=False, success=False,
             replan_state_hash_matches=0, replans_checked=0, exact_motion_steps=0,
             max_errors={}, violations={})

    def compare(key, ok, step, actual, expected):
        if not ok:
            first = v["violations"].setdefault(key, dict(count=0, first_control=step, actual=actual, expected=expected))
            first["count"] += 1

    def bounded(key, error, limit, step):
        error = float(error)
        require(np.isfinite(error), f"Nonfinite {key} at control {step}")
        v["max_errors"][key] = max(v["max_errors"].get(key, 0.), error)
        compare(key, error <= limit, step, error, limit)

    def pose_errors(step):
        state = obs["state"]
        require(state.shape == (8,) and state.dtype == np.float32 and np.isfinite(state).all(),
                f"Invalid public proprioception at {step}")
        bounded("wrist_position_m", np.linalg.norm(state[:3] - xyz), POSITION_M, step)
        bounded("wrist_angle_rad", (Rotation.from_quat(state[3:7]) * rot.inv()).magnitude(), ANGLE_RAD, step)
        bounded("jaw_m", abs(state[7] - jaw), POSITION_M, step)

    pose_errors(0)
    require(not v["violations"], "Opening proprioception exceeds the trajectory bounds", v["violations"])
    for i, action in enumerate(actions):
        step = i + 1
        if i % 5 == 0:
            plan = plans[i // 5]
            v["replans_checked"] += 1
            matches = state_sha(obs["state"]) == plan["state"]
            v["replan_state_hash_matches"] += int(matches)
            if not matches and "first_nonidentical_replan" not in v:
                v["first_nonidentical_replan"] = dict(control=i, archived=plan["state"], live=state_sha(obs["state"]))
        packet = env.arm.step(action)
        obs = env.observation(packet)
        motion = packet["sensor_motion"]
        expected_motion = data["motion"][i]
        issued = packet["issued_action"]
        done = bool(packet["terminated"] or packet["truncated"])
        reward = float(packet["reward"])
        success = bool(packet["success"])
        v.update(steps=step, terminal_reward=reward, native_return=v["native_return"] + reward, done=done,
                 success=success, terminated=bool(packet["terminated"]), truncated=bool(packet["truncated"]))
        compare("native_step", packet["worker_step_index"] == int(data["native_steps"][i]), step,
                packet["worker_step_index"], int(data["native_steps"][i]))
        proposal = data["chunks"][i // 5, i % 5]
        compare("issued_action", np.array_equal(issued, action) and np.array_equal(issued, proposal), step,
                issued.tolist(), action.tolist())
        compare("native_outcome", (reward, done, success) ==
                (data["rewards"][i], data["done"][i], data["success"][i]), step,
                [reward, done, success], [float(data["rewards"][i]), bool(data["done"][i]), bool(data["success"][i])])
        compare("prompt", obs["prompt"] == plans[0]["prompt"], step, obs["prompt"], plans[0]["prompt"])
        v["exact_motion_steps"] += int(np.array_equal(motion, expected_motion))
        delta = motion.astype(np.float64) - expected_motion
        bounded("delta_position_m", np.linalg.norm(delta[:3]), POSITION_M, step)
        bounded("delta_rotation_vector_rad", np.linalg.norm(delta[3:6]), ANGLE_RAD, step)
        bounded("delta_jaw_m", abs(delta[6]), POSITION_M, step)
        # sim_reset.tool_delta encodes R_after @ R_before.T in the world frame.
        xyz += expected_motion[:3]
        rot = Rotation.from_rotvec(expected_motion[3:6].astype(np.float64)) * rot
        jaw += float(expected_motion[6])
        pose_errors(step)
        geometry.sample(obs)
        if done:
            break
    compare("length", v["steps"] == len(actions), v["steps"], v["steps"], len(actions))
    compare("return", v["native_return"] == float(data["rewards"].sum()), v["steps"],
            v["native_return"], float(data["rewards"].sum()))


def replay(args, contract, layout, receipt, row):
    import numpy as np
    from expert import session
    from generated_layouts import LayoutRejected
    from geometry import Geometry
    from large_panel_audit import ResetMismatch
    from large_panel_sim import physical_reset

    v = row["physics_validation"]
    row["stage"] = "archive"
    state, data, archived, plans = load_episode(args, contract, layout, receipt, v)
    row["stage"] = "reset"
    with session(BODY, False) as env:
        row["runtime"] = runtime_identity(args, env, layout)
        try:
            obs, live, _ = physical_reset(env, BODY, False, state, layout)
        except (LayoutRejected, ResetMismatch) as exc:
            raise Rejected(str(exc), getattr(exc, "report", None)) from exc
        except ValueError as exc:
            # Explicit guards reached in the pinned reset are validation results.
            # A ValueError raised inside scipy, MuJoCo, or other code is an exception.
            origin = traceback.extract_tb(exc.__traceback__)[-1]
            if (Path(origin.filename).stem, origin.name) in {
                ("large_panel_protocol", "require"), ("large_panel_audit", "ensure"),
                ("generated_layouts", "reset_generated"), ("arm_env", "reset"),
                ("sim_reset", "transfer_scene"), ("sim_reset", "_match_tool_pose"),
            }:
                raise Rejected(str(exc)) from exc
            raise
        check_opening(obs, live, archived, plans[0], v)
        row["stage"] = "transitions"
        with Geometry(env, obs) as geometry:
            transitions(env, obs, geometry, data, archived, plans, v)
            row["stage"] = "geometry"
            result = geometry.result()
            metrics, trace = result["metrics"], result["trace"]
            if metrics["controls"] != v["steps"] or len(trace["time_s"]) != v["steps"] + 1:
                raise ValueError("Geometry trace length differs from reset plus executed controls")
    status = "rejected" if v["violations"] else "validated"
    if args.traces:
        path = args.out / status / f"{layout['layout_seed']}.npz"
        if any(a.dtype.hasobject for a in trace.values()):
            raise ValueError("Geometry returned a non-numeric trace")
        with path.open("xb") as stream:
            np.savez_compressed(stream, **trace)
        row.update(trace=str(path.relative_to(args.out)), trace_sha256=sha(path))
    row["geometry"] = metrics
    row["stage"] = "transitions"
    require(not v["violations"], "Replay differs from archived transitions", v["violations"])
    row["stage"] = "complete"


def failure(exc):
    if isinstance(exc, Rejected):
        return dict(status="rejected", reason=str(exc), detail=exc.details)
    return dict(status="exception", reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())


def label(seed, receipt=None):
    row = dict(arm=BODY, layout_id=1000000 + seed, layout_seed=seed, physics_validation={})
    if receipt:
        row["archived"] = {k: receipt[k] for k in ("success", "native_return", "steps", "tape_sha256")}
    return row


def episode_process(args, contract, layout, receipt):
    seed = layout["layout_seed"]
    row = label(seed, receipt)
    try:
        replay(args, contract, layout, receipt, row)
        row["status"] = "validated"
    except Exception as exc:
        row.update(failure(exc))
    save_case(args.out, row)


def save_case(out, row):
    directory = "exceptions" if row["status"] == "exception" else row["status"]
    path = out / directory / f"{row['layout_seed']}.json"
    row["record"] = str(path.relative_to(out))
    write(path, row)
    return row


def run_one(args, contract, layout, receipt):
    seed = layout["layout_seed"]
    try:
        worker = multiprocessing.get_context("spawn").Process(
            target=episode_process, args=(args, contract, layout, receipt))
        worker.start()
        worker.join()
        if worker.exitcode:
            raise RuntimeError(f"Native episode process exited with code {worker.exitcode}")
        files = [p for p in args.out.glob(f"*/{seed}.json")]
        if len(files) != 1:
            raise RuntimeError("Native episode process did not publish one record")
        return read(files[0])
    except Exception as exc:
        return save_case(args.out, dict(label(seed, receipt), stage="worker", **failure(exc)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkout", required=True, type=Path, help="immutable original evaluator Git worktree")
    parser.add_argument("--panel", required=True, type=Path, help="original large-panel-v3 root")
    parser.add_argument("--phase", choices=("dev",), default="dev")
    parser.add_argument("--out", required=True, type=Path, help="fresh diagnostic output directory")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1, metavar="1..8")
    parser.add_argument("--traces", action="store_true", help="retain the geometry arrays as compressed NPZ per case")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("run native replay in the allocated Linux simulator interpreter")
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LP_NUM_THREADS"):
        os.environ[key] = "1"
    args.checkout, args.panel, args.out = (p.resolve() for p in (args.checkout, args.panel, args.out))
    if any(args.out.is_relative_to(p) for p in (args.checkout, args.panel)):
        parser.error("--out must be outside the immutable evaluator and panel")
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ("validated", "rejected", "exceptions"):
        (args.out / name).mkdir()
    source = {}
    rows = {s: dict(label(s), status="pending") for s in SEEDS}
    output = dict(status="running", phase="dev", arm=BODY, source=source,
                  trajectory_limits=dict(position_m=POSITION_M, angle_rad=ANGLE_RAD),
                  observation_scope="Camera-free. Archived RGB/wrist hashes verified in the source opening only. "
                  "Replan proprio hashes counted directly; every pose also compared to archived opening plus motion. "
                  "Original tapes have no post-control plate poses or contact labels.")
    def publish():
        output["episodes"] = list(rows.values())
        output["counts"] = {k: sum(r["status"] == k for r in rows.values())
                            for k in ("pending", "validated", "rejected", "exception")}
        write(args.out / "report.json", output)

    publish()
    try:
        contract, layouts, receipts = prepare(args, source)
    except Exception as exc:
        error = failure(exc)
        output["input_failure"] = error
        for seed in SEEDS:
            rows[seed] = save_case(args.out, dict(label(seed), stage="shared_inputs", **error))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = [pool.submit(run_one, args, contract, layout, receipts[layout["layout_seed"]])
                       for layout in layouts]
            for future in as_completed(pending):
                row = future.result()
                if row["status"] == "validated":
                    # The per-case record retains compiled body and geom identities for decoding trace pairs.
                    row["geometry"] = {k: row["geometry"][k] for k in
                                       ("controls", "sampling", "physics_dt_s", "substeps_per_control", "summary")}
                else:
                    # Full partial observations, traceback and any geometry stay in their diagnostic record.
                    row = {k: v for k, v in row.items() if k not in ("traceback", "detail", "geometry")}
                rows[row["layout_seed"]] = row
                publish()
                print(f"{row['layout_seed']} {row['status']}", flush=True)
    output["status"] = "finished"
    publish()
    counts = output["counts"]
    print(json.dumps(counts, sort_keys=True), flush=True)
    return 1 if counts["exception"] else 2 if counts["rejected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
