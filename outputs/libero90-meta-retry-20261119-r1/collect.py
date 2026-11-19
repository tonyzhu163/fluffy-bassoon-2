"""Collect one native practice and two live, paired retries on standard LIBERO-90.

The learner reads <output>/<split>/taskNN-rNN/query.npy and the support, default,
and alternate episode directories. public.npz follows decision-pretraining.
raw.npz adds unrotated camera pixels and proprio at every replan and at the end,
all proposal/noise tensors, their control offsets and post-control quaternions.
All identities and hashes are evaluator metadata. Run with --plan for shard costs.
"""
import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from standard_sim import (CAMERAS, HORIZON, NATIVE_CADENCE, PUBLIC_SHAPES, RESOLUTION,
                          SETTLE, SUITE, Simulator, fingerprint)

HERE = Path(__file__).resolve().parent
STAGES = ("fitting", "development", "test")
CADENCES = {"default": NATIVE_CADENCE, "alternate": 10 if NATIVE_CADENCE == 5 else 5}
ARMS = tuple(CADENCES)
SCENE_SEED = 7
LOG_SEED = 101  # same prospective random logging design as learner.py
OPENPI_PIN = "215abfb217dbac7d5f1273282331b9b1866c0479"
LIBERO_PIN = "87edbd1b823e464bc66557d213cc45e9f88aa543"
READY_SHA = "ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170"
CHECKPOINT = Path("/mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1")
DEFAULT_OUTPUT = Path("/mowenjie/fluffy-bassoon-2/outputs") / HERE.name / "episodes"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    # Callers own the path, either by a lock or by an unpublished directory.
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, allow_nan=False, indent=2) + "\n")
    temp.replace(path)


@contextmanager
def lock(path):
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def manifest_rows(data):
    rows = json.loads(data)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a nonempty JSON list")
    seen, seeds, poses, splits = set(), set(), set(), {}
    for row in rows:
        if not isinstance(row, dict) or row.get("split") not in STAGES:
            raise ValueError(f"Invalid manifest row {row!r}")
        if any(type(row.get(k)) is not int for k in ("task_id", "repeat", "support_id", "query_id", "seed")):
            raise ValueError(f"Missing or noninteger manifest field in {row}")
        task, repeat, seed = row["task_id"], row["repeat"], row["seed"]
        ids = row["support_id"], row["query_id"]
        if not (0 <= task < 90 and repeat >= 0 and 0 <= seed < 2**32 and all(0 <= i < 50 for i in ids)):
            raise ValueError(f"Out-of-range manifest field in {row}")
        if (task, repeat) in seen or seed in seeds or splits.setdefault(task, row["split"]) != row["split"]:
            raise ValueError(f"Repeated history, seed, or task across splits in {row}")
        for i in ids:
            if (task, i) in poses:
                raise ValueError(f"Support and query reset banks must be disjoint within a task: {row}")
            poses.add((task, i))
        seen.add((task, repeat))
        seeds.add(seed)
    return rows


def select(rows, stage, index, shards):
    if not 0 <= index < shards:
        raise ValueError("Require 0 <= --shard-index < --shards")
    # Stage-local ordering and behavior draws stay invariant to job order, sharding and retries.
    panel = sorted((r for r in rows if r["split"] == stage), key=lambda r: (r["task_id"], r["repeat"]))
    if not panel:
        raise ValueError(f"No {stage} cases in manifest")
    logged = np.random.default_rng(LOG_SEED).integers(2, size=len(panel))
    return [(i, panel[i], ARMS[int(logged[i])]) for i in range(index, len(panel), shards)]


def trial_name(row):
    return f"task{row['task_id']:02d}-r{row['repeat']:02d}"


def policy_input(raw, prompt):
    # Arithmetic and order from examples/libero/main.py at OPENPI_PIN.
    from openpi_client import image_tools

    def image(key):
        rotated = np.ascontiguousarray(raw[key][::-1, ::-1])
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(rotated, 224, 224))

    quat = raw["robot0_eef_quat"]
    w = np.clip(quat[3], -1., 1.)
    den = np.sqrt(1. - w * w)
    rotation = np.zeros(3) if math.isclose(den, 0.) else quat[:3] * 2. * math.acos(w) / den
    return {"observation/image": image("agentview_image"), "observation/wrist_image": image("robot0_eye_in_hand_image"),
            "observation/state": np.concatenate((raw["robot0_eef_pos"], rotation, raw["robot0_gripper_qpos"])),
            "prompt": prompt}


class FrozenMotor:
    def __init__(self):
        import jax
        from openpi.policies import policy_config
        from openpi.shared import nnx_utils
        from openpi.training import config

        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform != "gpu":
            raise RuntimeError(f"Expected one allocated JAX GPU, found {devices}")
        cfg = config.get_config("pi05_libero")
        if (cfg.model.action_horizon, cfg.model.action_dim) != (10, 32):
            raise RuntimeError("pi05_libero proposal or noise shape changed")
        self.policy = policy_config.create_trained_policy(cfg, str(CHECKPOINT))
        if self.policy._is_pytorch_model:
            raise RuntimeError("pi05_libero must load the native JAX checkpoint")
        self.embed = nnx_utils.module_jit(self.policy._model.embed_prefix)

    def features(self, item):
        import jax
        import jax.numpy as jnp
        from openpi.models import model

        transformed = self.policy._input_transform(dict(item))
        batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], transformed)
        obs = model.preprocess_observation(None, model.Observation.from_dict(batch), train=False)
        embeds, mask, _ = self.embed(obs)
        pooled = jnp.where(mask[..., None], embeds.astype(jnp.float32), 0).sum(axis=1) / mask.sum(axis=1, keepdims=True)
        result = np.concatenate((item["observation/state"], np.asarray(pooled[0]))).astype(np.float32)
        if result.ndim != 1 or result.size <= 8 or not np.isfinite(result).all():
            raise ValueError("Invalid reset query feature")
        return result

    def propose(self, item, noise):
        actions = np.asarray(self.policy.infer(item, noise=noise)["actions"])
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise ValueError("Invalid native action proposal")
        return actions


def noise_bank(seed, role):
    # Full independent Gaussian tensors keyed only by numbered manifest seed and
    # support/query role. At a common physical replan time both arms use the same tensor.
    rng = np.random.default_rng(np.random.SeedSequence([seed, role]))
    return rng.standard_normal((HORIZON // min(CADENCES.values()), 10, 32), dtype=np.float32)


def save_raw(path, arrays):
    # Low compression level keeps the full 256px frames without penalizing short rollouts.
    with ZipFile(path, "x", compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for key, values in arrays.items():
            with archive.open(key + ".npy", "w") as stream:
                np.lib.format.write_array(stream, np.asarray(values), allow_pickle=False)


def rollout(sim, motor, folder, case, identity, raw, cadence, noise, shared_opening=None):
    folder.mkdir()
    started = time.monotonic()
    calls = 0
    inference_seconds = 0.
    issued, positions, gripper, rewards, quats = [], [], [], [], []
    proposals, offsets, draws = [], [], []
    frames = {k: [] for k in PUBLIC_SHAPES}
    initial_position = raw["robot0_eef_pos"].copy()
    while len(issued) < HORIZON:
        offset = len(issued)
        for key in frames:
            frames[key].append(raw[key])
        draw = noise[offset // min(CADENCES.values())]
        if offset == 0 and shared_opening is not None:
            proposal = shared_opening
        else:
            before = time.monotonic()
            proposal = motor.propose(policy_input(raw, identity["prompt"]), draw)
            inference_seconds += time.monotonic() - before
            calls += 1
        proposals.append(proposal)
        draws.append(draw)
        offsets.append(offset)
        commands = proposal[:min(cadence, HORIZON - offset)]
        step = sim.call("step", commands)
        n = len(step["rewards"])
        if not (0 < n <= len(commands)) or (n != len(commands) and not step["success"]):
            raise ValueError("Missing live controls")
        issued.extend(commands[:n])
        positions.extend(step["positions"])
        gripper.extend(step["gripper"])
        rewards.extend(step["rewards"])
        quats.extend(step["quats"])
        raw = step["observation"]
        if step["done"]:
            break
    success = step["success"]
    if not step["done"] or (not success and len(issued) != HORIZON) or bool(np.any(np.asarray(rewards) > 0)) != success:
        raise ValueError("Invalid native terminal or truncation")
    for key in frames:
        frames[key].append(raw[key])  # terminal observation, at control index len(issued)
    np.savez_compressed(folder / "public.npz", issued=issued, positions=positions, initial_position=initial_position,
                        gripper=gripper, rewards=rewards, waits=np.zeros(len(issued), bool), opening=proposals[0])
    save_raw(folder / "raw.npz", dict(frames, control_offsets=offsets, proposed_actions=proposals, noise=draws,
                                      post_control_quat=quats))
    record = dict(case=case, identity=identity, task_name=identity["task_name"], controls=len(issued),
                  settling_controls=SETTLE, waits=0, success=success, truncated=not success,
                  cadence=cadence, opening_sha256=fingerprint(proposals[0]), noise_bank_sha256=fingerprint(noise),
                  policy_calls=len(proposals), fresh_policy_calls=calls, reused_opening=shared_opening is not None,
                  inference_seconds=inference_seconds, rollout_and_write_seconds=time.monotonic() - started,
                  tape_bytes=sum((folder / name).stat().st_size for name in ("public.npz", "raw.npz")))
    write_json(folder / "result.json", record)
    return record, proposals[0]


def collect_trial(sim, motor, work, row, logged_arm):
    started = time.monotonic()

    def case(role):
        return dict(suite=SUITE, variant="BASE", task_id=row["task_id"], init_state_id=row[role + "_id"],
                    seed=SCENE_SEED, trial_seed=row["seed"], noise_role=role)

    support_case, query_case = case("support"), case("query")
    raw, previous = sim.call("reset", support_case)
    support, _ = rollout(sim, motor, work / "support", support_case, previous, raw,
                         NATIVE_CADENCE, noise_bank(row["seed"], 0))
    raw, identity = sim.call("reset", query_case)
    if identity["bank_state_sha256"] == previous["bank_state_sha256"] or identity["physics"] == previous["physics"]:
        raise ValueError("Support and query restored the same state")
    if previous["prompt"] != identity["prompt"]:
        raise ValueError("Standard task instruction changed between practice and retry")
    # The only query readout available to the scorer, persisted before either live retry.
    query = motor.features(policy_input(raw, identity["prompt"]))
    np.save(work / "query.npy", query, allow_pickle=False)
    behavior = dict(arm=logged_arm, probability=0.5, log_seed=LOG_SEED,
                    query_sha256=fingerprint(query), control_offset=0)
    write_json(work / "behavior.json", behavior)
    paired_noise = noise_bank(row["seed"], 1)
    default, opening = rollout(sim, motor, work / "default", query_case, identity, raw, NATIVE_CADENCE, paired_noise)
    raw, paired = sim.call("reset", query_case)
    if paired != identity:
        raise ValueError(f"Query branches have different reset fingerprints: {identity} vs {paired}")
    alternate, _ = rollout(sim, motor, work / "alternate", query_case, paired, raw, CADENCES["alternate"],
                           paired_noise, shared_opening=opening)
    branches = dict(default=default, alternate=alternate)
    episodes = (support, default, alternate)
    return dict(trial=trial_name(row), manifest_row=row, support=support, queries=branches, behavior=behavior,
                cost=dict(episodes=3, controls=sum(r["controls"] for r in episodes), settling_controls=3 * SETTLE,
                          policy_calls=sum(r["policy_calls"] for r in episodes),
                          fresh_policy_calls=sum(r["fresh_policy_calls"] for r in episodes), reset_feature_calls=1,
                          seconds=time.monotonic() - started, tape_bytes=sum(r["tape_bytes"] for r in episodes)))


def runtime_identity():
    for name, pin, paths in (
        ("OPENPI_ROOT", OPENPI_PIN, ("src", "packages/openpi-client", "examples/libero/main.py")),
        ("LIBERO_STD_ROOT", LIBERO_PIN, ("libero/libero",)),
    ):
        root = Path(os.environ[name])
        actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if actual != pin:
            raise RuntimeError(f"{name} must be at {pin}, found {actual}")
        subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", *paths], check=True)
    if sha((CHECKPOINT / "READY.json").read_bytes()) != READY_SHA:
        raise RuntimeError(f"Unexpected JAX checkpoint receipt at {CHECKPOINT}")
    for item in json.loads((CHECKPOINT / "READY.json").read_text())["files"]:
        if (CHECKPOINT / item["name"]).stat().st_size != item["bytes"]:
            raise RuntimeError(f"Missing or truncated checkpoint component: {item['name']}")
    return dict(openpi=OPENPI_PIN, libero=LIBERO_PIN, model="pi05_libero", checkpoint=str(CHECKPOINT),
                checkpoint_receipt_sha256=READY_SHA, simulator_python=os.environ["LIBERO_SIM_PYTHON"],
                model_python=sys.executable, standard_config=os.environ["LIBERO_CONFIG_PATH"])


def bind_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Output was bound to a different experiment: {path}")
    else:
        write_json(path, value)


def bind_design(root, data, runtime, args):
    root.mkdir(parents=True, exist_ok=True)
    design = dict(suite=SUITE, manifest_sha256=sha(data), runtime=runtime,
                  collector_sha256={name: sha((HERE / name).read_bytes())
                                    for name in ("collect.py", "standard_sim.py", "runner.sh")},
                  cadences=CADENCES, cameras=CAMERAS, camera_resolution=RESOLUTION, model_resolution=224,
                  settle_steps=SETTLE, horizon=HORIZON, scene_seed=SCENE_SEED, flow_steps=10,
                  noise="PCG64 SeedSequence([manifest seed, role=0 support/1 query]), indexed by control // 5",
                  policy_calls="replans needed for a standalone episode; fresh_policy_calls counts actual inference",
                  readout="settled public proprioception and mask-mean frozen image/language prefix, before retry")
    # JSON uses arrays for tuples. Keep the in-memory representation identical on resume.
    design["cameras"] = list(CAMERAS)
    with lock(root / ".design.lock"):
        bind_json(root / "design.json", design)
        manifest = root / "manifest.json"
        if manifest.exists():
            if manifest.read_bytes() != data:
                raise ValueError("Saved manifest differs from input")
        else:
            temp = root / ".manifest.tmp"
            temp.write_bytes(data)
            temp.replace(manifest)
        if args.stage == "test":
            if args.frozen_reader is None:
                raise ValueError("Registered test collection requires --frozen-reader RUN/frozen/identity.json")
            frozen = args.frozen_reader.read_bytes()
            reader = json.loads(frozen)
            if (reader["manifest_sha256"] != sha(data) or reader["suite"] != SUITE
                    or reader["branches"] != list(ARMS) or not reader["checkpoints"]):
                raise ValueError("Frozen reader identity disagrees with the test experiment")
            bind_json(root / "test-freeze.json", dict(reader_identity_sha256=sha(frozen), manifest_sha256=sha(data)))
    return sha((root / "design.json").read_bytes())


def read_finished(folder, row, design_sha):
    record = json.loads((folder / "result.json").read_text())
    if record["manifest_row"] != row or record["design_sha256"] != design_sha:
        raise ValueError(f"Cannot resume a different trial at {folder}")
    for arm in ("support", *ARMS):
        episode = record["support"] if arm == "support" else record["queries"][arm]
        if json.loads((folder / arm / "result.json").read_text()) != episode:
            raise ValueError(f"Published episode disagrees with trial at {folder}/{arm}")
        if sum((folder / arm / n).stat().st_size for n in ("raw.npz", "public.npz")) != episode["tape_bytes"]:
            raise ValueError(f"Published episode tapes changed at {folder}/{arm}")
    query = np.load(folder / "query.npy", allow_pickle=False)
    if (json.loads((folder / "behavior.json").read_text()) != record["behavior"]
            or fingerprint(query) != record["behavior"]["query_sha256"]):
        raise ValueError(f"Published reset readout changed at {folder}")
    return record


def run(args, data, selected, plan):
    root = args.output.resolve()
    runtime = runtime_identity()
    design_sha = bind_design(root, data, runtime, args)
    stage = root / args.stage
    locks = stage / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    jobs = root / ".jobs"
    jobs.mkdir(exist_ok=True)
    attempt = Path(tempfile.mkdtemp(dir=jobs, prefix=f"{args.stage}-{args.shard_index:04d}-"))
    started = time.monotonic()
    receipt = dict(plan, host=socket.gethostname(), pid=os.getpid(), started_unix=time.time(),
                   code_revision=subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                                                         text=True).strip(),
                   attempt=str(attempt), design_sha256=design_sha, status="running",
                   collected=0, resumed=0, fresh_policy_calls=0, controls=0, settling_controls=0)
    write_json(attempt / "job.json", receipt)
    print(json.dumps(receipt), flush=True)
    try:
        probe = attempt / "gpu.json"
        subprocess.run([os.environ["BAIGE_PROBE_PYTHON"], "-Bu", str(HERE.parents[1] / "scripts/[train]gpu_probe.py"),
                        "--expected-count", "1", "--purpose", "evaluation", "--baige-container", "--output", str(probe)],
                       check=True)
        gpu = json.loads(probe.read_text())["devices"][0]["cuda_name"]
        if "4090" not in gpu and "6000D" not in gpu:
            raise RuntimeError(f"Expected distributed RTX 4090 or RTX PRO 6000D, got {gpu}")
        with ExitStack() as stack, (attempt / "progress.jsonl").open("a", buffering=1) as progress:
            motor = None
            for i, row, logged_arm in selected:
                name = trial_name(row)
                final = stage / name
                with lock(locks / name):
                    if final.exists():
                        record = read_finished(final, row, design_sha)
                        receipt["resumed"] += 1
                        resumed = True
                    else:
                        if motor is None:
                            motor = FrozenMotor()
                            sim = stack.enter_context(Simulator())
                        # Stage on the same filesystem. A killed attempt keeps its partial raw files.
                        work = Path(tempfile.mkdtemp(dir=stage, prefix=f".partial-{name}-"))
                        write_json(work / "intent.json", dict(manifest_row=row, design_sha256=design_sha,
                                                              attempt=attempt.name, stage_index=i))
                        record = collect_trial(sim, motor, work, row, logged_arm)
                        record["design_sha256"] = design_sha
                        write_json(work / "result.json", record)
                        work.rename(final)
                        receipt["collected"] += 1
                        resumed = False
                        for key in ("fresh_policy_calls", "controls", "settling_controls"):
                            receipt[key] += record["cost"][key]
                event = dict(trial=name, stage_index=i, resumed=resumed, cost=record["cost"],
                             success=[record["support"]["success"], *(record["queries"][k]["success"] for k in ARMS)])
                progress.write(json.dumps(event) + "\n")
                print(json.dumps(event), flush=True)
                receipt["elapsed_seconds"] = time.monotonic() - started
                write_json(attempt / "job.json", receipt)
        receipt["status"] = "complete"
    except BaseException:
        receipt["status"] = "interrupted"
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        receipt["ended_unix"] = time.time()
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["one_gpu_wall_hours"] = receipt["elapsed_seconds"] / 3600
        write_json(attempt / "job.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="shared episode root for all stages/shards")
    parser.add_argument("--manifest", type=Path, default=HERE / "cases.json")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--shard-index", type=int, default=0, help="zero-based within the selected stage")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--frozen-reader", type=Path, help="test-stage learner identity JSON")
    parser.add_argument("--plan", action="store_true", help="validate manifest and print shard bounds without collecting")
    args = parser.parse_args()
    data = args.manifest.read_bytes()
    selected = select(manifest_rows(data), args.stage, args.shard_index, args.shards)
    count = len(selected)
    plan = dict(stage=args.stage, shard_index=args.shard_index, shards=args.shards, manifest_sha256=sha(data),
                histories=count, episodes=count * 3, max_controls=count * 3 * HORIZON, settling_controls=count * 3 * SETTLE,
                max_policy_calls=count * (2 * math.ceil(HORIZON / NATIVE_CADENCE) + math.ceil(HORIZON / CADENCES["alternate"])),
                first=trial_name(selected[0][1]) if selected else None, last=trial_name(selected[-1][1]) if selected else None)
    if args.plan or not count:
        print(json.dumps(plan))
        return
    run(args, data, selected, plan)


if __name__ == "__main__":
    main()
