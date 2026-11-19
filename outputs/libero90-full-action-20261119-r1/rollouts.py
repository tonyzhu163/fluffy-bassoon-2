"""Full-action, closed-loop retries on standard LIBERO-90. See IO.md."""
import argparse
from contextlib import ExitStack
import json
from multiprocessing import Pipe
from multiprocessing.connection import Connection
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
SISTER = HERE.parent / "libero90-meta-retry-20261119-r1"
sys.path.insert(1, str(SISTER))
from collect import (
    FrozenMotor as NativeMotor, SCENE_SEED, bind_json, lock, manifest_rows,
    policy_input, runtime_identity, sha, trial_name, write_json,
)
from standard_sim import HORIZON, NATIVE_CADENCE, SETTLE, Simulator, fingerprint

MANIFEST = SISTER / "cases.json"
MANIFEST_SHA = "398742aa78dbdd62334718c264f15dbbd0c041d2643c3d2571f19953ce25ca86"
LOGGED = ("query0", "query1")
LIVE = ("base", "matched", "meta_intact", "meta_cleared", "imitation_intact", "imitation_cleared")
STAGES = ("smoke", "fitting", "development", "live-development", "live-test")


class FrozenMotor(NativeMotor):
    def query_features(self, item):
        """The sister collector's state followed by mask-pooled frozen prefix."""
        return self.features(item)


def reader_process(fd, meta, imitation):
    from model import load_scorer

    with Connection(fd) as peer:
        try:
            scorers = {method: load_scorer(path, method) for method, path in
                       (("meta", Path(meta)), ("imitation", Path(imitation)))}
            peer.send((True, None))
            while True:
                method, support, state, embedding, proposals = peer.recv()
                scores = np.asarray(scorers[method](support, state, embedding, proposals), dtype=np.float32)
                if scores.shape != (2,) or not np.isfinite(scores).all():
                    raise ValueError("The frozen reader must return two finite logits")
                peer.send((True, scores.tolist()))
        except EOFError:
            pass
        except Exception:
            peer.send((False, traceback.format_exc()))
            raise


class Readers:
    def __init__(self, meta, imitation):
        self.paths = (meta, imitation)

    def __enter__(self):
        self.pipe, child = Pipe()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
                   PYTHONPATH=str(HERE.parents[1] / "src"), OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            self.process = subprocess.Popen(
                [env["READER_PYTHON"], "-Bu", __file__, "_reader", str(child.fileno()), *self.paths],
                env=env, pass_fds=(child.fileno(),))
        except BaseException:
            self.pipe.close()
            raise
        finally:
            child.close()
        try:
            self.receive()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def receive(self):
        ok, value = self.pipe.recv()
        if not ok:
            raise RuntimeError(f"Frozen reader failed\n{value}")
        return value

    def score(self, method, support, features, proposals):
        self.pipe.send((method, support, features[:8], features[8:], proposals.astype(np.float32, copy=False)))
        return np.array(self.receive(), dtype=np.float32)

    def __exit__(self, *exc):
        self.pipe.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


def policy_stream(arm):
    # Separate policy continuations on the fitting/development query reset.
    # The six live arms share the same independent candidate streams.
    return {"support": 0, "query0": 1, "query1": 2}.get(arm, 3)


def rng(seed, stream, channel):
    return np.random.default_rng(np.random.SeedSequence([seed, stream, channel]))


def checked_tape(issued, positions, gripper, rewards, initial, success):
    count = len(issued)
    arrays = dict(issued=np.asarray(issued), positions=np.asarray(positions), initial_position=np.asarray(initial),
                  gripper=np.asarray(gripper), rewards=np.asarray(rewards))
    shapes = dict(issued=(count, 7), positions=(count, 3), initial_position=(3,), gripper=(count, 2),
                  rewards=(count,))
    if any(a.shape != shapes[k] or not np.isfinite(a).all() for k, a in arrays.items()):
        raise ValueError("Incomplete public control tape")
    positive = np.flatnonzero(arrays["rewards"] > 0).tolist()
    if not 0 < count <= HORIZON or positive != ([count - 1] if success else []) or (not success and count != HORIZON):
        raise ValueError("Controls and native terminal reward disagree")
    return dict(arrays, controls=np.int32(count), terminal=np.bool_(success))


def rollout(sim, motor, readers, folder, arm, row, raw, identity, support, matched_views=None):
    folder.mkdir()
    started = time.monotonic()
    is_query = arm != "support"
    k = 1 if arm in ("base", "support") else 2
    stream = policy_stream(arm)
    generators = [rng(row["seed"], stream, slot) for slot in range(k)]
    coin = rng(row["seed"], stream, 2)
    initial = raw["robot0_eef_pos"].copy()
    issued, positions, gripper, rewards = [], [], [], []
    decisions = {key: [] for key in ("state", "embedding", "proposals", "scores", "index", "offset", "length")}
    if arm in ("matched", "base"):
        decisions["input_sha256"] = []
    times = dict(action_seconds=0., feature_seconds=0., reader_seconds=0., simulator_step_seconds=0.)
    ordinal = 0
    while len(issued) < HORIZON:
        item = policy_input(raw, identity["prompt"])
        if arm in ("matched", "base"):
            # Include both model input images, pre-cast proprioception and instruction.
            inputs = [fingerprint(item["observation/" + k]) for k in ("image", "wrist_image", "state")]
            view = sha(json.dumps([*inputs, item["prompt"]]).encode())
            decisions["input_sha256"].append(view)
        if arm == "base":
            hashes, embeddings = matched_views
            if ordinal >= len(hashes) or hashes[ordinal] != view:
                raise ValueError("Cannot reuse a readout from a different motor input")
            state = item["observation/state"].astype(np.float32)
            embedding = embeddings[ordinal]
        elif is_query:
            before = time.monotonic()
            features = motor.query_features(item)
            if features.size <= 8 or not np.array_equal(features[:8], item["observation/state"].astype(np.float32)):
                raise ValueError("pi05 query features do not start with the current physical state")
            times["feature_seconds"] += time.monotonic() - before
            state, embedding = features[:8], features[8:]
        before = time.monotonic()
        proposals = np.stack([motor.propose(item, g.standard_normal((10, 32), dtype=np.float32))
                              for g in generators])
        times["action_seconds"] += time.monotonic() - before
        scores = np.zeros(k, dtype=np.float32)
        if arm in LOGGED:
            index = int(coin.integers(2))
        elif arm in ("base", "matched", "support"):
            index = 0
        else:
            method, memory = arm.split("_")
            before = time.monotonic()
            scores = readers.score(method, support if memory == "intact" else None, features, proposals)
            times["reader_seconds"] += time.monotonic() - before
            index = int(np.argmax(scores))

        offset = len(issued)
        commands = proposals[index, :min(NATIVE_CADENCE, HORIZON - offset)]
        before = time.monotonic()
        step = sim.call("step", commands)
        times["simulator_step_seconds"] += time.monotonic() - before
        n = len(step["rewards"])
        if not 0 < n <= len(commands) or (n < len(commands) and not step["success"]):
            raise ValueError("Invalid number of executed LIBERO controls")
        if is_query:
            for key, value in (("state", state), ("embedding", embedding), ("proposals", proposals),
                               ("scores", scores), ("index", index), ("offset", offset), ("length", n)):
                decisions[key].append(value)
        issued.extend(commands[:n])
        positions.extend(step["positions"])
        gripper.extend(step["gripper"])
        rewards.extend(step["rewards"])
        ordinal += 1
        raw = step["observation"]
        if step["done"]:
            break
    success = bool(step["success"])
    if not step["done"]:
        raise ValueError("Missing native termination or time limit")
    tape = checked_tape(issued, positions, gripper, rewards, initial, success)
    np.savez_compressed(folder / "public.npz", **tape)
    if is_query:
        np.savez_compressed(folder / "decisions.npz", **decisions)
    record = dict(arm=arm, controls=len(issued), settling_controls=SETTLE, score=float(sum(rewards)),
                  success=success, truncated=not success, replans=ordinal, policy_calls=k * ordinal,
                  feature_calls=ordinal if arm not in ("base", "support") else 0, policy_stream=[row["seed"], stream],
                  identities=dict(issued=fingerprint(tape["issued"]), positions=fingerprint(tape["positions"]),
                                  rewards=fingerprint(tape["rewards"])),
                  arrays={p.name: sha(p.read_bytes()) for p in folder.glob("*.npz")},
                  elapsed_seconds=time.monotonic() - started, **times)
    write_json(folder / "result.json", record)
    return record, tape if arm == "support" else None


def collect_trial(sim, motor, readers, folder, row, arms, source, design_sha):
    started = time.monotonic()
    episodes, identities = {}, {}
    support = None
    reset_seconds = 0.
    # Only the base's saved diagnostic embeddings are reused. Each policy
    # still samples and executes its own actions on a newly reset simulator.
    execution_order = ("matched", "base", *arms[2:]) if arms == LIVE else arms
    matched_views = None
    for arm in ("support", *execution_order):
        role = "support" if arm == "support" else "query"
        case = dict(task_id=row["task_id"], init_state_id=row[role + "_id"], seed=SCENE_SEED)
        before = time.monotonic()
        raw, identity = sim.call("reset", case)
        reset_seconds += time.monotonic() - before
        if role in identities and identities[role] != identity:
            raise ValueError(f"Query reset drift in {trial_name(row)} / {arm}")
        identities[role] = identity
        if role == "query":
            practice = identities["support"]
            if practice["prompt"] != identity["prompt"] or any(
                    practice[key] == identity[key] for key in ("physics", "bank_state_sha256")):
                raise ValueError("Practice and retry must restore different poses of the same task")
        record, tape = rollout(sim, motor, readers, folder / arm, arm, row, raw, identity, support, matched_views)
        episodes[arm] = record
        if arm == "support":
            support = tape
        if arm == "matched":
            with np.load(folder / arm / "decisions.npz", allow_pickle=False) as archive:
                matched_views = archive["input_sha256"], archive["embedding"]
        if arm == "base" and record["identities"] != episodes["matched"]["identities"]:
            raise ValueError("Discarding the extra proposal changed the live native trajectory")
    if arms == LOGGED:
        # Both truly reset, and their opening readouts agree. Their policy draws must differ.
        with np.load(folder / "query0/decisions.npz") as a, np.load(folder / "query1/decisions.npz") as b:
            if any(not np.array_equal(a[key][0], b[key][0]) for key in ("state", "embedding")):
                raise ValueError("Different opening readout across fitting continuations")
            if np.array_equal(a["proposals"][0], b["proposals"][0]):
                raise ValueError("Independent fitting streams returned identical opening proposals")
    totals = {key: sum(r[key] for r in episodes.values()) for key in
              ("controls", "settling_controls", "policy_calls", "feature_calls", "action_seconds",
               "feature_seconds", "reader_seconds", "simulator_step_seconds")}
    return dict(manifest_row=row, source_pin=source, design_sha256=design_sha, env_seed=SCENE_SEED,
                reset=identities, episodes=episodes, elapsed_seconds=time.monotonic() - started,
                reset_seconds=reset_seconds, totals=totals)


def read_published(folder, row, arms, design_sha):
    trial = json.loads((folder / "trial.json").read_text())
    if trial["manifest_row"] != row or trial["design_sha256"] != design_sha:
        raise ValueError(f"Trial has a different source or contract at {folder}")
    for arm in ("support", *arms):
        record = trial["episodes"][arm]
        if record != json.loads((folder / arm / "result.json").read_text()):
            raise ValueError(f"Changed published result at {folder / arm}")
        for name, digest in record["arrays"].items():
            if sha((folder / arm / name).read_bytes()) != digest:
                raise ValueError(f"Changed published array at {folder / arm / name}")
    return trial


def run(args):
    live = args.stage.startswith("live-")
    smoke = args.stage == "smoke"
    host = socket.gethostname()
    if smoke:
        if not host.startswith("aibox-") or (args.shard_index, args.shards) != (0, 1):
            raise ValueError("Smoke uses one fitting trial on an allocated Baige development machine")
    elif not host.startswith("job-"):
        raise ValueError("Panel collection and live evaluations require a Baige Distributed Training job")
    if not 0 <= args.shard_index < args.shards:
        raise ValueError("Require 0 <= shard-index < shards")
    if live != bool(args.meta and args.imitation) or (not live and (args.meta or args.imitation)):
        raise ValueError("Supply both frozen weight files only for live evaluations")
    data = MANIFEST.read_bytes()
    if sha(data) != MANIFEST_SHA:
        raise ValueError("Expected the registered 48/12/30 manifest")
    split = "fitting" if smoke else args.stage.removeprefix("live-")
    rows = [r for r in manifest_rows(data) if r["split"] == split]
    rows = rows[:1] if smoke else rows[args.shard_index::args.shards]
    if not rows:
        raise ValueError("Empty rollout shard")
    arms = LIVE if live else LOGGED
    stage = args.output / "live" / split if live else args.output / split
    stage.mkdir(parents=True, exist_ok=True)
    weights = {k: sha(path.read_bytes()) for k, path in (("meta", args.meta), ("imitation", args.imitation))
               if path is not None}
    design = dict(version=1, source_pin=args.pin, runtime=runtime_identity(), weights_sha256=weights,
                  manifest_sha256=MANIFEST_SHA, stage=args.stage, query_arms=list(arms), scene_seed=SCENE_SEED,
                  policy_noise="PCG64 SeedSequence([manifest seed, policy stream, candidate slot]); "
                               "one float32 standard_normal([10,32]) per decision; coin on channel 2",
                  policy_streams={name: policy_stream(name) for name in ("support", *arms)},
                  horizon=HORIZON, cadence=NATIVE_CADENCE, settle=SETTLE)
    with lock(stage / ".design.lock"):
        bind_json(stage / "design.json", design)
    design_sha = sha((stage / "design.json").read_bytes())
    locks = stage / ".locks"
    locks.mkdir(exist_ok=True)
    started = time.monotonic()
    completed = resumed = 0
    with ExitStack() as resources:
        motor = readers = sim = None
        for row in rows:
            name = trial_name(row)
            final = stage / name
            with lock(locks / name):
                if final.exists():
                    record = read_published(final, row, arms, design_sha)
                    resumed += 1
                else:
                    if motor is None:
                        motor = FrozenMotor()
                        sim = resources.enter_context(Simulator())
                        if live:
                            readers = resources.enter_context(Readers(args.meta.resolve(), args.imitation.resolve()))
                            if any(sha(path.read_bytes()) != weights[k] for k, path in
                                   (("meta", args.meta), ("imitation", args.imitation))):
                                raise RuntimeError("Frozen weight file changed during reader startup")
                    work = Path(tempfile.mkdtemp(dir=stage, prefix=f".partial-{name}-"))
                    record = collect_trial(sim, motor, readers, work, row, arms, args.pin, design_sha)
                    write_json(work / "trial.json", record)
                    work.rename(final)
                    completed += 1
            print(json.dumps(dict(trial=name, fresh=completed, resumed=resumed, seconds=record["elapsed_seconds"],
                                  totals=record["totals"], success={k: v["success"] for k, v in record["episodes"].items()})),
                  flush=True)
    print(json.dumps(dict(stage=args.stage, completed=completed, resumed=resumed,
                          elapsed_seconds=time.monotonic() - started)), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["_reader"]:
        reader_process(int(sys.argv[2]), *sys.argv[3:])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("stage", choices=STAGES)
        parser.add_argument("--pin", required=True)
        parser.add_argument("--output", required=True, type=Path)
        parser.add_argument("--shard-index", type=int, default=0)
        parser.add_argument("--shards", type=int, default=1)
        parser.add_argument("--meta", type=Path)
        parser.add_argument("--imitation", type=Path)
        run(parser.parse_args())
