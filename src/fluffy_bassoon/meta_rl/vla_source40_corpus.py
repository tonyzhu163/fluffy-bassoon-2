"""Persistent official DSRL histories on source LIBERO40 with frozen π0.5.

One invocation owns one task and one case partition. SAC is never reset at case
boundaries. Train and validation invocations have independent replay/optimizers.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

import numpy as np

from .vla_source40_schedule import SOURCE_SUITES, content_hash, load_schedule


PI05_RECEIPT = "ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170"


def save_motor_rng(key):
    import jax
    names = {"fry": "threefry2x32", "threefry2x32": "threefry2x32",
             "rbg": "rbg", "urbg": "unsafe_rbg", "unsafe_rbg": "unsafe_rbg"}
    implementation = names.get(str(jax.random.key_impl(key)))
    if implementation is None:
        raise ValueError("Unsupported motor PRNG implementation")
    return dict(format="jax_motor_key_v1", implementation=implementation,
                key_data=np.asarray(jax.device_get(jax.random.key_data(key))).copy())


def restore_motor_rng(saved):
    import jax
    import jax.numpy as jnp
    if saved.get("format") != "jax_motor_key_v1":
        raise ValueError("Require an explicitly encoded motor PRNG key")
    return jax.random.wrap_key_data(jnp.asarray(saved["key_data"], dtype=jnp.uint32),
                                    impl=saved["implementation"])


class OnlineSourceSimulator:
    """Collect with fixed episode weights, then apply official SAC updates."""
    def __init__(self, sim, learner, sampler, observation_fn, *, warmup=500, batch=256, utd=20,
                 before_step=None, after_outcome=None, before_reset=None, after_reset=None):
        self.sim, self.metadata = sim, sim.metadata
        self.learner, self.sampler, self.observation_fn = learner, sampler, observation_fn
        self.warmup, self.batch, self.utd = warmup, batch, utd
        self.before_step, self.after_outcome = before_step, after_outcome
        self.after_reset = after_reset
        self.before_reset = before_reset
        self.provenance = []

    def reset(self, initial_state, scene_seed):
        self.provenance = []
        self.warming = self.learner.updates == 0
        if self.before_reset:
            self.before_reset(initial_state, scene_seed)
        result = self.sim.reset(initial_state, scene_seed)
        if self.after_reset:
            self.after_reset(initial_state, scene_seed, result)
        return result

    def choose(self, observation):
        warming = self.warming
        noise = (self.sampler.standard_normal(32).astype(np.float32) if warming else
                 np.clip(self.learner.action(observation), -1., 1.).astype(np.float32))
        self.pending = dict(observation=observation, noise=noise,
            stage="warmup_gaussian" if warming else "sac_online",
            replay_before=len(self.learner.replay), updates_before=self.learner.updates)
        return noise

    def step(self, actions):
        if self.before_step:
            self.before_step(actions)
        outcome = self.sim.step(actions)
        pending = self.pending
        if self.after_outcome:
            self.after_outcome(pending, outcome)
        reward = np.float32(0. if outcome["success"] else -1.)
        self.learner.replay.append(dict(observations=self.observation_fn(pending["observation"]),
            next_observations=self.observation_fn(outcome["observation"]),
            actions=pending["noise"][None].copy(), next_actions=pending["noise"][None].copy(),
            rewards=reward, masks=np.float32(not outcome["success"]),
            discount=np.float32(.999 ** outcome["primitive_steps"])))
        self.provenance.append(dict(source_stage=pending["stage"],
            source_updates_before=pending["updates_before"], source_updates_after=self.learner.updates,
            source_replay_before=pending["replay_before"], source_replay_after=len(self.learner.replay),
            source_update_metrics={}))
        return outcome

    def finish_episode(self):
        metrics = {}
        if len(self.learner.replay) > self.warmup:
            for _ in range(len(self.provenance) * self.utd):
                metrics = self.learner.update(self.batch, self.sampler)
        return metrics


def physical_costs(journal):
    """Known outcomes plus conservative charges for unresolved RPC intents."""
    outcomes = {p.stem: json.loads(p.read_text()) for p in journal.glob("decision-*.json")}
    known = sum(row["primitive_steps"] for row in outcomes.values())
    uncertain = sum(json.loads(p.read_text())["maximum_primitive_steps"]
                    for p in journal.glob("intent-*.json")
                    if "decision-" + p.stem[len("intent-"):] not in outcomes)
    return dict(known_primitive_steps=known, uncertain_primitive_steps_ceiling=uncertain,
                primitive_step_budget_charge=known + uncertain)


def paired_arrays(rows, labels):
    starts = np.asarray([i == 0 or rows[i - 1]["done"] for i in range(len(rows))], bool)
    base = dict(observations=np.stack([row["public_features"] for row in rows]).astype(np.float32),
        executed_noise=np.stack([row["executed_noise"] for row in rows]).astype(np.float32),
        rewards=np.asarray([row["reward"] for row in rows], np.float32),
        dones=np.asarray([row["done"] for row in rows], bool), episode_starts=starts,
        teacher_steps=np.asarray([row["source_updates_before"] for row in rows], np.int64))
    return dict(ad=dict(**base, target_noise=base["executed_noise"].copy()),
                dpt=dict(**base, target_noise=np.asarray(labels, np.float32)))


def has_pending_case(progress, case_count, target_chunks):
    # A recovery can cross target mid-case. It must finish that same case and
    # publication before stopping, rather than silently dropping its tail.
    return progress["case_index"] < case_count and (
        progress["episode_in_case"] > 0 or progress["chunks"] < target_chunks)


def publish_prefix(output, learner, contract, progress, checkpoint_sha256):
    """Version a complete prefix; raw warmup remains in every later prefix."""
    from .vla_history import save_dataset
    rows = []
    for episode in progress["episode_receipts"]:
        from .vla_libero import sha256
        if sha256(output / episode["trace"]) != episode["trace_sha256"]:
            raise ValueError("Raw source episode checksum changed before publication")
        with gzip.open(output / episode["trace"], "rb") as stream:
            rows.extend(pickle.load(stream)["rows"])
    if not rows or rows[-1]["source_updates_before"] == rows[0]["source_updates_before"]:
        return dict(status="raw_only", reason="No changing source-policy history yet; not labeled AD", episodes=len(progress["episode_receipts"]))
    labels = np.stack([np.asarray(learner.agent.eval_actions(
        {key: value[None] for key, value in row["sac_observation"].items()})).reshape(32) for row in rows])
    arrays = paired_arrays(rows, labels)
    directory = output / f"prefix-through-case-{progress['case_index']:05d}"
    if directory.exists():
        # An interrupted publication is not a committed dataset. Preserve it.
        destination = directory.with_name(directory.name + f"-uncommitted-{time.time_ns()}")
        directory.rename(destination)
    directory.mkdir()
    datasets = {}
    for objective, values in arrays.items():
        path = directory / f"{objective}.npz"
        manifest = save_dataset(path, [values], objective=objective, metadata=dict(
            source_contract=contract, task_key=contract["task"]["task_key"],
            global_task_split="source40", local_scene_split=contract["partition"],
            feature_contract=contract["feature_contract"], motor_sha256=contract["motor"]["receipt_sha256"],
            noise_contract=contract["noise_contract"], trial_ids=[contract["history_id"]],
            source_case_ids=[case["case_id"] for case in contract["cases"][:progress["case_index"] + 1]],
            source_generation_episodes=len(progress["episode_receipts"]), source_generation_chunks=len(rows),
            source_generation_primitive_steps=progress["primitive_steps"], source_updates=learner.updates,
            source_actor_checkpoint_sha256=checkpoint_sha256,
            teacher_status="candidate current actor; no independent expert/optimality claim",
            history_policy="Actual chronological source learning across cases, including all warmup; no artificial policy reset",
            deployment_shot_count="Not inferred from source lengths; target evaluation is one support episode plus one query",
            publication_rule="Choose one latest committed prefix per history; do not concatenate overlapping prefixes"))
        datasets[objective] = dict(path=str(path.relative_to(output)), sha256=manifest["sha256"])
    return dict(status="paired_ad_dpt", datasets=datasets, transitions=len(rows),
                episodes=len(progress["episode_receipts"]), case_count=progress["case_index"] + 1)


def validate_simulator(task, metadata):
    if (task["suite"] not in SOURCE_SUITES or not 0 <= task["task_id"] < 10
            or metadata.get("suite") != task["suite"] or metadata.get("task_id") != task["task_id"]
            or metadata.get("bddl_sha256") != task["bddl_sha256"]
            or Path(metadata.get("bddl_file", "")).stem != task["semantic_task"]
            or metadata.get("initial_states") != 50 or metadata.get("horizon") != 400
            or metadata.get("fresh_environment_on_reset") is not True):
        raise ValueError("Simulator must match the declared source40 task and fresh-environment reset contract")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("schedule", "output", "checkpoint", "openpi-root", "dsrl-root"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--task-key", required=True)
    ap.add_argument("--partition", choices=("train", "validation"), required=True)
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--max-primitive-steps", type=int, required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    schedule = load_schedule(args.schedule)
    tasks = [row for row in schedule["tasks"] if row["task_key"] == args.task_key]
    if len(tasks) != 1:
        raise ValueError("Task absent from the source-only schedule")
    if args.max_primitive_steps != schedule["maximum_primitive_steps_per_task_partition"]:
        raise ValueError("Explicit job budget must equal the declared complete-case ceiling")
    if sys.platform != "linux":
        raise ValueError("Collection/model computation requires allocated Linux")
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get("FLUFFY_GPU_EXECUTION", "baige"), os.environ)
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()):
        raise ValueError("Require the exact clean reviewed collection commit")
    from .vla_libero import FrozenOpenPI, OfficialSAC, SimulatorClient, run_episode, sac_observation, sha256, write_json
    task = tasks[0]
    cases = task["partitions"][args.partition]
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if not args.resume and (args.output / "contract.json").exists():
        raise ValueError("New collection requires a fresh output identity")
    import fcntl
    lock = (args.output / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (args.output / "raw").mkdir(exist_ok=True)
    started, stage, sim = time.monotonic(), "connecting", SimulatorClient(args.port)
    try:
        validate_simulator(task, sim.metadata)
        if sha256(args.checkpoint / "READY.json") != PI05_RECEIPT:
            raise ValueError("Require the exact official OpenPI pi05_libero checkpoint")
        args.model_config, args.feature_interface = "pi05_libero", "vla-prefix"
        stage = "motor_qualification"
        motor = FrozenOpenPI(args)
        journal = args.output / "raw" / "decision-journal"
        journal.mkdir(exist_ok=True)
        qualification_reset = journal / f"reset-qualification-{time.time_ns()}.json"
        write_json(qualification_reset, dict(kind="qualification", returned=False,
                   settling_steps=int(sim.metadata["settle_steps"])))
        initial = sim.reset(cases[0]["initial_state"], cases[0]["scene_seed"])["observation"]
        write_json(qualification_reset, dict(kind="qualification", returned=True,
                   settling_steps=int(sim.metadata["settle_steps"])))
        zeros = np.zeros((10, 32), np.float32)
        a, b = motor.infer(initial, sim.metadata["instruction"], zeros), motor.infer(initial, sim.metadata["instruction"], zeros.copy())
        delta = float(np.max(np.abs(a - b)))
        if not np.isfinite(delta) or delta > 1e-5 or motor.horizon != 10:
            raise RuntimeError("Official pi0.5 same-noise/native-shape qualification failed")
        write_json(args.output / "motor-qualification.json", dict(same_noise_max_delta=delta,
            native_horizon=motor.horizon, feature_parity=motor.feature_encoder.qualify(initial, sim.metadata["instruction"])))
        seed = int(content_hash(dict(task=args.task_key, partition=args.partition, seed=schedule["seed"]))[:8], 16)
        contract = dict(format="openpi_pi05_libero40_persistent_source_v1", source_commit=args.source_commit,
            schedule_sha256=schedule["schedule_sha256"], task=task, partition=args.partition,
            history_id=f"{args.task_key}/{args.partition}/seed-{schedule['seed']}", cases=cases,
            source_only=True, source_data_from_targets=False,
            motor=motor.metadata, feature_contract=motor.feature_encoder.contract,
            noise_contract="32D tied across native H10; warmup unbounded N(0,I), then SAC clipped[-1,1]; no rescaling",
            source_recipe=schedule["source_recipe"], learner_seed=seed,
            horizon=400, chunk=10, target_chunks=schedule["target_chunks_per_task_partition"],
            max_primitive_steps=args.max_primitive_steps,
            resets="Simulator resets each episode, same state/seed within case; SAC persists across all cases in this task+partition",
            source_generation_vs_deployment=schedule["deployment"],
            validation="Independent SAC/replay on disjoint validation cases; no validation interactions enter train replay or student gradients")
        if args.resume:
            if json.loads((args.output / "contract.json").read_text()) != contract:
                raise ValueError("Resumed source contract changed")
        else:
            write_json(args.output / "contract.json", contract)
        learner, sampler = OfficialSAC(args.dsrl_root, seed), np.random.default_rng(seed)
        recovery = args.output / "source-recovery.pkl"
        progress = dict(case_index=0, episode_in_case=0, primitive_steps=0, chunks=0,
            settling_steps=0, episode_receipts=[], publications=[])
        if args.resume:
            progress = learner.restore(recovery, sampler, contract)
            motor.policy._rng = restore_motor_rng(progress.pop("motor_rng"))
        physical_cost = physical_costs(journal)["primitive_step_budget_charge"]
        settling_cost = sum(json.loads(path.read_text())["settling_steps"] for path in journal.glob("reset-*.json"))
        inflight, reset_receipt = None, None
        def before_reset(initial_state, scene_seed):
            nonlocal settling_cost, reset_receipt
            reset_receipt = journal / f"reset-episode-{time.time_ns()}.json"
            write_json(reset_receipt, dict(kind="episode", returned=False,
                case_id=cases[progress["case_index"]]["case_id"], initial_state=initial_state,
                scene_seed=scene_seed, settling_steps=int(sim.metadata["settle_steps"])))
            settling_cost += int(sim.metadata["settle_steps"])
        def after_reset(initial_state, scene_seed, result):
            write_json(reset_receipt, dict(kind="episode", returned=True,
                case_id=cases[progress["case_index"]]["case_id"], initial_state=initial_state,
                scene_seed=scene_seed, initial_state_vector_sha256=result.get("initial_state_vector_sha256"),
                initial_observation_sha256=result["initial_observation_sha256"],
                settling_steps=int(sim.metadata["settle_steps"])))
        def before_step(actions):
            nonlocal physical_cost, inflight
            if physical_cost + len(actions) > args.max_primitive_steps:
                raise RuntimeError("Physical cost including interrupted attempts would exceed the job ceiling")
            inflight = dict(event_id=str(time.time_ns()), maximum_primitive_steps=len(actions))
            write_json(journal / f"intent-{inflight['event_id']}.json", dict(inflight,
                case_id=cases[progress["case_index"]]["case_id"],
                episode_in_case=progress["episode_in_case"]))
            physical_cost += len(actions)
        def after_outcome(pending, outcome):
            nonlocal physical_cost
            # Persist actual outcomes before an optimizer failure can discard
            # the incomplete episode. These are never padded into AD trials.
            event_id = f"decision-{inflight['event_id']}"
            trace = journal / (event_id + ".pkl.gz")
            receipt = dict(primitive_steps=int(outcome["primitive_steps"]),
                success=bool(outcome["success"]), done=bool(outcome["done"]),
                case_id=cases[progress["case_index"]]["case_id"], source_stage=pending["stage"],
                source_updates_before=pending["updates_before"], trace=trace.name, trace_sha256=None,
                trace_status="writing")
            write_json(journal / (event_id + ".json"), receipt)
            physical_cost += int(outcome["primitive_steps"]) - inflight["maximum_primitive_steps"]
            with gzip.open(trace, "wb", compresslevel=1) as stream:
                pickle.dump(dict(pending=pending, outcome=outcome,
                    case=cases[progress["case_index"]], episode_in_case=progress["episode_in_case"]), stream, protocol=5)
            write_json(journal / (event_id + ".json"), dict(receipt, trace_sha256=sha256(trace), trace_status="complete"))
        def save_boundary():
            progress["settling_steps"] = settling_cost
            learner.save(recovery, sampler, contract, **progress,
                         motor_rng=save_motor_rng(motor.policy._rng))
            write_json(args.output / "progress.json", dict(progress, source_updates=learner.updates,
                replay_rows=len(learner.replay), latest_recovery_sha256=sha256(recovery)))
            costs = physical_costs(journal)
            resets = [json.loads(p.read_text()) for p in journal.glob("reset-*.json")]
            write_json(args.output / "index.json", dict(progress,
                format="openpi_pi05_libero40_source_history_index_v1", completed=False,
                source_only=True, task_key=args.task_key, partition=args.partition,
                schedule_sha256=schedule["schedule_sha256"], source_contract_sha256=content_hash(contract),
                motor_receipt_sha256=motor.metadata["receipt_sha256"], feature_contract=motor.feature_encoder.contract,
                noise_contract=contract["noise_contract"],
                actual_primitive_steps=costs["known_primitive_steps"],
                physical_cost_accounting=costs,
                committed_learning_history_primitive_steps=progress["primitive_steps"],
                actual_settling_steps=sum(p["settling_steps"] for p in resets if p["returned"]),
                settling_step_budget_charge=settling_cost,
                uncertain_settling_steps_ceiling=sum(p["settling_steps"] for p in resets if not p["returned"]),
                source_updates=learner.updates,
                latest_recovery_sha256=sha256(recovery), collection_cost_count_once=True))
        if not args.resume:
            save_boundary()
        wrapped = OnlineSourceSimulator(sim, learner, sampler, sac_observation,
            before_step=before_step, after_outcome=after_outcome,
            before_reset=before_reset, after_reset=after_reset)
        while has_pending_case(progress, len(cases), contract["target_chunks"]):
            case = cases[progress["case_index"]]
            stage = f"case_{progress['case_index']}"
            while progress["episode_in_case"] < case["episodes"]:
                rows, summary = run_episode(wrapped, motor, wrapped.choose,
                    initial_state=case["initial_state"], scene_seed=case["scene_seed"],
                    chunk=10, bounded=False, record_features=True)
                if (not rows or not rows[-1]["done"] or len(rows) != len(wrapped.provenance)
                        or summary["primitive_steps"] > 400 or any(row["public_features"].shape != (4104,) for row in rows)):
                    raise RuntimeError("Require complete real episodes and every shared prefix feature")
                update_metrics = wrapped.finish_episode()
                for row, provenance in zip(rows, wrapped.provenance):
                    row.update(provenance, case_id=case["case_id"], source_partition=args.partition,
                               source_episode_in_case=progress["episode_in_case"],
                               source_updates_after_episode=learner.updates)
                previous = [receipt for receipt in progress["episode_receipts"] if receipt["case_id"] == case["case_id"]]
                if not isinstance(summary.get("initial_state_vector_sha256"), str):
                    raise RuntimeError("Fresh simulator omitted the actual nominal state-vector hash")
                if previous and summary.get("initial_state_vector_sha256") != previous[0].get("initial_state_vector_sha256"):
                    raise RuntimeError("Nominal fixed case state changed across retries")
                progress["primitive_steps"] += int(summary["primitive_steps"])
                progress["chunks"] += len(rows)
                if progress["primitive_steps"] > args.max_primitive_steps:
                    raise RuntimeError("Exceeded the declared complete-case interaction ceiling")
                ordinal = len(progress["episode_receipts"])
                trace = args.output / "raw" / f"episode-{ordinal:06d}.pkl.gz"
                if trace.exists():
                    trace.rename(trace.with_name(trace.name + f"-uncommitted-{time.time_ns()}"))
                with gzip.open(trace, "wb", compresslevel=1) as stream:
                    pickle.dump(dict(rows=rows, summary=summary, case=case,
                        source_contract_sha256=content_hash(contract)), stream, protocol=5)
                progress["episode_receipts"].append(dict(summary, case_id=case["case_id"],
                    episode_in_case=progress["episode_in_case"], trace=str(trace.relative_to(args.output)),
                    trace_sha256=sha256(trace), source_updates=learner.updates,
                    source_update_metrics=update_metrics,
                    source_stages=dict(__import__("collections").Counter(row["source_stage"] for row in rows))))
                progress["episode_in_case"] += 1
                save_boundary()
                print(json.dumps(dict(case_id=case["case_id"], episode_in_case=progress["episode_in_case"],
                    chunks=progress["chunks"], primitive_steps=progress["primitive_steps"], source_updates=learner.updates)), flush=True)
            checkpoint = args.output / f"case-{progress['case_index']:05d}-source.pkl"
            learner.save(checkpoint, sampler, contract, **progress,
                         motor_rng=save_motor_rng(motor.policy._rng))
            publication = publish_prefix(args.output, learner, contract, progress, sha256(checkpoint))
            progress["publications"].append(dict(publication, through_case=progress["case_index"],
                source_checkpoint=checkpoint.name, source_checkpoint_sha256=sha256(checkpoint)))
            progress["case_index"] += 1
            progress["episode_in_case"] = 0
            save_boundary()
        final = dict(json.loads((args.output / "index.json").read_text()), completed=True,
            target_chunks_reached=progress["chunks"] >= contract["target_chunks"],
            source_updates=learner.updates, motor_calls=motor.calls,
            feature_calls=motor.feature_encoder.calls, feature_seconds=motor.feature_encoder.elapsed_seconds,
            elapsed_seconds=time.monotonic() - started, collection_cost_count_once=True,
            deployment_shots="one support plus query is a separate evaluation protocol")
        final["smoke_qualified"] = bool(final["target_chunks_reached"] and learner.updates > 0
            and any(publication["status"] == "paired_ad_dpt" for publication in progress["publications"]))
        write_json(args.output / "index.json", final)
        return 0 if final["smoke_qualified"] else 3
    except Exception as error:
        write_json(args.output / "failure.json", dict(stage=stage, error=repr(error),
            traceback=traceback.format_exc(), recovery="source-recovery.pkl is the last complete physical episode",
            elapsed_seconds=time.monotonic() - started))
        raise
    finally:
        sim.close()


if __name__ == "__main__":
    raise SystemExit(main())
