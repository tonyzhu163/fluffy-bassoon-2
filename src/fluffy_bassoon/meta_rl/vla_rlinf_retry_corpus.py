"""Publish genuine fresh-DSRL learning trials with fixed RLinf-130 features.

One complete four-episode trial is exported immediately as paired AD/DPT data.
The final source actor supplies candidate query labels, not an assumed expert.
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

from .vla_rlinf_libero import FEATURE_KIND, MODEL_SHA256


def validate_source_task(task, metadata):
    states = [0, 1, 2, 3, 36, 37]
    if (task["split"] != "train" or not task["teacher_collection_allowed"]
            or not set(states).issubset(task["source_state_indices"])
            or metadata["suite"] != task["suite"] or metadata["task_id"] != task["task_id"]
            or metadata["bddl_sha256"] != task["bddl_sha256"]
            or Path(metadata["bddl_file"]).stem != task["semantic_task"] or metadata["horizon"] != 400
            or metadata.get("fresh_environment_on_reset") is not True):
        raise ValueError("Require an original-LIBERO training task, source scenes and exact frozen BDDL")


def scenes(seed):
    return [dict(trial_id=f"{split}-state-{state}", split=split, initial_state=state,
                 scene_seed=seed + state * 1000) for split, bank in
            (("train", [0, 1, 2, 3]), ("validation", [36, 37])) for state in bank]


def trial_arrays(rows, starts, teacher_steps, candidate_actions):
    base = dict(observations=np.stack([row["public_features"] for row in rows]).astype(np.float32),
        executed_noise=np.stack([row["executed_noise"] for row in rows]).astype(np.float32),
        rewards=np.asarray([row["reward"] for row in rows], np.float32),
        dones=np.asarray([row["done"] for row in rows], bool),
        episode_starts=np.asarray(starts, bool), teacher_steps=np.asarray(teacher_steps, np.int64))
    return dict(ad=dict(**base, target_noise=base["executed_noise"].copy()),
                dpt=dict(**base, target_noise=np.asarray(candidate_actions, np.float32)))


def reset_consistency(first, current):
    """Keep the declared nominal scene fixed; measure rendered replay variation."""
    nominal = current.get("initial_state_vector_sha256")
    if not isinstance(nominal, str) or not nominal:
        raise ValueError("Corpus reset omitted its actual nominal initial-state identity")
    if first and nominal != first["initial_state_vector_sha256"]:
        raise ValueError("The nominal initial-state vector changed within one trial")
    return not first or current["initial_observation_sha256"] == first["initial_observation_sha256"]


def assemble_local_pilot(index, root):
    """A single task's local scene splits; never merge a different backbone."""
    if (index.get("format") != "rlinf130_retry_corpus_index_v1" or index.get("model_sha256") != MODEL_SHA256
            or index.get("feature_contract", {}).get("kind") != FEATURE_KIND):
        raise ValueError("Only the explicit shared RLinf-130 feature contract can enter this pilot")
    from .vla_history import load_dataset, save_dataset
    root = Path(root)
    gathered = {objective: {split: [] for split in ("train", "validation")} for objective in ("ad", "dpt")}
    ids = {split: [] for split in ("train", "validation")}
    for trial in index["trials"]:
        split = trial["scene"]["split"]
        ids[split].append(trial["trial_id"])
        for objective in gathered:
            receipt = trial["datasets"][objective]
            data, manifest = load_dataset(root / receipt["path"], expected_sha256=receipt["sha256"])
            metadata = manifest["metadata"]
            if (manifest["objective"] != objective or metadata["feature_contract"] != index["feature_contract"]
                    or metadata["model_sha256"] != MODEL_SHA256 or metadata["task_key"] != index["task_key"]
                    or metadata["global_task_split"] != "train" or metadata["local_scene_split"] != split
                    or manifest["observation_dim"] != 4104 or manifest["noise_dim"] != 32):
                raise ValueError("A trial changed the common RLinf motor/features or task/scene assignment")
            gathered[objective][split].extend(data)
    if not all(ids.values()) or len(set(ids["train"] + ids["validation"])) != sum(map(len, ids.values())):
        raise ValueError("Require distinct complete trials in both local scene banks")
    paths = {objective: {} for objective in gathered}
    for objective in gathered:
        for split in gathered[objective]:
            path = root / f"pilot-{objective}-{split}.npz"
            receipt = save_dataset(path, gathered[objective][split], objective=objective, metadata=dict(
                feature_contract=index["feature_contract"], model_sha256=MODEL_SHA256,
                motor_sha256=index["motor_receipt_sha256"], task_key=index["task_key"],
                noise_contract=index["noise_contract"],
                frozen_split_sha256=index["frozen_split_sha256"], global_task_split="train",
                local_scene_split=split, trial_ids=ids[split], total_source_primitive_steps=index["actual_primitive_steps"],
                actual_settling_steps=index["actual_settling_steps"],
                scope="local scenes of one original-LIBERO source training task; not semantic task OOD"))
            paths[objective][split] = dict(path=path.name, sha256=receipt["sha256"])
    return paths


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "checkpoint", "rlinf-root", "openpi-root", "dsrl-root", "frozen-split"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--suite", default="libero_90")
    p.add_argument("--task-id", type=int, default=47)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if sys.platform != "linux":
        raise ValueError("Corpus model computation requires allocated Linux")
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get("FLUFFY_GPU_EXECUTION", "baige"), os.environ)
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()):
        raise ValueError("Require the exact clean reviewed corpus source")
    from .vla_cnd_corpus import load_frozen_split
    from .vla_history import save_dataset, validate_trial
    from .vla_libero import OfficialSAC, SimulatorClient, append_json, run_episode, sha256, write_json
    from .vla_noise_headroom import RecordingSimulator
    from .vla_retry_teacher import append_episode_to_replay
    from .vla_rlinf_libero import FrozenRLinfPi05Libero
    split, tasks = load_frozen_split(args.frozen_split)
    matching = [row for row in tasks.values() if row["suite"] == args.suite and row["task_id"] == args.task_id]
    if len(matching) != 1 or matching[0]["split"] != "train":
        raise ValueError("Corpus task is outside the frozen source-training bank")
    task = matching[0]
    plan = scenes(args.seed)
    if len(plan) * 4 * 400 != 9600:
        raise ValueError("Unexpected complete-trial interaction budget")
    args.output.mkdir(parents=True, exist_ok=False)
    import fcntl
    lock = (args.output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started, stage, total_steps, total_settling, published = time.monotonic(), "connecting", 0, 0, []
    sim = SimulatorClient(args.port)
    try:
        validate_source_task(task, sim.metadata)
        args.model_config, args.feature_interface = "rlinf_pi05_libero130", "vla-prefix"
        stage = "motor_qualification"
        motor = FrozenRLinfPi05Libero(args)
        obs = sim.reset(0, args.seed + 100000)["observation"]
        total_settling += int(sim.metadata["settle_steps"])
        zeros = np.zeros((10, 32), np.float32)
        first, second = motor.infer(obs, sim.metadata["instruction"], zeros), motor.infer(obs, sim.metadata["instruction"], zeros.copy())
        delta = float(np.max(np.abs(first - second)))
        write_json(args.output / "motor-noise-parity.json", dict(maximum_delta=delta, passed=delta <= 1e-5))
        if not np.isfinite(delta) or delta > 1e-5:
            raise RuntimeError("RLinf motor changed under identical initial noise")
        write_json(args.output / "feature-parity.json", motor.feature_encoder.qualify(obs, sim.metadata["instruction"]))
        if motor.feature_encoder.contract["kind"] != FEATURE_KIND:
            raise ValueError("A different feature backend cannot enter this corpus")
        contract = dict(format="rlinf130_retry_corpus_v1", source_commit=args.source_commit,
            dsrl_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.dsrl_root, text=True).strip(),
            task=task, frozen_split_sha256=split["metadata_sha256"], motor=motor.metadata,
            simulator={key: value for key, value in sim.metadata.items() if key != "pid"},
            feature_contract=motor.feature_encoder.contract,
            noise="32D clipped[-1,1] official DSRL coordinate repeated over native H10",
            attempts=4, scene_plan=plan, max_primitive_steps=9600, chunk=10, horizon=400,
            batch_size=32, utd=20, warmup_chunks=0, prior_training_primitive_steps=0,
            source="fresh full common SAC initialization restored independently per scene; update between attempts",
            labels="AD=actual executed noise; DPT=final candidate actor query action; no assumed expert or monotonicity",
            split="all trials belong to the source training task; local scene labels retained separately")
        write_json(args.output / "contract.json", contract)
        stage = "fresh_source_initialization"
        prior = OfficialSAC(args.dsrl_root, args.seed)
        sampler = np.random.default_rng(args.seed)
        prior_path = args.output / "shared-fresh-source-recovery.pkl"
        prior.save(prior_path, sampler, contract, episodes=0, primitive_steps=0)
        prior_hash = sha256(prior_path)
        del prior
        recorder = RecordingSimulator(sim)
        index = dict(format="rlinf130_retry_corpus_index_v1", completed=False,
            task_key=task["task_key"], global_task_split="train", frozen_split_sha256=split["metadata_sha256"],
            model_sha256=MODEL_SHA256, motor_receipt_sha256=motor.metadata["receipt_sha256"],
            feature_contract=motor.feature_encoder.contract, objectives=["ad", "dpt"], trials=published,
            noise_contract=contract["noise"],
            actual_primitive_steps=0, actual_settling_steps=total_settling,
            candidate_teacher_competence="not independently measured; training outcomes are not final-actor evaluation")
        write_json(args.output / "index.json", index)
        for ordinal, scene in enumerate(plan):
            stage = f"trial_{ordinal}_collection"
            directory = args.output / f"trial-{ordinal:03d}"
            directory.mkdir()
            branch_seed = int(np.random.SeedSequence([args.seed, ordinal, 1853]).generate_state(1)[0])
            sampler = np.random.default_rng(branch_seed)
            learner = OfficialSAC(args.dsrl_root, branch_seed)
            learner.restore(prior_path, sampler, contract)
            sampler = np.random.default_rng(branch_seed)
            branch_contract = dict(parent=contract, scene=scene, branch_seed=branch_seed,
                                   shared_prior_sha256=prior_hash)
            write_json(directory / "contract.json", branch_contract)
            all_rows, starts, steps, receipts, trial_steps = [], [], [], [], 0
            for attempt in range(4):
                updates_before = learner.updates
                rows, result = run_episode(recorder, motor, learner.action, initial_state=scene["initial_state"],
                    scene_seed=scene["scene_seed"], chunk=10, bounded=True, record_features=True)
                if (not rows or not rows[-1]["done"] or result["primitive_steps"] > 400
                        or len(recorder.observations) != len(rows) + 1
                        or any(row["public_features"].shape != (4104,) for row in rows)):
                    raise ValueError("Require a complete episode with real shared RLinf features")
                reset_matches = reset_consistency(receipts[0] if receipts else None, result)
                trial_steps += int(result["primitive_steps"])
                total_steps += int(result["primitive_steps"])
                total_settling += int(sim.metadata["settle_steps"])
                if total_steps > 9600:
                    raise RuntimeError("Source corpus exceeded its declared interaction budget")
                append_episode_to_replay(learner, rows)
                metrics = {}
                for _ in range(len(rows) * 20):
                    metrics = learner.update(32, sampler)
                recovery = directory / f"source-recovery-{attempt + 1:03d}.pkl"
                learner.save(recovery, sampler, branch_contract, episodes=attempt + 1,
                    primitive_steps=trial_steps, shared_prior_training_steps=0)
                trace = directory / f"episode-{attempt + 1:03d}.pkl.gz"
                with gzip.open(trace, "xb", compresslevel=1) as stream:
                    pickle.dump(dict(rows=rows, public_observations=recorder.observations,
                        summary=result, attempt=attempt + 1, source_updates_before=updates_before,
                        source_updates_after=learner.updates, source_checkpoint_sha256=sha256(recovery)), stream, protocol=5)
                receipt = dict(result, attempt=attempt + 1, source_updates_before=updates_before,
                    source_updates_after=learner.updates, trace_sha256=sha256(trace), metrics=metrics,
                    initial_observation_matches_first=reset_matches)
                receipts.append(receipt)
                all_rows.extend(rows)
                starts.extend([True] + [False] * (len(rows) - 1))
                steps.extend([updates_before] * len(rows))
                append_json(args.output / "episodes.jsonl", dict(receipt, trial_id=scene["trial_id"],
                    local_scene_split=scene["split"], global_task_split="train"))
                progress = dict(completed_trials=len(published), completed_episodes=ordinal * 4 + attempt + 1,
                    actual_primitive_steps=total_steps, actual_settling_steps=total_settling,
                    current_trial=scene, source_updates=learner.updates,
                    latest_checkpoint=str(recovery.relative_to(args.output)))
                write_json(directory / "progress.json", dict(progress, episodes=receipts))
                write_json(args.output / "progress.json", progress)
                index.update(actual_primitive_steps=total_steps, actual_settling_steps=total_settling,
                    pending_trial=scene["trial_id"], pending_trial_completed_episodes=attempt + 1)
                write_json(args.output / "index.json", index)
                print(json.dumps(progress), flush=True)
            stage = f"trial_{ordinal}_publication"
            candidate = np.stack([np.asarray(learner.agent.eval_actions(
                {key: value[None] for key, value in row["sac_observation"].items()})).reshape(32)
                for row in all_rows]).astype(np.float32)
            paired = trial_arrays(all_rows, starts, steps, candidate)
            datasets = {}
            for objective, arrays in paired.items():
                validate_trial(arrays, objective=objective)
                path = directory / f"{objective}.npz"
                manifest = save_dataset(path, [arrays], objective=objective, metadata=dict(
                    trial_ids=[scene["trial_id"]], feature_contract=motor.feature_encoder.contract,
                    motor_sha256=motor.metadata["receipt_sha256"], model_sha256=MODEL_SHA256,
                    noise_contract=contract["noise"], source_contract=branch_contract,
                    frozen_split_sha256=split["metadata_sha256"], task_key=task["task_key"],
                    global_task_split="train", local_scene_split=scene["split"],
                    total_source_primitive_steps=trial_steps, source_updates=learner.updates,
                    candidate_teacher_sha256=sha256(recovery),
                    teacher_status="candidate final actor; independently measured competence pending"))
                datasets[objective] = dict(path=str(path.relative_to(args.output)), sha256=manifest["sha256"])
            published.append(dict(trial_id=scene["trial_id"], scene=scene, primitive_steps=trial_steps,
                episodes=receipts, datasets=datasets, source_checkpoint=str(recovery.relative_to(args.output)),
                source_checkpoint_sha256=sha256(recovery)))
            index.update(actual_primitive_steps=total_steps, actual_settling_steps=total_settling, trials=published,
                         pending_trial=None, pending_trial_completed_episodes=0)
            write_json(args.output / "index.json", index)
            print(json.dumps(dict(published_complete_trial=scene["trial_id"], datasets=datasets)), flush=True)
        if sha256(prior_path) != prior_hash:
            raise RuntimeError("The common fresh source initialization changed")
        index.update(pilot_datasets=assemble_local_pilot(index, args.output))
        index.update(completed=True, elapsed_seconds=time.monotonic() - started,
            motor_calls=motor.calls, feature_calls=motor.feature_encoder.calls,
            feature_seconds=motor.feature_encoder.seconds)
        write_json(args.output / "index.json", index)
        write_json(args.output / "summary.json", index)
        return 0
    except Exception as error:
        write_json(args.output / "failure.json", dict(stage=stage, error=repr(error), traceback=traceback.format_exc(),
            published_trials=len(published), actual_primitive_steps=total_steps,
            actual_settling_steps=total_settling, elapsed_seconds=time.monotonic() - started))
        raise
    finally:
        sim.close()


if __name__ == "__main__":
    raise SystemExit(main())
