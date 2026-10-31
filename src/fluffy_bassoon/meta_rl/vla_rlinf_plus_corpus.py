"""Train-only Plus source histories for the approved clean/augmented comparison.

Six fresh SAC trials, four complete attempts each, shared frozen RLinf-130
features. Final actors label DPT queries as candidates, not assumed experts.
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

from .vla_plus_worker import CASES, SOURCE_TRAINING_CASES


CASE_ORDER = (2387, 1438)
NOISE_CONTRACT = "32D clipped[-1,1] official DSRL coordinate repeated over native H10"


def source_plan(seed):
    return [dict(trial_id=f"plus-source-{case}-state-{state}", split="train",
        case_index=case, initial_state=state, scene_seed=seed + state * 1000)
        for case in CASE_ORDER for state in (0, 1, 2)]


def validate_source_case(metadata, task, expected_index):
    case = metadata.get("plus_case", {})
    if (metadata.get("benchmark_kind") != "plus" or metadata.get("source_training") is not True
            or metadata.get("development") is not False or metadata.get("role") != "source_training"
            or case.get("role") != "source_training" or case.get("runtime_index") != expected_index
            or expected_index not in SOURCE_TRAINING_CASES or expected_index in CASES
            or metadata.get("task_id") != expected_index or metadata.get("horizon") != 280
            or metadata.get("settle_steps") != 10 or metadata.get("initial_states") != 50
            or metadata.get("fresh_environment_on_reset") is not True
            or metadata.get("origin_base", {}).get("train_task_key") != task["task_key"]
            or case.get("attachment_train_lineage", {}).get("goal_binding_sha256") != task["goal_binding_sha256"]
            or task["split"] != "train" or not task["teacher_collection_allowed"]
            or not {0, 1, 2}.issubset(task["source_state_indices"])):
        raise ValueError("Only the declared Plus TRAIN cases and original training-goal lineage may supply this corpus")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "checkpoint", "rlinf-root", "openpi-root", "dsrl-root", "frozen-split", "source-cases"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--ports", type=int, nargs=2, default=[8765, 8766])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        raise ValueError("Corpus computation and tests require allocated Linux")
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get("FLUFFY_GPU_EXECUTION", "baige"), os.environ)
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()):
        raise ValueError("Require exact clean reviewed Plus source")
    from .vla_cnd_corpus import load_frozen_split
    from .vla_history import save_dataset, validate_trial
    from .vla_libero import OfficialSAC, SimulatorClient, append_json, run_episode, sha256, write_json
    from .vla_noise_headroom import RecordingSimulator, trajectory_digest
    from .vla_retry_teacher import append_episode_to_replay
    from .vla_rlinf_libero import FEATURE_KIND, MODEL_SHA256, FrozenRLinfPi05Libero
    from .vla_rlinf_retry_corpus import reset_consistency, trial_arrays
    split, tasks = load_frozen_split(args.frozen_split)
    task = tasks["libero_90:47"]
    declaration = json.loads(args.source_cases.read_text())
    if (declaration.get("format") != "rlinf130_plus_source_cases_v3"
            or declaration.get("case_indices") != list(CASE_ORDER)
            or declaration.get("excluded_development_indices") != sorted(CASES)
            or declaration.get("source_states") != [0, 1, 2] or declaration.get("attempts") != 4
            or declaration.get("maximum_primitive_steps") != 6720
            or declaration.get("frozen_original_split_sha256") != split["metadata_sha256"]):
        raise ValueError("Source case declaration differs from the approved train-only plan")
    plan = source_plan(args.seed)
    if len(plan) * 4 * 280 != 6720:
        raise ValueError("Unexpected complete-trial source budget")
    args.output.mkdir(parents=True, exist_ok=False)
    import fcntl
    lock = (args.output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sims, published = {}, []
    started, stage, total_steps, total_settling = time.monotonic(), "connecting", 0, 0
    try:
        for case, port in zip(CASE_ORDER, args.ports, strict=True):
            sims[case] = SimulatorClient(port)
            validate_source_case(sims[case].metadata, task, case)
        args.model_config, args.feature_interface = "rlinf_pi05_libero130", "vla-prefix"
        stage = "motor_qualification"
        motor = FrozenRLinfPi05Libero(args)
        for case, sim in sims.items():
            obs = sim.reset(0, args.seed + 100000)["observation"]
            total_settling += int(sim.metadata["settle_steps"])
            noise = np.zeros((10, 32), np.float32)
            first = motor.infer(obs, sim.metadata["instruction"], noise)
            second = motor.infer(obs, sim.metadata["instruction"], noise.copy())
            delta = float(np.max(np.abs(first - second)))
            write_json(args.output / f"motor-parity-{case}.json", dict(maximum_delta=delta, passed=delta <= 1e-5))
            if not np.isfinite(delta) or delta > 1e-5:
                raise RuntimeError("RLinf motor changed under identical noise")
            write_json(args.output / f"feature-parity-{case}.json",
                motor.feature_encoder.qualify(obs, sim.metadata["instruction"]))
        if motor.horizon != 10 or motor.feature_encoder.contract["kind"] != FEATURE_KIND:
            raise ValueError("Require the exact common RLinf130 motor/feature interface")
        contract = dict(format="rlinf130_plus_source_corpus_v1", source_commit=args.source_commit,
            dsrl_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.dsrl_root, text=True).strip(),
            benchmark_kind="plus", role="source_training", task=task,
            frozen_split_sha256=split["metadata_sha256"], source_cases_sha256=sha256(args.source_cases),
            motor=motor.metadata, feature_contract=motor.feature_encoder.contract,
            simulators={str(case): {k: v for k, v in sim.metadata.items() if k != "pid"} for case, sim in sims.items()},
            noise=NOISE_CONTRACT, scene_plan=plan, attempts=4, max_primitive_steps=6720,
            chunk=10, horizon=280, batch_size=32, utd=20, warmup_chunks=0,
            prior_training_primitive_steps=0, validation="none; clean validation remains shared by both data arms",
            labels="AD actual executed noise; DPT final candidate query action; no assumed competence or monotonicity")
        write_json(args.output / "contract.json", contract)
        stage = "fresh_source_initialization"
        learner = OfficialSAC(args.dsrl_root, args.seed)
        sampler = np.random.default_rng(args.seed)
        prior_path = args.output / "shared-fresh-source-recovery.pkl"
        learner.save(prior_path, sampler, contract, episodes=0, primitive_steps=0)
        prior_hash = sha256(prior_path)
        del learner
        index = dict(format="rlinf130_plus_source_corpus_index_v1", completed=False,
            benchmark_kind="plus", role="source_training", task_key=task["task_key"], global_task_split="train",
            frozen_split_sha256=split["metadata_sha256"], source_cases_sha256=sha256(args.source_cases),
            model_sha256=MODEL_SHA256, motor_receipt_sha256=motor.metadata["receipt_sha256"],
            feature_contract=motor.feature_encoder.contract, noise_contract=NOISE_CONTRACT,
            objectives=["ad", "dpt"], trials=published, actual_primitive_steps=0,
            actual_settling_steps=total_settling, excluded_development_indices=sorted(CASES),
            validation="none; common clean-only validation is not modified",
            candidate_teacher_competence="final actor candidate; no independent expert competence assumed")
        write_json(args.output / "index.json", index)
        for ordinal, scene in enumerate(plan):
            stage = f"trial_{ordinal}_collection"
            case = scene["case_index"]
            sim, case_metadata = sims[case], sims[case].metadata["plus_case"]
            recorder = RecordingSimulator(sim)
            directory = args.output / f"trial-{ordinal:03d}"
            directory.mkdir()
            branch_seed = int(np.random.SeedSequence([args.seed, ordinal, 1853]).generate_state(1)[0])
            sampler = np.random.default_rng(branch_seed)
            learner = OfficialSAC(args.dsrl_root, branch_seed)
            learner.restore(prior_path, sampler, contract)
            sampler = np.random.default_rng(branch_seed)
            branch_contract = dict(parent=contract, scene=scene, plus_case=case_metadata,
                branch_seed=branch_seed, shared_prior_sha256=prior_hash)
            write_json(directory / "contract.json", branch_contract)
            all_rows, starts, steps, receipts, trial_steps = [], [], [], [], 0
            behavior_path = prior_path
            for attempt in range(4):
                updates_before = learner.updates
                behavior_hash = sha256(behavior_path)
                rows, result = run_episode(recorder, motor, learner.action,
                    initial_state=scene["initial_state"], scene_seed=scene["scene_seed"],
                    chunk=10, bounded=True, record_features=True)
                if (not rows or not rows[-1]["done"] or result["primitive_steps"] > 280
                        or len(recorder.observations) != len(rows) + 1
                        or any(row["public_features"].shape != (4104,) for row in rows)):
                    raise ValueError("Require a complete actual Plus episode and common RLinf features")
                reset_matches = reset_consistency(receipts[0] if receipts else None, result)
                trial_steps += int(result["primitive_steps"])
                total_steps += int(result["primitive_steps"])
                total_settling += int(sim.metadata["settle_steps"])
                if total_steps > 6720:
                    raise RuntimeError("Plus source corpus exceeded its approved interaction budget")
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
                        summary=result, scene=scene, plus_case=case_metadata, attempt=attempt + 1,
                        behavior_checkpoint_sha256=behavior_hash, source_updates_before=updates_before,
                        source_updates_after=learner.updates, source_checkpoint_sha256=sha256(recovery)), stream, protocol=5)
                receipt = dict(result, attempt=attempt + 1, source_updates_before=updates_before,
                    source_updates_after=learner.updates, trace_sha256=sha256(trace),
                    trace=str(trace.relative_to(args.output)), metrics=metrics,
                    behavior_checkpoint_sha256=behavior_hash,
                    trajectory_sha256=trajectory_digest(rows, recorder.observations),
                    initial_observation_matches_first=reset_matches)
                receipts.append(receipt)
                all_rows.extend(rows)
                starts.extend([True] + [False] * (len(rows) - 1))
                steps.extend([updates_before] * len(rows))
                behavior_path = recovery
                append_json(args.output / "episodes.jsonl", dict(receipt,
                    trial_id=scene["trial_id"], case_index=case, global_task_split="train", local_scene_split="train"))
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
            datasets = {}
            for objective, arrays in trial_arrays(all_rows, starts, steps, candidate).items():
                validate_trial(arrays, objective=objective)
                path = directory / f"{objective}.npz"
                manifest = save_dataset(path, [arrays], objective=objective, metadata=dict(
                    trial_ids=[scene["trial_id"]], model_sha256=MODEL_SHA256,
                    feature_contract=motor.feature_encoder.contract, motor_sha256=motor.metadata["receipt_sha256"],
                    noise_contract=NOISE_CONTRACT, source_contract=branch_contract,
                    frozen_split_sha256=split["metadata_sha256"], task_key=task["task_key"],
                    benchmark_kind="plus", source_training=True, plus_case=case_metadata, case_index=case,
                    global_task_split="train", local_scene_split="train", total_source_primitive_steps=trial_steps,
                    source_updates=learner.updates, candidate_teacher_sha256=sha256(recovery),
                    teacher_status="final candidate actor; independent competence not assumed"))
                datasets[objective] = dict(path=str(path.relative_to(args.output)), sha256=manifest["sha256"])
            published.append(dict(trial_id=scene["trial_id"], scene=scene, plus_case=case_metadata,
                case_index=case, primitive_steps=trial_steps, episodes=receipts, datasets=datasets,
                source_checkpoint=str(recovery.relative_to(args.output)), source_checkpoint_sha256=sha256(recovery)))
            index.update(trials=published, actual_primitive_steps=total_steps, actual_settling_steps=total_settling,
                pending_trial=None, pending_trial_completed_episodes=0)
            write_json(args.output / "index.json", index)
            print(json.dumps(dict(published_complete_trial=scene["trial_id"], datasets=datasets)), flush=True)
            del learner
        if sha256(prior_path) != prior_hash:
            raise RuntimeError("Shared fresh source initialization changed")
        index.update(completed=True, elapsed_seconds=time.monotonic() - started, motor_calls=motor.calls,
            feature_calls=motor.feature_encoder.calls, feature_seconds=motor.feature_encoder.seconds)
        write_json(args.output / "index.json", index)
        write_json(args.output / "summary.json", index)
        return 0
    except Exception as error:
        write_json(args.output / "failure.json", dict(stage=stage, error=repr(error), traceback=traceback.format_exc(),
            published_trials=len(published), actual_primitive_steps=total_steps, actual_settling_steps=total_settling,
            elapsed_seconds=time.monotonic() - started))
        raise
    finally:
        for sim in sims.values():
            try:
                sim.close()
            except (EOFError, OSError):
                pass  # A failed worker must not overwrite the saved source failure.


if __name__ == "__main__":
    raise SystemExit(main())
