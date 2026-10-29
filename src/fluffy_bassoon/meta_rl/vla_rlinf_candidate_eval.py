"""Closed-loop development evaluation of trained RLinf-130 steering candidates.

Full and episode-reset inference use the same trained weights. Static policies
are trained separately on the same source data. These declared development
cases do not constitute a blind benchmark or new-goal generalization test.
"""
from __future__ import annotations

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

from .vla_libero import OfficialSAC, SimulatorClient, append_json, run_episode, sha256, write_json
from .vla_shifted_poc import replay_rows, validate_development_case


HISTORY_METHODS = {
    "ad_full": ("ad_full", "full"),
    "ad_episode": ("ad_full", "episode"),
    "ad_static": ("ad_static", "static"),
    "dpt_full": ("dpt_full", "full"),
    "dpt_episode": ("dpt_full", "episode"),
    "dpt_static": ("dpt_static", "static"),
}
SAC_METHODS = ("source_sac_frozen", "source_sac_online")
NOISE_CONTRACT = "32D clipped[-1,1] official DSRL coordinate repeated over native H10"


def checked_entry(root, entry):
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256(path) != entry["sha256"]:
        raise ValueError("Candidate registry path or artifact hash differs")
    return path


def validate_registry(registry, motor, *, students=True):
    if (registry.get("format") != "rlinf130_trained_arm_v1" or registry.get("completed") is not True
            or registry.get("arm") not in ("clean", "augmented")
            or registry.get("model_sha256") != motor.metadata["model_sha256"]
            or registry.get("feature_contract") != motor.feature_encoder.contract
            or registry.get("noise_contract") != NOISE_CONTRACT or not registry.get("collection_cost")):
        raise ValueError("Require a completed, matching RLinf-130 source-data arm")
    if students:
        for key in ("ad_full", "ad_static", "dpt_full", "dpt_static"):
            entry = registry.get("models", {}).get(key, {})
            if (entry.get("actual_updates") != 500 or entry.get("requested_updates") != 500
                    or entry.get("primary", {}).get("update") != 500
                    or entry.get("primary", {}).get("selection") != "final_update"
                    or entry.get("objective") != key.split("_")[0]
                    or entry.get("mode") != key.split("_")[1]):
                raise ValueError("Require all four final-update500 full/static candidates")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "checkpoint", "rlinf-root", "openpi-root", "dsrl-root", "arm-registry"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--sac-registry", type=Path)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--benchmark", choices=("plus", "pro"), required=True)
    p.add_argument("--ports", type=int, nargs="+", required=True)
    p.add_argument("--methods", nargs="+", choices=("native", *HISTORY_METHODS, *SAC_METHODS),
                   default=["native", *HISTORY_METHODS])
    p.add_argument("--attempts", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=280)
    p.add_argument("--max-primitive-steps", type=int, default=20160)
    p.add_argument("--sample-history", action="store_true", help="Default primary uses deterministic means")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.platform != "linux":
        raise ValueError("Robot and model computation requires allocated Linux")
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get("FLUFFY_GPU_EXECUTION", "baige"), os.environ)
    if (len(set(args.ports)) != len(args.ports) or len(set(args.methods)) != len(args.methods)
            or not args.ports or not args.methods or args.attempts != 4 or args.horizon != 280
            or not 0 <= args.seed < 2**32
            or len(args.ports) * len(args.methods) * args.attempts * args.horizon > args.max_primitive_steps
            or (set(args.methods) & set(SAC_METHODS) and not args.sac_registry)):
        raise ValueError("Invalid declared K4 development evaluation or interaction budget")
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()):
        raise ValueError("Require exact clean reviewed evaluation source")
    from .vla_cnd_corpus import load_frozen_split
    from .vla_history import FrozenHistoryController, load_policy
    from .vla_rlinf_libero import FrozenRLinfPi05Libero
    _, tasks = load_frozen_split(root / "outputs/libero-pi05-corpus-20261029-r1/split.json")
    args.output.mkdir(parents=True, exist_ok=False)
    import fcntl
    lock = (args.output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    clients, episodes = [], []
    stage, primitive, settling, started = "connecting", 0, 0, time.monotonic()
    try:
        for port in args.ports:
            sim = SimulatorClient(port)
            clients.append(sim)
            validate_development_case(sim.metadata, args.benchmark, tasks, args.horizon)
        args.model_config, args.feature_interface = "rlinf_pi05_libero130", "vla-prefix"
        stage = "motor_and_candidate_loading"
        motor = FrozenRLinfPi05Libero(args)
        registry = json.loads(args.arm_registry.read_text())
        validate_registry(registry, motor)
        models, artifacts = {}, {}
        for key in sorted({HISTORY_METHODS[m][0] for m in args.methods if m in HISTORY_METHODS}):
            entry = registry["models"][key]
            path = checked_entry(args.arm_registry.parent, entry["primary"])
            policy, manifest = load_policy(path, expected_sha256=entry["primary"]["sha256"], device="cuda:0")
            saved = manifest.get("contract", {})
            if (manifest.get("update") != 500 or saved.get("mode") != entry["mode"]
                    or saved.get("objective") != entry["objective"]
                    or saved.get("train_sha256") != entry["train_sha256"]
                    or saved.get("validation_sha256") != entry["validation_sha256"]
                    or saved.get("train_metadata", {}).get("feature_contract") != motor.feature_encoder.contract
                    or policy.config.observation_dim != 4104 or policy.config.noise_dim != 32):
                raise ValueError("Exported candidate provenance differs from the source-data arm")
            models[key], artifacts[key] = policy, dict(path=str(path), sha256=sha256(path), manifest=manifest)
        sac_registry, sac_checkpoint, source_contract = None, None, None
        if set(args.methods) & set(SAC_METHODS):
            sac_registry = json.loads(args.sac_registry.read_text())
            validate_registry(sac_registry, motor, students=False)
            if (sac_registry["arm"] != registry["arm"]
                    or sac_registry["collection_cost"] != registry["collection_cost"]
                    or sac_registry["source_sac"]["updates"] != 5000):
                raise ValueError("Source SAC must use the same source-data arm and complete5000updates")
            sac_checkpoint = checked_entry(args.sac_registry.parent, sac_registry["source_sac"]["checkpoint"])
            source_contract_path = checked_entry(args.sac_registry.parent, sac_registry["source_sac"]["contract"])
            source_contract = json.loads(source_contract_path.read_text())
        contract = dict(format="rlinf130_candidate_development_eval_v1", source_commit=args.source_commit,
            benchmark=args.benchmark, arm=registry["arm"], arm_registry_sha256=sha256(args.arm_registry),
            source_sac_registry_sha256=None if args.sac_registry is None else sha256(args.sac_registry),
            collection_cost=registry["collection_cost"], candidates=artifacts, motor=motor.metadata,
            feature_contract=motor.feature_encoder.contract, noise_contract=NOISE_CONTRACT,
            methods=args.methods, seed=args.seed, attempts=4, horizon=280, chunk=10,
            max_primitive_steps=args.max_primitive_steps,
            selection="final-update500; development results do not select blind-test models",
            history_action_selection="sampled Gaussian" if args.sample_history else "deterministic Gaussian mean",
            source_sac_action_selection="frozen: deterministic mean; online: conventional stochastic SAC",
            source_sac_migration="Independent per-case source checkpoint clone, retaining source replay/optimizer/RNG; new target contract records migration",
            history_deployment="Weights frozen; context reset per case; full retains across K4; episode clears at episode boundaries; trained static clears every query",
            scope="One-goal-family development smoke; no broad meta-learning or blind-test performance claim",
            cases=[{k: v for k, v in sim.metadata.items() if k != "pid"} for sim in clients])
        write_json(args.output / "contract.json", contract)
        for case_index, sim in enumerate(clients):
            scene_seed = args.seed + case_index * 1000
            obs = sim.reset(0, scene_seed)["observation"]
            settling += sim.metadata["settle_steps"]
            qualification = motor.feature_encoder.qualify(obs, sim.metadata["instruction"])
            zero = np.zeros((10, 32), np.float32)
            delta = float(np.max(np.abs(motor.infer(obs, sim.metadata["instruction"], zero)
                                      - motor.infer(obs, sim.metadata["instruction"], zero.copy()))))
            if not np.isfinite(delta) or delta > 1e-5:
                raise RuntimeError("Motor noise replay failed")
            write_json(args.output / f"qualification-case-{case_index:02d}.json",
                       dict(motor_max_delta=delta, features=qualification, passed=True))
            for method in args.methods:
                stage = f"case{case_index}_{method}"
                directory = args.output / f"case-{case_index:02d}" / method
                directory.mkdir(parents=True)
                rng, sampler = np.random.default_rng(scene_seed + 279011), np.random.default_rng(scene_seed)
                learner, controller = None, None
                if method in HISTORY_METHODS:
                    import torch
                    key, mode = HISTORY_METHODS[method]
                    generator = torch.Generator(device="cuda:0").manual_seed(scene_seed + 937)
                    controller = FrozenHistoryController(models[key], mode=mode, generator=generator)
                    choose = lambda obs: controller.act(motor.feature_encoder(obs, sim.metadata["instruction"]),
                                                       deterministic=not args.sample_history)
                elif method in SAC_METHODS:
                    learner = OfficialSAC(args.dsrl_root, args.seed)
                    learner.restore(sac_checkpoint, sampler, source_contract)
                    write_json(directory / "source-migration.json", dict(checkpoint_sha256=sha256(sac_checkpoint),
                        source_contract=source_contract, target_contract_sha256=sha256(args.output / "contract.json"),
                        source_updates=learner.updates, retained_source_replay_rows=len(learner.replay)))
                    choose = lambda obs: learner.action(obs, deterministic=method == "source_sac_frozen")
                else:
                    choose = lambda _: np.zeros(32, np.float32)
                local_steps = 0
                source_updates = 0 if learner is None else learner.updates
                for attempt in range(1, 5):
                    updates_before = 0 if learner is None else learner.updates
                    rows, result = run_episode(sim, motor, choose, initial_state=0, scene_seed=scene_seed,
                        chunk=10, native=method == "native", rng=rng, record_features=False,
                        feedback=None if controller is None else controller.observe)
                    primitive += result["primitive_steps"]
                    local_steps += result["primitive_steps"]
                    settling += sim.metadata["settle_steps"]
                    metrics = {}
                    if method == "source_sac_online":
                        learner.replay.extend(replay_rows(rows))
                        for _ in range(len(rows) * 20):
                            metrics = learner.update(32, sampler)
                    if learner is not None:
                        learner.save(directory / f"recovery-{attempt:03d}.pkl", sampler, contract,
                            case_index=case_index, method=method, episodes=attempt, primitive_steps=local_steps,
                            prior_source_updates=source_updates, rollout_noise_rng=rng.bit_generator.state)
                    if controller is not None:
                        from .warmstart import atomic_save
                        atomic_save(directory / f"context-{attempt:03d}.pt", dict(
                            policy_sha256=artifacts[HISTORY_METHODS[method][0]]["sha256"],
                            state=controller.state, previous_noise=controller.previous_noise,
                            previous_reward=controller.previous_reward, previous_done=controller.previous_done,
                            episode_start=controller.episode_start, pending=controller.pending,
                            generator=controller.generator.get_state(), rollout_noise_rng=rng.bit_generator.state,
                            numpy_rng=np.random.get_state(), python_rng=__import__("random").getstate(),
                            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                            contract=contract, case_index=case_index, method=method, episodes=attempt))
                    trace = directory / f"episode-{attempt:03d}.pkl.gz"
                    with gzip.open(trace, "xb", compresslevel=1) as stream:
                        pickle.dump(dict(rows=rows, summary=result), stream, protocol=5)
                    receipt = dict(result, method=method, benchmark=args.benchmark, arm=registry["arm"],
                        case_index=case_index, attempt=attempt, trial_seed=scene_seed,
                        cumulative_deployment_steps=local_steps, source_updates_before=updates_before,
                        source_updates_after=0 if learner is None else learner.updates,
                        deployment_optimizer_updates=0 if learner is None else learner.updates - source_updates,
                        trace_sha256=sha256(trace), metrics=metrics, completed=True)
                    episodes.append(receipt)
                    append_json(args.output / "episodes.jsonl", receipt)
                    write_json(args.output / "progress.json", dict(completed_episodes=len(episodes),
                        expected_episodes=len(clients) * len(args.methods) * 4,
                        primitive_steps=primitive, settling_steps=settling, last_episode=receipt))
                    print(json.dumps(receipt), flush=True)
                    if primitive > args.max_primitive_steps:
                        raise RuntimeError("Exceeded declared candidate evaluation interaction ceiling")
                del learner, controller
        write_json(args.output / "summary.json", dict(completed=True, episodes=len(episodes),
            primitive_steps=primitive, settling_steps=settling, elapsed_seconds=time.monotonic() - started,
            success_by_case_method={f"{i}/{m}": dict(episodes=4, successes=sum(
                bool(r["success"]) for r in episodes if r["case_index"] == i and r["method"] == m))
                for i in range(len(clients)) for m in args.methods}, scope=contract["scope"]))
        return 0
    except Exception as error:
        write_json(args.output / "failure.json", dict(stage=stage, error=repr(error),
            traceback=traceback.format_exc(), completed_episodes=len(episodes), primitive_steps=primitive))
        raise
    finally:
        for sim in clients:
            try:
                sim.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
