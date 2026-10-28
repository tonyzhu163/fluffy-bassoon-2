"""Bounded balanced-data, flow-interface and support→fresh-probe qualification.

This is a saved-artifact gate on the existing native interface, not a new PPO
training schedule. All simulation and fitting require an allocated Linux GPU.
"""

import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from fluffy_bassoon.evaluation_stages import sha256, write_json
from fluffy_bassoon.rl import maniskill_pi05 as ms
from fluffy_bassoon.rl.pi05_bridge_policy import StochasticPi05BridgePolicy
from fluffy_bassoon.rl.pi05_feature_bridge import Pi05FeatureBridge
from .bridge_distillation import (corrected_action_label, distillation_loss, export_history, load_history,
                                  update_distillation)
from .checkpoint import restore_checkpoint, rng_state, save_checkpoint
from .pi05_bridge_train import (admit_gpu, collect_live_trials, parser as live_parser,
                                qualify_replay)
from .warmstart import NATIVE, atomic_save, export_warmstart, validate_weights


PROTOCOL = "native_pi05_balanced_support_probe_gate_v1"


@torch.no_grad()
def sampled_action_distances(motor, observation, actions, record, theta_degrees):
    """Compare an already executed-path sample with paired native/label actions."""
    noise = record["noisy_actions"][0].to(actions.device)
    native = motor.sample_actions(observation, num_steps=4, noise=noise)
    target = corrected_action_label(motor, observation.state, native, theta_degrees)
    decoded_target = motor.decode_actions(target, observation.state)
    steps = decoded_target.shape[1]
    result = {}
    for name, candidate in (("bridge", actions), ("zero_bridge", native)):
        error = candidate[:, :steps, :7] - target[:, :steps, :7]
        decoded_error = motor.decode_actions(candidate, observation.state) - decoded_target
        result[name] = dict(model_action_mse=float(error.square().mean()),
            decoded_action_mse=float(decoded_error.square().mean()),
            decoded_xy_mse=float(decoded_error[..., :2].square().mean()))
    return dict(num_steps=4, executed_action_steps=steps, paired_initial_noise=True, distances=result)


def bridge_policy(core, *, width=64, noise_scale=.05, residual_scale=1., device="cuda"):
    config = dict(core=core, width=width, ticks=4 if core == "persistent_ctm" else 1,
                  residual_scale=residual_scale)
    policy = StochasticPi05BridgePolicy(Pi05FeatureBridge(**config),
        noise_scale=noise_scale, active_dims=7).to(device).eval()
    return policy, config


def native_contract(args, source, motor_hash, policy, bridge_config):
    return dict(protocol=PROTOCOL, source_commit=source, core=bridge_config["core"],
        bridge=bridge_config, gaussian_policy=policy.config(), task="PutOnPlateInScene25Main-v3",
        objects=[0, 1], train_theta_degrees=[-45., 0., 45.], train_overlays=list(range(12)),
        attempts=4, episode_steps=80, action_chunk=5, denoise_steps=4,
        feedback="unrotated 7D requested command; previous terminal success; boundary; fixed clocks",
        reward="success_once latched within attempt, emitted only on its final chunk",
        motor_sha256=args.expected_model_sha256, norm_stats_sha256=sha256(args.norm_stats),
        frozen_motor_fingerprint=motor_hash,
        rlinf_sources={name: sha256(args.rlinf_root / name) for name in (
            "rlinf/models/embodiment/openpi/pi0.py", "rlinf/models/embodiment/openpi/env_io.py",
            "rlinf/models/embodiment/openpi/modules/gemma.py", "rlinf/models/embodiment/openpi/checkpoint.py")})


def fit_candidate(policy, training, validation, *, output, contract, provenance,
                  updates, fit_seconds, learning_rate, seed, should_stop, resume=None):
    """Keep a complete resume snapshot even when the declared wall cap intervenes."""
    output.mkdir(parents=True, exist_ok=False)
    sampler = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=0.)
    contract = dict(contract, phase="native_dpt_flow_warmstart", seed=seed,
                    learning_rate=learning_rate, data=provenance)
    counters = dict(teacher_simulator_steps=provenance["simulator_steps"], optimizer_steps=0,
                    replayed_policy_chunks=0, validation_policy_chunks=0)
    iteration, started = 0, time.monotonic()
    if resume is not None:
        iteration, counters = restore_checkpoint(resume, policy=policy, optimizer=optimizer,
                                                environment_rng=sampler, contract=contract)
    write_json(output / "config.json", dict(contract=contract, target_updates=updates,
                                           max_elapsed_seconds=fit_seconds))
    def snapshot():
        return save_checkpoint(output / "latest.pt", policy=policy, optimizer=optimizer,
            environment_rng=sampler, iteration=iteration, counters=counters, contract=contract)
    @torch.no_grad()
    def evaluate():
        losses = [float(distillation_loss(policy, batch)) for batch in validation]
        counters["validation_policy_chunks"] += sum(batch["rewards"].numel() for batch in validation)
        return dict(validation_flow_matching_loss=float(np.mean(losses)),
                    behavioral_status="pending actual four-Euler live evaluation")
    digest = snapshot()
    with (output / "metrics.jsonl").open("x", buffering=1) as stream:
        result = evaluate()
        stream.write(json.dumps(dict(iteration=0, evaluation=result), allow_nan=False) + "\n")
        while iteration < updates and not should_stop() and time.monotonic() - started < fit_seconds:
            batch = training[int(sampler.integers(len(training)))]
            metrics = update_distillation(policy, optimizer, batch)
            iteration += 1
            counters["optimizer_steps"] += 1
            counters["replayed_policy_chunks"] += batch["rewards"].numel()
            row = dict(iteration=iteration, elapsed_seconds=time.monotonic() - started, **metrics)
            if iteration % 10 == 0:
                digest = snapshot()
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        result = evaluate()
        digest = snapshot()
    portable = export_warmstart(output / "warmstart.pt", family=NATIVE, policy=policy,
        contract=contract, iteration=iteration, counters=counters, data=provenance,
        evaluation=result) if iteration else None
    summary = dict(iteration=iteration, completed=iteration == updates, counters=counters,
        checkpoint_sha256=digest, warmstart_sha256=portable, evaluation=result,
        elapsed_seconds=time.monotonic() - started)
    write_json(output / "summary.json", summary)
    return summary


def load_direct_candidate(path, digest, policy, contract):
    """Read only a checksum-pinned completed-boundary snapshot from the other run."""
    if sha256(path) != digest:
        raise ValueError("Direct PPO snapshot hash differs")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    parent = saved.get("contract", {})
    keys = ("task", "objects", "train_theta_degrees", "train_overlays", "attempts",
            "episode_steps", "action_chunk", "denoise_steps", "feedback", "reward", "core",
            "bridge", "gaussian_policy", "motor_sha256", "norm_stats_sha256", "rlinf_sources")
    if (saved.get("boundary") != "between_trials" or saved.get("iteration", 0) < 1
            or any(parent.get(key) != contract[key] for key in keys)
            or sha256(path) != digest):
        raise ValueError("Direct PPO checkpoint does not match the evaluation interface")
    validate_weights(policy, saved.get("policy"))
    policy.load_state_dict(saved["policy"], strict=True)
    return dict(checkpoint_sha256=digest, iteration=saved["iteration"], source_commit=parent["source_commit"],
                training_counters=saved["counters"], evaluation_only=True)


def parser():
    p = live_parser()
    p.description = __doc__
    p.set_defaults(num_envs=6, attempts=4, updates=100, max_elapsed_seconds=4800.)
    p.add_argument("--fit-seconds", type=float, default=600.)
    p.add_argument("--direct-ctm-checkpoint", type=Path)
    p.add_argument("--direct-ctm-sha256")
    p.add_argument("--direct-xl-checkpoint", type=Path)
    p.add_argument("--direct-xl-sha256")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.platform != "linux" or args.num_envs != 6 or args.attempts != 4 or args.width != 64:
        raise ValueError("This bounded gate requires allocated Linux, six lanes, K4 and width64")
    if (args.resume or args.warmstart or args.collect_dpt or args.updates < 1
            or not math.isfinite(args.fit_seconds) or not 0 < args.fit_seconds <= 600
            or not math.isfinite(args.max_elapsed_seconds) or not 0 < args.max_elapsed_seconds <= 4800
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or args.threads < 1 or args.seed < 0 or args.seed >= 2**32
            or args.objects != [0, 1] or args.noise_scale != .05 or args.residual_scale != 1.):
        raise ValueError("Gate arguments differ from the bounded qualification contract")
    for name in ("ctm", "xl"):
        if (getattr(args, f"direct_{name}_checkpoint") is None) != (getattr(args, f"direct_{name}_sha256") is None):
            raise ValueError("Direct PPO references require path and SHA256 together")
    root = Path(__file__).resolve().parents[3]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if (source != args.source_commit
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()):
        raise ValueError("Require the exact clean reviewed source")
    if sha256(args.model_path / "model.safetensors") != args.expected_model_sha256:
        raise ValueError("Frozen motor input hash differs")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started, stopped, env = time.monotonic(), False, None
    phases, costs = {}, dict(simulator_steps=0, policy_chunks=0, additional_label_motor_calls=0,
                            additional_flow_matching_suffix_calls=0, native_prefix_queries=0,
                            native_suffix_queries=0, bridge_context_updates=0)
    def should_stop():
        return stopped or time.monotonic() - started >= args.max_elapsed_seconds
    def signal_stop(signum, frame):
        nonlocal stopped
        stopped = True
    for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        signal.signal(getattr(signal, name), signal_stop)
    def receipt(phase, value):
        phases[phase] = value
        write_json(args.output_dir / "progress.json", dict(phases=phases, costs=costs,
            elapsed_seconds=time.monotonic() - started, source_commit=source))
        print(json.dumps(dict(phase=phase, result=value, costs=costs,
                              elapsed_seconds=time.monotonic() - started), allow_nan=False), flush=True)
    try:
        admit_gpu(args, root, args.output_dir)
        sys.path.insert(0, str(args.rlinf_root))
        from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv
        import rlinf.envs.sim.maniskill.tasks  # noqa: F401
        from rlinf.models import get_model
        from .pi05_balanced_plan import make_balanced_history_manifest
        from .pi05_interface_gate import residual_headroom, slice_history_lanes
        torch.set_num_threads(args.threads)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        motor_rng = torch.Generator(device="cuda").manual_seed(args.seed + 1)
        label_rng = torch.Generator(device="cuda").manual_seed(args.seed + 17)
        env = ManiskillEnv(ms.env_config(args.rlinf_root, 6, args.seed), 6, seed_offset=0,
                          total_num_processes=1, worker_info=None, record_metrics=True)
        ms.record_renderer_binding(env, args.output_dir)
        motor = get_model(ms.model_config(args.model_path, args.norm_stats)).cuda().eval().requires_grad_(False)
        if (motor.num_steps, motor.action_horizon, motor.action_dim) != (4, 8, 32):
            raise ValueError("Native motor differs from qualified four-Euler 8x32 contract")
        frozen_before = ms.model_fingerprint(motor)
        policy, config = bridge_policy("persistent_ctm")
        contract = native_contract(args, source, frozen_before, policy, config)
        write_json(args.output_dir / "config.json", dict(contract=contract, hostname=os.uname().nodename,
            job_id=ms.job_id(), max_elapsed_seconds=args.max_elapsed_seconds,
            fit_updates=args.updates, fit_seconds=args.fit_seconds,
            runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                         cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))))
        manifest = make_balanced_history_manifest(env.env.unwrapped.basic_obj_infos, seed=args.seed)
        write_json(args.output_dir / "balanced-plan.json", manifest)
        batches = []
        for index, plan in enumerate(manifest["plans"]):
            if should_stop():
                break
            write_json(args.output_dir / f"trial-plan-{index+1}.json", plan)
            phase_start = time.monotonic()
            rollout = collect_live_trials(policy, motor, env, plan, generator=motor_rng, dpt_generator=label_rng)
            replay = qualify_replay(policy, rollout)
            export_history(args.output_dir / "histories", rollout, contract=contract, iteration=index+1)
            history_manifest = json.loads((args.output_dir / "histories/manifest.json").read_text())
            entry = history_manifest["histories"][-1]
            data = load_history(args.output_dir / "histories" / entry["file"], entry["sha256"],
                                history_manifest["compatibility"])
            batches.append(data)
            costs["simulator_steps"] += rollout["simulator_steps"]
            costs["policy_chunks"] += rollout["policy_chunks"]
            costs["additional_label_motor_calls"] += rollout["policy_chunks"]
            costs["additional_flow_matching_suffix_calls"] += rollout["policy_chunks"]
            costs["native_prefix_queries"] += 3 * rollout["policy_chunks"]
            costs["native_suffix_queries"] += 9 * rollout["policy_chunks"]
            costs["bridge_context_updates"] += rollout["policy_chunks"]
            write_json(args.output_dir / f"action-audit-{index+1}.json", rollout["audit"])
            receipt(f"collection-{index+1}", dict(seconds=time.monotonic()-phase_start,
                replay=replay, outcomes=rollout["outcomes"].tolist(),
                residual_headroom=residual_headroom(data, residual_scale=1., active_dims=7),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), history=entry))
            del rollout
        if len(batches) != 2:
            receipt("bounded_stop", dict(reason="cap or signal before both balanced batches"))
            write_json(args.output_dir / "summary.json", dict(phases=phases, costs=costs,
                source_commit=source, bounded_stop=True, elapsed_seconds=time.monotonic()-started,
                conclusion="incomplete balanced collection; no candidates fitted"))
            return
        # The private balanced plan is saved separately from public records. It
        # selects teacher heads and fixes disjoint train/validation batches only.
        history_manifest["train_indices"] = [0]
        history_manifest["validation_indices"] = [1]
        history_manifest["balanced_plan_sha256"] = sha256(args.output_dir / "balanced-plan.json")
        write_json(args.output_dir / "histories/manifest.json", history_manifest)
        provenance = dict(manifest_sha256=sha256(args.output_dir / "histories/manifest.json"),
            histories=history_manifest["histories"], train_indices=[0], validation_indices=[1],
            balanced_plan_sha256=history_manifest["balanced_plan_sha256"],
            simulator_steps=costs["simulator_steps"], split="disjoint complete histories; all theta/object cells")
        candidates = {}
        # Privileged heads remove theta identification from the interface gate.
        # They still execute the same bounded four-Euler residual path.
        for theta in (-45., 0., 45.):
            if should_stop():
                break
            indices = [i for i, value in enumerate(manifest["plans"][0]["theta_degrees"]) if value == theta]
            val_indices = [i for i, value in enumerate(manifest["plans"][1]["theta_degrees"]) if value == theta]
            teacher, teacher_config = bridge_policy("persistent_ctm")
            teacher_contract = native_contract(args, source, frozen_before, teacher, teacher_config)
            teacher_data = dict(provenance, private_teacher_theta=theta,
                                simulator_steps=2*len(indices)*4*ms.STEPS)
            name = f"privileged-theta-{int(theta):+d}"
            result = fit_candidate(teacher, [slice_history_lanes(batches[0], indices)],
                [slice_history_lanes(batches[1], val_indices)], output=args.output_dir/name,
                contract=teacher_contract, provenance=teacher_data, updates=args.updates,
                fit_seconds=args.fit_seconds, learning_rate=args.learning_rate, seed=args.seed,
                should_stop=should_stop)
            candidates[name] = teacher
            receipt(name, result)
        for core in ("persistent_ctm", "transformer_xl"):
            if should_stop():
                break
            student, student_config = bridge_policy(core)
            student_contract = native_contract(args, source, frozen_before, student, student_config)
            result = fit_candidate(student, [batches[0]], [batches[1]], output=args.output_dir/core,
                contract=student_contract, provenance=provenance, updates=args.updates,
                fit_seconds=args.fit_seconds, learning_rate=args.learning_rate, seed=args.seed,
                should_stop=should_stop)
            candidates[core] = student
            receipt(core, result)
        del batches
        env.env.close()
        env = None
        if not should_stop():
            env = ManiskillEnv(ms.env_config(args.rlinf_root, 2, args.seed), 2, seed_offset=0,
                              total_num_processes=1, worker_info=None, record_metrics=True)
            binding_dir = args.output_dir / "interface-evaluation"
            binding_dir.mkdir()
            ms.record_renderer_binding(env, binding_dir)
            infos = env.env.unwrapped.basic_obj_infos
            _, _, positions, orientations = ms.check_object_infos(infos)
            gate_raw = {}
            for theta in (-45., 0., 45.):
                name = f"privileged-theta-{int(theta):+d}"
                if name not in candidates or should_stop():
                    break
                rng = np.random.default_rng(910000 + int(theta))
                scene = [dict(object=obj, overlay=12 + obj,
                    position=int(rng.integers(positions)), orientation=int(rng.integers(orientations))) for obj in (0, 1)]
                for row in scene:
                    row["episode_id"] = ms.episode_id(infos, row["object"], row["overlay"], row["position"], row["orientation"])
                plan = dict(theta_degrees=[theta, theta], scenes=[scene], reset_seeds=[710000], split="teacher_fresh_scene_gate")
                write_json(binding_dir / f"plan-{int(theta):+d}.json", plan)
                gate_raw[str(theta)] = {}
                for condition in ("bridge", "zero_bridge", "external_oracle"):
                    if should_stop():
                        break
                    evaluated = candidates[name] if condition == "bridge" else policy
                    before = ms.model_fingerprint(evaluated)
                    def diagnostic(motor, observation, actions, record, attempt, chunk):
                        if attempt == 0 and chunk == 0:
                            costs["additional_label_motor_calls"] += 2
                            costs["native_prefix_queries"] += 2
                            costs["native_suffix_queries"] += 8
                            return sampled_action_distances(motor, observation, actions, record, [theta, theta])
                    rollout = collect_live_trials(evaluated, motor, env, plan,
                        generator=torch.Generator(device="cuda").manual_seed(810000), deterministic=True,
                        execute_labels=condition == "external_oracle",
                        diagnostic_callback=diagnostic if condition == "bridge" else None)
                    if ms.model_fingerprint(evaluated) != before:
                        raise RuntimeError("Frozen interface evaluation mutated teacher weights")
                    gate_raw[str(theta)][condition] = rollout["outcomes"].tolist()
                    costs["simulator_steps"] += rollout["simulator_steps"]
                    costs["policy_chunks"] += rollout["policy_chunks"]
                    costs["native_prefix_queries"] += rollout["policy_chunks"]
                    costs["native_suffix_queries"] += 4 * rollout["policy_chunks"]
                    costs["bridge_context_updates"] += rollout["policy_chunks"]
                    if condition == "external_oracle":
                        costs["additional_label_motor_calls"] += rollout["policy_chunks"]
                        costs["native_prefix_queries"] += rollout["policy_chunks"]
                        costs["native_suffix_queries"] += 4 * rollout["policy_chunks"]
                    write_json(binding_dir / "outcomes.json", gate_raw)
                    write_json(binding_dir / f"audit-{int(theta):+d}-{condition}.json", rollout["audit"])
                    receipt(f"interface-{theta}-{condition}", dict(outcomes=rollout["outcomes"].tolist()))
                    del rollout
            write_json(binding_dir / "summary.json", dict(raw=gate_raw,
                interpretation="small same-interface privileged-fit competence screen; not an optimal controller or ICRL result"))
            env.env.close()
            env = None
        if not should_stop():
            from .pi05_balanced_plan import make_heldout_probe_plan
            from .pi05_fresh_probe import evaluate_fresh_probes
            env = ManiskillEnv(ms.env_config(args.rlinf_root, 4, args.seed), 4, seed_offset=0,
                              total_num_processes=1, worker_info=None, record_metrics=True)
            probe_dir = args.output_dir / "fresh-probes"
            probe_dir.mkdir()
            ms.record_renderer_binding(env, probe_dir)
            # New module's plan adapter is kept explicit here to make K4 and the
            # private theta/object pairing reviewable in the saved JSON.
            probe_manifest = make_heldout_probe_plan(env.env.unwrapped.basic_obj_infos, seed=410000 + args.seed)
            write_json(probe_dir / "plan.json", probe_manifest)
            comparisons = []
            for core, short in (("persistent_ctm", "ctm"), ("transformer_xl", "xl")):
                if core in candidates:
                    comparisons.append(("dpt-"+short, candidates[core], None, core))
            for core, short in (("persistent_ctm", "ctm"), ("transformer_xl", "xl")):
                direct_path = getattr(args, f"direct_{short}_checkpoint")
                if direct_path:
                    direct, direct_config = bridge_policy(core)
                    direct_contract = native_contract(args, source, frozen_before, direct, direct_config)
                    parent = load_direct_candidate(direct_path, getattr(args, f"direct_{short}_sha256"), direct, direct_contract)
                    comparisons.append(("direct-"+short, direct, parent, core))
            for name, evaluated, parent, core in comparisons:
                if should_stop():
                    break
                before = ms.model_fingerprint(evaluated)
                support_generator = torch.Generator(device="cuda").manual_seed(420000)
                def progress(result):
                    write_json(probe_dir / f"{name}-progress.json", result)
                def boundary(state, metadata):
                    atomic_save(probe_dir / f"{name}-boundary-{state.attempt_index}.pt",
                        dict(state=state, metadata=metadata, source_commit=source,
                             policy_fingerprint=before, rng=rng_state(),
                             probe_plan_sha256=sha256(probe_dir / "plan.json")))
                result = evaluate_fresh_probes(evaluated, motor, env,
                    probe_manifest["support_plan"], probe_manifest["probe_plans"],
                    support_generator=support_generator,
                    probe_seeds=probe_manifest["probe_seeds"], history_batch_id="heldout-support-batch-0",
                    opposite_indices=probe_manifest["opposite_indices"], should_stop=should_stop,
                    on_progress=progress, on_boundary=boundary)
                if ms.model_fingerprint(evaluated) != before:
                    raise RuntimeError("Frozen support-probe evaluation mutated learner weights")
                for key in ("simulator_steps", "native_prefix_queries", "native_suffix_queries", "bridge_context_updates"):
                    costs[key] += result["total_counts"][key]
                costs["policy_chunks"] += result["total_counts"]["action_chunks"]
                write_json(probe_dir / f"{name}.json", dict(result=result, parent=parent,
                    checkpoint_sha256=phases.get(core, {}).get("warmstart_sha256") if parent is None else parent["checkpoint_sha256"]))
                receipt(f"fresh-probe-{name}", result)
        frozen_after = ms.model_fingerprint(motor)
        if frozen_after != frozen_before or sha256(args.model_path / "model.safetensors") != args.expected_model_sha256:
            raise RuntimeError("Frozen motor changed during bounded qualification")
        write_json(args.output_dir / "summary.json", dict(phases=phases, costs=costs,
            source_commit=source, frozen_motor_before=frozen_before, frozen_motor_after=frozen_after,
            bounded_stop=should_stop(), elapsed_seconds=time.monotonic()-started,
            conclusion="diagnostic qualification; no automatic PPO extension or adaptation claim"))
    except BaseException as error:
        write_json(args.output_dir / "failure.json", dict(error_type=type(error).__name__, error=str(error),
            phases=phases, costs=costs, source_commit=source, elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        if env is not None:
            env.env.close()


if __name__ == "__main__":
    main()
