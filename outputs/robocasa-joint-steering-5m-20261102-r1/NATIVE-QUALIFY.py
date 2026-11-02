#!/usr/bin/env python3
"""One allocated Baige GPU, one TRAIN reset, numerical qualification only.

Invoke under an external 600-second process timeout. The reviewed profile
binds code/source_commit, runtime_path/runtime_commit, model_directory,
model_sha256, model_config_sha256, norm_path and norm_sha256. No policy control
is dispatched, no scientific PPO update is made, and no environment is edited.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

PUBLIC_MODEL_SHA256 = "661bf172a2ae9dfca75216697eeaa543408bcbdfa53c98132759b93e0e8993a5"
PUBLIC_NORM_SHA256 = "4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def timed_out(signum, frame):
    raise TimeoutError(f"Qualification interrupted by signal{signum}; external600s bound remains authoritative")


def cgroup_state():
    fields = ("cpu.max", "memory.max", "memory.current", "memory.peak")
    return {name: (Path("/sys/fs/cgroup") / name).read_text().strip()
            for name in fields if (Path("/sys/fs/cgroup") / name).exists()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-hostname", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)  # Refuse to overwrite any earlier attempt.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output / "console.log")])
    started = time.time()
    profile = {}
    environment = journal = motor = torch = None
    result = dict(format="joint_steering_single_gpu_native_qualification_v1", completed=False,
        status="preparing", hostname=socket.gethostname(), expected_hostname=args.expected_hostname,
        pid=os.getpid(), started_unix=started, started_utc=datetime.now(timezone.utc).isoformat(),
        scientific_policy_controls=0, scientific_ppo_updates=0, policy_episodes=0,
        charged_to_arm="velocity", arms={}, scope="Shared numerical qualification; one TRAIN reset only",
        input_hashes=dict(script=sha(__file__)), external_timeout_seconds=600)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, timed_out)
    signal.setitimer(signal.ITIMER_REAL, 570)

    def stage(name):
        result["stage"] = name
        logging.info("stage=%s elapsed_seconds=%.3f", name, time.time() - started)
        write_json(output / "PROGRESS.json", dict(stage=name, completed=False,
            last_check_unix=time.time(), hostname=result["hostname"], pid=result["pid"]))

    try:
        stage("input_admission")
        if sys.platform != "linux" or socket.gethostname() != args.expected_hostname:
            raise RuntimeError("Require the exact named allocated Linux scheduler container")
        result["input_hashes"]["profile"] = sha(args.profile)
        profile = json.loads(args.profile.read_text())
        required = ("code", "source_commit", "runtime_path", "runtime_commit", "model_directory",
                    "model_sha256", "model_config_sha256", "norm_path", "norm_sha256")
        binding = {key: profile[key] for key in required}
        code, runtime = Path(binding["code"]).resolve(), Path(binding["runtime_path"]).resolve()
        if binding["model_sha256"] != PUBLIC_MODEL_SHA256 or binding["norm_sha256"] != PUBLIC_NORM_SHA256:
            raise ValueError("Require the original released Human300 motor and published native normalization")
        if (profile.get("qualification_script_sha256") is not None
                and profile["qualification_script_sha256"] != result["input_hashes"]["script"]):
            raise ValueError("Qualification artifact differs from the reviewed script")
        result["bindings"] = binding
        for key, path in (("model_sha256", Path(binding["model_directory"]) / "model.safetensors"),
                          ("model_config_sha256", Path(binding["model_directory"]) / "config.json"),
                          ("norm_sha256", Path(binding["norm_path"]))):
            actual = sha(path)
            if actual != binding[key]:
                raise ValueError(f"Exact initializer input differs: {key}")
            result["input_hashes"][key] = actual
        config = json.loads((Path(binding["model_directory"]) / "config.json").read_text())
        expected_config = dict(pi05=True, action_horizon=50, action_dim=32,
            paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
        if any(config.get(key) != value for key, value in expected_config.items()):
            raise ValueError("Released original pi0.5 H50/32D architecture required")
        protected = {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "LYCO_BAIGE_GPU_MAP",
                     "LYCO_BAIGE_ISOLATED_UUID", "MUJOCO_EGL_DEVICE_ID"}
        if any(key in profile.get("native_env", {}) for key in protected):
            raise ValueError("Native profile must not override the scheduled GPU allocation or its attestation")
        os.environ.update({key: str(value) for key, value in profile.get("native_env", {}).items()})
        os.environ.update(RLINF_PATH=str(runtime), ROBOCASA_NATIVE_OUTPUT=str(output / "native-runtime"),
            ROBOCASA_PI05_TORCH_MODEL=binding["model_directory"], ROBOCASA_PI05_NATIVE_NORM=binding["norm_path"],
            NCCL_P2P_DISABLE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        sys.path[:0] = [str(code / "src"), str(runtime)]
        os.environ["PYTHONPATH"] = os.pathsep.join((str(code / "src"), str(runtime), os.environ.get("PYTHONPATH", "")))
        from fluffy_bassoon.meta_rl.rlinf_native_ppo import checked_checkout
        checked_checkout(code, binding["source_commit"])
        checked_checkout(runtime, binding["runtime_commit"])
        from fluffy_bassoon.baige_gpu import validate_container
        if len(validate_container(os.environ, socket.gethostname())) != 1:
            raise ValueError("Exactly one scheduler-assigned GPU is required")
        result["storage_before"] = dict(zip(("total_bytes", "used_bytes", "free_bytes"), shutil.disk_usage(output)))
        result["cgroup_before"] = cgroup_state()

        stage("actual_single_gpu_probe")
        probe_path = output / "gpu-probe.json"
        probe_command = [sys.executable, "-B", str(code / "scripts/[train]gpu_probe.py"),
            "--expected-count", "1", "--output", str(probe_path), "--baige-container",
            "--purpose", "training", "--baige-exclusive-training"]
        process = subprocess.run(probe_command, capture_output=True, text=True, timeout=90)
        (output / "gpu-probe.log").write_text(process.stdout + "\n" + process.stderr)
        process.check_returncode()
        probe = json.loads(probe_path.read_text())
        if (probe["status"] != "passed" or probe["hostname"] != args.expected_hostname
                or len(probe["devices"]) != 1 or probe["allocation_type"] != "baige_container"):
            raise ValueError("Actual GPU probe differs from the named one-GPU allocation")
        device = probe["devices"][0]
        os.environ.update(LYCO_BAIGE_GPU_MAP=str(probe_path), LYCO_BAIGE_ISOLATED_UUID=device["uuid"],
                          MUJOCO_EGL_DEVICE_ID=str(device["allocation_ordinal"]))
        # Keep the scheduler's original CUDA/NVIDIA visibility; do not invent
        # a device namespace or reorder its single attested GPU.
        result["gpu_probe"] = dict(path=str(probe_path), sha256=sha(probe_path), device=device)

        import numpy as np
        import torch as torch_module
        torch = torch_module
        from fluffy_bassoon.baige_gpu import verify_selected_cuda
        from fluffy_bassoon.meta_rl.interface_ppo import InterfaceSpec
        from fluffy_bassoon.meta_rl.joint_steering_ppo import (
            JointActionJournal, cases_for, make_policy, native_scale_calibration,
            qualify_optimizer_restore, spec_arguments)
        from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import (
            _load_native, _qualify_cached, _RecordingEnvironment, _save, _versions)
        from fluffy_bassoon.meta_rl.residual_ppo import update
        from fluffy_bassoon.meta_rl.rlinf_native_ppo import cpu_tree, snapshot_rng, tensor_hash
        from fluffy_bassoon.meta_rl.robocasa_residual import combine_observations
        import imageio.v2 as imageio
        verify_selected_cuda(torch)
        torch.cuda.reset_peak_memory_stats(0)

        stage("load_original_frozen_native_once")
        load_started = time.time()
        motor, factory, signature = _load_native(binding, code)
        if any(signature[key] != binding[key] for key in ("model_sha256", "model_config_sha256", "norm_sha256")):
            raise ValueError("Loaded native initializer signature differs")
        native_versions = tensor_hash(_versions(motor.model))
        result["native_frozen_signature"] = signature
        result["native_load_seconds"] = time.time() - load_started
        case = cases_for("train", 1)[0]
        if case["task"] != "OpenDrawer":
            raise ValueError("Qualification is exactly one actual TRAIN OpenDrawer reset")
        result["case"] = case
        journal = JointActionJournal(output / "accounting", rank=0, category="train", scope="shared-native-qualification")
        environment = _RecordingEnvironment(factory, [case], horizon=None,
            output=output / "videos", monitoring_root=output / "monitoring", journal=journal,
            writer_factory=imageio.get_writer, rank=0, episode_id_prefix="reset-only-",
            canonical_root=output / "canonical", source=dict(phase="native_qualification", charged_to_arm="velocity",
                source_commit=binding["source_commit"], runtime_commit=binding["runtime_commit"]))
        stage("one_charged_train_reset")
        observation = environment.reset()  # The ONLY simulator reset; never step_chunk.
        result["initial"] = _save(output / "initial.pt", dict(public=cpu_tree(observation),
            raw=cpu_tree(environment.native.current_raw_obs), evidence=cpu_tree(environment.initial_evidence)))
        result["input_hashes"].update(public_observation=tensor_hash(observation),
            raw_observation=tensor_hash(environment.native.current_raw_obs),
            physical_snapshot=tensor_hash(environment.initial_evidence["snapshot"]["physical"]))
        repeated = combine_observations([observation] * 4)

        for arm in ("velocity", "kv_all_layers"):
            arm_started = time.time()
            arm_output = output / arm
            arm_output.mkdir()
            spec = InterfaceSpec(**spec_arguments(arm))
            receipt = result["arms"][arm] = dict(interface=spec.record(), scientific_ppo_updates=0,
                scientific_policy_controls=0, repeated_observation_rows=4, native_resets_added=0)
            stage(arm + "/b1_public_scale_calibration")
            receipt["scale_calibration"] = native_scale_calibration(motor, observation, spec)
            write_json(arm_output / "SCALE-CALIBRATION.json", receipt["scale_calibration"])

            stage(arm + "/b4_cached_zero_and_influence")
            features, prepared = motor.prepare_batch(repeated)
            noise = torch.randn((4, 50, 32), generator=torch.Generator().manual_seed(921211))
            receipt["cached_qualification"] = _qualify_cached(motor, prepared, noise, spec, .5)
            if arm == "kv_all_layers" and (receipt["cached_qualification"]["consumed_cache_layers"] != 18
                    or len(receipt["cached_qualification"]["cache_targets"]) != 36):
                raise ValueError("Require consumed changes to both K/V in all18 native layers")
            # Consume the original port's still-pending token, with explicit
            # zero code and the SAME Gaussian. This is inference, no dispatch.
            zero_chunks = motor.execute_batch(prepared, noise=noise,
                steering=np.zeros((4, spec.steering_dim), np.float32), spec=spec)
            if motor._prepared is not None or features.shape != (4, 2064):
                raise ValueError("Qualification must consume its prefix and retain native public2064 features")
            receipt["cached_outputs"] = _save(arm_output / "cached-outputs.pt", dict(
                features=features, native_noise=noise, zero_chunks=zero_chunks,
                qualification=receipt["cached_qualification"], observation_sha256=tensor_hash(repeated)))
            write_json(arm_output / "CACHE-QUALIFICATION.json", receipt["cached_qualification"])

            stage(arm + "/synthetic_actor_value_update_and_cuda_adam_restore")
            policy = make_policy(arm, device="cuda:0", seed=921301)
            feature_rows = np.repeat(features[:1], 24, axis=0)
            policy.fit_feature_normalization(feature_rows)
            optimizer = torch.optim.Adam(policy.parameters(), lr=policy.config.learning_rate)
            generator = torch.Generator().manual_seed(921302)
            tensors = torch.as_tensor(feature_rows, device="cuda:0", dtype=torch.float32)
            actor_noise = torch.randn((24, spec.steering_dim), generator=torch.Generator().manual_seed(921303)).to("cuda:0")
            control, logs, values = policy.act(tensors, action_noise=actor_noise)
            ends = torch.tensor([False] * 5 + [True], device="cuda:0")[:, None].repeat(4, 1)
            rollout = dict(features=tensors[:, None], controls=control[:, None], log_probs=logs[:, None],
                values=values[:, None], ends=ends, rewards=ends.to(torch.float32))
            receipt["synthetic_update"] = update(policy, optimizer, rollout, generator=generator)
            if receipt["synthetic_update"]["optimizer_steps"] < 1 or not optimizer.state:
                raise ValueError("Numerical qualification did not initialize actual CUDA Adam moments")
            for state in optimizer.state.values():
                if state["exp_avg"].device.type != "cuda" or state["exp_avg_sq"].device.type != "cuda":
                    raise ValueError("Adam moments must reside on the actual allocated GPU")
            learner_before = tensor_hash(dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(),
                                               generator=generator.get_state()))
            rng_before = tensor_hash(snapshot_rng())
            receipt["adam_restore"] = qualify_optimizer_restore(policy, optimizer, generator, feature_rows,
                arm=arm, interface_spec=asdict(spec))
            if (learner_before != tensor_hash(dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(),
                                                   generator=generator.get_state()))
                    or rng_before != tensor_hash(snapshot_rng())):
                raise ValueError("Shadow Adam qualification changed the numerical learner/RNG boundary")
            receipt["synthetic_boundary"] = _save(arm_output / "synthetic-boundary.pt", dict(
                policy_config=asdict(policy.config), policy=policy.state_dict(), optimizer=optimizer.state_dict(),
                generator=generator.get_state(), rng=snapshot_rng(), public_features=feature_rows,
                synthetic_rollout=rollout, scientific_ppo_updates=0, numerical_ppo_updates=1,
                scope="Synthetic rewards only; never scientific source data or policy episodes"))
            if native_versions != tensor_hash(_versions(motor.model)):
                raise ValueError("Original frozen native tensor version signature changed")
            torch.cuda.synchronize()
            receipt["elapsed_seconds"] = time.time() - arm_started
            receipt["completed"] = True
            write_json(arm_output / "RESULT.json", receipt)
            del policy, optimizer, tensors, rollout, actor_noise, control, logs, values
            torch.cuda.empty_cache()

        counts = environment._counts()
        if environment.steps != 0 or counts["controls"] != counts["reset_controls"]:
            raise ValueError("Qualification must execute zero policy controls")
        if native_versions != tensor_hash(_versions(motor.model)) or any(parameter.requires_grad for parameter in motor.model.parameters()):
            raise ValueError("Native model must remain frozen throughout both arms")
        result["native_versions_unchanged"] = True
        result["native_counts"] = counts
        result["completed"] = True
        result["status"] = "passed"
    except BaseException:
        result.update(completed=False, status="failed", error=traceback.format_exc())
        logging.error("Qualification failed\n%s", result["error"])
        # Publish failure before cleanup; a blocked native close must not hide it.
        write_json(output / "FAILED.json", result)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if environment is not None:
            try:
                if environment.native is not None:
                    result["native_counts"] = environment._counts()
                environment.close()
                if hasattr(environment, "writer"):
                    scope = dict(full_physical_episode=False, scoreable_episode=False, reset_only=True,
                        policy_controls=environment.steps, frames=environment.writer.count,
                        videos=environment.writer.refs(), charged_to_arm="velocity")
                    write_json(output / "RESET-ONLY-SCOPE.json", scope)
                    result["reset_only_scope"] = scope
            except BaseException:
                result.update(completed=False, status="failed", cleanup_error=traceback.format_exc())
                logging.error("Native cleanup failed\n%s", result["cleanup_error"])
        if journal is not None:
            try:
                result["escrow"] = journal.snapshot()
                result["charged_controls"] = journal.charged()
                if result["completed"] and result["escrow"]["pending"]:
                    raise ValueError("Successful qualification left an unresolved reset reservation")
            except BaseException:
                result.update(completed=False, status="failed", accounting_error=traceback.format_exc())
        result["cgroup_after"] = cgroup_state()
        if torch is not None and torch.cuda.is_initialized():
            try:
                result["cuda_memory"] = dict(allocated_bytes=torch.cuda.memory_allocated(0),
                    reserved_bytes=torch.cuda.memory_reserved(0), peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
            except BaseException:
                result["cuda_memory_error"] = traceback.format_exc()
            try:
                sample = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                result["cuda_process_telemetry"] = dict(returncode=sample.returncode,
                    stdout=sample.stdout, stderr=sample.stderr, sampled_unix=time.time())
            except BaseException:
                result["cuda_process_telemetry_error"] = traceback.format_exc()
        result["ended_unix"] = time.time()
        result["process_gpu_seconds"] = result["ended_unix"] - started
        if "allocated_start_unix" in profile:
            result["allocated_gpu_seconds_through_check"] = result["ended_unix"] - float(profile["allocated_start_unix"])
        result["allocation_release_not_included"] = True
        result["ended_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "RESULT.json", result)
        if not result["completed"]:
            write_json(output / "FAILED.json", result)
        logging.info("status=%s elapsed_seconds=%.3f charged_controls=%s", result["status"],
                     result["process_gpu_seconds"], result.get("charged_controls"))
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
