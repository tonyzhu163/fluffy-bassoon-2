#!/usr/bin/env python3
"""Finite allocated-GPU calibration or 48-live-lane stress, no scientific SGD.

Use a fresh output for each phase. `calibrate` collects five reset-only TRAIN
observations and writes CALIBRATION.json. `stress` requires its reviewed
SHA-bound profile.calibration_reference and runs one or both arms, 48 full600
physical episodes per arm through the production history/video collector.
External timeout must cover the finite --deadline-unix including cleanup.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback


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


def process_identity(pid):
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=pid, name=Path(f"/proc/{pid}/comm").read_text().strip(), start_ticks=int(state[19]),
            interpreter_helper=any(value in command for value in (b"multiprocessing.resource_tracker", b"multiprocessing.forkserver")))
    except (OSError, ValueError, IndexError):
        return None


def descendants():
    pending, seen, rows = [os.getpid()], set(), []
    while pending:
        owner = pending.pop()
        for file in Path(f"/proc/{owner}/task").glob("*/children"):
            try:
                children = [int(value) for value in file.read_text().split()]
            except OSError:
                continue
            for pid in children:
                if pid not in seen:
                    seen.add(pid)
                    pending.append(pid)
                    identity = process_identity(pid)
                    if identity is not None:
                        rows.append(identity)
    return rows


def process_resources():
    cgroup = {}
    for name in ("cpu.max", "memory.max", "memory.current", "memory.peak", "pids.current", "pids.max", "pids.peak"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            cgroup[name] = path.read_text().strip()
    shm = os.statvfs("/dev/shm")
    return dict(cgroup=cgroup, fd_count=len(list(Path("/proc/self/fd").iterdir())),
        nofile_limit=list(resource.getrlimit(resource.RLIMIT_NOFILE)),
        shm_total_bytes=shm.f_blocks * shm.f_frsize, shm_free_bytes=shm.f_bavail * shm.f_frsize,
        descendants=descendants(), maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


class ResourceWatch:
    def __init__(self, output, uuid):
        self.path, self.uuid = Path(output) / "resources.jsonl", uuid
        self.stop = threading.Event()
        self.phase = "loading"
        self.summary = dict(by_phase={}, owned_children={}, samples=0, gpu_samples=0, peak_gpu_memory_mib=0., peak_fd_count=0, peak_descendants=0,
                            peak_cgroup_memory_bytes=0)
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with self.path.open("x") as stream:
            while not self.stop.is_set():
                row = dict(unix=time.time(), phase=self.phase)
                try:
                    row.update(process_resources())
                    self.summary["owned_children"].update({str(child["pid"]): child for child in row["descendants"]})
                    self.summary["peak_fd_count"] = max(self.summary["peak_fd_count"], row["fd_count"])
                    self.summary["peak_descendants"] = max(self.summary["peak_descendants"], len(row["descendants"]))
                    self.summary["peak_cgroup_memory_bytes"] = max(self.summary["peak_cgroup_memory_bytes"],
                        int(row["cgroup"].get("memory.current", "0")))
                    sample = subprocess.run(["nvidia-smi", "--id=" + self.uuid,
                        "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                        text=True, capture_output=True, timeout=3, check=True)
                    memory, utilization = map(float, sample.stdout.strip().split(","))
                    row.update(gpu_memory_mib=memory, gpu_utilization_percent=utilization)
                    self.summary["gpu_samples"] += 1
                    self.summary["peak_gpu_memory_mib"] = max(self.summary["peak_gpu_memory_mib"], memory)
                    phase = self.summary["by_phase"].setdefault(row["phase"], dict(samples=0, peak_gpu_memory_mib=0., peak_fd_count=0, peak_descendants=0))
                    phase["samples"] += 1
                    phase["peak_gpu_memory_mib"] = max(phase["peak_gpu_memory_mib"], memory)
                    phase["peak_fd_count"] = max(phase["peak_fd_count"], row["fd_count"])
                    phase["peak_descendants"] = max(phase["peak_descendants"], len(row["descendants"]))
                except Exception as error:
                    row["telemetry_error"] = repr(error)
                self.summary["samples"] += 1
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                self.stop.wait(2)

    def close(self):
        self.stop.set()
        self.thread.join(5)
        if self.thread.is_alive():
            raise RuntimeError("Resource watcher failed to release its bounded subprocess")


def interrupted(signum, frame):
    raise TimeoutError(f"Finite qualification interrupted by signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--phase", choices=("calibrate", "stress"), required=True)
    parser.add_argument("--arm", choices=("velocity", "kv_all_layers", "both"), default="both")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output / "console.log")])
    started = time.time()
    result = dict(format="joint_scale50_single_gpu_qualification_v1", completed=False, status="preparing",
        phase=args.phase, arm=args.arm, hostname=socket.gethostname(), pid=os.getpid(),
        started_unix=started, scientific_updates=0, policy_episodes_admitted_to_PPO=0,
        input_hashes=dict(script=sha(__file__)), arms={}, partial_artifacts_retained=True)
    profile, environments, watch, journal, torch = {}, [], None, None, None
    calibration = None

    def stage(name):
        result["stage"] = name
        if watch is not None:
            watch.phase = name
        logging.info("stage=%s elapsed=%.3f", name, time.time() - started)
        write_json(output / "PROGRESS.json", dict(stage=name, last_check_unix=time.time(), completed=False))

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, interrupted)
    try:
        if sys.platform != "linux" or socket.gethostname() != args.expected_hostname:
            raise RuntimeError("Require the exact allocated Linux scheduled hostname")
        profile = json.loads(args.profile.read_text())
        result["input_hashes"]["profile"] = sha(args.profile)
        prior = float(profile["prior_qualification_gpu_seconds"])
        allocated = float(profile["allocated_start_unix"])
        hard_end = allocated + 2400 - prior
        if (not 0 <= prior < 2400 or not math.isfinite(allocated) or not math.isfinite(args.deadline_unix)
                or not time.time() < args.deadline_unix <= hard_end - 30):
            raise ValueError("Deadline must leave cleanup inside2400 lifetime qualification GPU-seconds")
        signal.setitimer(signal.ITIMER_REAL, max(.001, args.deadline_unix - time.time()))
        binding = {key: profile[key] for key in ("code", "source_commit", "runtime_path", "runtime_commit",
            "model_directory", "model_sha256", "model_config_sha256", "norm_path", "norm_sha256")}
        result["bindings"] = binding
        storage = os.statvfs(output)
        result["storage_before"] = dict(total_bytes=storage.f_blocks * storage.f_frsize,
                                       free_bytes=storage.f_bavail * storage.f_frsize)
        if (binding["model_sha256"] != "661bf172a2ae9dfca75216697eeaa543408bcbdfa53c98132759b93e0e8993a5"
                or binding["norm_sha256"] != "4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee"):
            raise ValueError("Only the released original Human300 motor/native normalization is authorized")
        code, runtime = Path(binding["code"]), Path(binding["runtime_path"])
        for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "LYCO_BAIGE_GPU_MAP", "LYCO_BAIGE_ISOLATED_UUID", "MUJOCO_EGL_DEVICE_ID"):
            if key in profile.get("native_env", {}):
                raise ValueError("Profile cannot override scheduler GPU identity/visibility")
        os.environ.update({key: str(value) for key, value in profile.get("native_env", {}).items()})
        os.environ.update(RLINF_PATH=str(runtime), ROBOCASA_NATIVE_OUTPUT=str(output / "native-runtime"),
            ROBOCASA_PI05_TORCH_MODEL=binding["model_directory"], ROBOCASA_PI05_NATIVE_NORM=binding["norm_path"],
            NCCL_P2P_DISABLE="1")
        sys.path[:0] = [str(code / "src"), str(runtime)]
        os.environ["PYTHONPATH"] = os.pathsep.join((str(code / "src"), str(runtime), os.environ.get("PYTHONPATH", "")))
        from fluffy_bassoon.meta_rl.rlinf_native_ppo import checked_checkout
        checked_checkout(code, binding["source_commit"])
        checked_checkout(runtime, binding["runtime_commit"])
        for key, path in (("model_sha256", Path(binding["model_directory"]) / "model.safetensors"),
                          ("model_config_sha256", Path(binding["model_directory"]) / "config.json"),
                          ("norm_sha256", Path(binding["norm_path"]))):
            if sha(path) != binding[key]:
                raise ValueError(f"Frozen initializer differs: {key}")
        stage("one_actual_allocated_gpu_probe")
        probe_path = output / "gpu-probe.json"
        with (output / "gpu-probe.log").open("x") as log:
            subprocess.run([sys.executable, "-B", str(code / "scripts/[train]gpu_probe.py"),
                "--expected-count", "1", "--output", str(probe_path), "--baige-container", "--purpose", "training",
                "--baige-exclusive-training"], stdout=log, stderr=subprocess.STDOUT, timeout=90, check=True)
        probe = json.loads(probe_path.read_text())
        if probe["status"] != "passed" or probe["hostname"] != args.expected_hostname or len(probe["devices"]) != 1:
            raise ValueError("Unexpected actual GPU probe")
        device = probe["devices"][0]
        os.environ.update(LYCO_BAIGE_GPU_MAP=str(probe_path), LYCO_BAIGE_ISOLATED_UUID=device["uuid"],
                          MUJOCO_EGL_DEVICE_ID=str(device["allocation_ordinal"]))
        import numpy as np
        import torch as torch_module
        torch = torch_module
        from fluffy_bassoon.meta_rl.interface_ppo import InterfaceSpec
        from fluffy_bassoon.meta_rl.joint_steering_ppo import (TASKS, JointActionJournal, cases_for,
            make_policy, native_scale_calibration, qualify_learning_rates, select_scale, spec_arguments)
        from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import (_load_native, _RecordingEnvironment,
            _qualify_cached, _save, _versions, collect_bank, native_video_writer)
        from fluffy_bassoon.meta_rl.rlinf_native_ppo import cpu_tree, snapshot_rng, tensor_hash
        import imageio.v2 as imageio
        watch = ResourceWatch(output, device["uuid"])
        watch.thread.start()
        stage("load_original_native_once")
        motor, factory, signature = _load_native(binding, code)
        result["native_model_frozen_signature"] = signature
        versions = tensor_hash(_versions(motor.model))
        result["resources_before_environments"] = process_resources()
        torch.cuda.reset_peak_memory_stats(0)
        journal = JointActionJournal(output / "accounting", rank=0, category="train", scope="qualification-" + args.phase)
        canonical = output / "canonical"

        def make_environments(cases, root, lanes, label):
            rows = [_RecordingEnvironment(factory, cases, horizon=None, output=root / "videos" / f"lane-{lane:02d}",
                monitoring_root=root / "monitoring" / f"lane-{lane:02d}", journal=journal,
                writer_factory=native_video_writer, rank=0, episode_id_prefix=label + f"-l{lane:02d}-",
                canonical_root=canonical, source=dict(phase="qualification", scientific_updates=0,
                    source_commit=binding["source_commit"], runtime_commit=binding["runtime_commit"])) for lane in range(lanes)]
            environments.extend(rows)
            return rows

        if args.phase == "calibrate":
            if args.arm != "both":
                raise ValueError("Five task calibration measures both arms on the same five public resets")
            default = cases_for("train", 1)
            cases = profile.get("calibration_cases", [next(case for case in default if case["task"] == task) for task in TASKS])
            if len(cases) != 5 or {case["task"] for case in cases} != set(TASKS):
                raise ValueError("Calibrate exactly one public TRAIN reset per ID task")
            public, references = [], []
            receipts = {arm: [] for arm in ("velocity", "kv_all_layers")}
            for index, case in enumerate(cases):
                root = output / f"reset-{index:02d}-{case['task']}"
                env = make_environments([case], root, 1, f"reset{index}")[0]
                stage("reset_only/" + case["task"])
                try:
                    observation = env.reset()
                    initial = _save(root / "initial.pt", dict(public=cpu_tree(observation),
                        raw=cpu_tree(env.native.current_raw_obs), evidence=cpu_tree(env.initial_evidence)))
                    public.append(cpu_tree(observation))
                    references.append(dict(case=case, initial=initial, public_sha256=tensor_hash(observation),
                        physical_sha256=tensor_hash(env.initial_evidence["snapshot"]["physical"])))
                    for arm in receipts:
                        receipts[arm].append(native_scale_calibration(motor, observation, InterfaceSpec(**spec_arguments(arm))))
                    counts = env._counts()
                    if env.steps != 0 or counts["controls"] != counts["reset_controls"]:
                        raise ValueError("Calibration dispatched unexpected physical policy controls")
                    references[-1]["native_counts"] = counts
                finally:
                    env.close()
                scope = dict(reset_only=True, scoreable_episode=False, policy_controls=0,
                    frames=env.writer.count, videos=env.writer.refs(), case=case)
                write_json(root / "RESET-ONLY-SCOPE.json", scope)
                references[-1]["reset_scope"] = scope
                _save(root / "scale-receipts.pt", {arm: rows[-1] for arm, rows in receipts.items()})
            calibration = dict(completed=False, phase="calibrate", scientific_updates=0, public_cases=references,
                source_commit=profile["source_commit"], runtime_commit=profile["runtime_commit"],
                native_model_frozen_signature=signature, arms={})
            for arm in receipts:
                selected = select_scale(receipts[arm])
                specification = dict(spec_arguments(arm), fraction=selected["selected_fraction"])
                feature_rows, cache_rows = [], []
                for index, observation in enumerate(public):
                    features, prepared = motor.prepare_batch(observation)
                    noise = torch.randn((1, 50, 32), generator=torch.Generator().manual_seed(921211 + index))
                    cache = _qualify_cached(motor, prepared, noise, InterfaceSpec(**specification), .5)
                    motor.execute_batch(prepared, noise=noise, steering=np.zeros((1, specification["steering_dim"])),
                                        spec=InterfaceSpec(**specification))
                    feature_rows.append(features[0])
                    cache_rows.append(cache)
                public_features = np.stack(feature_rows)
                _save(output / f"{arm}-public-features.pt", dict(features=public_features,
                    source_public_cases=references, cached_native_qualification=cache_rows))
                optimizer = qualify_learning_rates(arm, public_features, specification, device="cuda:0")
                calibration["arms"][arm] = dict(scale=selected, optimizer=optimizer,
                    interface_spec=specification, raw_task_scale_receipts=receipts[arm], cached_qualification=cache_rows)
                write_json(output / "CALIBRATION.json", calibration)
                if not optimizer["passed"]:
                    raise ValueError("No preregistered optimizer candidate passed for " + arm)
            result["new_reset_calls"] = 5
            result["physical_policy_controls"] = 0
        else:
            reference = profile["calibration_reference"]
            if sha(reference["path"]) != reference["sha256"]:
                raise ValueError("Calibration reference differs")
            calibration = json.loads(Path(reference["path"]).read_text())
            if not calibration["completed"] or set(calibration["arms"]) != {"velocity", "kv_all_layers"}:
                raise ValueError("Stress needs completed five-task numerical qualification of both arms")
            if any(calibration[key] != profile[key] for key in ("source_commit", "runtime_commit")):
                raise ValueError("Calibration code/runtime differs; require fresh qualification or an explicit reviewed migration")
            for key in ("model_sha256", "model_config_sha256", "norm_sha256"):
                if calibration["native_model_frozen_signature"][key] != signature[key]:
                    raise ValueError("Calibration used a different frozen initializer")
            result["calibration_reference"] = reference
            arms = ("velocity", "kv_all_layers") if args.arm == "both" else (args.arm,)
            for index, arm in enumerate(arms):
                qualified = calibration["arms"][arm]
                if not qualified["optimizer"]["passed"]:
                    raise ValueError("Unqualified optimizer cannot enter a physical stress bank")
                specification = qualified["interface_spec"]
                if specification["fraction"] != qualified["scale"]["selected_fraction"]:
                    raise ValueError("Interface fraction differs from its task-balanced qualification")
                policy = make_policy(arm, device="cuda:0", seed=921401,
                    optimizer_config=qualified["optimizer"]["selected_optimizer_config"])
                cases = profile.get("stress_cases", cases_for("train", 1)[:48])
                if len(cases) != 48 or len({(case["task"], case["native_seed"]) for case in cases}) != 48 or any(case["task"] not in TASKS for case in cases):
                    raise ValueError("Stress is exactly48 distinct TRAIN cases over approved600-control ID tasks")
                root = output / arm
                journal = JointActionJournal(output / "accounting", rank=0, category="train", scope="qualification-stress-" + arm)
                live = make_environments(cases, root, 48, arm)
                request = dict(sequence=index + 1, phase="qualification", iteration=0, policy_iteration=0,
                    cases=cases, lanes=48, synchronize_initial_lanes=True, spec=specification,
                    deterministic=False, zero_steering=False, source_commit=binding["source_commit"],
                    runtime_commit=binding["runtime_commit"], allocation_id=str(os.getpid()),
                    native_cuda_uuid=signature["cuda_uuid"])
                stage("stress_48_live/" + arm)
                before = time.time()
                torch.cuda.reset_peak_memory_stats(0)
                collection = collect_bank(policy, motor, live, request=request, output=root / "history", rank=0,
                    deadline_unix=args.deadline_unix, qualification_path=root / "CACHE-QUALIFICATION.json")
                elapsed = time.time() - before
                if (collection["execution"]["peak_active_lanes"] != 48 or collection["execution"]["native_batch_sizes"][0] != 48
                        or len(collection["episodes"]) != 48 or any(episode["horizon"] != 600 for episode in collection["episodes"])):
                    raise ValueError("Actual48-live/B48/full600 evidence incomplete")
                record = dict(completed=True, arm=arm, scientific_updates=0, physical_policy_controls=28800,
                    elapsed_seconds=elapsed, controls_per_second=28800 / elapsed,
                    execution=collection["execution"], interface=InterfaceSpec(**specification).record(),
                    optimizer_config=asdict(policy.config), history=_save(root / "collection.pt", collection),
                    resources_after=process_resources(),
                    sampled_phase_peaks=dict(watch.summary["by_phase"].get("stress_48_live/" + arm, {})),
                    torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                    torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
                # The single-GPU allocation qualifies the two arms sequentially.
                # Release all48 native/render/video children before constructing
                # the next arm; retaining both banks would test96 live simulators.
                arm_cleanup_errors = []
                for environment in live:
                    try:
                        environment.close()
                    except BaseException as error:
                        arm_cleanup_errors.append(repr(error))
                if arm_cleanup_errors:
                    raise RuntimeError("Arm cleanup failed before the next bank: " + repr(arm_cleanup_errors))
                record["resources_after_lane_release"] = process_resources()
                record["lanes_closed_before_next_arm"] = True
                write_json(root / "RESULT.json", record)
                result["arms"][arm] = record
                del policy, collection
                torch.cuda.empty_cache()
            result["new_reset_calls"] = 48 * len(arms)
            result["physical_policy_controls"] = 28800 * len(arms)
        if tensor_hash(_versions(motor.model)) != versions or any(parameter.requires_grad for parameter in motor.model.parameters()):
            raise ValueError("Frozen native weights/buffers changed")
        result.update(completed=True, status="passed", native_versions_unchanged=True)
    except BaseException:
        result.update(completed=False, status="failed", error=traceback.format_exc())
        logging.error("Qualification failed\n%s", result["error"])
        write_json(output / "FAILED.json", result)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        cleanup_errors = []
        for env in environments:
            try:
                env.close()
            except BaseException as error:
                cleanup_errors.append(repr(error))
        if watch is not None:
            try:
                watch.close()
                result["resource_peaks"] = watch.summary
                result["resource_log"] = dict(path=str(watch.path), sha256=sha(watch.path))
            except BaseException as error:
                cleanup_errors.append(repr(error))
        if journal is not None:
            result["escrow"] = journal.snapshot()
            result["charged_controls"] = journal.charged()
            if result["completed"] and result["escrow"]["pending"]:
                cleanup_errors.append("Unresolved physical reservations at successful boundary")
        if sys.platform == "linux":
            def remaining_owned():
                known = {} if watch is None else dict(watch.summary["owned_children"])
                known.update({str(row["pid"]): row for row in descendants()})
                rows = []
                for old in known.values():
                    current = process_identity(old["pid"])
                    if current is not None and current["start_ticks"] == old["start_ticks"]:
                        rows.append(current)
                return rows
            end = time.time() + 10
            children = remaining_owned()
            while any(not child["interpreter_helper"] for child in children) and time.time() < end:
                time.sleep(.1)
                children = remaining_owned()
            result["child_release"] = dict(all_native_video_children_released=not any(not child["interpreter_helper"] for child in children),
                remaining=children, interpreter_helpers_exit_with_process=True)
            result["resources_final"] = process_resources()
            if not result["child_release"]["all_native_video_children_released"]:
                cleanup_errors.append("Owned native/video children remain after bounded release audit")
        if cleanup_errors:
            result.update(completed=False, status="failed", cleanup_errors=cleanup_errors)
        result["ended_unix"] = time.time()
        result["process_gpu_seconds"] = result["ended_unix"] - started
        if "allocated_start_unix" in profile:
            result["allocated_gpu_seconds_through_check"] = max(0., result["ended_unix"] - float(profile["allocated_start_unix"]))
            result["qualification_campaign_gpu_seconds_upper"] = math.ceil(
                float(profile["prior_qualification_gpu_seconds"]) + result["allocated_gpu_seconds_through_check"])
            if result["qualification_campaign_gpu_seconds_upper"] > 2400:
                result.update(completed=False, status="failed", budget_overrun=True)
        if args.phase == "calibrate" and calibration is not None:
            calibration["completed"] = result["completed"]
            calibration["native_versions_unchanged"] = result.get("native_versions_unchanged", False)
            calibration["child_release"] = result.get("child_release")
            write_json(output / "CALIBRATION.json", calibration)
            result["calibration"] = dict(path=str(output / "CALIBRATION.json"), sha256=sha(output / "CALIBRATION.json"))
        result["allocation_release_not_included"] = True
        write_json(output / "RESULT.json", result)
        if not result["completed"]:
            write_json(output / "FAILED.json", result)
        logging.info("status=%s elapsed=%.3f charged_controls=%s", result["status"], result["process_gpu_seconds"], result.get("charged_controls"))
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
