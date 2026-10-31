"""Two independent CPU software phases. Never allocates or controls source jobs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET


PHASE_TESTS = [
    ["tests/test_robotwin_trained_pair_continuation.py",
     "tests/test_robotwin_trained_pair_plan.py",
     "tests/test_robotwin_trained_pair_preflight.py",
     "tests/test_robotwin_trained_pair_runtime.py",
     "tests/test_robotwin_trained_pair_audit.py",
     "tests/test_robotwin_trained_pair_continuation_spawn.py"],
    ["tests/test_vla_cached_scalar_motor.py"],
]
LIBERO_PIN = "26b55eae2a5ce6545cdc97b3a875cc6e3a7b46fa"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def exact_clean_git(root, pin):
    root = Path(root).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", pin):
        raise ValueError("Unresolved or invalid source pin: " + str(pin))
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if actual != pin or dirty:
        raise ValueError("Code root is not the exact clean pin: " + str(root))
    return dict(path=str(root), source_commit=actual, git_clean=True)


def cgroup_value(name):
    path = Path("/sys/fs/cgroup") / name
    return path.read_text().strip() if path.is_file() else None


def probe(output, expected_hostname):
    # This runs in a separate allocated CPU Python process. Importing torch
    # here queries actual device visibility; it constructs no model/tensor.
    if platform.system() != "Linux" or socket.gethostname() != expected_hostname or not expected_hostname.startswith("job-"):
        raise RuntimeError("CPU qualification requires its explicit allocated job hostname")
    incoming = {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")}
    import torch
    count = torch.cuda.device_count()
    available = torch.cuda.is_available()
    initialized = torch.cuda.is_initialized()
    receipt = dict(checked_at_utc=now(), hostname=socket.gethostname(), expected_hostname=expected_hostname,
        platform=platform.platform(), python=sys.executable, torch_version=torch.__version__,
        incoming_device_visibility=incoming, cuda_device_count=count,
        cuda_available=available, cuda_initialized=initialized,
        affinity_cpus=len(os.sched_getaffinity(0)), cgroup_cpu_max=cgroup_value("cpu.max"),
        cgroup_memory_max=cgroup_value("memory.max"), requested_resources=dict(gpus=0, cpus=4, memory_gib=16),
        status="passed" if count == 0 and not available and not initialized else "rejected_nonzero_or_initialized_cuda")
    write(output, receipt)
    return 0 if receipt["status"] == "passed" else 3


def junit_receipt(path, selected):
    document = ET.parse(path).getroot()
    cases = list(document.iter("testcase"))
    counts = dict(tests=len(cases), failures=0, errors=0, skipped=0)
    modules = {Path(item).stem: 0 for item in selected}
    for case in cases:
        for tag, key in (("failure", "failures"), ("error", "errors"), ("skipped", "skipped")):
            counts[key] += int(case.find(tag) is not None)
        names = case.attrib.get("classname", "").split(".")
        for name in modules:
            if name in names:
                modules[name] += 1
    counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    accepted = bool(cases) and all(modules.values()) and not any(counts[key] for key in ("failures", "errors", "skipped"))
    return dict(path=str(path.resolve()), sha256=sha(path), counts=counts,
                selected_file_case_counts=modules, accepted=bool(accepted))


def validate_reservation(value, output):
    if value.get("schema") != "joint_cpu_software_qualification_v1" or value.get("status") != "ready_pending_owner_review_not_submitted":
        raise ValueError("Reservation is not the pinned owner-review artifact")
    if value["resources"] != dict(gpus=0, cpus=4, memory_gib=16, scheduler_cap_seconds=600):
        raise ValueError("Qualification resource envelope changed")
    if (value["external_timeout_seconds"], value["termination_grace_seconds"], value["scheduler_margin_seconds"]) != (560, 10, 30):
        raise ValueError("Qualification wall envelope changed")
    if Path(value["output"]).resolve() != output.resolve() or output.name != value["run_id"]:
        raise ValueError("Fresh output differs from reviewed reservation")
    if len(value["phases"]) != 2 or [phase["tests"] for phase in value["phases"]] != PHASE_TESTS:
        raise ValueError("Only the two reviewed CPU test selections are allowed")
    if value["phases"][1]["source_commit"] != LIBERO_PIN:
        raise ValueError("LIBERO sole-owner candidate changed")
    if len({str(Path(phase["code_root"]).resolve()) for phase in value["phases"]}) != 2:
        raise ValueError("Phases require separate pinned code roots")
    for phase in value["phases"]:
        if not re.fullmatch(r"[0-9a-f]{40}", phase["source_commit"]):
            raise ValueError("Production owner source pin remains unresolved")
        for relative in phase["tests"] + phase.get("fixture_dependency_files_not_separately_collected", []):
            if not (Path(phase["code_root"]) / relative).is_file():
                raise FileNotFoundError("Missing selected test/fixture dependency: " + relative)


def qualify(args):
    started = time.monotonic()
    output = args.output.resolve()
    summary = dict(status="running", started_at_utc=now(), phases=[], expected_gpu_count=0,
                   robot_actions=0, source_process_controls=0)
    try:
        if platform.system() != "Linux" or socket.gethostname() != args.expected_hostname or not args.expected_hostname.startswith("job-"):
            raise RuntimeError("Run only in the explicit allocated Linux CPU job")
        if (output / "summary.json").exists() or (output / "actual-cpu-allocation.json").exists():
            raise FileExistsError("Qualification output already has execution evidence")
        value = json.loads(args.reservation.read_text())
        validate_reservation(value, output)
        if Path(sys.executable).resolve() != Path(value["python"]).resolve():
            raise RuntimeError("Use the existing qualified Python runtime")
        launcher_root = Path(__file__).resolve().parents[2]
        summary["launcher"] = exact_clean_git(launcher_root, args.launcher_pin)
        summary["reservation"] = dict(path=str(args.reservation.resolve()), sha256=sha(args.reservation))
        summary["source_identities"] = [exact_clean_git(phase["code_root"], phase["source_commit"]) for phase in value["phases"]]
        summary["disk_free_bytes"] = shutil.disk_usage(output).free
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="", PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        environment.pop("PYTHONPATH", None)
        probe_path = output / "actual-cpu-allocation.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--probe-output", str(probe_path),
                   "--expected-hostname", args.expected_hostname]
        with (output / "actual-cpu-allocation.log").open("w") as log:
            receipt = subprocess.run(command, cwd=output, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if receipt.returncode or not probe_path.exists() or json.loads(probe_path.read_text())["status"] != "passed":
            raise RuntimeError("Actual zero-GPU CPU allocation admission failed before tests")
        summary["actual_cpu_allocation"] = dict(path=str(probe_path), sha256=sha(probe_path))
        # The receipt above observes scheduler visibility before this defence.
        environment["CUDA_VISIBLE_DEVICES"] = ""
        for index, phase in enumerate(value["phases"], 1):
            directory = output / f"phase-{index}-{phase['name']}"
            directory.mkdir()
            code = Path(phase["code_root"]).resolve()
            phase_env = dict(environment, PYTHONPATH=os.pathsep.join(str(code / name) for name in ("src", "tests", "scripts/diagnostics")))
            before = exact_clean_git(code, phase["source_commit"])
            junit = directory / "junit.xml"
            command = [sys.executable, "-m", "pytest", "-q", "-rA", "-o", "addopts=",
                "-o", "xfail_strict=true", "-p", "no:cacheprovider",
                "--junitxml=" + str(junit), "--basetemp=" + str(directory / "tmp"), *phase["tests"]]
            write(directory / "intent.json", dict(name=phase["name"], source=before,
                command=command, pythonpath=phase_env["PYTHONPATH"], started_at_utc=now()))
            phase_started = time.monotonic()
            with (directory / "pytest.log").open("w") as log:
                process = subprocess.run(command, cwd=code, env=phase_env, stdout=log, stderr=subprocess.STDOUT)
            result = dict(name=phase["name"], returncode=process.returncode,
                          elapsed_seconds=time.monotonic()-phase_started, log=str(directory / "pytest.log"), source_before=before)
            try:
                result["source_after"] = exact_clean_git(code, phase["source_commit"])
                result["junit"] = junit_receipt(junit, phase["tests"])
                result["accepted"] = process.returncode == 0 and result["junit"]["accepted"]
            except Exception as error:
                result.update(accepted=False, validation_error=dict(type=type(error).__name__, message=str(error)))
            write(directory / "receipt.json", result)
            summary["phases"].append(result)
            write(output / "progress.json", summary)
            # Independent second phase still runs after ordinary test failure.
            # The outer GNU timeout bounds both phases and their fake children.
        summary["launcher_after"] = exact_clean_git(launcher_root, args.launcher_pin)
        summary["status"] = "passed" if all(phase["accepted"] for phase in summary["phases"]) else "failed"
    except Exception as error:
        summary.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        traceback.print_exc()
    summary.update(finished_at_utc=now(), elapsed_seconds=time.monotonic()-started)
    write(output / "summary.json", summary)
    print(json.dumps(dict(status=summary["status"], phases=len(summary["phases"]),
                          elapsed_seconds=summary["elapsed_seconds"]), allow_nan=False), flush=True)
    return 0 if summary["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reservation", type=Path)
    parser.add_argument("--launcher-pin")
    parser.add_argument("--expected-hostname")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe-output", type=Path)
    parser.add_argument("--finish-only", action="store_true")
    parser.add_argument("--external-exit-code", type=int)
    args = parser.parse_args()
    if args.probe_output:
        return probe(args.probe_output, args.expected_hostname)
    if args.finish_only:
        write(args.output / "launch-outcome.json", dict(finished_at_utc=now(), external_exit_code=args.external_exit_code,
            external_timeout_seconds=560, termination_grace_seconds=10, scheduler_cap_seconds=600,
            summary_present=(args.output / "summary.json").exists(),
            interpretation="Timeout/nonzero/partial JUnit is not qualification completion; no automatic retry."))
        return 0
    if not all((args.reservation, args.launcher_pin, args.expected_hostname, args.output)):
        parser.error("Reservation, exact launcher pin, expected job hostname and fresh output are required")
    return qualify(args)


if __name__ == "__main__":
    raise SystemExit(main())
