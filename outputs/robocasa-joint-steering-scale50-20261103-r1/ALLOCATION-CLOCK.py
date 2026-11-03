#!/usr/bin/env python3
"""Bind a fresh attempt to scheduler evidence before any GPU/model work.

This is run bookkeeping, not a scheduler client. The owner observes the normal
console and writes ALLOCATION.json once. A fresh queued observation before the
allocation is a conservative lower bound; a dated scheduler allocation event
may provide an earlier bound. The whole admitted interval is charged.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    run = Path(sys.argv[1])
    submitted = float(os.environ["TASK_SUBMITTED_UNIX"])
    started = float(os.environ["TASK_PROCESS_STARTED"])
    expected = os.environ["TASK_EXPECTED_JOB_NAME"]
    assert os.environ["AIHC_JOB_NAME"] == expected
    assert math.isfinite(submitted) and submitted <= started <= time.time()
    request_path = run / "ALLOCATION-REQUEST.json"
    request = dict(expected_job_name=expected, hostname=socket.gethostname(),
        process_started_unix=started, queue_submitted_unix=submitted,
        requested_GPUs=8, requested_CPU=168, requested_RAM_GiB=1664,
        requested_SHM_GiB=384, native_source_commit=os.environ["TASK_CODE_PIN"],
        launcher_sha256=digest(os.environ["TASK_WRAPPER_SOURCE"]),
        no_GPU_probe_model_or_physical_calls_started=True,
        receipt_deadline_unix=started + 300)
    write_once(request_path, request)
    receipt_path = run / "ALLOCATION.json"
    while not receipt_path.exists():
        if time.time() >= started + 300:
            write_once(run / "ALLOCATION-CLOCK-FAILED.json", dict(
                reason="No verified scheduler allocation receipt within300seconds",
                process_started_unix=started, ended_unix=time.time(),
                no_GPU_probe_model_or_physical_calls_started=True,
                no_automatic_retry=True))
            raise TimeoutError("Require owner-verified allocation clock")
        time.sleep(1)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["request_sha256"] == digest(request_path)
    assert receipt["expected_job_name"] == expected
    assert receipt["hostname"] == socket.gethostname()
    assert receipt["process_started_unix"] == started
    assert isinstance(receipt["scheduler_job_id"], str) and receipt["scheduler_job_id"].startswith("job-")
    assert receipt["observed_requested_GPUs"] == 8
    assert receipt["observed_allocated_instance"] is True
    assert receipt["evidence_path"] and receipt["evidence_sha256"]
    # The pre-refresh time of a freshly verified QUEUED state is before its
    # allocation. Charging from that lower bound counts the full allocation,
    # plus at most the final observation interval of unallocated queue time.
    bounds = [receipt[key] for key in ("last_verified_queued_lower_bound_unix",
        "scheduler_allocation_lower_bound_unix") if receipt.get(key) is not None]
    assert bounds and all(isinstance(value, (float, int)) and math.isfinite(value)
        and submitted <= value <= started for value in bounds)
    origin = min(bounds)
    assert receipt["accounting_start_unix"] == origin
    write_once(run / "ALLOCATION-CLOCK.json", dict(
        accounting_start_unix=origin, queue_submitted_unix=submitted,
        process_started_unix=started, receipt_path=str(receipt_path),
        receipt_sha256=digest(receipt_path), request_sha256=digest(request_path),
        scheduler_job_id=receipt["scheduler_job_id"],
        unallocated_queue_seconds_excluded=origin - submitted,
        accounting_semantics="All8GPUs charged from conservative verified allocation lower bound; startup, receipt wait and release remain charged"))


if __name__ == "__main__":
    main()
