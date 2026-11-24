"""Complete only the first eligible wrist arm for the immutable batch054 left arm."""
import argparse
from hashlib import file_digest, sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
from unittest.mock import patch
from zipfile import ZipFile

import reference as ref

SCIENCE = "ac28987f221e336cef2ea64dc16dfffc9f0e74c9"
DRAIN = "6e51cd532ec9c5b7ca285a051660bd181374fae8"
RECOVERY = "d6de8c725cfaf2dbb01a6a9c4895197441b8a6af"
RECOVERY_SHA = "c46a5996c99f28b93c965873ed1fd7e40f58e782a972f9afaea1127bfaeb4041"
DRAIN_SHA = "7638ca269111d3bf505038dc7dbb1bba87c6585c68ba0666a2f68262351a8d53"
ORIGINS = {"retry02": ("job-7nq10tvhsewa", 20550, RECOVERY_SHA),
           "retry04": ("job-0ttm7yo8nha0", 5250, RECOVERY_SHA),
           "retry05": ("job-c2yodm5r90kh", 5250, DRAIN_SHA)}
LEFT = "retry02"
BATCH = "batch054"
KEYS = [("spatial_0", i, 0) for i in range(21, 25)]
LEFT_FILES = ("contract.json", "support.json", "support.npz", "opening.npz", "translation.json",
              "translation/traces/retry.npz")
HISTORICAL_JOURNALS = ("support.jsonl", "translation.jsonl", "translation_wrist.jsonl")
ORIGIN_FILES = ("launch.txt", "launcher.sha256", "exit.txt", "gpu.json", "rank0.json")
PIXELS = {f"{kind}/3/obs/pixels/image2" for kind in ("raw", "mapped")}


class PixelRejection(Exception):
    pass


class BudgetExhausted(Exception):
    pass


def sha(path):
    ref.require(path.is_file() and not path.is_symlink(), f"Missing or mutable reference {path}")
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def immutable_checkout(path, pin, scope):
    git = ["git", "-C", str(path)]
    ref.require(subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() == pin,
                "Wrong operational checkout")
    subprocess.run([*git, "diff", "--quiet", "HEAD", "--", scope], check=True)


def admission(where, expected_units, terminal):
    job, budget, launcher = ORIGINS[where.name]
    host = job + "-master-0"
    pin, seen_host, phase, count, started, seconds = (where / "launch.txt").read_text().split()
    ref.require((pin, seen_host, phase, count, int(seconds)) == (SCIENCE, host, "full", "1", budget),
                "Source launch contract differs")
    digest, launcher_path = (where / "launcher.sha256").read_text().split(maxsplit=1)
    ref.require(digest == launcher == sha(Path(launcher_path.strip())), "Source launcher hash differs")
    gpu = ref.read(where / "gpu.json")
    rank = ref.read(where / "rank0.json")
    device, = gpu["devices"]
    ref.require(gpu["hostname"] == host and gpu["status"] == "passed" and gpu["purpose"] == "training" and
                gpu["allocation_type"] == "baige_container" and gpu["exclusive_training_admission"] and
                device["cuda_name"] == "NVIDIA GeForce RTX 4090" and device["allocation_ordinal"] == 0 and
                rank["device"] == device and rank["rank"] == 0 and rank["phase"] == "full" and
                rank["batches"] == expected_units and isinstance(rank["egl"], int), "Source native admission differs")
    if terminal:
        code, start, ended, cards, total = map(int, (where / "exit.txt").read_text().split())
        ref.require(code == 1 and int(started) == start and cards == 1 and 0 < total == ended - start <= budget,
                    "Source partial attempt lacks expected terminal receipt")
        with (where / "rank0.log").open() as log:
            strict = guard = False
            for line in log:
                strict |= "All keys loaded successfully!" in line
                guard |= "Paired opening differs at raw/3/obs/pixels/image2" in line
        ref.require(strict and guard, "Source strict native policy or raw mismatch receipt is absent")
    return host, rank["egl"]


def validate_left(path, retry):
    # The full validator first validates support/opening and then loops over arms.
    # Keep native arm indexing and treatment dispatch, and visit only the available arm.
    class OriginalArms(tuple):
        def __iter__(self):
            yield self[0]

    with patch.object(retry, "ARMS", OriginalArms(retry.ARMS)):
        retry.validate(path, KEYS)


def load_opening(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as tape:
        return {key: tape[key] for key in tape.files}


def receipts(arrays):
    return {key: dict(dtype=str(value.dtype), shape=list(value.shape), sha256=sha256(value.tobytes()).hexdigest())
            for key, value in arrays.items()}


def differences(expected, inputs, identical):
    return sorted((expected.keys() ^ inputs.keys()) | {key for key in expected.keys() & inputs.keys()
                                                      if not identical(expected[key], inputs[key])})


def diagnose(earliest, alternate, identical):
    import numpy as np
    raw = "raw/3/obs/pixels/image2"
    mapped = "mapped/3/obs/pixels/image2"
    ref.require(set(differences(earliest, alternate, identical)) == PIXELS and
                identical(earliest[raw], earliest[mapped]) and identical(alternate[raw], alternate[mapped]),
                "Known opening fingerprints differ outside the wrist RGB")
    a, b = earliest[raw], alternate[raw]
    ref.require(a.shape == b.shape == (256, 256, 3) and a.dtype == b.dtype == np.uint8 and
                np.argwhere(np.any(a != b, axis=-1)).tolist() == [[79, 153]] and
                np.abs(a[79, 153].astype(int) - b[79, 153]).tolist() == [1, 1, 1],
                "Known diagnostic pixel fingerprint differs")


def source_hashes(store, retry, units, journal):
    result, tapes = {}, {}
    for name in (LEFT, "retry04"):
        where = store / "attempts" / name
        path = where / BATCH
        admission(where, units, terminal=True)
        ref.require(not (path / "translation_wrist.json").exists() and
                    not (path / "translation_wrist").exists(), "Source wrist arm already scored")
        for filename in ("support.npz", "opening.npz", "translation/traces/retry.npz"):
            with ZipFile(path / filename) as archive:
                ref.require(archive.testzip() is None, "Damaged original native tape")
        validate_left(path, retry)
        left = ref.read(path / "translation.json")
        contract = ref.read(path / "contract.json")
        ref.require(journal(path / "support.jsonl", "support") == contract["support_budget"] and
                    journal(path / "translation.jsonl", "translation") == left["budget"] and
                    journal(path / "translation_wrist.jsonl", "translation_wrist") == {"reset": 4} and
                    left["opening_sha256"] == sha(path / "opening.npz"), "Source native interaction journal differs")
        tapes[name] = load_opening(path / "opening.npz")
        source_files = {name: sha(where / name) for name in ORIGIN_FILES}
        source_files.update({f"{BATCH}/{name}": sha(path / name) for name in LEFT_FILES + HISTORICAL_JOURNALS})
        result[name] = source_files
    diagnose({k:v for k,v in tapes[LEFT].items() if k != "proposal"},
             {k:v for k,v in tapes["retry04"].items() if k != "proposal"}, retry.identical)
    return result, tapes


def validate_previous(store, retry, grid, units, drain):
    admitted, original, timings, all_groups = {}, [], [], []
    for group in sorted((store / "batches").glob("batch[0-9][0-9][0-9]")):
        number = int(group.name[5:])
        producer = ref.read(group / "producer.json")
        name = producer["attempt"]
        ref.require(name in ("retry02", "retry05") and number in units and number != 54,
                    "Prior producer or enrollment is unregistered")
        scheduled = units if name == "retry02" else list(range(55, 70))
        if name not in admitted:
            admitted[name] = admission(store / "attempts" / name, scheduled, terminal=(name == "retry02"))
        host, egl = admitted[name]
        ref.require(number in scheduled and producer["host"] == host and producer["egl"] == egl and
                    producer["rank"] == 0 and producer["code"] == SCIENCE, "Prior producer admission differs")
        sidecar = group / "dispatch.json"
        if name == "retry05":
            receipt = ref.read(sidecar)
            ref.require(receipt["dispatch_pin"] == DRAIN and receipt["scheduler_id"] == ORIGINS[name][0],
                        "Drain dispatch receipt differs")
        drain.validate_batch(group, grid[number], retry, published=True)
        if name == "retry02":
            original.append(number)
            timings.append(producer["ended"] - producer["started"])
        all_groups.append(number)
    ref.require(len(original) == 54 and all(math.isfinite(n) and n > 0 for n in timings),
                "Original sealed cohort or native timings differ")
    return dict(original=original, verified=all_groups, whole_batch_max_seconds=math.ceil(max(timings)))


def open_candidate(work, candidate, vla, support, retry, earliest, alternate, deadline, reserve, attempt):
    """Call the unchanged guard in the native runner and preserve its compared leaves."""
    from reground.envs.libero_env import max_steps
    from reground.eval import jobs, rollout

    right = retry.ARMS[1]
    began = time.monotonic()
    record = {"candidate": candidate.name, "status": "structural_failure"}
    ref.write(candidate / "begin.json", dict(at_epoch=time.time(), remaining_seconds=deadline-began))
    marker = attempt / "sequential-first-wrist-control.json"
    exact = False

    class AuditedRetry(retry.Retry):
        def call(self, idx, op, args):
            if op == "step" and not marker.exists():
                ref.require(exact, "Wrist step before exact opening")
                with marker.open("x") as stream:
                    json.dump(dict(candidate=candidate.name, epoch=time.time()), stream)
                    stream.flush()
                    os.fsync(stream.fileno())
            return super().call(idx, op, args)

        def predict_chunk(self, obs, tasks, seeds, query):
            nonlocal exact
            if query:
                return super().predict_chunk(obs, tasks, seeds, query)
            inputs = self.opening
            try:
                proposal = super().predict_chunk(obs, tasks, seeds, query)
            except ValueError as error:
                changed = differences(earliest, inputs, retry.identical)
                record.update(accepted_guard=False, opening=receipts(inputs), differences=changed,
                              before_controls=(not marker.exists()))
                if (str(error) == "Paired opening differs at raw/3/obs/pixels/image2" and not marker.exists()
                        and set(changed) == PIXELS
                        and all(retry.identical(inputs[key], alternate[key]) for key in PIXELS)):
                    raise PixelRejection from error
                raise
            record.update(accepted_guard=True, opening=receipts(inputs), differences=[])
            ref.require(not differences(earliest, inputs, retry.identical), "Native guard readback differs")
            if time.monotonic() + reserve > deadline:
                raise BudgetExhausted("Exact opening found after the terminal-rollout reserve cutoff")
            exact = True
            return proposal

    try:
        with retry.fresh(candidate, right, len(KEYS)) as (pool, meter):
            initial_pids = [proc.pid for proc in pool.native.procs]
            bridge = AuditedRetry(vla, pool, KEYS, support, right, work)
            original_call = bridge.call

            def checked_call(idx, op, args):
                result = original_call(idx, op, args)
                if op == "reset":
                    ref.require(initial_pids == [proc.pid for proc in pool.native.procs] and
                                all(row["max_steps"] == max_steps(KEYS[i][0]) for i, row in enumerate(result)),
                                "Candidate reset respawned or changed the native horizon")
                return result

            bridge.call = checked_call
            children = tuple(pool.native.procs)
            try:
                episodes = [rollout.Episode(*key) for key in KEYS]
                speed_limit = jobs.speed_limit("pi05")
                ref.require(speed_limit and speed_limit > 0, "Missing native safety envelope")
                result = rollout.run_batch(bridge, bridge, episodes, bridge, retry.CELL, speed_limit=speed_limit)
            finally:
                record.update(pids=initial_pids, budget=meter.counts.copy())
        ref.require(marker.exists() and exact, "Native terminal return had no eligible first control")
        jobs._save_traces(work / right / "traces/retry.npz", result.traces)
        ref.write(work / f"{right}.json", dict(rows=result.rows, arm=right, budget=meter.counts,
                  queries=bridge.queries, opening_sha256=sha(work / "opening.npz"), speed_limit=speed_limit))
        record["status"] = "terminal"
    except PixelRejection:
        ref.require(record["budget"] == {"reset": len(KEYS)}, "Rejected candidate recorded extra interactions")
        record["status"] = "ineligible_pixel"
    except BudgetExhausted:
        ref.require(not marker.exists(), "Budget exhausted after first wrist control")
        record["status"] = "exact_without_terminal_reserve"
        raise
    finally:
        for proc in locals().get("children", ()):
            proc.join(timeout=5)
        record.update(elapsed_seconds=time.monotonic()-began,
                      child_exitcodes=[proc.exitcode for proc in locals().get("children", ())])
        ref.write(candidate / "candidate.json", record)
        ref.require(all(code == 0 for code in record["child_exitcodes"]), "Candidate pool did not exit cleanly")
    return record


def stage_left(store, work, origins):
    source = store / "attempts" / LEFT / BATCH
    work.mkdir()
    for name in LEFT_FILES:
        src, dest = source / name, work / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        ref.require(sha(dest) == origins[LEFT][f"{BATCH}/{name}"], "Adopted left bytes differ")


def check_sources_unchanged(store, origins):
    for name, files in origins.items():
        where = store / "attempts" / name
        ref.require(all(sha(where / filename) == digest for filename, digest in files.items()),
                    "Source bytes changed during recovery")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    store, attempt = args.store, args.attempt
    ref.configure()
    import retry
    import worker
    from reground.eval import jobs
    import symmeta_adapter as stream

    ref.require(os.environ["RETRY_PIN"] == SCIENCE and Path(retry.__file__).resolve().parent.name == f"src-{SCIENCE}",
                "Unexpected scientific import path")
    drain_path = store.parent / f"drain-src-{DRAIN}"
    immutable_checkout(drain_path, DRAIN, "recovery/drain")
    sys.path.insert(0, str(drain_path / "recovery/drain"))
    import dispatch as drain

    grid = retry.schedule()
    units = worker.selected(grid, "full")
    ref.require(grid[54] == KEYS and len(grid) == 70, "Frozen enrollment differs")
    if not args.verify_only:
        ref.require(not list((store / "attempts").glob("*/sequential-first-wrist-control.json")),
                    "A prior sequential attempt already dispatched a first wrist control")
        ref.require(not (store / "batches" / BATCH).exists(), "Batch054 already sealed")
    origins, tapes = source_hashes(store, retry, units, drain.journal)
    earliest = {key: value for key, value in tapes[LEFT].items() if key != "proposal"}
    alternate = {key: tapes["retry04"][key] for key in PIXELS}
    validation_started = time.monotonic()
    previous = validate_previous(store, retry, grid, units, drain)
    validation_seconds = math.ceil(time.monotonic() - validation_started)
    checkpoint = Path(os.environ["SYMMETA_CHECKPOINT"])
    ref.require(sha(checkpoint) == stream.WEIGHTS, "Original seed-41 checkpoint bytes differ")
    ref.write(attempt / "preflight.json", dict(source_attempt=LEFT, source_sha256=origins, cohort=previous,
              keys=KEYS, science_pin=SCIENCE, prior_validation_seconds=validation_seconds, seed41_sha256=stream.WEIGHTS, native_policy=ref.POLICY,
              proposal=receipts({"proposal": tapes[LEFT]["proposal"]})["proposal"], exact_leaves=receipts(earliest)))
    if args.verify_only:
        return 0

    import torch
    from fluffy_bassoon.baige_gpu import cuda_pci, pci

    gpu = ref.read(attempt / "gpu.json")
    device, = gpu["devices"]
    host = socket.gethostname()
    ref.require(gpu["hostname"] == host and gpu["status"] == "passed" and gpu["exclusive_training_admission"] and
                gpu["allocation_type"] == "baige_container" and gpu["purpose"] == "training" and
                torch.cuda.device_count() == 1 and device["allocation_ordinal"] == 0 and
                pci(device["pci_bus_id"]) == cuda_pci(0) and
                torch.cuda.get_device_name(0) == device["cuda_name"] == "NVIDIA GeForce RTX 4090" and
                str(torch.cuda.get_device_properties(0).uuid) == device["cuda_uuid"], "Wrong native GPU")
    torch.cuda.set_device(0)
    egl = ref.egl_device(0)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl)
    ref.write(attempt / "rank0.json", dict(rank=0, egl=egl, phase="full", batches=[54], device=device))
    work = attempt / BATCH
    stage_left(store, work, origins)
    started = time.time()
    payload = int(os.environ["SEQ_PAYLOAD_SECONDS"])
    deadline = time.monotonic() + int(os.environ["SEQ_STARTED_EPOCH"]) + payload - time.time()
    reserve = previous["whole_batch_max_seconds"] + validation_seconds
    support = ref.read(work / "support.json")
    ctx = jobs.WorkerCtx("pi05", "cuda:0", len(KEYS), lambda m: print(m, flush=True))
    tried, longest = [], 0
    try:
        if time.monotonic() + reserve >= deadline:
            raise BudgetExhausted("No time for conservative terminal-rollout reserve")
        ref.strict_policy(ctx)
        while time.monotonic() + reserve + longest < deadline:
            candidate = attempt / f"{BATCH}-reset-{len(tried)+1:04d}"
            candidate.mkdir()
            record = open_candidate(work, candidate, ctx.vla, support, retry, earliest, alternate,
                                    deadline, reserve, attempt)
            tried.append(record)
            longest = max(longest, math.ceil(record["elapsed_seconds"]))
            if record["status"] == "terminal":
                break
            ref.require(record["status"] == "ineligible_pixel", "Unexpected retry condition")
        else:
            raise BudgetExhausted("No further candidates fit the remaining search budget")
    except BudgetExhausted as error:
        ref.write(attempt / "sequential-status.json", dict(state="budget_exhausted_before_wrist_control",
                  reason=str(error), wall_seconds=payload, terminal_reserve_seconds=reserve,
                  candidate_records=len(list(attempt.glob(f"{BATCH}-reset-*/candidate.json")))))
        return 3
    finally:
        ctx.close()

    accepted = attempt / tried[-1]["candidate"]
    right_journal = "translation_wrist.jsonl"
    (accepted / right_journal).rename(work / right_journal)
    retry.validate(work, KEYS)
    check_sources_unchanged(store, origins)
    post = validate_previous(store, retry, grid, units, drain)
    ref.require(post == previous, "Prior sealed cohort changed during isolated recovery")
    ref.require(not (store / "batches" / BATCH).exists(), "Competing batch054 publication")
    ref.write(work / "producer.json", dict(rank=0, egl=egl, host=host, attempt=attempt.name,
              code=SCIENCE, started=started, ended=time.time()))
    retained = (*LEFT_FILES, "translation_wrist.json", "translation_wrist/traces/retry.npz", right_journal,
                "producer.json")
    receipt = dict(batch=54, keys=KEYS, source_attempt=LEFT, science_pin=SCIENCE,
                   operational_pin=os.environ["SEQ_PIN"], native_primary_file=sha(Path(retry.__file__)),
                   seed41_sha256=stream.WEIGHTS, source_sha256=origins, prior_producers=post,
                   source_journals={key: f"attempts/{LEFT}/{BATCH}/{key}.jsonl" for key in ("support", "translation")},
                   accepted=accepted.name, candidate_sha256={item["candidate"]: sha(attempt / item["candidate"] / "candidate.json")
                                                            for item in tried},
                   first_control_marker_sha256=sha(attempt / "sequential-first-wrist-control.json"),
                   files={name: sha(work / name) for name in retained}, wall_seconds=payload,
                   terminal_reserve_seconds=reserve, shell_receipt_sha256=sha(attempt / "launcher.sha256"))
    ref.write(work / "sequential.json", receipt)
    work.rename(store / "batches" / BATCH)
    ref.write(attempt / "sequential-status.json", dict(state="published", sealed_groups=1+len(post["verified"]),
              aggregate_ready=len(post["verified"]) == 69, accepted=accepted.name, candidates=len(tried),
              receipt_sha256=sha(store / "batches" / BATCH / "sequential.json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
