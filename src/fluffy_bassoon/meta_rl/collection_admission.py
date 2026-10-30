"""Atomic replay admission and fail-closed recovery for execution groups."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
import hashlib
import fcntl
from pathlib import Path


RESUME_BLOCKER = "ORDINARY-RESUME-BLOCKED.json"


def assert_resume_allowed(path, *, checkpoint=None):
    """Check before constructing a GPU runtime, including relocated recoveries."""
    path = Path(path).resolve()
    root = path if path.is_dir() else path.parent
    if (root / RESUME_BLOCKER).exists():
        raise RuntimeError("Ordinary resume blocked by a quarantined collection group; forensic recovery required")
    if checkpoint is not None:
        states = checkpoint.get("states", {})
        if (checkpoint.get("resume_blocked") or checkpoint.get("rejected_group")
                or checkpoint.get("metadata", {}).get("resume_blocked")
                or any(state.get("rejected_group") for state in states.values())):
            raise RuntimeError("Ordinary resume blocked by rejected-group checkpoint metadata")
        vector_root=checkpoint.get("parallel_vector_root")
        if vector_root:
            canonical=Path(vector_root)
            if not canonical.is_dir():
                raise RuntimeError("Canonical parallel collection journal is unavailable")
            for journal in canonical.glob("group-*.json"):
                if json.loads(journal.read_text()).get("status") not in ("checkpointed","cancelled_before_dispatch"):
                    raise RuntimeError("Uncheckpointed native group blocks ordinary resume, including relocated copies")
        group=checkpoint.get("parallel_collection_group")
        if group:
            journal=Path(group["journal_path"])
            committed=json.loads(journal.read_text())
            checkpoint_path=path/"coordinator-recovery.pt" if path.is_dir() else path
            digest=hashlib.sha256()
            with checkpoint_path.open("rb") as stream:
                for block in iter(lambda:stream.read(1<<20),b""):digest.update(block)
            if (committed.get("status")!="checkpointed" or committed.get("group_id")!=group.get("group_id")
                    or committed.get("checkpoint_sha256")!=digest.hexdigest()):
                raise RuntimeError("Ordinary resume blocked by an unacknowledged physical collection group")


def campaign_checkpoint_fence(budget):
    snapshot=budget.snapshot()
    return dict(canonical_root=str(budget.root.resolve()),
        **{key:snapshot[key] for key in ("journal_events","journal_sha256","charged_actions","decision_count")})


def assert_campaign_checkpoint_fence(campaign_root, checkpoint, *, live_budget=None):
    """Read canonical history before GPU setup; stale copies cannot refund it."""
    fence=checkpoint.get("campaign_journal_fence")
    root=Path(campaign_root).resolve()
    if not fence or fence.get("canonical_root")!=str(root):
        raise RuntimeError("Parallel recovery requires its original canonical campaign journal fence")
    lock=None
    try:
        if live_budget is None:
            lock=(root/"campaign.lock").open("r")
            try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another process owns the canonical campaign; resume is blocked") from error
        else:
            live_budget._require_open()
            if live_budget.root.resolve()!=root:
                raise RuntimeError("Live campaign differs from the recovery fence")
        paths=sorted((root/"events").glob("*.json"))
        limit=fence["journal_events"]
        if type(limit) is not int or limit<1 or len(paths)<limit:
            raise RuntimeError("Canonical campaign journal is shorter than the recovery fence")
        previous=None; charged=0; decisions={}
        for index,path in enumerate(paths):
            if path.name!=f"{index:010d}.json":
                raise RuntimeError("Canonical campaign journal sequence is incomplete")
            event=json.loads(path.read_text());recorded=event.pop("sha256",None)
            encoded=json.dumps(event,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
            digest=hashlib.sha256(encoded).hexdigest()
            if recorded!=digest or event.get("sequence")!=index or event.get("previous_sha256")!=previous:
                raise RuntimeError("Canonical campaign journal integrity failed")
            previous=digest
            if index>=limit:
                if event.get("kind") not in ("run",):
                    raise RuntimeError("Uncheckpointed physical campaign events block ordinary resume")
                continue
            kind=event["kind"]
            if kind=="reserve":
                granted=event["granted"];decisions[event["decision_id"]]=granted;charged+=granted
            elif kind=="resolve":
                charged+=event["actual_actions"]-decisions[event["decision_id"]]
                decisions[event["decision_id"]]=event["actual_actions"]
            elif kind=="receipt":charged+=event["receipt"]["actual_actions"]
            if index==limit-1 and (digest!=fence["journal_sha256"]
                    or charged!=fence["charged_actions"] or len(decisions)!=fence["decision_count"]):
                raise RuntimeError("Canonical campaign fence hash/counters differ from recovery")
    finally:
        if lock is not None:
            fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()


def _fsync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def admit_collected_group(state, collected, *, output_dir, method, capture_rng,
                          save_payload, log_prob_tolerance=2e-4, value_tolerance=1e-5):
    """Charge every executed lane before validating; admit all lanes or none.

    `collected` already satisfies the collector's exact physical-count/task
    contract. `save_payload(payload,path)` is normally torch.save. It is
    injectable so lifecycle/admission tests need no GPU runtime or simulator.
    """
    if state.get("rejected_group"):
        raise RuntimeError("A rejected collection group already blocks this learner")
    if not collected:
        raise ValueError("A collected execution group cannot be empty")
    learner = state["learner"]
    total = sum(int(result["primitive_steps"]) for _, result in collected)
    state["steps"] += total
    for _, result in collected:
        state["task_steps"][result["task"]] += int(result["primitive_steps"])
    learner.primitive_steps = state["steps"]
    # Preserve the collection-end streams even if a faulty replay validator
    # itself consumes randomness. No policy/optimizer update occurs here.
    rng = capture_rng()
    sampler = deepcopy(learner.sampler.bit_generator.state)
    schedule = deepcopy(state["schedule"].state_dict())
    errors = []
    rejected = False
    for rollout, _ in collected:
        try:
            metrics = learner.replay_error(rollout)
            log_error = float(metrics["log_prob_max_error"])
            value_error = float(metrics["value_max_error"])
            valid = (math.isfinite(log_error) and math.isfinite(value_error)
                     and 0 <= log_error <= log_prob_tolerance
                     and 0 <= value_error <= value_tolerance)
            errors.append(dict(metrics=metrics, passed=valid))
            rejected |= not valid
        except Exception as error:
            rejected = True
            errors.append(dict(passed=False, error=f"{type(error).__name__}: {error}"))
    if not rejected:
        state["pending"].extend(collected)
        return errors

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(method=method, lanes=len(collected), primitive_steps=total,
                    source_steps=state["steps"], quarantine_file="quarantine.pt",
                    resume_blocked=True, status="quarantine_pending")
    state["rejected_group"] = metadata
    # The durable blocker precedes the larger payload: a failed payload write
    # must never make an ordinary checkpoint look safe to resume.
    _atomic_json(output_dir / RESUME_BLOCKER, metadata)
    payload = dict(format="rejected_execution_group_v1", metadata=metadata,
                   collected_group=collected, replay_errors=errors,
                   learner=learner.state_dict(), sampler=sampler, rng=rng,
                   schedule=schedule, source_steps=state["steps"],
                   task_steps=deepcopy(state["task_steps"]),
                   previously_pending_rollouts=list(state["pending"]))
    temporary = output_dir / "quarantine.pt.tmp"
    save_payload(payload, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(output_dir / "quarantine.pt")
    _fsync_directory(output_dir)
    metadata["status"] = "quarantined"
    _atomic_json(output_dir / RESUME_BLOCKER, metadata)
    raise RuntimeError("Batched old-policy replay failed; entire executed group quarantined and ordinary resume blocked")


def quarantine_collection_failure(state, *, output_dir, method, error, capture_rng,
                                  save_payload, details=None, collected=None):
    """Block incomplete native/corpus groups without refunding physical work."""
    if state.get("rejected_group"):
        return
    if collected:
        state["steps"]+=sum(int(result["primitive_steps"]) for _,result in collected)
        for _,result in collected:
            state["task_steps"][result["task"]]+=int(result["primitive_steps"])
        state["learner"].primitive_steps=state["steps"]
    root=Path(output_dir).resolve()
    root.mkdir(parents=True,exist_ok=True)
    metadata=dict(method=method,status="collection_failed",resume_blocked=True,
        reason=f"{type(error).__name__}: {error}",source_steps=state["steps"],
        quarantine_file="collection-failure.pt")
    state["rejected_group"]=metadata
    _atomic_json(root/RESUME_BLOCKER,metadata)
    payload=dict(format="failed_execution_group_v1",metadata=metadata,details=details,
        collected_group=collected,learner=state["learner"].state_dict(),rng=capture_rng(),
        sampler=deepcopy(state["learner"].sampler.bit_generator.state),
        schedule=deepcopy(state["schedule"].state_dict()),source_steps=state["steps"],
        task_steps=deepcopy(state["task_steps"]),previously_pending_rollouts=list(state["pending"]))
    temporary=root/"collection-failure.pt.tmp"
    save_payload(payload,temporary)
    with temporary.open("rb") as stream: os.fsync(stream.fileno())
    temporary.replace(root/"collection-failure.pt")
    _fsync_directory(root)
