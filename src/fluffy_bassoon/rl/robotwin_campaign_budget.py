"""Durable action accounting shared by all runs in one RoboTwin campaign.

Reserve before issuing simulator control. A completed receipt releases unused
actions; a crashed/unresolved call keeps its entire reservation. Journal
events are immutable, fsynced files. The replaceable snapshot is only a view:
recovery always replays the journal while holding a process-exclusive lock.
No model or simulator dependency is imported here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


class CampaignBudgetError(RuntimeError):
    pass


@dataclass(frozen=True)
class Reservation:
    decision_id: str
    granted: int
    execute: bool
    status: str
    actual_actions: int | None = None


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _count(value, name, *, positive=False):
    if type(value) is not int or value < int(positive):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'} integer")
    return value


def _name(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class CampaignBudget:
    """Serial campaign ledger, including prior runs and non-training actions.

    ``source_caps`` limits each method only in phase ``source``. Optional
    ``phase_caps`` additionally limits aggregate validation/evaluation phases.
    Every phase counts toward ``total_actions``. Callers must inspect
    ``Reservation.execute``: replaying a recorded ID never authorizes another
    simulator call. Each allocated run should use a unique run_id.
    """
    schema = "robotwin_campaign_actions_v1"

    def __init__(self, root, *, campaign_id, total_actions, source_caps, contract,
                 run_id, source_commit, phase_caps=None):
        self.root = Path(root)
        self.run_id = _name(run_id, "run_id")
        self.source_commit = _name(source_commit, "source_commit")
        if not re.fullmatch(r"[0-9a-f]{40}", self.source_commit):
            raise ValueError("source_commit must be a full lowercase Git commit")
        caps = {str(k): _count(v, "source cap") for k, v in source_caps.items()}
        phases = {str(k): _count(v, "phase cap") for k, v in (phase_caps or {}).items()}
        for key in (*caps, *phases):
            _name(key, "cap name")
            if "/" in key:
                raise ValueError("Cap names cannot contain slashes")
        if not isinstance(contract, dict):
            raise ValueError("contract must be a JSON object")
        self.header = json.loads(_canonical(dict(
            schema=self.schema, campaign_id=_name(campaign_id, "campaign_id"),
            total_actions=_count(total_actions, "total_actions", positive=True),
            source_caps=caps, phase_caps=phases, contract=contract)))
        self.closed = False
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = (self.root / "campaign.lock").open("a+")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock.close()
            self.closed = True
            raise CampaignBudgetError("Another process owns this campaign ledger") from error
        try:
            self.events = self.root / "events"
            self.events.mkdir(exist_ok=True)
            self._sequence = 0
            self._previous_hash = None
            self._state = dict(decisions={}, receipts={}, runs={})
            self._unresolved = {}
            self._counts = dict(charged_actions=0, actual_actions=0,
                                uncertain_actions=0, pending_actions=0,
                                by_phase={}, by_method={})
            files = sorted(self.events.glob("*.json"))
            if not files:
                self._record(dict(kind="header", header=self.header))
            else:
                for index, path in enumerate(files):
                    if path.name != f"{index:010d}.json":
                        raise CampaignBudgetError("Campaign journal sequence has a gap or unknown file")
                    event = json.loads(path.read_text())
                    recorded_hash = event.pop("sha256", None)
                    actual_hash = hashlib.sha256(_canonical(event).encode()).hexdigest()
                    if (recorded_hash != actual_hash or event.get("sequence") != index
                            or event.get("previous_sha256") != self._previous_hash):
                        raise CampaignBudgetError(f"Campaign journal integrity failed: {path.name}")
                    self._apply(event)
                    self._previous_hash = actual_hash
                    self._sequence += 1
            previous_run = self._state["runs"].get(self.run_id)
            if previous_run is not None and previous_run != self.source_commit:
                raise CampaignBudgetError("A recorded run_id cannot change its source commit")
            # The former process no longer holds the lock. Pending simulator
            # intents may have executed, so recovery cannot safely refund them.
            for decision_id, decision in list(self._state["decisions"].items()):
                if decision["status"] == "pending":
                    self._record(dict(kind="uncertain", decision_id=decision_id,
                                      recovered_by_run=self.run_id))
            if previous_run is None:
                self._record(dict(kind="run", run_id=self.run_id,
                                  source_commit=self.source_commit))
            self._write_snapshot()
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if not self.closed:
            self.closed = True
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()

    def _require_open(self):
        if self.closed:
            raise CampaignBudgetError("Campaign ledger is closed")

    def _record(self, payload):
        self._require_open()
        event = dict(payload, sequence=self._sequence,
                     previous_sha256=self._previous_hash,
                     created_at_utc=datetime.now(timezone.utc).isoformat())
        digest = hashlib.sha256(_canonical(event).encode()).hexdigest()
        encoded = (_canonical(dict(event, sha256=digest)) + "\n").encode()
        descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=self.events)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            # link() refuses to replace a prior event, unlike replace(). A
            # partial temporary is never interpreted as a committed intent.
            os.link(temporary, self.events / f"{self._sequence:010d}.json")
            _fsync_directory(self.events)
        finally:
            os.unlink(temporary)
        self._apply(event)
        self._previous_hash = digest
        self._sequence += 1
        self._write_snapshot()

    def _apply(self, event):
        kind = event["kind"]
        if kind == "header":
            if self._sequence != 0 or event["header"] != self.header:
                raise CampaignBudgetError("Campaign contract or action limits changed")
        elif kind == "run":
            if event["run_id"] in self._state["runs"]:
                raise CampaignBudgetError("Duplicate run registration in campaign journal")
            self._state["runs"][event["run_id"]] = event["source_commit"]
        elif kind == "reserve":
            identity = event["decision_id"]
            if identity in self._state["decisions"]:
                raise CampaignBudgetError("Duplicate simulator intent in campaign journal")
            self._state["decisions"][identity] = dict(event["request"],
                granted=event["granted"], status="pending", actual_actions=None)
            self._unresolved[identity] = self._state["decisions"][identity]
            self._add_charge(self._state["decisions"][identity])
        elif kind == "resolve":
            decision = self._state["decisions"][event["decision_id"]]
            if decision["status"] != "pending":
                raise CampaignBudgetError("Only a pending simulator intent can resolve")
            actual = _count(event["actual_actions"], "actual_actions")
            if actual > decision["granted"]:
                raise CampaignBudgetError("Executed actions exceed the reserved upper bound")
            self._add_charge(decision, -1)
            decision.update(status="resolved", actual_actions=actual)
            self._add_charge(decision)
            self._unresolved.pop(event["decision_id"])
        elif kind == "uncertain":
            decision = self._state["decisions"][event["decision_id"]]
            if decision["status"] != "pending":
                raise CampaignBudgetError("Only a pending intent becomes uncertain")
            self._add_charge(decision, -1)
            decision["status"] = "uncertain"
            self._add_charge(decision)
        elif kind == "receipt":
            identity = event["import_id"]
            if identity in self._state["receipts"]:
                raise CampaignBudgetError("Duplicate prior receipt in campaign journal")
            self._state["receipts"][identity] = deepcopy(event["receipt"])
            self._add_charge(self._state["receipts"][identity])
        else:
            raise CampaignBudgetError(f"Unknown campaign event kind: {kind}")
        if self._counts["charged_actions"] > self.header["total_actions"]:
            raise CampaignBudgetError("Campaign journal exceeds its total action limit")

    def _add_charge(self, item, multiplier=1):
        status = item.get("status", "resolved")
        charge = multiplier * (item["actual_actions"] if status == "resolved" else item["granted"])
        self._counts["charged_actions"] += charge
        self._counts["actual_actions" if status == "resolved" else status + "_actions"] += charge
        phase, method = item["phase"], item["method"]
        self._counts["by_phase"][phase] = self._counts["by_phase"].get(phase, 0) + charge
        key = f"{phase}/{method}"
        self._counts["by_method"][key] = self._counts["by_method"].get(key, 0) + charge

    def remaining(self, *, phase, method):
        self._require_open()
        _name(phase, "phase"); _name(method, "method")
        if "/" in phase or "/" in method:
            raise ValueError("phase and method names cannot contain slashes")
        if phase == "source" and method not in self.header["source_caps"]:
            raise CampaignBudgetError("Source method is absent from the declared method caps")
        totals = self._counts
        allowance = self.header["total_actions"] - totals["charged_actions"]
        if phase in self.header["phase_caps"]:
            allowance = min(allowance, self.header["phase_caps"][phase]
                            - totals["by_phase"].get(phase, 0))
        if phase == "source" and method in self.header["source_caps"]:
            allowance = min(allowance, self.header["source_caps"][method]
                            - totals["by_method"].get(f"source/{method}", 0))
        return max(0, allowance)

    def reserve(self, decision_id, requested_actions, *, phase, method, task,
                episode, chunk, policy_version, metadata=None):
        """Durably authorize one bounded simulator call before it executes."""
        self._require_open()
        decision_id = _name(decision_id, "decision_id")
        if type(episode) not in (str, int) or episode == "":
            raise ValueError("episode must be a nonempty string or integer")
        request = json.loads(_canonical(dict(
            requested_actions=_count(requested_actions, "requested_actions", positive=True),
            phase=_name(phase, "phase"), method=_name(method, "method"),
            task=_name(task, "task"), episode=episode, chunk=_count(chunk, "chunk"),
            policy_version=_name(policy_version, "policy_version"),
            run_id=self.run_id, source_commit=self.source_commit,
            metadata={} if metadata is None else metadata)))
        previous = self._state["decisions"].get(decision_id)
        if previous is not None:
            # A new immutable allocation may encounter an old decision while
            # replaying its cursor. Keep the original run/source receipt and
            # refuse reexecution without treating a new executor as new data.
            semantic = {key: value for key, value in request.items()
                        if key not in ("run_id", "source_commit")}
            saved_request = {key: previous[key] for key in semantic}
            if saved_request != semantic:
                raise CampaignBudgetError("A recorded decision_id cannot change its intent")
            return Reservation(decision_id, previous["granted"], False,
                               previous["status"], previous["actual_actions"])
        granted = min(requested_actions, self.remaining(phase=phase, method=method))
        if not granted:
            return Reservation(decision_id, 0, False, "exhausted")
        self._record(dict(kind="reserve", decision_id=decision_id,
                          request=request, granted=granted))
        return Reservation(decision_id, granted, True, "pending")

    def resolve(self, decision_id, actual_actions):
        self._require_open()
        actual_actions = _count(actual_actions, "actual_actions")
        previous = self._state["decisions"].get(decision_id)
        if previous is None:
            raise CampaignBudgetError("No prior simulator reservation for this decision_id")
        if previous["status"] == "resolved" and previous["actual_actions"] == actual_actions:
            return deepcopy(previous)
        if previous["status"] != "pending":
            raise CampaignBudgetError("A resolved or uncertain intent cannot change its action charge")
        if actual_actions > previous["granted"]:
            raise CampaignBudgetError("Executed actions exceed the reserved upper bound")
        self._record(dict(kind="resolve", decision_id=decision_id, actual_actions=actual_actions))
        return deepcopy(self._state["decisions"][decision_id])

    def import_receipt(self, import_id, actual_actions, receipt_sha256, *,
                       phase="prior", method="qualification", metadata=None):
        """Charge a pinned prior-run receipt exactly once across retries."""
        self._require_open()
        import_id = _name(import_id, "import_id")
        if not isinstance(receipt_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", receipt_sha256):
            raise ValueError("Prior receipt must have a SHA-256 identity")
        receipt = json.loads(_canonical(dict(
            actual_actions=_count(actual_actions, "actual_actions"),
            receipt_sha256=receipt_sha256, phase=_name(phase, "phase"),
            method=_name(method, "method"), metadata={} if metadata is None else metadata)))
        previous = self._state["receipts"].get(import_id)
        if previous is not None:
            if previous != receipt:
                raise CampaignBudgetError("A prior import_id cannot change its receipt/count")
            return deepcopy(previous)
        if actual_actions > self.remaining(phase=phase, method=method):
            raise CampaignBudgetError("Prior actions do not fit the remaining campaign allowance")
        self._record(dict(kind="receipt", import_id=import_id, receipt=receipt))
        return deepcopy(receipt)

    def snapshot(self, *, include_decisions=False):
        self._require_open()
        # Historical decisions stay in the journal, not rewritten every chunk.
        # This avoids quadratic shared-filesystem writes during long runs.
        result = dict(header=self.header, receipts=self._state["receipts"],
                      runs=self._state["runs"], **self._counts,
                      decision_count=len(self._state["decisions"]),
                      unresolved_decisions=self._unresolved,
                      journal_events=self._sequence,
                      journal_sha256=self._previous_hash)
        if include_decisions:
            result["decisions"] = self._state["decisions"]
        return deepcopy(result)

    def _write_snapshot(self):
        encoded = (_canonical(self.snapshot()) + "\n").encode()
        descriptor, temporary = tempfile.mkstemp(prefix=".snapshot-", dir=self.root)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.root / "snapshot.json")
            _fsync_directory(self.root)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
