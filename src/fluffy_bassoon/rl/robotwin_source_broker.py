"""Local, single-owner action admission for independent source processes.

Only the parent opens CampaignBudget. Episode holds are prospective capacity,
never fabricated physical charges. The original ledger journals actual chunk
intents/resolutions; this separate append-only journal records ownership first.
No simulator, tensor library, socket listener or scheduler is used here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

FORMAT = "robotwin_source_budget_broker_v1"


class SourceBrokerError(RuntimeError):
    pass


class SourceAdmissionStop(SourceBrokerError):
    """No physical reset was admitted."""


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("Expected a nonnegative integer action count")
    return value


class SourceBudgetBroker:
    """Call only from one parent event loop; connections identify the worker.

    ``is_live`` must query the parent's Process objects, never worker claims.
    Restoring the journal is audit/recovery only: dead active groups retain
    their holds and block fresh admission pending explicit owner reconciliation.
    This does not rewrite the canonical ledger or adopt an interrupted episode.
    """

    def __init__(self, budget, root, *, method_limits, episode_bound, is_live, recover=False):
        self.budget, self.root, self.is_live = budget, Path(root), is_live
        self.episode_bound = _count(episode_bound)
        if not self.episode_bound or not method_limits:
            raise ValueError("Require finite whole-episode bounds and method/phase quotas")
        self.limits = deepcopy(method_limits)
        for method, phases in self.limits.items():
            if not method or set(phases) != {"source", "source_validation"}:
                raise ValueError("Only explicitly bound source and validation lineages")
            if any(_count(n) < self.episode_bound for n in phases.values()):
                raise ValueError("Every phase quota must cover a complete episode")
        self.header = budget.header
        self.workers, self.groups, self.intents, self.episodes = {}, {}, {}, {}
        self.sequence, self.tip, self.stopping = 0, None, False
        self.root.mkdir(parents=True, exist_ok=True)
        expected = dict(format=FORMAT, action_ledger_root=str(budget.root.resolve()),
            action_ledger_header=budget.header, source_commit=budget.source_commit,
            run_id=budget.run_id, method_limits=self.limits, episode_bound=self.episode_bound)
        files = sorted(self.root.glob("[0-9]*.json"))
        if files:
            if not recover:
                raise SourceBrokerError("Existing broker ownership requires explicit recovery; no fresh launch")
            for path in files:
                row = json.loads(path.read_text())
                body = {k: v for k, v in row.items() if k != "sha256"}
                digest = hashlib.sha256(_bytes(body)).hexdigest()
                if (path.name != f"{self.sequence:010d}.json" or row["sequence"] != self.sequence
                        or row["previous_sha256"] != self.tip or row["sha256"] != digest):
                    raise SourceBrokerError("Broken broker ownership journal")
                if self.sequence == 0 and row["event"] != dict(kind="header", header=expected):
                    raise SourceBrokerError("Broker recovery contract differs")
                self._apply(row["event"])
                self.sequence += 1
                self.tip = digest
            # Ledger recovery conservatively converts pending charges to
            # uncertain. Ownership never converts them back or refunds them.
            self._record(dict(kind="recovered"))
        else:
            self._record(dict(kind="header", header=expected))

    def _apply(self, event):
        kind = event["kind"]
        if kind == "header":
            return
        if kind == "register":
            self.workers[event["worker"]] = dict(pid=event["pid"], exited=False)
        elif kind == "admit":
            group = deepcopy(event["group"])
            self.groups[event["worker"]] = group
            for spec in group["specs"]:
                self.episodes[spec["id"]] = dict(worker=event["worker"], status="admitted")
        elif kind == "intent":
            self.intents[event["decision_id"]] = dict(worker=event["worker"],
                request=deepcopy(event["request"]), granted=None, actual_actions=None, status="intent")
        elif kind == "reserved":
            row = self.intents[event["decision_id"]]
            row.update(status="pending", granted=event["granted"])
            self.groups[row["worker"]]["hold"] -= event["granted"]
        elif kind == "resolved":
            row = self.intents[event["decision_id"]]
            row.update(status="resolved", actual_actions=event["actual_actions"])
            self.groups[row["worker"]]["hold"] += row["granted"] - event["actual_actions"]
        elif kind == "complete":
            group = self.groups[event["worker"]]
            self.episodes[event["episode"]].update(status="complete", receipt=event["receipt"])
            group["cursor"] += 1
            if group["cursor"] == len(group["specs"]):
                self.groups[event["worker"]] = None  # Release only unissued capacity.
        elif kind == "exited":
            self.workers[event["worker"]]["exited"] = True
        elif kind == "recovered":
            for row in self.workers.values():
                row["exited"] = True
        elif kind == "stop":
            self.stopping = True
        else:
            raise SourceBrokerError("Unknown broker journal event")

    def _record(self, event):
        body = dict(sequence=self.sequence, previous_sha256=self.tip, event=event)
        digest = hashlib.sha256(_bytes(body)).hexdigest()
        target = self.root / f"{self.sequence:010d}.json"
        # Exclusive creation prevents accidental adoption/overwrite. A partial
        # event is detected on recovery and blocks execution conservatively.
        with target.open("xb") as handle:
            handle.write(_bytes(dict(body, sha256=digest)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        descriptor = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._apply(event)
        self.sequence += 1
        self.tip = digest

    def register(self, worker, pid):
        if worker not in self.limits or worker in self.workers or type(pid) is not int or pid < 1:
            raise SourceBrokerError("Require one new isolated process per bound lineage")
        self._record(dict(kind="register", worker=worker, pid=pid))

    def _live(self, worker):
        row = self.workers.get(worker)
        return bool(row and not row["exited"] and self.is_live(worker, row["pid"]))

    def _require_worker(self, worker):
        if not self._live(worker):
            raise SourceBrokerError("Worker is not a known live parent-owned process")

    def snapshot(self, worker=None, *, include_decisions=False):
        snapshot = self.budget.snapshot(include_decisions=include_decisions)
        excluded = []
        if worker is not None:
            self._require_worker(worker)
            for identity, decision in list(snapshot["unresolved_decisions"].items()):
                owner = self.intents.get(identity)
                if (decision["status"] == "pending" and owner is not None
                        and owner["status"] == "pending" and owner["worker"] != worker
                        and self._live(owner["worker"]) and self.groups.get(owner["worker"]) is not None):
                    excluded.append(identity)
                    del snapshot["unresolved_decisions"][identity]
        snapshot["source_broker"] = dict(format=FORMAT, journal_events=self.sequence,
            journal_sha256=self.tip, known_live_foreign_pending=excluded,
            active_groups=deepcopy(self.groups), worker_processes=deepcopy(self.workers))
        return snapshot

    def _hazards(self):
        snapshot = self.budget.snapshot()
        for identity, decision in snapshot["unresolved_decisions"].items():
            owner = self.intents.get(identity)
            if (decision["status"] != "pending" or owner is None or owner["status"] != "pending"
                    or not self._live(owner["worker"]) or self.groups.get(owner["worker"]) is None):
                return True
        return any(group is not None and not self._live(worker) for worker, group in self.groups.items())

    def _used(self, worker, phase):
        return sum(row["actual_actions"] if row["status"] == "resolved" else (row["granted"] or 0)
            for row in self.intents.values() if row["worker"] == worker and row["request"]["phase"] == phase)

    def remaining(self, worker, *, phase, method):
        self._require_worker(worker)
        if method != worker or phase not in self.limits[worker]:
            raise SourceBrokerError("A worker cannot borrow another lineage/phase identity")
        # Canonical remaining already subtracts all pending chunk grants.
        # Subtract only FOREIGN prospective holds so an admitted worker can
        # see its own guaranteed complete-episode capacity at the legacy gate.
        total_hold = phase_hold = 0
        for owner, group in self.groups.items():
            if owner != worker and group is not None:
                total_hold += group["hold"]
                phase_hold += group["hold"] if group["phase"] == phase else 0
        snap = self.budget.snapshot()
        phase_free = self.header["phase_caps"][phase] - snap["by_phase"].get(phase, 0) - phase_hold
        global_free = self.header["total_actions"] - snap["charged_actions"] - total_hold
        method_free = (self.header["source_caps"][method] - snap["by_method"].get(f"source/{method}", 0)
                       if phase == "source" else global_free)
        return max(0, min(global_free, phase_free, method_free,
            self.limits[worker][phase] - self._used(worker, phase)))

    def admit(self, worker, *, specs, phase):
        self._require_worker(worker)
        if self.stopping or self._hazards():
            raise SourceAdmissionStop("Stop/uncertain/dead physical ownership blocks a fresh reset")
        if self.groups.get(worker) is not None or phase not in self.limits[worker]:
            raise SourceBrokerError("Only one finite source episode/panel per lineage")
        if (not specs or len(specs) != (2 if phase == "source_validation" else 1)
                or any(s["method"] != worker or s["horizon"] != self.episode_bound
                    or s.get("phase") != phase for s in specs)
                or len({s["id"] for s in specs}) != len(specs)
                or any(s["id"] in self.episodes for s in specs)):
            raise SourceBrokerError("Exact fresh whole source episode or fixed two-case panel required")
        # Remaining includes this worker's quota and excludes every other hold.
        needed = self.episode_bound * len(specs)
        if self.remaining(worker, phase=phase, method=worker) < needed:
            raise SourceAdmissionStop("Complete episode/panel exceeds canonical or method/phase capacity")
        self._record(dict(kind="admit", worker=worker, group=dict(
            phase=phase, specs=deepcopy(specs), hold=needed, cursor=0)))
        return dict(admitted=True, actions=needed)

    def reserve(self, worker, decision_id, requested_actions, **request):
        self._require_worker(worker)
        group = self.groups.get(worker)
        if group is None:
            raise SourceBrokerError("No whole episode was admitted")
        spec = group["specs"][group["cursor"]]
        owned = [row for row in self.intents.values()
            if row["worker"] == worker and row["request"]["episode"] == spec["id"]]
        episode_charge = sum(row["actual_actions"] if row["status"] == "resolved" else (row["granted"] or 0)
                             for row in owned)
        expected = f"{self.budget.run_id}:{worker}:{spec['id']}:{len(owned)}"
        if (decision_id in self.intents or decision_id != expected
                or request.get("method") != worker or request.get("phase") != group["phase"]
                or request.get("episode") != spec["id"] or request.get("task") != spec["task"]
                or request.get("chunk") != len(owned)
                or any(row["status"] != "resolved" for row in owned)
                or type(requested_actions) is not int
                or not 0 < requested_actions <= min(32, group["hold"], self.episode_bound-episode_charge)):
            raise SourceBrokerError("Duplicate, foreign, nonsequential or unadmitted physical intent")
        self._record(dict(kind="intent", worker=worker, decision_id=decision_id,
            request=dict(request, requested_actions=requested_actions)))
        reservation = self.budget.reserve(decision_id, requested_actions, **request)
        if not reservation.execute or reservation.granted != requested_actions:
            raise SourceBrokerError("Canonical duplicate/uncertain/short grant; execution forbidden")
        self._record(dict(kind="reserved", decision_id=decision_id, granted=reservation.granted))
        return reservation

    def resolve(self, worker, decision_id, actual_actions):
        self._require_worker(worker)
        row = self.intents.get(decision_id)
        if row is None or row["worker"] != worker or row["status"] != "pending":
            raise SourceBrokerError("Resolve only the worker's one live pending intent")
        actual_actions = _count(actual_actions)
        result = self.budget.resolve(decision_id, actual_actions)
        self._record(dict(kind="resolved", decision_id=decision_id, actual_actions=actual_actions))
        return result

    def complete(self, worker, *, episode, receipt):
        self._require_worker(worker)
        group = self.groups.get(worker)
        if group is None or group["specs"][group["cursor"]]["id"] != episode:
            raise SourceBrokerError("Wrong physical episode boundary")
        rows = [row for row in self.intents.values()
            if row["worker"] == worker and row["request"]["episode"] == episode]
        if (not rows or any(row["status"] != "resolved" for row in rows)
                or receipt.get("complete") is not True or receipt.get("chunks") != len(rows)
                or receipt.get("primitive_steps") != sum(row["actual_actions"] for row in rows)
                or not receipt.get("episode_manifest_ref")):
            raise SourceBrokerError("Canonical complete receipt and all resolved physical chunks required")
        self._record(dict(kind="complete", worker=worker, episode=episode, receipt=deepcopy(receipt)))

    def exited(self, worker):
        if worker in self.workers and not self.workers[worker]["exited"]:
            self._record(dict(kind="exited", worker=worker))

    def stop(self):
        if not self.stopping:
            self._record(dict(kind="stop"))

    def dispatch(self, worker, operation, arguments):
        if operation not in {"snapshot", "remaining", "admit", "reserve", "resolve", "complete", "guard"}:
            raise SourceBrokerError("Unknown local budget operation")
        if operation == "guard":
            self._require_worker(worker)
            return dict(stopping=self.stopping)
        return getattr(self, operation)(worker, **arguments)


class SourceBudgetClient:
    """One synchronous local Pipe per worker; parent serializes only accounting."""
    def __init__(self, connection, *, worker, header):
        self.connection, self.worker, self.header = connection, worker, header
        self.sequence = 0

    def call(self, operation, **arguments):
        sequence = self.sequence
        self.sequence += 1
        self.connection.send(dict(sequence=sequence, operation=operation, arguments=arguments))
        response = self.connection.recv()
        if response.get("sequence") != sequence:
            raise SourceBrokerError("Out-of-order parent budget response")
        if not response["ok"]:
            cls = SourceAdmissionStop if response["error_type"] == "SourceAdmissionStop" else SourceBrokerError
            raise cls(response["error"])
        return response["result"]

    def snapshot(self, *, include_decisions=False):
        return self.call("snapshot", include_decisions=include_decisions)

    def remaining(self, *, phase, method):
        return self.call("remaining", phase=phase, method=method)

    def reserve(self, decision_id, requested_actions, **request):
        return self.call("reserve", decision_id=decision_id, requested_actions=requested_actions, **request)

    def resolve(self, decision_id, actual_actions):
        return self.call("resolve", decision_id=decision_id, actual_actions=actual_actions)
