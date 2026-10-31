"""Prepared CPU qualification; execute only on allocated Linux resources.

Synthetic accounting/process tests never qualify a frontend or source corpus.
"""
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import sys

import pytest

from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
from fluffy_bassoon.rl.robotwin_source_broker import (
    SourceAdmissionStop, SourceBrokerError, SourceBudgetBroker, SourceBudgetClient,
)

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Allocated Linux CPU qualification only")
METHODS = ("sac_calibration_minus_train0", "sac_calibration_plus_train0", "sac_calibration_minus_train1")


def spec(method, episode=0, phase="source", case=0):
    return dict(id=f"{method}:{phase}:{episode}:{case}", method=method,
                task="place_phone", horizon=400, phase=phase)


def campaign(root, *, total=10000, source=10000):
    return CampaignBudget(root, campaign_id="synthetic-accounting", total_actions=total,
        source_caps=dict.fromkeys(METHODS, 2000), phase_caps=dict(source=source, source_validation=10000),
        contract=dict(test_only=True), run_id="fixture", source_commit="a"*40)


def broker(budget, root, live, **kwargs):
    owner = SourceBudgetBroker(budget, root, method_limits={m: dict(source=800, source_validation=800)
        for m in METHODS}, episode_bound=400, is_live=lambda method, pid: live.get(method) == pid, **kwargs)
    if not kwargs.get("recover"):
        for i, method in enumerate(METHODS):
            live[method] = i+10
            owner.register(method, i+10)
    return owner


def reserve(owner, method, episode, chunk=0, requested=32):
    identity = f"fixture:{method}:{episode['id']}:{chunk}"
    result = owner.reserve(method, identity, requested, phase=episode["phase"], method=method,
        task=episode["task"], episode=episode["id"], chunk=chunk, policy_version="synthetic-only")
    return identity, result


def finish(owner, method, episode, *, actions=400):
    remaining, chunks = actions, 0
    while remaining:
        requested = min(32, remaining)
        identity, result = reserve(owner, method, episode, chunks, requested)
        assert result.execute and result.granted == requested
        owner.resolve(method, identity, requested)
        remaining -= requested
        chunks += 1
    owner.complete(method, episode=episode["id"], receipt=dict(complete=True, chunks=chunks,
        primitive_steps=actions, episode_manifest_ref=dict(path="synthetic-not-admissible", sha256="a"*64)))


def test_two_full_episodes_admitted_together_without_overspend(tmp_path):
    live = {}
    with campaign(tmp_path/"ledger", total=800) as budget:
        owner = broker(budget, tmp_path/"ownership", live)
        a, b, c = (spec(m) for m in METHODS)
        owner.admit(METHODS[0], specs=[a], phase="source")
        owner.admit(METHODS[1], specs=[b], phase="source")
        with pytest.raises(SourceAdmissionStop):
            owner.admit(METHODS[2], specs=[c], phase="source")
        ai, _ = reserve(owner, METHODS[0], a)
        bi, _ = reserve(owner, METHODS[1], b)
        snapshot = owner.snapshot(METHODS[0])
        assert snapshot["charged_actions"] == 64
        assert set(snapshot["unresolved_decisions"]) == {ai}
        assert snapshot["source_broker"]["known_live_foreign_pending"] == [bi]
        assert set(budget.snapshot()["unresolved_decisions"]) == {ai, bi}
        owner.resolve(METHODS[0], ai, 32)
        owner.resolve(METHODS[1], bi, 32)
        # The entire admitted400 is still available after each reservation,
        # not a partial native episode produced by racing chunk reservations.
        for method, episode in ((METHODS[0], a), (METHODS[1], b)):
            for chunk in range(1, 13):
                identity, result = reserve(owner, method, episode, chunk, 16 if chunk == 12 else 32)
                owner.resolve(method, identity, result.granted)
            owner.complete(method, episode=episode["id"], receipt=dict(complete=True, chunks=13,
                primitive_steps=400, episode_manifest_ref=dict(path="synthetic", sha256="a"*64)))
        assert budget.snapshot()["actual_actions"] == 800
        assert not budget.snapshot()["unresolved_decisions"]


def test_phase_cap_and_per_method_phase_quota_are_independent(tmp_path):
    live = {}
    with campaign(tmp_path/"ledger", source=600) as budget:
        owner = broker(budget, tmp_path/"ownership", live)
        owner.admit(METHODS[0], specs=[spec(METHODS[0])], phase="source")
        with pytest.raises(SourceAdmissionStop):
            owner.admit(METHODS[1], specs=[spec(METHODS[1])], phase="source")
    with campaign(tmp_path/"other-ledger") as budget:
        owner = broker(budget, tmp_path/"other-ownership", live)
        method = METHODS[0]
        for number in (0, 1):
            episode = spec(method, number)
            owner.admit(method, specs=[episode], phase="source")
            finish(owner, method, episode)
        with pytest.raises(SourceAdmissionStop):
            owner.admit(method, specs=[spec(method, 2)], phase="source")
        panel = [spec(method, 0, "source_validation", case) for case in (0, 1)]
        owner.admit(method, specs=panel, phase="source_validation")
        finish(owner, method, panel[0])
        assert owner.groups[method]["cursor"] == 1
        with pytest.raises(SourceBrokerError):
            owner.admit(method, specs=[spec(method, 3)], phase="source")
        finish(owner, method, panel[1])
        assert owner.groups[method] is None


def test_short_actual_grant_returns_only_unused_hold_after_verified_boundary(tmp_path):
    live = {}
    with campaign(tmp_path/"ledger", total=800) as budget:
        owner = broker(budget, tmp_path/"ownership", live)
        episode = spec(METHODS[0])
        owner.admit(METHODS[0], specs=[episode], phase="source")
        identity, _ = reserve(owner, METHODS[0], episode)
        owner.resolve(METHODS[0], identity, 1)
        assert owner.groups[METHODS[0]]["hold"] == 399
        owner.complete(METHODS[0], episode=episode["id"], receipt=dict(complete=True, chunks=1,
            primitive_steps=1, episode_manifest_ref=dict(path="synthetic", sha256="a"*64)))
        assert budget.snapshot()["actual_actions"] == 1 and owner.groups[METHODS[0]] is None


def test_unknown_foreign_pending_and_dead_known_pending_never_hidden(tmp_path):
    live = {}
    with campaign(tmp_path/"ledger") as budget:
        owner = broker(budget, tmp_path/"ownership", live)
        episode = spec(METHODS[0])
        owner.admit(METHODS[0], specs=[episode], phase="source")
        identity, _ = reserve(owner, METHODS[0], episode)
        del live[METHODS[0]]  # Actual parent process liveness, not a worker message.
        assert identity in owner.snapshot(METHODS[1])["unresolved_decisions"]
        with pytest.raises(SourceAdmissionStop):
            owner.admit(METHODS[1], specs=[spec(METHODS[1])], phase="source")
        unknown = budget.reserve("unknown:foreign", 32, phase="source", method=METHODS[2],
            task="place_phone", episode="unknown", chunk=0, policy_version="unknown")
        assert unknown.execute
        assert "unknown:foreign" in owner.snapshot(METHODS[1])["unresolved_decisions"]


def test_duplicate_foreign_and_nonsequential_decisions_refuse_before_new_charge(tmp_path):
    live = {}
    with campaign(tmp_path/"ledger") as budget:
        owner = broker(budget, tmp_path/"ownership", live)
        episode = spec(METHODS[0])
        owner.admit(METHODS[0], specs=[episode], phase="source")
        identity, _ = reserve(owner, METHODS[0], episode)
        for method, chunk in ((METHODS[0], 0), (METHODS[0], 2), (METHODS[1], 0)):
            with pytest.raises(SourceBrokerError):
                reserve(owner, method, episode, chunk)
        assert budget.snapshot()["decision_count"] == 1
        with pytest.raises(SourceBrokerError):
            owner.resolve(METHODS[1], identity, 1)


def test_crash_recovery_retains_durable_owner_uncertain_cost_and_forbids_replay(tmp_path):
    live = {}
    with campaign(tmp_path/"ledger") as budget:
        owner = broker(budget, tmp_path/"ownership", live)
        episode = spec(METHODS[0])
        owner.admit(METHODS[0], specs=[episode], phase="source")
        identity, _ = reserve(owner, METHODS[0], episode)
        events = [json.loads(p.read_text())["event"]["kind"] for p in sorted(owner.root.glob("*.json"))]
        assert events[-2:] == ["intent", "reserved"]
    with campaign(tmp_path/"ledger") as budget:
        with pytest.raises(SourceBrokerError, match="explicit recovery"):
            broker(budget, tmp_path/"ownership", live)
        restored = broker(budget, tmp_path/"ownership", live, recover=True)
        assert restored.intents[identity]["worker"] == METHODS[0]
        assert restored.groups[METHODS[0]]["hold"] == 368
        assert restored._hazards()
        assert budget.snapshot()["uncertain_actions"] == 32
        assert identity in restored.snapshot()["unresolved_decisions"]
        with pytest.raises(SourceBrokerError):
            reserve(restored, METHODS[0], episode)
        duplicate = budget.reserve(identity, 32, phase="source", method=METHODS[0], task=episode["task"],
            episode=episode["id"], chunk=0, policy_version="synthetic-only")
        assert not duplicate.execute and duplicate.status == "uncertain"


def test_crash_between_canonical_grant_and_broker_reply_exposes_unknown_intent(monkeypatch, tmp_path):
    live = {}
    with campaign(tmp_path/"ledger") as budget:
        owner = broker(budget, tmp_path/"ownership", live)
        episode = spec(METHODS[0])
        owner.admit(METHODS[0], specs=[episode], phase="source")
        record = owner._record
        def fail_after_ledger(event):
            if event["kind"] == "reserved":
                raise OSError("injected ownership journal failure after canonical grant")
            return record(event)
        monkeypatch.setattr(owner, "_record", fail_after_ledger)
        with pytest.raises(OSError, match="injected"):
            reserve(owner, METHODS[0], episode)
        pending = budget.snapshot()["unresolved_decisions"]
        assert len(pending) == 1 and budget.snapshot()["pending_actions"] == 32
        assert set(owner.snapshot(METHODS[1])["unresolved_decisions"]) == set(pending)
        assert owner._hazards()
        with pytest.raises(SourceAdmissionStop):
            owner.admit(METHODS[1], specs=[spec(METHODS[1])], phase="source")


def _process_fixture(connection, method, header, directory):
    client = SourceBudgetClient(connection, worker=method, header=header)
    episode = spec(method)
    client.call("admit", specs=[episode], phase="source")
    identity = f"fixture:{method}:{episode['id']}:0"
    grant = client.reserve(identity, 32, phase="source", method=method, task=episode["task"],
        episode=episode["id"], chunk=0, policy_version="process-fixture")
    # Parent withholds both reserve replies until both independent intents are
    # durably pending: this is a concurrency test, not a timing/sleep test.
    assert grant.execute
    Path(directory).mkdir()
    Path(directory, "private-state.json").write_text(json.dumps(dict(pid=os.getpid(), method=method)))
    client.resolve(identity, 1)
    client.call("complete", episode=episode["id"], receipt=dict(complete=True, chunks=1,
        primitive_steps=1, episode_manifest_ref=dict(path="synthetic", sha256="a"*64)))
    connection.send(dict(operation="finished"))
    connection.close()


def test_real_spawned_processes_overlap_pending_intents_and_write_disjoint_shards(tmp_path):
    context = mp.get_context("spawn")
    processes, connections, delayed = {}, {}, []
    with campaign(tmp_path/"ledger", total=800) as budget:
        owner = SourceBudgetBroker(budget, tmp_path/"ownership",
            method_limits={m: dict(source=400, source_validation=800) for m in METHODS[:2]},
            episode_bound=400, is_live=lambda m, pid: processes[m].pid == pid and processes[m].is_alive())
        try:
            for method in METHODS[:2]:
                parent, child = context.Pipe()
                process = context.Process(target=_process_fixture,
                    args=(child, method, budget.header, str(tmp_path/method)))
                process.start(); child.close()
                processes[method], connections[parent] = process, method
                owner.register(method, process.pid)
            while connections:
                ready = wait(list(connections), timeout=10)
                assert ready, "Local worker protocol made no progress"
                for connection in ready:
                    message = connection.recv()
                    method = connections[connection]
                    if message["operation"] == "finished":
                        owner.exited(method)
                        connection.close()
                        del connections[connection]
                        continue
                    result = owner.dispatch(method, message["operation"], message["arguments"])
                    response = dict(sequence=message["sequence"], ok=True, result=result)
                    if message["operation"] == "reserve":
                        delayed.append((connection, response))
                        if len(delayed) == 2:
                            assert len(budget.snapshot()["unresolved_decisions"]) == 2
                            for channel, reply in delayed:
                                channel.send(reply)
                    else:
                        connection.send(response)
            for process in processes.values():
                process.join(timeout=10)
                assert process.exitcode == 0
        finally:
            for connection in connections:
                connection.close()
            for process in processes.values():
                if process.is_alive():
                    process.terminate()  # Synthetic CPU children owned only by this test.
                process.join(timeout=10)
        states = [json.loads((tmp_path/m/"private-state.json").read_text()) for m in METHODS[:2]]
        assert len({s["pid"] for s in states}) == 2
        assert budget.snapshot()["actual_actions"] == 2
        assert not budget.snapshot()["unresolved_decisions"]
