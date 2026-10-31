"""Campaign accounting persistence tests; run on allocated Linux only."""
import hashlib
import json

import pytest

from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget, CampaignBudgetError


COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40
RECEIPT = hashlib.sha256(b"prior simulator receipt").hexdigest()


def ledger(path, **changes):
    args = dict(campaign_id="robotwin-one-million", total_actions=100,
                source_caps={"ppo": 40, "sac": 40}, contract={"motor": "frozen", "K": 4},
                run_id="run-1", source_commit=COMMIT)
    args.update(changes)
    return CampaignBudget(path, **args)


def reserve(budget, decision="episode-1-chunk-0", count=32, **changes):
    args = dict(phase="source", method="ppo", task="handover_mic",
                episode="trial-1-episode-1", chunk=0, policy_version="ppo-update-0")
    args.update(changes)
    return budget.reserve(decision, count, **args)


def test_all_methods_phases_and_prior_actions_share_one_ceiling(tmp_path):
    with ledger(tmp_path) as budget:
        budget.import_receipt("earlier-run", 10, RECEIPT)
        budget.import_receipt("earlier-run", 10, RECEIPT)  # Reopening/importing does not double count.
        decision = reserve(budget)
        assert decision.execute and decision.granted == 32
        assert budget.snapshot()["charged_actions"] == 42
        budget.resolve(decision.decision_id, 18)  # Early success refunds only unused upper bound.
        assert budget.remaining(phase="source", method="ppo") == 22
        assert budget.snapshot()["actual_actions"] == 28
        last_ppo = reserve(budget, "ppo-last", count=32)
        assert last_ppo.execute and last_ppo.granted == 22
        budget.resolve("ppo-last", 22)
        assert not reserve(budget, "ppo-exhausted").execute
        sac = reserve(budget, "sac-1", count=40, method="sac")
        assert sac.granted == 40
        budget.resolve("sac-1", 40)
        evaluation = reserve(budget, "eval-1", count=32, phase="target", method="native")
        assert evaluation.granted == 10
        budget.resolve("eval-1", 10)
        exhausted = reserve(budget, "no-budget", phase="validation", method="native")
        assert exhausted.granted == 0 and exhausted.status == "exhausted"
        assert budget.snapshot()["actual_actions"] == budget.snapshot()["charged_actions"] == 100


def test_crashed_intent_is_not_refunded_or_reexecuted_after_recovery(tmp_path):
    with ledger(tmp_path) as original:
        intent = reserve(original, count=32)
        assert intent.execute
        # The simulator may have executed any subset. Deliberately no resolve.
    with ledger(tmp_path) as recovered:
        state = recovered.snapshot()
        assert state["charged_actions"] == state["uncertain_actions"] == 32
        assert state["actual_actions"] == state["pending_actions"] == 0
        replay = reserve(recovered, count=32)
        assert not replay.execute and replay.status == "uncertain"
        with pytest.raises(CampaignBudgetError, match="uncertain"):
            recovered.resolve(replay.decision_id, 0)
        assert recovered.remaining(phase="source", method="ppo") == 8
    with ledger(tmp_path, run_id="run-2", source_commit=OTHER_COMMIT) as next_run:
        assert next_run.snapshot()["uncertain_actions"] == 32
        assert next_run.snapshot()["runs"] == {"run-1": COMMIT, "run-2": OTHER_COMMIT}
        assert not reserve(next_run).execute
        assert reserve(next_run, "run-2-first").granted == 8


def test_resolved_receipts_are_idempotent_and_ids_cannot_change(tmp_path):
    with ledger(tmp_path) as budget:
        first = reserve(budget)
        pending_again = reserve(budget)
        assert first.execute and not pending_again.execute and pending_again.status == "pending"
        budget.resolve(first.decision_id, 0)  # A preterminal simulator can execute no action.
        assert budget.resolve(first.decision_id, 0)["actual_actions"] == 0
        resolved_again = reserve(budget)
        assert not resolved_again.execute and resolved_again.status == "resolved"
        assert resolved_again.actual_actions == 0
        with pytest.raises(CampaignBudgetError, match="change its intent"):
            reserve(budget, count=31)
        with pytest.raises(CampaignBudgetError, match="change its action charge"):
            budget.resolve(first.decision_id, 1)
        budget.import_receipt("prior", 5, RECEIPT, metadata={"path": "prior/AUDIT.json"})
        with pytest.raises(CampaignBudgetError, match="change its receipt"):
            budget.import_receipt("prior", 6, RECEIPT, metadata={"path": "prior/AUDIT.json"})
        assert budget.snapshot()["charged_actions"] == 5


def test_concurrent_driver_and_changed_recovery_contract_are_rejected(tmp_path):
    with ledger(tmp_path):
        with pytest.raises(CampaignBudgetError, match="Another process"):
            ledger(tmp_path, run_id="competing-run")
    for change in ({"total_actions": 101}, {"contract": {"motor": "changed", "K": 4}},
                   {"source_caps": {"ppo": 41, "sac": 40}},
                   {"campaign_id": "different-campaign"}):
        with pytest.raises(CampaignBudgetError, match="contract or action limits"):
            ledger(tmp_path, **change)
    with pytest.raises(CampaignBudgetError, match="run_id cannot change"):
        ledger(tmp_path, source_commit=OTHER_COMMIT)
    with ledger(tmp_path) as recovered:
        assert recovered.snapshot()["runs"] == {"run-1": COMMIT}


def test_snapshot_is_a_cache_and_journal_corruption_fails_closed(tmp_path):
    with ledger(tmp_path) as budget:
        reserve(budget, count=10)
        budget.resolve("episode-1-chunk-0", 7)
    (tmp_path / "snapshot.json").write_text('{"charged_actions":0}')
    with ledger(tmp_path) as recovered:
        assert recovered.snapshot()["charged_actions"] == 7
        snapshot = json.loads((tmp_path / "snapshot.json").read_text())
        assert snapshot["charged_actions"] == 7
        assert "decisions" not in snapshot  # Historical journal is never rewritten each chunk.
    events = sorted((tmp_path / "events").glob("*.json"))
    original = events[-1].read_bytes()
    damaged = json.loads(original)
    damaged["actual_actions"] = 0
    events[-1].write_text(json.dumps(damaged))
    with pytest.raises(CampaignBudgetError, match="integrity"):
        ledger(tmp_path)


def test_phase_caps_include_all_methods_and_reject_oversized_actuals(tmp_path):
    with ledger(tmp_path, phase_caps={"validation": 12}) as budget:
        check = reserve(budget, "check-a", count=10, phase="validation", method="ppo")
        with pytest.raises(CampaignBudgetError, match="upper bound"):
            budget.resolve(check.decision_id, 11)
        budget.resolve(check.decision_id, 7)
        check_b = reserve(budget, "check-b", count=10, phase="validation", method="sac")
        assert check_b.granted == 5
        budget.resolve(check_b.decision_id, 5)
        assert budget.remaining(phase="validation", method="native") == 0
        assert budget.remaining(phase="source", method="ppo") == 40
        with pytest.raises(CampaignBudgetError, match="remaining campaign"):
            budget.import_receipt("prior-too-large", 89, RECEIPT)
        with pytest.raises(CampaignBudgetError, match="absent from the declared"):
            reserve(budget, "undeclared-arm", method="sac_typo")


@pytest.mark.parametrize("count", [True, -1, 1.5, 0])
def test_invalid_reservation_never_publishes_an_intent(tmp_path, count):
    with ledger(tmp_path) as budget:
        before = budget.snapshot()["journal_events"]
        with pytest.raises(ValueError, match="positive integer"):
            reserve(budget, count=count)
        assert budget.snapshot()["journal_events"] == before
        assert budget.snapshot()["charged_actions"] == 0
