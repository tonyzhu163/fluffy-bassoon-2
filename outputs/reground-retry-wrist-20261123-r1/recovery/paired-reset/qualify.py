"""One-core Mesa reset qualification and inert scheduling fixtures. No model or scored actions."""
import argparse
from contextlib import contextmanager
import fcntl
import multiprocessing as mp
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import admission
import dispatch
import reference as ref


def scheduler(root):
    store, ledger = root / "store", root / "ledger"
    batches = store / "batches"
    batches.mkdir(parents=True)
    (store / "attempts").mkdir()
    for batch in (*range(54), 55, 58):
        work = batches / f"batch{batch:03d}"
        work.mkdir()
        (work / "producer.json").write_text("sentinel: existing outcomes must never be read")
    selected = root / "selection.json"
    grant = dict(batches=[54, 55, 56, 57, 58], remaining_wall_seconds=1800, remaining_gpu_seconds=1600,
                 startup_and_scheduler_seconds=60, score_seconds_per_batch=60, pair_seconds=30,
                 ordinary_continuation="qualification fixture", budget_evidence="qualification arithmetic")
    admission.write(selected, grant)
    lock = store / "active.lock"
    with lock.open("a") as owner, lock.open("a") as contender:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise AssertionError("Global lock permits a competing worker")
        except BlockingIOError:
            pass
        plan = admission.reserve(store, ledger, selected, "retry801", 420, 360, "fixture")
    assert plan["pair_cap"] == (1600 - 60 - 5 * 60) // (5 * 30) == 8
    prior = store / "attempts/retry801"
    (prior / "batch054").mkdir()
    (prior / "batch054/translation.json").write_text("sentinel: partial outcome must never be read")
    plan = admission.reserve(store, ledger, selected, "retry802", 590, 530, "fixture")
    states = admission.statuses(store, plan)
    assert states == {54: "prior_attempt_quarantine", 55: "sealed_parked", 56: "unstarted",
                      57: "unstarted", 58: "sealed_parked"}
    for label, path, allotment in (("budget", selected, 600), ("selection", root / "changed.json", 120)):
        if label == "selection":
            admission.write(path, dict(grant, batches=[56]))
        try:
            admission.reserve(store, ledger, path, "retry803", allotment, allotment - 60, "fixture")
            raise AssertionError(f"Admitted with changed {label}")
        except ValueError:
            pass
    calls, validator = [], []
    grid = [[(b, lane, 0) for lane in range(4)] for b in range(70)]
    def run_pair(vla, pipe, reader, keys, work, cap, admit):
        assert admit() and cap == 8 and not list((store / "batches").glob(work.name))
        batch = keys[0][0]
        calls.append(batch)
        return "draw_cap_quarantine" if batch == 56 else "publishable"

    @contextmanager
    def reader():
        yield None, None

    def validate(work, keys):
        validator.append(keys[0][0])

    code = dispatch.dispatch(store / "attempts/retry802", store, plan, grid, SimpleNamespace(validate=validate),
                             run_pair, SimpleNamespace(vla=None), reader, lambda ctx: None,
                             {"egl": 0, "science_pin": "fixture"})
    progress = admission.read(store / "attempts/retry802/paired-reset-progress.json")
    assert code == 3 and calls == [56, 57] and validator == [57]
    assert progress["quarantined_batches"] == 2 and progress["quarantined_starts"] == 8
    assert (batches / "batch057/producer.json").is_file()
    return dict(global_lock=True, sparse_calls=calls, validator_called=validator, statuses=progress["statuses"],
                cap=plan["pair_cap"], quarantine=progress["quarantined_starts"], exit=code)


def resets(root, fixtures_only=False):
    import numpy as np
    import pairing
    import retry
    from opening import file_hash
    from reground.eval import rollout
    from reground.policies.vla import VLA

    mp.set_forkserver_preload(["preload_env"])
    keys = retry.schedule()[54]

    class InertPolicy:
        _noise = VLA._noise
        device = "cpu"
        chunk_size = 50
        noise_dim = 32

        def __init__(self):
            self.forward = 0

        def predict_chunk(self, obs, tasks, seeds, query):
            assert query == 0
            assert len(obs) == len(tasks) == len(seeds) == 4
            assert self._noise(seeds, query).shape == (4, 50, 32)
            self.forward += 1
            return np.zeros((4, 50, 7), dtype=np.float32)

    class BeforeControl(Exception):
        pass

    def consumer(work, policy, seen):
        def observe(candidates, episodes):
            receipt = ref.read(work / "prequery.json")
            assert receipt["state"] == "accepted" and receipt["arm_receipts"][0] == receipt["arm_receipts"][1]
            assert policy.forward == 0
            assert all(candidate.bridge.pool.meter.counts == {"reset": 4} for candidate in candidates)
            assert set(candidates[0].pids).isdisjoint(candidates[1].pids)
            for candidate in candidates:
                candidate.check_alive()
                assert all(not a.flags.writeable for a in candidate.snapshot.values())

            def stop_before_control(x, proposal):
                seen.append(proposal.copy())
                raise BeforeControl

            for candidate in candidates:
                for held in candidates:
                    held.check_alive()
                bridge = candidate.bridge
                with patch.object(bridge, "on_chunk", stop_before_control):
                    try:
                        rollout.run_batch(bridge, bridge, episodes, bridge, retry.CELL)
                        raise AssertionError("Native loop failed to stop before controls")
                    except BeforeControl:
                        pass
                assert "step" not in bridge.pool.meter.counts
                ref.write(work / f"{bridge.arm}.json", dict(opening_sha256=file_hash(work / "opening.npz")))
            assert policy.forward == 1 and np.array_equal(*seen), "Both arms must share one opening proposal"
        return observe

    fixture_tools = SimpleNamespace(starts=0, closed=0, mode="retry")
    fixture_support = [dict(estimate=[72.0, 1.0, 1], initial_sim_sha256="separate support") for _ in keys]

    class FixturePool:
        def __init__(self, n):
            fixture_tools.starts += 1
            self.serial = fixture_tools.starts
            self.procs = [SimpleNamespace(pid=100 * self.serial + i, is_alive=lambda: True,
                          join=lambda timeout: None, exitcode=0) for i in range(n)]

        def call(self, idx, op, args):
            changed = self.serial % 2 == 0 and (self.serial == 2 or fixture_tools.mode == "cap")
            if op == "full_state":
                return [dict(task=keys[i][0], state={"sim": np.array([i, 1])}) for i in idx]
            assert op == "reset", f"Unexpected native action: {op}"
            assert args == [(k[0], retry.CELL, k[1], 0, True) for k in keys]
            rows = []
            for i in idx:
                wrist = np.zeros((256, 256, 3), np.uint8)
                if i == 3 and changed:
                    wrist[79, 153] = 1
                x = np.zeros(25)
                if (changed and fixture_tools.mode == "physical" or
                        self.serial >= 3 and fixture_tools.mode == "cross_draw"):
                    x[0] = 1
                if changed and fixture_tools.mode == "broad":
                    wrist[80, 153] = 1
                obs = dict(pixels=dict(image=wrist.copy() if fixture_tools.mode == "external" else np.zeros_like(wrist),
                                       image2=wrist),
                           robot_state=dict(eef=dict(pos=np.zeros(3)), gripper=dict(qpos=np.zeros(2))))
                rows.append(dict(x=x, obs=obs, description="fixture", max_steps=280))
            return rows

        def close(self):
            fixture_tools.closed += 1

    fixtures = {}
    for mode in ("retry", "cap", "physical", "broad", "external", "cross_draw"):
        fixture_tools.starts = fixture_tools.closed = 0
        fixture_tools.mode = mode
        work = root / mode
        work.mkdir()
        policy = InertPolicy()
        seen = []
        try:
            result = pairing.pair(policy, keys, fixture_support, work, 3 if mode == "retry" else 2,
                                  lambda: True, consumer(work, policy, seen), FixturePool)
        except ValueError as exc:
            assert mode in ("physical", "broad", "external", "cross_draw") and "outside" in str(exc)
            result = "structural_halt"
        assert result == {"retry": "publishable", "cap": "draw_cap_quarantine"}.get(mode, "structural_halt")
        assert fixture_tools.closed == fixture_tools.starts == (4 if mode in ("retry", "cap", "cross_draw") else 2)
        if mode == "retry":
            assert (work / "draw001/cleanup.json").is_file() and not (work / "draw003").exists()
            assert ref.read(work / "draw001/images.json")["raw/3/obs/pixels/image2"]["yx"] == [[79, 153]]
            assert ref.read(work / "prequery.json")["ordinal"] == 2
            assert pairing.verify_prequery(work, keys)["rejected_draws"] == 1
        assert policy.forward == (1 if mode == "retry" else 0)
        fixtures[mode] = result

    if fixtures_only:
        return dict(fixture_results=fixtures, scored_actions=0, weights_loaded=0)

    work = root / "native"
    work.mkdir()
    policy = InertPolicy()
    seen = []
    begin = time.monotonic()
    result = pairing.pair(policy, keys, fixture_support, work, 1, lambda: True, consumer(work, policy, seen))
    elapsed = time.monotonic() - begin
    assert result == "publishable", f"One-draw Mesa native reset result: {result}"
    assert pairing.verify_prequery(work, keys)["rejected_draws"] == 0
    arms = [ref.read(work / "draw001" / f"{arm}-opening.json") for arm in retry.ARMS]
    return dict(fixture_results=fixtures, native_keys=keys, native_result=result,
                native_seconds=elapsed, native_arm_timings=[a["seconds"] for a in arms],
                native_receipts=[file_hash(work / "draw001" / f"{arm}-opening.json") for arm in retry.ARMS],
                stub_forward_calls=policy.forward, scored_actions=0, weights_loaded=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("--fixtures-only", action="store_true")
    parser.add_argument("--verify-native", type=Path)
    args = parser.parse_args()
    target = args.target
    target.mkdir(exist_ok=True)
    ref.configure()
    admission.write(target / "scheduler.json", scheduler(target))
    admission.write(target / "reset.json", resets(target, args.fixtures_only))
    if args.verify_native:
        import pairing
        import retry
        admission.write(target / "native-readback.json", pairing.verify_prequery(args.verify_native, retry.schedule()[54]))
