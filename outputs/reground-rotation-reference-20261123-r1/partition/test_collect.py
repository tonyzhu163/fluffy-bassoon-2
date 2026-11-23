"""Run with the pinned Linux interpreter and CUDA hidden. Only writes under private /tmp."""
import argparse
from dataclasses import asdict
import hashlib
import multiprocessing as mp
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import collect
import partition
import reference as ref


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def use_store(root):
    ref.STORE, ref.NATIVE = root, root / "runs"


def schedules():
    from reground.data import babble
    from util import resolve_tasks
    result = {}
    for phase, boundary, stop in [("collect", 17, 25), ("babble", 33, 50)]:
        all_jobs = []
        for seeds in ([0], [1, 2, 3, 4]):
            for limits, reverse in [((0, boundary), False), ((boundary, stop), True)]:
                chunk = collect.selected(phase, seeds, limits, reverse)
                assert [j["shard"] for j in chunk] == sorted([j["shard"] for j in chunk], reverse=reverse)
                for rank in range(3):
                    all_jobs.extend(chunk[rank::3])
        actual = {ref.label(j): j for j in all_jobs}
        assert len(actual) == len(all_jobs)
        native = []
        for cell in ref.CELLS:
            if phase == "collect":
                native.extend(json.loads(subprocess.check_output(
                    [sys.executable, "-B", str(ref.SOURCE / "main.py"), "collect", "--cell", cell,
                     "--n", "100000", "--dry-run"], cwd=ref.SOURCE)))
            else:
                pool = MagicMock()
                capture = pool.Pool.return_value.__enter__.return_value.imap_unordered
                with patch("multiprocessing.get_context", return_value=pool):
                    for seed in ref.SEEDS:
                        babble.run_babble(cell, seed, 100000, resolve_tasks("all"))
                        for cell_, seed_, shard, n, tasks, cfg in capture.call_args.args[1]:
                            assert cfg == asdict(babble.BabbleCfg())
                            native.append(dict(kind=phase, policy="pi05", cell=cell_, seed=seed_, shard=shard,
                                               n=n, tasks=tasks))
        assert actual == {ref.label(j): j for j in native}
        result[phase] = dict(native_jobs=len(native), no_duplicates=True, seed_ranges=[[0], [1, 2, 3, 4]],
                             shard_ranges=[[0, boundary], [boundary, stop]], lanes_per_assignment=3)
    return result


def fixture_jobs(root):
    import numpy as np
    import torch
    from reground.eval import jobs, rollout
    assert not torch.cuda.is_available()
    ctx = jobs.WorkerCtx("pi05", "cpu", ref.BATCH, lambda _: None)
    ctx._vla = SimpleNamespace(revision=ref.POLICY)
    ctx._pool = object()
    calls = []
    def simulate(vla, pool, episodes, ctrl, cell, **kw):
        calls.append([e.scene_seed for e in episodes])
        traces = []
        for e in episodes:
            rng = np.random.default_rng(e.scene_seed)
            n = int(rng.integers(200, 300))
            traces.append(dict(task=e.task, X=rng.normal(size=(n + 1, 25)).astype("f"),
                               A=rng.normal(size=(n, 7)).astype("f")))
        return SimpleNamespace(traces=traces)
    rows = []
    for seeds, limits, reverse in [([0, 3], (0, 1), False), ([0, 3], (24, 25), True)]:
        for j in collect.selected("collect", seeds, limits, reverse):
            use_store(root / "native")
            path = ref.output(j)
            with patch.object(rollout, "run_batch", simulate):
                ref.perform(j, ctx, root / "baseline" / ref.label(j), path)
            expected = digest(path)
            meta = ref.read(path.with_suffix(".json"))
            use_store(root / "partition")
            dest = ref.output(j)
            calls.clear()
            def interrupt(*args, **kw):
                if len(calls) == 1:
                    raise InterruptedError
                return simulate(*args, **kw)
            with patch.object(rollout, "run_batch", interrupt):
                try:
                    collect.one(j, ctx, root / "killed")
                except InterruptedError:
                    pass
                else:
                    raise AssertionError("Fault did not trigger")
            assert not dest.exists()
            checkpoint = ref.STORE / "batches" / ref.label(j) / "batch0000.pt"
            stamp = checkpoint.stat().st_mtime_ns
            calls.clear()
            with patch.object(rollout, "run_batch", simulate):
                event, facts = collect.one(j, ctx, root / "retry")
            assert event == "published" and len(calls) == facts["episodes"] // 4 - 1
            assert digest(dest) == expected and checkpoint.stat().st_mtime_ns == stamp
            actual = ref.read(dest.with_suffix(".json"))
            assert all(meta[k] == actual[k] for k in ("git_sha", "vla_revision", "seed", "cell", "x0_fixed", "policy", "n"))
            mt = tuple(dest.with_suffix(s).stat().st_mtime_ns for s in (".npz", ".json"))
            assert collect.one(j, None, root / "skip")[0] == "existing"
            assert mt == tuple(dest.with_suffix(s).stat().st_mtime_ns for s in (".npz", ".json"))
            # Model death between the sidecar and marker hard links.
            dest.unlink()
            assert collect.one(j, None, root / "relink")[0] == "recovered"
            assert digest(dest) == expected
            rows.append(dict(unit=ref.label(j), npz_sha256=expected, **facts, replayed_batches=1))
    j = collect.selected("collect", [2], (1, 2), False)[0]
    script = """import sys,json,reference as r; from pathlib import Path
r.STORE=Path(sys.argv[1])
with r.claim(json.loads(sys.argv[2])) as mine:
 print(mine,flush=True)
 sys.stdin.read()
"""
    with subprocess.Popen([sys.executable, "-B", "-c", script, str(ref.STORE), json.dumps(j)],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as owner:
        assert owner.stdout.readline().strip() == "True"
        lock = ref.STORE / "locks" / (ref.label(j) + ".lock")
        inode = lock.stat().st_ino
        try:
            assert collect.one(j, None, root / "locked") == ("busy", {})
        finally:
            owner.stdin.close()
    with ref.claim(j) as fresh:
        assert fresh and lock.stat().st_ino == inode
    return rows


def batch_racer(store, job, barrier, marker):
    import numpy as np
    from reground.eval import rollout
    use_store(store)
    episodes = [rollout.Episode(t, None, 0, scene_seed=i) for i, t in enumerate(job["tasks"][:4])]

    def generate(*args, **kw):
        barrier.wait(30)
        return SimpleNamespace(traces=[dict(X=np.full((2, 25), marker, np.float32),
                                            A=np.full((1, 7), marker, np.float32), task=e.task)
                                       for e in episodes])

    with patch.object(rollout, "run_batch", generate), collect.batches(job, lambda _: None):
        result = rollout.run_batch(None, None, episodes, object(), job["cell"])
    (store / f"observer{marker}").write_text(str(int(result.traces[0]["A"][0, 0])))


def same_batch(root):
    import torch
    store = root / "batch-race"
    job = collect.selected("collect", [1], (0, 1), False)[0]
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    contenders = [ctx.Process(target=batch_racer, args=(store, job, barrier, i)) for i in (1, 2)]
    for p in contenders:
        p.start()
    try:
        for p in contenders:
            p.join(60)
            assert p.exitcode == 0
    finally:
        for p in contenders:
            if p.is_alive():
                p.kill()
            p.join()
    checkpoint = store / "batches" / ref.label(job) / "batch0000.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    marker = int(saved["result"].traces[0]["A"][0, 0])
    assert [int((store / f"observer{i}").read_text()) for i in (1, 2)] == [marker, marker]
    return dict(competing_batch_markers=[1, 2], consumed_by_both=marker,
                retained_batch_sha256=digest(checkpoint))


def budget(root):
    os.environ.update(REFERENCE_CODE_PIN="fixture", REFERENCE_PHASE="fit", REFERENCE_STARTED=str(int(time.time())))
    canonical = root / "canonical"
    attempts = canonical / "attempts"
    def args(name, **kw):
        opts = dict(phase="fit", seeds=[0], namespace="isolated", descending=False, shards=None,
                    gpus=1, wall=600, cap=1, attempt=attempts / name)
        return SimpleNamespace(**{**opts, **kw})
    bad = args("bad", gpus=2)
    with patch.object(partition, "CANONICAL", canonical):
        cases = [bad, args("bad", cap=40.01), args("bad", namespace="../canonical"),
                 args("bad", phase="collect", shards=[24, 26])]
        for a in cases:
            try:
                partition.validate(a)
            except ValueError:
                pass
            else:
                raise AssertionError(a)
        a = args("fresh")
        partition.validate(a)
        a.attempt.mkdir(parents=True)
        ref.write(attempts / "old/reservation.json", {"max_gpu_seconds": 3200})
        try:
            partition.reserve(a)
        except ValueError as e:
            assert "3200+600" in str(e)
        else:
            raise AssertionError("Ceiling bypassed")
        ref.write(attempts / "old/exit.json", {"allocated_gpu_seconds": 2800})
        b = args("retry", seeds=[4])
        b.attempt.mkdir()
        partition.reserve(b)
        partition.read_contract(b)
        row = ref.read(b.attempt / "reservation.json")
        assert row["prior_gpu_seconds"] == 2800 and row["max_gpu_seconds"] == 600
        assert partition.store_for("canonical") == canonical
        assert len({partition.store_for(n) for n in ("canonical", "a", "b")}) == 3
    return dict(prior_open=3200, requested=600, limit=3600, rejected=True, prior_closed=2800,
                max_charge=row["max_gpu_seconds"], admitted=True)


def main():
    assert platform.system() == "Linux" and os.environ["CUDA_VISIBLE_DEVICES"] == ""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arg = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="reground-partition-fixture-", dir="/tmp") as scratch:
        root = Path(scratch)
        use_store(root / "init")
        ref.configure()
        facts = dict(upstream=ref.PIN, policy=ref.POLICY, reference=partition.NATIVE_RUNNER,
                     platform=platform.platform(), python=platform.python_version(), gpu_hours=0,
                     plan=schedules(), collect=fixture_jobs(root), budget=budget(root), batch_race=same_batch(root),
                     original_claim_contended_then_reacquired=True, canonical_lock_inode_unchanged=True,
                     final_skip_kept_inode_times=True, interrupted_pair_recovered=True)
        ref.write(arg.output, facts)


if __name__ == "__main__":
    main()
