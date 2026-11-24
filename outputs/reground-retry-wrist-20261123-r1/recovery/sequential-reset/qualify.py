"""Exercise the pinned query-zero guard and recovery logic without simulation or weights."""
from contextlib import contextmanager
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import reference as ref
import sequential as seq


def tree(pairs):
    root = {}
    for path, value in pairs:
        *parents, leaf = path.split("/")
        branch = root
        missing = leaf == "None"
        if missing:
            leaf = parents.pop()
        for name in parents:
            branch = branch.setdefault(name, {})
        branch[leaf] = None if missing else value.copy()
    return root


def main():
    ref.configure()
    import retry
    from reground.eval import jobs, rollout

    output, store = map(Path, sys.argv[1:])
    source = store / "attempts" / seq.LEFT / seq.BATCH
    first = seq.load_opening(source / "opening.npz")
    proposal = first.pop("proposal")
    alt = seq.load_opening(store / "attempts/retry04" / seq.BATCH / "opening.npz")
    alt.pop("proposal")
    seq.diagnose(first, alt, retry.identical)
    support = ref.read(source / "support.json")
    proc = SimpleNamespace(pid=127, exitcode=0, join=lambda **_: None)

    class Native:
        procs = [proc] * 4
        def __init__(self, opening):
            self.opening = opening

        def call(self, idx, op, args):
            if op == "reset":
                return [dict(obs=tree((k.removeprefix(f"raw/{i}/obs/"), v) for k,v in self.opening.items()
                                           if k.startswith(f"raw/{i}/obs/")), x=self.opening["X0"][i].copy(),
                             description=self.opening["raw_tasks"][i], max_steps=280) for i in idx]
            if op == "full_state":
                return [dict(task="spatial_0", state=tree((k.removeprefix(f"sim/{i}/"), v) for k,v in self.opening.items()
                                                        if k.startswith(f"sim/{i}/"))) for i in idx]
            assert op == "step"
            return [None] * len(idx)

    class FakeVLA:
        def _noise(self, seeds, query):
            assert seeds == first["noise_seeds"].tolist() and query == 0
            return torch.from_numpy(first["noise"].copy())

    def fixture(name, opening, cutoff=500, stop_after_control=False):
        attempt = output / name
        attempt.mkdir()
        candidate = attempt / "batch054-reset-0001"
        candidate.mkdir()
        copies = {seq.LEFT: {f"{seq.BATCH}/{f}": seq.sha(source / f) for f in seq.LEFT_FILES}}
        work = attempt / seq.BATCH
        seq.stage_left(store, work, copies)

        @contextmanager
        def fresh(path, arm, size):
            meter = retry.Meter(path / f"{arm}.jsonl", arm)
            try:
                yield retry.Pool(Native(opening), meter), meter
            finally:
                meter.log.close()

        def run(vla, pool, episodes, _ctrl, cell, **kwargs):
            idx = list(range(4))
            initial = pool.call(idx, "reset", [(e.task, cell, e.init_id, e.scene_seed, True) for e in episodes])
            observed = vla.predict_chunk([o["obs"] for o in initial], [o["description"] for o in initial],
                                         [e.noise_seed for e in episodes], 0)
            np.testing.assert_array_equal(proposal, observed)
            pool.call(idx, "step", [None] * 4)
            if stop_after_control:
                raise RuntimeError("synthetic native interruption after dispatch")
            return SimpleNamespace(traces={}, rows=[])

        with patch.object(retry, "fresh", fresh), patch.object(rollout, "run_batch", run), patch.object(jobs, "_save_traces"):
            return seq.open_candidate(work, candidate, FakeVLA(), support, retry, first,
                                      {k: alt[k] for k in seq.PIXELS}, time.monotonic()+cutoff, 100, attempt)

    assert fixture("known_pixel", alt)["status"] == "ineligible_pixel"
    assert not (output / "known_pixel/sequential-first-wrist-control.json").exists()
    structural = {**alt, "X0": alt["X0"].copy()}
    structural["X0"][0, 0] += 1
    try:
        fixture("unknown_difference", structural)
    except ValueError:
        assert not (output / "unknown_difference/sequential-first-wrist-control.json").exists()
    else:
        raise AssertionError("Unexpected physical mismatch admitted")
    assert fixture("exact", first)["status"] == "terminal"
    try:
        fixture("wall_cutoff", first, cutoff=20)
    except seq.BudgetExhausted:
        assert not (output / "wall_cutoff/sequential-first-wrist-control.json").exists()
    else:
        raise AssertionError("Exact opening ignored terminal budget reserve")
    try:
        fixture("post_dispatch", first, stop_after_control=True)
    except RuntimeError:
        assert (output / "post_dispatch/sequential-first-wrist-control.json").exists()
    else:
        raise AssertionError("Post-dispatch failure did not propagate")
    ref.write(output / "qualification.json", dict(known_pixel_rejected=True, physical_mismatch_terminal=True,
              exact_guard_and_shared_native_proposal=True, wall_cutoff_before_control=True,
              permanent_marker_before_control=True, no_model_instantiation=True, gpu_hours=0))


if __name__ == "__main__":
    main()
