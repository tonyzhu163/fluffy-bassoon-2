"""Private Baige PFS fixture. Competing processes deliberately bypass the advisory lock."""
import contextlib
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from unittest.mock import patch

import collect
import reference as ref


def set_store(root):
    ref.STORE, ref.NATIVE = root, root / "runs"


def worker(attempt, store, j, barrier):
    """Each writer has a distinct payload. Interleaving their two files is detectable."""
    import numpy as np
    from reground.data import babble
    set_store(store)
    def make(job, ctx, stage, dest):
        mark = int(attempt.name)
        episodes = [dict(X=np.full((251, 25), mark, np.float32), A=np.full((250, 7), mark, np.float32),
                         task=job["tasks"][(j["shard"] + i) % len(j["tasks"])]) for i in range(16)]
        babble.save_episodes(dest, episodes, dict(seed=job["seed"], cell=job["cell"], policy="pi05",
            n=4000, git_sha=ref.PIN, vla_revision=ref.POLICY, x0_fixed=True, marker=mark))
        barrier.wait(timeout=30)
    with patch.object(ref, "claim", lambda _: contextlib.nullcontext(True)), patch.object(ref, "perform", make):
        event, _ = collect.one(j, None, attempt)
        (attempt / "event").write_text(event)


def race(root):
    import numpy as np
    set_store(root / "race")
    j = collect.selected("collect", [1], (24, 25), True)[0]
    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(2)
    procs = [ctx.Process(target=worker, args=(root / str(n), ref.STORE, j, barrier)) for n in (1, 2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    path = ref.output(j)
    events = [(root / str(n) / "event").read_text() for n in (1, 2)]
    assert sorted(events) == ["published", "recovered"]
    mark = ref.read(path.with_suffix(".json"))["marker"]
    with np.load(path) as data:
        assert (data["X"] == mark).all() and (data["A"] == mark).all()
    hashes = [ref.read(root / str(n) / "partial" / ref.label(j) / "candidate.json")["sha256"][".npz"]
              for n in (1, 2)]
    assert len(set(hashes)) == 2 and hashes[mark-1] == hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(events=events, metadata_and_data_marker=mark, contenders=2, candidate_npz_sha256=hashes)


def babble(root):
    rows = []
    for seed, shard, cell in [(0, 0, "none/0"), (4, 49, "rot_z/90")]:
        j = next(j for j in ref.units("babble", [seed]) if j["shard"] == shard and j["cell"] == cell)
        j = dict(j, n=1)  # One whole real simulator episode in the qualification fixture.
        set_store(root / "native")
        ref.perform(j, None, root / "native-work" / ref.label(j), ref.output(j))
        a = ref.output(j)
        set_store(root / "partition")
        event, facts = collect.one(j, None, root / "partition-attempt")
        b = ref.output(j)
        assert event == "published" and a.read_bytes() == b.read_bytes()
        assert collect.one(j, None, root / "retry")[0] == "existing"
        ma, mb = ref.read(a.with_suffix(".json")), ref.read(b.with_suffix(".json"))
        assert ma.pop("wall_s") > 0 and mb.pop("wall_s") > 0 and ma == mb
        rows.append(dict(unit=ref.label(j), npz_sha256=hashlib.sha256(a.read_bytes()).hexdigest(), **facts))
    return rows


if __name__ == "__main__":
    assert platform.system() == "Linux" and os.environ["CUDA_VISIBLE_DEVICES"] == ""
    with tempfile.TemporaryDirectory(dir="/mowenjie/tmp", prefix="reground-partition-") as fixture:
        root = Path(fixture)
        set_store(root / "initial")
        ref.configure()
        facts = dict(filesystem=subprocess.check_output(["stat", "-f", "-c", "%T", fixture], text=True).strip(),
                     scope="single Linux host, separate processes on shared PFS", race=race(root),
                     native_babble=babble(root), gpu_hours=0)
        ref.write(Path(sys.argv[1]), facts)
