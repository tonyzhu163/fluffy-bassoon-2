"""One-core CPU dispatch qualification against native code and synthetic inputs."""

from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import dispatch as d


def main(out, scored, prior_rank):
    import os
    d.ref.configure()
    import numpy as np
    import torch
    import worker

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and not torch.cuda.is_available()
    assert len(os.sched_getaffinity(0)) == 1
    retry, stream = worker.retry, worker.stream
    grid = retry.schedule()
    original = worker.selected(grid, "full")
    assert original == d.ref.read(prior_rank)["batches"]
    _, host, *_ = (prior_rank.parent / "launch.txt").read_text().split()
    assert host == "job-0ttm7yo8nha0-master-0"
    units = d.assigned(grid, original)
    expected = [("spatial_0", i, 0) for i in range(25, 50)]
    expected += [("spatial_2", i, 0) for i in range(15, 50)]
    assert [key for b in units for key in grid[b]] == expected
    assert grid[d.PARKED] == [("spatial_0", i, 0) for i in range(21, 25)]
    live = {}
    d.preflight(scored, grid, units, lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("Live assigned input appeared. CPU qualification must stay outcome-blind.")), live)
    assert not live

    # Native children start and close in the qualified forkserver path without resetting any start.
    pool_test = out / "pool"
    pool_test.mkdir()
    with d.verified_pool(retry.fresh, pool_test, "dispatch_cpu_teardown", 4) as (pool, _):
        assert len(pool.native.procs) == 4
        assert all(child.is_alive() for child in pool.native.procs)

    counts = {"readers_opened": 0, "readers_closed": 0}
    current = {"failure": None}
    image = np.zeros((256, 256, 3), np.uint8)

    class Simulator:
        procs = [SimpleNamespace(pid=i, exitcode=0, join=lambda timeout: None) for i in range(4)]

        def __init__(self, stage):
            self.stage, self.steps = stage, 0

        def call(self, idx, op, args):
            if op == "reset":
                self.keys = [(task, start) for task, _, start, _, _ in args]
            if op == "full_state":
                return [dict(task=self.keys[i][0], state={"sim": np.array([self.keys[i][1]])}) for i in idx]
            if op == "summary":
                return [dict(speeds=np.zeros(self.steps), displaced=0, fallen=0, contact_steps=0) for _ in idx]
            assert op in ("reset", "step")
            self.steps += int(op == "step")
            rows = []
            for i in idx:
                wrist = image
                if i == 3 and self.stage == retry.ARMS[1] and current["failure"]:
                    wrist = image.copy()
                    wrist[17, 19, 1] = 1
                obs = dict(pixels={"image": image, "image2": wrist}, robot_state={
                    "eef": {"pos": np.zeros(3)}, "gripper": {"qpos": np.zeros(2)}})
                # This field is ordered after the raw wrist pixel and must still be checked.
                if i == 3 and self.stage == retry.ARMS[1] and current["failure"] == "state":
                    obs["robot_state"]["gripper"]["qpos"][0] = 1
                rows.append(dict(x=np.zeros(25, np.float32), obs=obs, description="CPU dispatch fixture",
                                 max_steps=1, success=op == "step" and self.stage != "support"))
            return rows

    class Reader:
        def send(self, message):
            self.message = message
            if message[0] == "begin":
                self.chunk = [0] * message[1]

        def recv(self):
            if self.message[0] == "begin":
                return None
            lane, before, after, actions = self.message[1]
            self.chunk[lane] += 1
            return dict(chunk=self.chunk[lane], issued_mean=actions.mean(axis=0).tolist(),
                        next_adapter=list(stream.IDENTITY),
                        before_eef=before["robot0_eef_pos"].tolist(), before_jaw_qpos=before["robot0_gripper_qpos"].tolist(),
                        after_eef=after["robot0_eef_pos"].tolist(), after_jaw_qpos=after["robot0_gripper_qpos"].tolist())

    @contextmanager
    def fake_pool(work, stage, n):
        assert n == 4
        meter = retry.Meter(work / f"{stage}.jsonl", stage)
        try:
            yield retry.Pool(Simulator(stage), meter), meter
        finally:
            meter.log.close()

    @contextmanager
    def fake_reader():
        counts["readers_opened"] += 1
        try:
            yield Reader(), {"weights_sha256": stream.WEIGHTS}
        finally:
            if current["failure"] == "pixel":
                assert sys.exc_info()[0] is None
            counts["readers_closed"] += 1

    class Policy:
        revision = d.ref.POLICY

        def _noise(self, seeds, query):
            return torch.zeros((len(seeds), 50, 7))

        def predict_chunk(self, obs, tasks, seeds, query):
            return self._noise(seeds, query).numpy()

    fixture = out / "fixture"
    batches = fixture / "batches"
    batches.mkdir(parents=True)
    for i in range(d.PARKED):
        (batches / f"batch{i:03d}").mkdir()
    attempts = fixture / "attempts"
    attempts.mkdir()
    ctx = SimpleNamespace(_vla=Policy(), vla=Policy())
    provenance = d.provenance(retry, stream)
    native_pair = retry.run_pair
    scored_fixture = []

    def simulate(policy, pipe, reader, keys, work, pool_factory):
        batch = int(work.name[5:])
        scored_fixture.append(batch)
        current["failure"] = MODE.get(batch)
        return native_pair(policy, pipe, reader, keys, work, pool_factory)

    def dispatch(attempt_id, selected):
        attempt = attempts / attempt_id
        attempt.mkdir()
        return d.drain(attempt, 0, 0, grid, selected, ctx, retry, stream, provenance)

    with patch.object(retry, "fresh", fake_pool), patch.object(retry, "run_pair", simulate), \
            patch.object(stream, "reader_process", fake_reader):
        MODE = {}
        assert dispatch("retry90", [56]) == 0
        MODE = {55: "pixel"}
        assert dispatch("retry91", [55, 56, 57]) == 3
        a = attempts / "retry91"
        facts = d.ref.read(a / "dispatch.json")["statuses"]
        assert [facts[str(i)] for i in (54, 55, 56, 57)] == [
            "enrolled_parked_for_qualification", "unpublished_raw_guard", "validated_existing", "published"]
        assert scored_fixture == [56, 55, 57] and counts == {"readers_opened": 3, "readers_closed": 3}
        assert not (batches / "batch055").exists() and (a / "batch055" / "translation.json").is_file()
        assert d.ref.read(a / "batch055-error.json")["message"] == "Paired opening differs at raw/3/obs/pixels/image2"
        assert d.journal(a / "batch055" / "translation_wrist.jsonl", "translation_wrist") == {"reset": 4}
        MODE = {58: "state"}
        try:
            dispatch("retry92", [58, 59])
        except ValueError as exc:
            assert "Structural opening difference at raw/3/obs/robot_state/gripper/qpos" == str(exc)
        else:
            raise AssertionError("Structural mismatch failed to halt")
        assert scored_fixture == [56, 55, 57, 58] and not (batches / "batch058").exists()
        with (batches / "batch057" / "contract.json").open("a") as f:
            f.write(" ")  # legal JSON, altered bytes
        try:
            dispatch("retry93", [55, 57])
        except ValueError as exc:
            assert "Dispatch receipt or artifact bytes differ" in str(exc)
        else:
            raise AssertionError("Tampered bytes failed to halt")
        assert scored_fixture == [56, 55, 57, 58]

    d.ref.write(out / "cpu-qualification.json", dict(provenance, cpu_affinity=sorted(os.sched_getaffinity(0)),
                batches=units, query_keys=expected, parked_keys=grid[d.PARKED], released_cell=retry.CELL,
                support_keys=[retry.support_key(key) for key in expected], gpu_hours=0,
                real_scored_rows_read=0, real_scored_rows_written=0, prior_rank_enrollment_matches=True,
                original_sealed_count=54, native_forkserver_children_reaped=4,
                sparse_resume=True, post_guard_reader_restart=True, continuation_after_pixel_guard=True,
                fail_closed_on_additional_state_difference=True, fail_closed_on_tampered_complete_bytes=True,
                attempt_journals_and_exception_preserved=True))
    print("PASS CPU membership, fresh pool teardown, sparse resume, real guard, structural halt and byte validation")


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:]))
