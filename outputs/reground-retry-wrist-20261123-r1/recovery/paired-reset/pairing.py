"""First exact native opening under the frozen four-lane wrist protocol."""
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
import os
import time
from unittest.mock import patch

import numpy as np

import reference as ref
from opening import file_hash, identical, observations
from reground.eval import jobs, rollout
import retry
import symmeta_adapter as stream


RULE = "first-exact-forkserver-native-v1"


def receipt(a):
    return dict(shape=list(a.shape), dtype=a.dtype.str, sha256=sha256(a.tobytes()).hexdigest())


def hashes(fields):
    return {key: receipt(a) for key, a in sorted(fields.items())}


def differences(left, right):
    return sorted(key for key in left.keys() | right.keys()
                  if key not in left or key not in right or not identical(left[key], right[key]))


def wrist_keys(n):
    return {f"{prefix}/{i}/obs/pixels/image2" for prefix in ("raw", "mapped") for i in range(n)}


def image_diagnostic(left, right, keys, work):
    """Keep compact typed pixel deltas for changed image leaves, with complete hashes elsewhere."""
    values = {}
    for key in keys:
        if key not in left or key not in right:
            continue
        a, b = left[key], right[key]
        if (a.shape != b.shape or a.dtype != np.uint8 or b.dtype != np.uint8
                or a.ndim != 3 or a.shape[-1] != 3 or not a.size):
            continue
        delta = b.astype(np.int16) - a
        coords = np.argwhere(np.any(delta != 0, axis=2))
        yx = coords[:16]
        values[key] = dict(pixels=len(coords), channel_counts=np.count_nonzero(delta, axis=(0, 1)).tolist(),
                           max_abs=int(np.max(np.abs(delta))), yx=yx.tolist(),
                           left=a[yx[:, 0], yx[:, 1]].tolist(), right=b[yx[:, 0], yx[:, 1]].tolist())
    ref.write(work / "images.json", values)
    return values


def retryable(left, right, changes, diagnostic, n):
    if len(changes) != 2 or not set(changes) <= wrist_keys(n):
        return False
    mapped, raw = changes
    return (mapped == "mapped/" + raw.removeprefix("raw/") and raw in diagnostic and
            diagnostic[raw]["pixels"] == 1 and diagnostic[raw]["max_abs"] == 1 and
            all(identical(arm[mapped], arm[raw]) for arm in (left, right)))


def raw_arrays(obs):
    return {"raw/" + key: a for key, a in observations(obs).items()}


def freeze_arrays(value):
    if isinstance(value, dict):
        for child in value.values():
            freeze_arrays(child)
    elif isinstance(value, np.ndarray):
        value.flags.writeable = False


def mapped_arrays(bridge, obs):
    mapped = [stream.policy_observation(o, anchor, estimate[0])
              for o, anchor, estimate in zip(obs, bridge.anchor, bridge.estimates, strict=True)]
    return {"mapped/" + key: a for key, a in observations(mapped).items()}


class HeldRetry(retry.Retry):
    accepted = False
    delivered = False

    def park(self, indices, reset):
        self.indices, self.reset_args = indices, reset
        self.rows = super().call(indices, "reset", reset)
        return self.rows

    def call(self, idx, op, args):
        ref.require(self.accepted, "Native episode call before exact-pair admission")
        if op == "reset":
            ref.require(not self.delivered and idx == self.indices and args == self.reset_args,
                        "Native loop changed or repeated the held reset")
            self.delivered = True
            return self.rows
        ref.require(self.queries, f"Native {op} before query zero")
        return super().call(idx, op, args)

    def predict_chunk(self, obs, tasks, seeds, query):
        ref.require(self.accepted and self.delivered, "Policy query before admission or held reset")
        if query:
            return super().predict_chunk(obs, tasks, seeds, query)
        want = self.snapshot
        checked = raw_arrays(obs)
        checked.update(mapped_arrays(self, obs), tasks=np.array(tasks), noise_seeds=np.array(seeds))
        ref.require(not differences(checked, {key: want[key] for key in checked}),
                    "Held observation or input changed on entry to the native query")
        ref.require(not differences(raw_arrays(obs), {key: a for key, a in want.items() if key.startswith("raw/")}),
                    "Mapping mutated a held native input")
        ref.require(identical(self.noise.detach().cpu().numpy(), want["noise"]), "Held seeded noise changed")
        calls = 0

        def supplied(actual, q):
            nonlocal calls
            ref.require(calls == 0 and q == 0 and list(actual) == list(seeds), "Native noise request differs")
            calls += 1
            return self.noise

        with patch.object(self.vla, "_noise", supplied):
            proposal = super().predict_chunk(obs, tasks, seeds, query)
        ref.require(calls == 1, "Native query bypassed prechecked noise")
        ref.require(not differences(raw_arrays(obs), {key: a for key, a in want.items() if key.startswith("raw/")}),
                    "Native query mutated a held observation")
        return proposal


@dataclass
class Candidate:
    bridge: HeldRetry
    pids: list
    seconds: dict

    @property
    def snapshot(self):
        return self.bridge.snapshot

    def check_alive(self):
        procs = self.bridge.pool.native.procs
        ref.require([p.pid for p in procs] == self.pids and all(p.is_alive() for p in procs),
                    "Held worker exited or was replaced")


class TimedPool(retry.Pool):
    def __init__(self, native, meter):
        super().__init__(native, meter)
        self.receive_seconds = {}

    def call(self, idx, op, args):
        if op not in {"reset", "full_state"}:
            return super().call(idx, op, args)
        start = time.monotonic()
        try:
            return super().call(idx, op, args)
        finally:
            self.receive_seconds[op] = time.monotonic() - start


def close_pool(native):
    native.close()
    for process in native.procs:
        process.join(timeout=5)
        ref.require(process.exitcode == 0, f"Native child {process.pid} did not exit cleanly")


def capture(stack, vla, episodes, keys, support, arm, work, directory, factory):
    begin = time.monotonic()
    meter = retry.Meter(directory / f"{arm}.jsonl", arm)
    stack.callback(meter.log.close)
    native = factory(len(keys))
    stack.callback(close_pool, native)
    pids = [process.pid for process in native.procs]
    ready = time.monotonic()
    bridge = HeldRetry(vla, TimedPool(native, meter), keys, support, arm, work)
    idx = list(range(len(episodes)))
    reset = [(e.task, retry.CELL, e.init_id, e.scene_seed, True) for e in episodes]
    try:
        rows = bridge.park(idx, reset)
    except Exception:
        ref.write(directory / f"{arm}-rpc-failure.json", dict(pids_before=pids,
                  pids_after=[p.pid for p in native.procs], counts=meter.counts))
        raise
    received = time.monotonic()
    candidate = Candidate(bridge, pids, {})
    if meter.counts.get("internal_reset_retry") or [p.pid for p in native.procs] != pids:
        ref.write(directory / f"{arm}-rpc-failure.json", dict(pids_before=pids,
                  pids_after=[p.pid for p in native.procs], counts=meter.counts))
        raise ValueError("Implicit native reset retry invalidates the candidate")
    candidate.check_alive()
    obs = [row["obs"] for row in rows]
    for value in obs:
        freeze_arrays(value)
    seeds = [e.noise_seed for e in episodes]
    bridge.noise = vla._noise(seeds, 0)
    fields = dict(bridge.opening, tasks=np.array([row["description"] for row in rows]),
                  native_horizon=np.array([row["max_steps"] for row in rows]),
                  noise_seeds=np.array(seeds), chunk_seeds=np.array([retry.chunk_seed(seed, 0) for seed in seeds]),
                  noise=bridge.noise.detach().cpu().numpy().copy())
    fields.update(mapped_arrays(bridge, obs))
    for array in fields.values():
        array.flags.writeable = False
    bridge.snapshot = fields
    changed = differences(raw_arrays(obs), {key: a for key, a in fields.items() if key.startswith("raw/")})
    candidate.seconds.update(pool_start=ready-begin, reset_receive=bridge.pool.receive_seconds["reset"],
                             state_receive=bridge.pool.receive_seconds["full_state"],
                             mapping_and_noise=time.monotonic()-received)
    ref.write(directory / f"{arm}-opening.json", dict(pids=pids, seconds=candidate.seconds,
              counts=meter.counts, mapping_mutation=changed, leaves=hashes(fields)))
    ref.require(not changed, "Mapping mutated the native reset arrays")
    return candidate


def pair(vla, keys, support, work, cap, admit, consume, pool_factory=retry.ImportedEnvPool):
    """Admission and consumption run in one scope while both accepted pools remain alive."""
    episodes = [rollout.Episode(*key) for key in keys]
    sensor = wrist_keys(len(keys))
    baseline = None
    for ordinal in range(1, cap + 1):
        if not admit():
            return "budget_quarantine"
        round_dir = work / f"draw{ordinal:03d}"
        round_dir.mkdir()
        begin = time.monotonic()
        record = dict(rule=RULE, keys=keys, ordinal=ordinal, state="started")
        ref.write(round_dir / "pair.json", record)
        with ExitStack() as stack:
            left, right = [capture(stack, vla, episodes, keys, support, arm, work, round_dir, pool_factory)
                           for arm in retry.ARMS]
            left.check_alive()
            right.check_alive()
            a, b = left.snapshot, right.snapshot
            both = differences(a, b)
            if baseline is None:
                baseline = {key: val for key, val in a.items() if key not in sensor}
            drift = [differences(baseline, {key: val for key, val in x.items() if key not in sensor})
                     for x in (a, b)]
            record.update(differences=both, cross_draw_nonwrist_differences=drift,
                          baseline_nonwrist=hashes(baseline),
                          arm_opening_sha256=[file_hash(round_dir / f"{arm}-opening.json") for arm in retry.ARMS],
                          decision_seconds=time.monotonic()-begin)
            if both or any(drift):
                image = image_diagnostic(a, b, both, round_dir)
                record["images_sha256"] = file_hash(round_dir / "images.json")
            allowed = not any(drift) and (not both or retryable(a, b, both, image, len(keys)))
            record["state"] = "accepted" if allowed and not both else "wrist_rejected" if allowed else "structural_halt"
            ref.write(round_dir / "pair.json", record)
            if not allowed:
                raise ValueError(f"Opening outside diagnosed wrist variation; see {round_dir}")
            if not both:
                # Move the original reset journals, so the unmodified validator sees the accepted physical resets.
                for arm in retry.ARMS:
                    (round_dir / f"{arm}.jsonl").rename(work / f"{arm}.jsonl")
                ref.write(work / "prequery.json", dict(record, arm_receipts=[hashes(a), hashes(b)],
                          worker_pids=[left.pids, right.pids], accepted_directory=round_dir.name))
                for candidate in (left, right):
                    candidate.check_alive()
                    candidate.bridge.accepted = True
                consume((left, right), episodes)
                return "publishable"
        ref.write(round_dir / "cleanup.json", dict(seconds=time.monotonic()-begin, closed=True))
    return "draw_cap_quarantine"


def verify_prequery(work, keys):
    """Read back a complete accepted pair using only the opening, draw and cost receipts."""
    prequery = ref.read(work / "prequery.json")
    arms = prequery["arm_receipts"]
    ref.require(prequery["rule"] == RULE and prequery["keys"] == list(map(list, keys)) and
                prequery["state"] == "accepted" and not prequery["differences"] and
                not any(prequery["cross_draw_nonwrist_differences"]) and arms[0] == arms[1],
                "Published opening is not the exact admitted pair")
    rounds = sorted(work.glob("draw[0-9]*"), key=lambda path: int(path.name[4:]))
    ref.require(len(rounds) == prequery["ordinal"] and
                [p.name for p in rounds] == [f"draw{i:03d}" for i in range(1, len(rounds) + 1)] and
                rounds[-1].name == prequery["accepted_directory"], "Draw sequence differs")
    for index, directory in enumerate(rounds):
        draw = ref.read(directory / "pair.json")
        accepted = index == len(rounds) - 1
        mode = "accepted" if accepted else "wrist_rejected"
        ref.require(draw["state"] == mode and not any(draw["cross_draw_nonwrist_differences"]) and
                    draw["arm_opening_sha256"] ==
                    [file_hash(directory / f"{arm}-opening.json") for arm in retry.ARMS],
                    "Draw digest or ordering differs")
        if accepted:
            ref.require(arms == [ref.read(directory / f"{arm}-opening.json")["leaves"] for arm in retry.ARMS],
                        "Accepted arm digests differ")
        else:
            ref.require(ref.read(directory / "cleanup.json")["closed"] and
                        draw["images_sha256"] == file_hash(directory / "images.json"),
                        "Rejected draw lacks cleanup or image evidence")
    with np.load(work / "opening.npz", allow_pickle=False) as tape:
        fields = set(arms[0]) - {"native_horizon"}
        ref.require(set(tape.files) == fields | {"proposal"} and
                    all(arms[0][key] == receipt(tape[key]) for key in fields), "Native opening changed after admission")
    return dict(prequery_sha256=file_hash(work / "prequery.json"), opening_sha256=file_hash(work / "opening.npz"),
                rejected_draws=len(rounds)-1)


def run_pair(vla, pipe, reader, keys, work, cap, admit):
    with retry.fresh(work, "support", len(keys)) as (pool, meter):
        support = retry.calibrate(vla, pool, pipe, keys, work)
        support_budget = meter.counts.copy()
    ref.require(not support_budget.get("internal_reset_retry"), "Implicit support reset retry invalidates calibration")
    ref.write(work / "contract.json", dict(keys=keys, label=0, code=os.environ["RETRY_PIN"], reader=reader,
              upstream=ref.PIN, policy=vla.revision, cell=retry.CELL, batch=len(keys), support_budget=support_budget))
    lim = jobs.speed_limit("pi05")
    ref.require(lim and lim > 0, "Missing reference unsafe speed envelope")

    def consume(candidates, episodes):
        for candidate in candidates:
            candidate.check_alive()
            bridge = candidate.bridge
            arm = bridge.arm
            result = rollout.run_batch(bridge, bridge, episodes, bridge, retry.CELL, speed_limit=lim)
            jobs._save_traces(work / arm / "traces/retry.npz", result.traces)
            ref.write(work / f"{arm}.json", dict(rows=result.rows, arm=arm, budget=bridge.pool.meter.counts,
                      queries=bridge.queries, opening_sha256=file_hash(work / "opening.npz"), speed_limit=lim))

    status = pair(vla, keys, support, work, cap, admit, consume)
    if status == "publishable":
        ref.write(work / "paired-reset-verified.json", verify_prequery(work, keys))
    return status
