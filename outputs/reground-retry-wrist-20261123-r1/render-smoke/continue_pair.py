"""Exact-opening continuation using pinned ReGround rollout, support and scoring."""
import argparse
from contextlib import ExitStack
from functools import partial
import os
from pathlib import Path
from unittest.mock import patch

import reference as ref


def validate_sealed(attempt, retry):
    store = attempt.parent.parent / "batches"
    grid = retry.schedule()
    originals = []
    for path in sorted(store.glob("batch[0-9][0-9][0-9]")):
        batch = int(path.name[5:])
        ref.require(batch < len(grid), f"Unexpected sealed batch {batch}")
        retry.validate(path, grid[batch])
        producer = ref.read(path / "producer.json")
        if producer["attempt"] == "retry02":
            ref.require(producer["code"] == os.environ["RETRY_PIN"], "Original producer scientific pin differs")
            originals.append(batch)
        else:
            bridge_receipt = ref.read(path / "prequery.json")
            ref.require(not bridge_receipt["differences"] and bridge_receipt["arms"][0] == bridge_receipt["arms"][1],
                        "Existing continuation lacks matching prequery receipt")
    ref.require(len(originals) == 54 and 54 not in originals, "Original sealed cohort differs")
    return originals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("smoke", type=Path)
    parser.add_argument("smoke_pin")
    parser.add_argument("--validate-originals-only", action="store_true")
    args = parser.parse_args()
    ref.configure()
    import numpy as np
    from reground.eval import jobs, rollout
    from opening import file_hash, observations
    import retry
    from smoke import CASES, DiagnosticPool, compare, leaf_receipt
    import symmeta_adapter as stream
    import worker

    if args.validate_originals_only:
        validated = validate_sealed(args.attempt, retry)
        ref.write(args.attempt / "validated-originals.json", dict(batches=validated, pin=os.environ["RETRY_PIN"]))
        return

    plan = ref.read(args.smoke / "paired-report.json")
    ref.require(plan["code"] == args.smoke_pin and plan["cases"] == list(map(list, CASES)), "Wrong qualification pin/cases")
    ref.require([row["mode"] for row in plan["trials"]] == ["forkserver-native", "spawn-native", "spawn-seed0"],
                "Incomplete physical comparison")
    selected = next((row for row in plan["trials"] if row["exact"]), None)
    ref.require(selected is not None, "Qualification found no byte-exact physical protocol")
    ref.require(selected["support_match"] and len(selected["trials"]) == 2 and
                {row["lifecycle"] for row in selected["trials"]} == {"sequential", "parked"} and
                all(not (row["opening"] or row["processed"] or row["mutation_detected"] or row["repeat_render_drift"])
                    and any(not any(gaps) for gaps in row["archive"].values()) for row in selected["trials"]),
                "Selected protocol lacks a complete exact comparison")
    transport, seed_name = selected["mode"].split("-")
    seed = 0 if seed_name == "seed0" else None

    archived_work = args.attempt.parent / "retry04" / "batch054"
    archive_fields = ("support", "query", "initial_sim_sha256", "estimate", "scene_seed", "noise_seed")
    archived_support = [{key: row[key] for key in archive_fields} for row in ref.read(archived_work / "support.json")]
    archived_openings = []
    for name in ("retry02", "retry04"):
        with np.load(args.attempt.parent / name / "batch054" / "opening.npz", allow_pickle=False) as tape:
            archived_openings.append({key: tape[key] for key in tape.files if key != "proposal"})

    class ParkedRetry(retry.Retry):
        def park(self, idx, reset_args):
            self.held = idx, reset_args, super().call(idx, "reset", reset_args)
            return self.held[2]

        def call(self, idx, op, call_args):
            if op != "reset":
                return super().call(idx, op, call_args)
            ref.require(self.held is not None, "Duplicate physical reset")
            lanes, expected, original = self.held
            ref.require(idx == lanes and call_args == expected, "Native runner changed registered reset")
            self.held = None
            return original

        def predict_chunk(self, obs, tasks, seeds, query):
            if query:
                return super().predict_chunk(obs, tasks, seeds, query)
            before = observations(obs)
            proposal = super().predict_chunk(obs, tasks, seeds, query)
            ref.require(not compare(before, observations(obs)), "Native query zero mutated its raw observations")
            return proposal

    class ScoringPool(DiagnosticPool):
        def __init__(self, n):
            self.transport, self.constructor_seed = transport, seed
            rollout.EnvPool.__init__(self, n)

    def run_pair(vla, pipe, reader, keys, work):
        critical = keys == CASES
        with retry.fresh(work, "support", len(keys)) as (pool, meter):
            support = retry.calibrate(vla, pool, pipe, keys, work)
            support_budget = meter.counts.copy()
        if critical:
            ref.require([{key: row[key] for key in archive_fields} for row in support] == archived_support,
                        "Fresh batch054 support does not reproduce the original enrollment and estimate")
        ref.write(work / "contract.json", dict(keys=keys, label=0, code=os.environ["RETRY_PIN"], reader=reader,
                  upstream=ref.PIN, policy=vla.revision, cell=retry.CELL, batch=len(keys), support_budget=support_budget))
        lim = jobs.speed_limit("pi05")
        ref.require(lim and lim > 0, "Missing native speed limit")
        idx = list(range(len(keys)))
        episodes = [rollout.Episode(*key) for key in keys]
        seeds = [episode.noise_seed for episode in episodes]
        reset_args = [(episode.task, retry.CELL, episode.init_id, episode.scene_seed, True) for episode in episodes]
        with ExitStack() as stack:
            bridges, receipts = [], []
            for arm in retry.ARMS:
                meter = retry.Meter(work / f"{arm}.jsonl", arm)
                stack.callback(meter.log.close)
                native = ScoringPool(len(keys))
                stack.callback(native.close)
                bridge = ParkedRetry(vla, retry.Pool(native, meter), keys, support, arm, work)
                rows = bridge.park(idx, reset_args)
                raw = [row["obs"] for row in rows]
                before = observations(raw)
                mapped = [stream.policy_observation(obs, anchor, est[0])
                          for obs, anchor, est in zip(raw, bridge.anchor, bridge.estimates, strict=True)]
                ref.require(not compare(before, observations(raw)), "Mapping mutated the raw reset")
                fields = dict(bridge.opening)
                fields.update({"mapped/" + key: value for key, value in observations(mapped).items()})
                fields.update(tasks=np.array([row["description"] for row in rows]), noise_seeds=np.array(seeds),
                              chunk_seeds=np.array([retry.chunk_seed(x, 0) for x in seeds]),
                              noise=vla._noise(seeds, 0).detach().cpu().numpy().copy())
                bridges.append(bridge)
                receipts.append(fields)
            changes = compare(*receipts)
            archive_match = [not compare(archive, receipts[0]) for archive in archived_openings] if critical else []
            ref.write(work / "prequery.json", dict(
                operational_pin=os.environ["CONTINUATION_PIN"], qualification_pin=args.smoke_pin,
                smoke_report_sha256=file_hash(args.smoke / "paired-report.json"), physical_protocol=selected["mode"],
                differences=changes, batch054_archive_exact=archive_match, arms=[{key: leaf_receipt(value) for key, value in fields.items()}
                                           for fields in receipts]))
            ref.require(not changes, "Fresh paired inputs differ before policy query zero")
            ref.require(not critical or any(archive_match), "Live batch054 opening differs from both native archives")
            for bridge in bridges:
                arm = bridge.arm
                result = rollout.run_batch(bridge, bridge, episodes, bridge, retry.CELL, speed_limit=lim)
                jobs._save_traces(work / arm / "traces/retry.npz", result.traces)
                ref.write(work / (arm + ".json"), dict(rows=result.rows, arm=arm, budget=bridge.pool.meter.counts,
                          queries=bridge.queries, opening_sha256=file_hash(work / "opening.npz"), speed_limit=lim))

    originals = validate_sealed(args.attempt, retry)
    ref.write(args.attempt / "validated-originals.json", dict(batches=originals, pin=os.environ["RETRY_PIN"],
              operational_pin=os.environ["CONTINUATION_PIN"], smoke_pin=args.smoke_pin, physical_protocol=selected["mode"]))
    with patch.object(retry, "run_pair", run_pair):
        worker.run(args.attempt, 0, "full")


if __name__ == "__main__":
    main()
