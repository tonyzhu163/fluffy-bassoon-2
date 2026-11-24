#!/usr/bin/env bash
# Run on Baige CPU. The production PFS store is read-only for this audit.
set -euo pipefail
[[ $# == 5 && $1 =~ ^(first|full|partial)$ && $2 =~ ^(retry[0-9]+|render-cont[0-9]+)$ && $3 =~ ^job-[a-z0-9]+$ && $4 =~ ^[1-9][0-9]*$ ]]
mode=$1
registry=$5
root=/mowenjie/fluffy-bassoon-2/outputs/reground-retry-wrist-20261123-r1
store=$root/scored
native_attempt=$store/attempts/$2
RETRY_PIN=ac28987f221e336cef2ea64dc16dfffc9f0e74c9
host=$3-master-0
export RETRY_PIN CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
scratch=$(mktemp -d /tmp/paper-wrist-native-readback.XXXXXXXX)
trap 'rm -rf -- "$scratch"' EXIT
here=$root/src-$RETRY_PIN
attempt=$scratch
source "$here/env.sh"
core=$(/usr/bin/python3 -IB -c 'import os; print(min(os.sched_getaffinity(0)))')
taskset -c "$core" "$py" -Bu - "$mode" "$native_attempt" "$scratch" "$host" "$4" "$registry" <<'PY'
import contextlib
from hashlib import file_digest, sha256
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import subprocess

mode, attempt_dir, scratch_dir, host, payload, registry = sys.argv[1:]
authorized = json.loads(registry)
attempt, scratch = Path(attempt_dir), Path(scratch_dir)
store = attempt.parent.parent
with contextlib.redirect_stdout(sys.stderr):
    import numpy as np
    import reference as ref
    ref.configure()
    import retry
    import summarize
    from worker import selected
    from reground.eval.jobs import load_traces

    def sha(path):
        with path.open("rb") as stream:
            return file_digest(stream, "sha256").hexdigest()

    grid = retry.schedule()
    pins = os.environ["RETRY_PIN"]
    assignments = selected(grid, "full")
    origins = {}
    donor_journals = set()

    def bridge_files(where, expected):
        checkout = store.parent / f"render-src-{expected['bridge_pin']}"
        folder = checkout / "outputs/reground-retry-wrist-20261123-r1/render-smoke"
        smoke = store.parent / "render-smoke/attempts" / expected["smoke_id"] / "paired-report.json"
        observed = []
        for line in (where / "bridge.sha256").read_text().splitlines():
            digest, filename = line.split(maxsplit=1)
            observed.append((digest, Path(filename.strip())))
        paths = [folder / name for name in ("continue.sh", "continue_pair.py", "smoke.py")] + [smoke]
        assert [path for _, path in observed] == paths
        assert [digest for digest, _ in observed] == [sha(path) for path in paths]
        assert observed[0][0] == expected["launcher_sha256"] and observed[-1][0] == expected["smoke_sha256"]
        git = ["git", "-C", str(checkout)]
        assert subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() == expected["bridge_pin"]
        subprocess.run([*git, "diff", "--quiet", "HEAD", "--", str(folder.relative_to(checkout))], check=True)
        plan = ref.read(smoke)
        assert plan["code"] == expected["smoke_pin"] and plan["cases"] == [list(key) for key in grid[54]]
        return next(trial["mode"] for trial in plan["trials"] if trial["exact"])

    def admission_for(source):
        if source not in origins:
            expected = authorized[source]
            where = store / "attempts" / source
            origin = f"{expected['job']}-master-0"
            wall = str(expected["payload_seconds"])
            probe = ref.read(where / "gpu.json")
            its_rank = ref.read(where / "rank0.json")
            device, = probe["devices"]
            assert probe["hostname"] == origin
            assert probe["status"] == "passed" and probe["exclusive_training_admission"]
            assert probe["allocation_type"] == "baige_container"
            assert device["cuda_name"] == "NVIDIA GeForce RTX 4090" and device["allocation_ordinal"] == 0
            assert its_rank["device"] == device and its_rank["rank"] == 0 and its_rank["phase"] == "full"
            scheduled = expected["batches"]
            if scheduled == "full":
                scheduled = assignments
            assert its_rank["batches"] == scheduled and set(scheduled) <= set(assignments)
            assert len(set(scheduled)) == len(scheduled)
            record = {"host": origin, "rank": its_rank, "launcher_sha256": expected["launcher_sha256"]}
            if (where / "launch.txt").is_file():
                pin, host_seen, phase, cards, start, wall_seen = (where / "launch.txt").read_text().split()
                launcher_sha, _ = (where / "launcher.sha256").read_text().split(maxsplit=1)
                assert (pin, host_seen, phase, cards, wall_seen) == (pins, origin, "full", "1", wall)
                assert launcher_sha == expected["launcher_sha256"]
                record["launch"] = [pin, host_seen, phase, cards, start, wall_seen]
            else:
                assert "bridge_pin" in expected
            if "bridge_pin" in expected:
                assert scheduled == [54]
                record["physical_protocol"] = bridge_files(where, expected)
                record["bridge_pin"] = expected["bridge_pin"]
                if (where / "exit.txt").is_file() and "launch" not in record:
                    code, start, end, cards, seconds = (where / "exit.txt").read_text().split()
                    assert cards == "1" and int(end) - int(start) == int(seconds)
                    record["launch"] = [pins, origin, "full", cards, start, wall]
                    record["launch_metadata_source"] = "bridge hashes, GPU admission, exit and scheduler handoff"
            origins[source] = record
            if "dispatch_pin" in expected:
                dispatch = ref.read(where / "dispatch.json")
                assert dispatch["attempt"] == source and dispatch["assigned"] == scheduled
                assert dispatch["science_pin"] == pins and dispatch["scheduler_id"] == expected["job"]
                assert dispatch["dispatch_pin"] == expected["dispatch_pin"]
                assert dispatch["native_hashes"]["retry"] == sha(Path(retry.__file__))
                assert dispatch["native_hashes"]["dispatch_launcher"] == record["launcher_sha256"]
                origins[source]["dispatch"] = dispatch
            if "sequential_pin" in expected:
                checkout = store.parent / f"sequential-src-{expected['sequential_pin']}"
                git = ["git", "-C", str(checkout)]
                assert subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() == expected["sequential_pin"]
                subprocess.run([*git, "diff", "--quiet", "HEAD", "--", "recovery/sequential-reset"], check=True)
                file = checkout / "recovery/sequential-reset/sequential.py"
                expected_sha, reported_file = (where / "implementation.sha256").read_text().split(maxsplit=1)
                assert file == Path(reported_file.strip()) and sha(file) == expected_sha
                record["sequential_pin"] = expected["sequential_pin"]
        return origins[source]

    current = admission_for(attempt.name)
    assert current["host"] == host
    assert authorized[attempt.name]["payload_seconds"] == int(payload)
    batches = sorted((store / "batches").glob("batch[0-9][0-9][0-9]"))
    assert {path.name for path in batches} <= {f"batch{i:03d}" for i in assignments}
    if mode == "full":
        assert len(batches) == len(grid) == 70
        assert (attempt / "exit.txt").read_text().split()[0] == "0"
    if mode == "partial":
        assert (attempt / "exit.txt").exists()
    if mode == "first":
        own = [path for path in batches if ref.read(path / "producer.json")["attempt"] == attempt.name]
        path = min(own, key=lambda item: ref.read(item / "producer.json")["ended"])
        old = {int(path.name[5:]) for path in batches if path not in own}
        assert int(path.name[5:]) == next(index for index in current["rank"]["batches"] if index not in old)
        batches = [path]
    verified = []
    for path in batches:
        producer = ref.read(path / "producer.json")
        assert producer["attempt"] in authorized
        origin = admission_for(producer["attempt"])
        assert producer["host"] == origin["host"]
        assert producer["rank"] == 0 and producer["egl"] == origin["rank"]["egl"]
        assert producer["code"] == ref.read(path / "contract.json")["code"] == pins
        index = int(path.name[5:])
        assert index in origin["rank"]["batches"]
        keys = grid[index]
        if mode != "full":  # The registered summarizer independently validates every full-panel batch.
            retry.validate(path, keys)
        entry = {"batch": path.name, "producer": producer, "keys": keys}
        if "dispatch" in origin:
            sidecar = ref.read(path / "dispatch.json")
            dispatch = origin["dispatch"]
            assert all(sidecar[key] == dispatch[key] for key in (
                "science_pin", "recovery_pin", "dispatch_pin", "scheduler_id", "native_hashes", "cell", "scene_seed"
            ))
            assert sidecar["batch"] == index and sidecar["attempt"] == producer["attempt"]
            assert sidecar["keys"] == [list(key) for key in keys]
            assert sidecar["support_keys"] == [list(retry.support_key(key)) for key in keys]
            if mode != "first":
                assert dispatch["statuses"][str(index)] == "published"
            files = {str(file.relative_to(path)) for file in path.rglob("*") if file.is_file()}
            assert files == set(sidecar["files"]) | {"dispatch.json"}
            assert all(sha(path / name) == digest for name, digest in sidecar["files"].items())
            entry["dispatch_sha256"] = sha(path / "dispatch.json")
        if "bridge_pin" in origin:
            expected = authorized[producer["attempt"]]
            pv = ref.read(path / "prequery.json")
            assert pv["operational_pin"] == origin["bridge_pin"] and pv["qualification_pin"] == expected["smoke_pin"]
            assert pv["smoke_report_sha256"] == expected["smoke_sha256"]
            assert pv["physical_protocol"] == origin["physical_protocol"] and any(pv["batch054_archive_exact"])
            assert not pv["differences"] and pv["arms"][0] == pv["arms"][1]
            with np.load(path / "opening.npz", allow_pickle=False) as tape:
                frozen = {name: dict(shape=list(tape[name].shape), dtype=str(tape[name].dtype),
                                     sha256=sha256(tape[name].tobytes()).hexdigest())
                          for name in tape.files if name != "proposal"}
            assert frozen == pv["arms"][0]
            entry["prequery_sha256"] = sha(path / "prequery.json")
        if "sequential_pin" in origin:
            where = store / "attempts" / producer["attempt"]
            sidecar = ref.read(path / "sequential.json")
            expected = authorized[producer["attempt"]]
            assert sidecar["batch"] == index and sidecar["keys"] == [list(key) for key in keys]
            assert sidecar["source_attempt"] == expected["donors"][str(index)]
            assert sidecar["science_pin"] == pins and sidecar["operational_pin"] == origin["sequential_pin"]
            assert sidecar["native_primary_file"] == sha(Path(retry.__file__))
            assert sidecar["shell_receipt_sha256"] == sha(where / "launcher.sha256")
            for name, artifacts in sidecar["source_sha256"].items():
                assert name in authorized
                original = store / "attempts" / name
                assert all(sha(original / filename) == digest for filename, digest in artifacts.items())
            assert all(sha(path / filename) == digest for filename, digest in sidecar["files"].items())
            donor = sidecar["source_sha256"][sidecar["source_attempt"]]
            for filename in ("contract.json", "support.json", "support.npz", "opening.npz", "translation.json",
                             "translation/traces/retry.npz"):
                assert sidecar["files"][filename] == donor[f"{path.name}/{filename}"]
            assert list(sidecar["candidate_sha256"])[-1] == sidecar["accepted"]
            assert set(sidecar["candidate_sha256"]) == {
                file.parent.name for file in where.glob(f"{path.name}-reset-*/candidate.json")
            }
            for who, digest in sidecar["candidate_sha256"].items():
                file = where / who / "candidate.json"
                assert sha(file) == digest
                candidate = ref.read(file)
                if who == sidecar["accepted"]:
                    assert candidate["status"] == "terminal" and candidate["accepted_guard"] and not candidate["differences"]
                    with np.load(path / "opening.npz", allow_pickle=False) as tape:
                        original_leaves = {
                            key: dict(shape=list(tape[key].shape), dtype=str(tape[key].dtype),
                                      sha256=sha256(tape[key].tobytes()).hexdigest())
                            for key in tape.files if key != "proposal"
                        }
                    assert candidate["opening"] == original_leaves
                else:
                    assert candidate["status"] == "ineligible_pixel" and candidate["before_controls"]
                    assert candidate["budget"] == {"reset": len(keys)}
            markers = list(where.glob("*sequential-first-wrist-control*.json"))
            marker, = [file for file in markers if sha(file) == sidecar["first_control_marker_sha256"]]
            assert ref.read(marker)["candidate"] == sidecar["accepted"]
            for stage, relative in sidecar["source_journals"].items():
                storage_journal = store / relative
                assert storage_journal.is_file() and stage in {"support", "translation"}
                assert storage_journal.parent == store / "attempts" / sidecar["source_attempt"] / path.name
                assert storage_journal.name == f"{stage}.jsonl" and not (path / storage_journal.name).exists()
                donor_journals.add(storage_journal)
            entry["sequential_sha256"] = sha(path / "sequential.json")
            entry["sequential_candidate_count"] = len(sidecar["candidate_sha256"])
        if mode == "first":
            reports = {arm: ref.read(path / f"{arm}.json") for arm in retry.ARMS}
            traces = [load_traces(path / arm) for arm in retry.ARMS]
            changes = []
            for key in keys:
                left, right = [trace[key]["A"][:10] for trace in traces]
                prefix = min(len(left), len(right))
                assert prefix > 0
                np.testing.assert_array_equal(left[:prefix, [0, 1, 2, 5, 6]], right[:prefix, [0, 1, 2, 5, 6]])
                changes.append(float(np.abs(left[:prefix, 3:5] - right[:prefix, 3:5]).max()))
            entry.update(
                support_keys=[row["support"] for row in ref.read(path / "support.json")],
                opening_sha256=reports[retry.ARMS[0]]["opening_sha256"],
                max_opening_wrist_xy_change=changes,
                native_pairs_validated=len(keys),
            )
        verified.append(entry)
    receipt = {
        "job": host.removesuffix("-master-0"),
        "code": pins,
        "attempt": attempt.name,
        "mode": mode,
        "checked_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "device": current["rank"]["device"],
        "rank": current["rank"],
        "sources": origins,
        "verified_batches": verified,
    }
    if mode == "full":
        for name in ("batches", "attempts"):
            (scratch / name).symlink_to(store / name, target_is_directory=True)
        (scratch / "readback-full").mkdir()
        summarize.report(scratch, "full")
        dest = scratch / "readback-full"
        cases = json.loads((dest / "cases.json").read_text())
        summary = json.loads((dest / "summary.json").read_text())
        if donor_journals:
            sealed_journals = list((store / "batches").glob("batch*/*.jsonl"))
            partial_journals = list((store / "attempts").glob("*/batch*/*.jsonl"))
            physical = sealed_journals + partial_journals
            assert donor_journals <= set(partial_journals) and not donor_journals.intersection(sealed_journals)
            assert len(physical) == len({(file.stat().st_dev, file.stat().st_ino) for file in physical})
            summary["intended_physical_interactions"] = summarize.journals(sealed_journals + sorted(donor_journals))
            summary["donor_journals_counted_once"] = [str(path.relative_to(store)) for path in sorted(donor_journals)]
        for source, origin in origins.items():
            if source not in summary["attempts"]:
                exit_fields = (store / "attempts" / source / "exit.txt").read_text().split()
                summary["attempts"][source] = {"launch": origin["launch"], "exit": exit_fields,
                                                "source": origin["launch_metadata_source"]}
        assert summary["code"] == receipt["code"] and summary["new_pairs"] == len(cases) == 280
        assert len({(row["task"], row["init_id"], row["rep"]) for row in cases}) == 280
        assert summary["new_seed_labels"] == [0] and summary["original_rows_per_arm"] == 1400
        assert summary["distinct_support_layouts"] == 120
        for goal in ("goal_1", "goal_2"):
            assert summary["paired"]["success"][goal]["n_unique_starts"] == 35
        receipt.update(summary=summary, cases=cases)
    elif mode == "partial":
        journals = list((store / "batches").glob("batch*/*.jsonl"))
        journals.extend((store / "attempts").glob("*/batch*/*.jsonl"))
        receipt["physical_interaction"] = summarize.journals(journals)
        receipt["exit"] = (attempt / "exit.txt").read_text().split()
print(json.dumps(receipt, indent=2))
PY
