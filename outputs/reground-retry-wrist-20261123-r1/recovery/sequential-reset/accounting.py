"""Audit sequential provenance and produce outcome-blind physical budgets after sealing."""
from pathlib import Path
import sys

import reference as ref
import sequential as seq


def main():
    store, output = map(Path, sys.argv[1:])
    ref.configure()
    import retry
    import summarize

    batch = store / "batches" / seq.BATCH
    receipt = ref.read(batch / "sequential.json")
    ref.require(receipt["batch"] == 54 and receipt["keys"] == list(map(list, seq.KEYS)) and
                receipt["source_attempt"] == seq.LEFT and receipt["science_pin"] == seq.SCIENCE,
                "Sequential provenance differs")
    producer = ref.read(batch / "producer.json")
    attempt = store / "attempts" / producer["attempt"]
    seq.immutable_checkout(store.parent / f"sequential-src-{receipt['operational_pin']}",
                           receipt["operational_pin"], "recovery/sequential-reset")
    ref.require(seq.sha(attempt / "launcher.sha256") == receipt["shell_receipt_sha256"] and
                seq.sha(attempt / "sequential-first-wrist-control.json") == receipt["first_control_marker_sha256"],
                "Missing admitted-wrist receipt")
    seq.check_sources_unchanged(store, receipt["source_sha256"])
    ref.require(all(seq.sha(batch / name) == digest for name, digest in receipt["files"].items()),
                "Published paired bytes differ")
    ref.require(all(seq.sha(batch / name) == receipt["source_sha256"][seq.LEFT][f"{seq.BATCH}/{name}"]
                    for name in seq.LEFT_FILES), "Published left differs from the fixed original")
    candidates = []
    for name, digest in sorted(receipt["candidate_sha256"].items()):
        path = attempt / name / "candidate.json"
        ref.require(seq.sha(path) == digest, "Candidate provenance differs")
        candidates.append(ref.read(path))
    accepted = candidates[-1]
    expected = seq.load_opening(batch / "opening.npz")
    expected.pop("proposal")
    ref.require(accepted["opening"] == seq.receipts(expected) and accepted["candidate"] == receipt["accepted"] and
                accepted["status"] == "terminal" and accepted["accepted_guard"] and not accepted["differences"] and
                all(row["status"] == "ineligible_pixel" and row["before_controls"] and
                    row["budget"] == {"reset": len(seq.KEYS)} for row in candidates[:-1]),
                "The first exact wrist candidate was not used")
    paths = sorted((store / "batches").glob("batch*/*.jsonl"))
    partial = sorted((store / "attempts").glob("*/batch*/*.jsonl"))
    adopted = [store / receipt["source_journals"][arm] for arm in ("support", "translation")]
    ref.require(all(path in partial and path not in paths for path in adopted), "Adopted journals would double-count")
    full = paths + partial
    ref.require(len(full) == len({(p.stat().st_dev, p.stat().st_ino) for p in full}),
                "Physical evidence journal alias would double-count")
    retry.validate(batch, seq.KEYS)
    groups = list((store / "batches").glob("batch[0-9][0-9][0-9]"))
    ref.write(output, dict(sealed_groups=len(groups), aggregate_ready=len(groups) == 70,
                          sequential_sha256=seq.sha(batch / "sequential.json"),
                          intended_physical_interactions=summarize.journals(paths + adopted),
                          all_sealed_and_partial_physical_interactions=summarize.journals(full)))


if __name__ == "__main__":
    main()
