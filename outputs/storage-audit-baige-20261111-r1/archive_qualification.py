"""Move the verified failed MIKASA qualification archive to BOS."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import time

from archive_rollouts import BOS, WORK, digest, has_readers, read, require, stamp, write

SOURCE = Path("/mowenjie/fluffy-bassoon/outputs/mikasa-fullft-reset-s1000-new-20261107-r3/qualification-step2-preserved.tar.zst")
ORIGINAL = SOURCE.with_name("qualification-step2-archive.json")
COLD = BOS / "strong-plant-archives" / WORK.name / "mikasa-failed-qualification"
ARCHIVE = COLD / SOURCE.name
RECEIPT = COLD / "RECEIPT.json"


def prepare():
    require(SOURCE.resolve() == SOURCE and not has_readers([SOURCE]), "Check the source and its readers")
    failed = read(SOURCE.with_name("state.json"))
    successor = SOURCE.parent.with_name("mikasa-fullft-reset-s1000-new-20261107-r4")
    require(failed["state"] == "failed", "Qualification is still live")
    require(read(successor / "retry-plan.json")["production_uses_qualification_checkpoint"] is False,
            "Production depends on the failed qualification")
    original = read(ORIGINAL)
    before = stamp(SOURCE)
    require(original["state"] == "complete_verified" and original["full_member_hashes_verified"]
            and original["archive"] == str(SOURCE) and original["archive_bytes"] == before["st_size"],
            "Missing original verified archive receipt")
    COLD.mkdir(parents=True, exist_ok=True)
    expected = original["archive_sha256"]
    require(digest(SOURCE, None if ARCHIVE.exists() else ARCHIVE) == expected, "Source checksum mismatch")
    staged = WORK / SOURCE.name
    staged.unlink(missing_ok=True)
    try:
        require(digest(ARCHIVE, staged) == expected and staged.stat().st_size == before["st_size"],
                "Cold archive readback failed")
        require(digest(staged) == expected, "Restored archive checksum mismatch")
    finally:
        staged.unlink(missing_ok=True)
    require(stamp(SOURCE) == before, "Source changed during verification")
    raw = ORIGINAL.read_bytes()
    cold_original = COLD / ORIGINAL.name
    cold_original.write_bytes(raw)
    require(cold_original.read_bytes() == raw, "Original qualification receipt readback failed")
    receipt = {
        "source": str(SOURCE), "archive": str(ARCHIVE), "bytes": before["st_size"],
        "sha256": expected, "source_stat": before, "allocated_bytes": before["st_blocks"] * 512,
        "metadata_sha256": hashlib.sha256(raw).hexdigest(), "qualification_job": failed["job_id"],
        "cold_readback_verified": True, "restored_file_verified": True, "verified_unix": time.time(),
    }
    write(RECEIPT, receipt)
    write(WORK / "MIKASA-ARCHIVE.json", receipt)


def release():
    scheduler = read(WORK / "MIKASA-SCHEDULER.json")
    require(scheduler["selected_consumers"] == [] and time.time() - scheduler["observed_unix"] < 600,
            "Refresh the scheduler and confirm there are no live consumers")
    receipt = read(WORK / "MIKASA-ARCHIVE.json")
    require(receipt == read(RECEIPT) and ARCHIVE.stat().st_size == receipt["bytes"], "Verified copy missing")
    require(stamp(SOURCE) == receipt["source_stat"] and not has_readers([SOURCE]), "Source changed or in use")
    pointer = SOURCE.with_name(SOURCE.name + ".cold.json")
    write(pointer, receipt)
    require(stamp(SOURCE) == receipt["source_stat"], "Source changed before release")
    SOURCE.unlink()
    require(not SOURCE.exists() and ORIGINAL.is_file() and read(pointer) == receipt,
            "Post-release verification failed")
    receipt.update(state="complete", released_unix=time.time(), scheduler=scheduler,
                   filesystem=shutil.disk_usage("/mowenjie")._asdict())
    write(WORK / "MIKASA-RELEASE.json", receipt)
    write(COLD / "RELEASE.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "release"))
    args = parser.parse_args()
    require(os.path.ismount(BOS), f"BOS is not mounted: {BOS}")
    (prepare if args.phase == "prepare" else release)()
