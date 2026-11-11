"""Restore a single verified archive part to its original Baige paths."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile

PFS = Path("/mowenjie")
BUFFER = 8 * 1024**2


def checksum(path, output=None):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(BUFFER), b""):
            digest.update(block)
            if output is not None:
                output.write(block)
    return digest.hexdigest()


def restore_member(incoming, row):
    original = Path(row["source"])
    if original.is_file() and not original.is_symlink() and checksum(original) == row["sha256"]:
        return
    if os.path.lexists(original) or original.resolve() != original:
        raise FileExistsError(original)
    metadata = row["source_stat"]
    with tempfile.NamedTemporaryFile(dir=original.parent, prefix=".restore-") as staged:
        if checksum(incoming, staged) != row["sha256"] or staged.tell() != row["bytes"]:
            raise RuntimeError(f"Invalid restored bytes for {original}")
        staged.flush()
        os.fsync(staged.fileno())
        if checksum(Path(staged.name)) != row["sha256"]:
            raise RuntimeError(f"Failed PFS readback for {original}")
        os.fchown(staged.fileno(), metadata["st_uid"], metadata["st_gid"])
        os.fchmod(staged.fileno(), stat.S_IMODE(metadata["st_mode"]))
        os.utime(staged.name, ns=(metadata["st_mtime_ns"], metadata["st_mtime_ns"]))
        os.link(staged.name, original)
    print(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_bytes())
    rows = receipt["rows"]
    archive = Path(receipt["archive"])
    for row in rows:
        path = Path(row["source"])
        relative = path.relative_to(PFS)
        if len(relative.parts) < 4 or relative.parts[1] != "outputs" or ".." in relative.parts:
            raise ValueError(f"Unexpected restore path: {path}")
    if receipt["archive_type"] == "file" and len(rows) == 1:
        restore_member(archive, rows[0])
        return
    if receipt["archive_type"] != "tar.zst":
        raise ValueError("Unknown archive format")
    with tempfile.TemporaryDirectory(prefix="baige-archive-restore-") as scratch:
        staging = Path(scratch)
        local = staging / "part.tar.zst"
        with local.open("xb") as copy:
            actual = checksum(archive, copy)
        if actual != receipt["archive_sha256"] or local.stat().st_size != receipt["archive_bytes"]:
            raise RuntimeError(f"Invalid archive: {archive}")
        extracted = staging / "extracted"
        extracted.mkdir()
        subprocess.run(["tar", "--acls", "--xattrs", "-I", "zstd", "-xf", str(local),
                        "-C", str(extracted)], check=True)
        for row in rows:
            restore_member(extracted / Path(row["source"]).relative_to(PFS), row)


if __name__ == "__main__":
    main()
