"""Verify and release the seven previously audited BOS repeats."""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import stat
import time

base = Path("/mnt/bos/adgfyl")
archive = base / "strong-plant-archives"
evidence = archive / "outputs-retention-20261113-r1"
mapping = evidence / "BOS-REPEAT-REPLACEMENTS-20261113-r1.json"
receipt = evidence / "BOS-REPEAT-RELEASE-20261113-r1.json"
plan = json.loads(Path("PLAN.json").read_bytes())


def publish(path, value):
    data = (json.dumps(value, indent=2) + "\n").encode()
    with path.open("xb") as f:
        f.write(data)
    if path.read_bytes() != data:
        raise IOError(f"Readback failed for {path}")


def stamp(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.resolve() != path:
        raise ValueError(f"Unexpected object {path}")
    return info.st_size, info.st_mtime_ns, info.st_ino


def checked(path, size, digest):
    before = stamp(path)
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if before[0] != size or actual != digest or stamp(path) != before:
        raise IOError(f"Archive verification failed for {path}")
    return before


def check_pair(row):
    source, survivor = (base / row[key] for key in ("remove", "keep"))
    source.relative_to(archive)
    survivor.relative_to(archive)
    if source == survivor:
        raise ValueError(f"Identical paths for {source}")
    old = checked(source, row["bytes"], row["remove_sha256"])
    keep = checked(survivor, row["bytes"], row["keep_sha256"])
    if "manifest_sha256" in row:
        manifests = [
            p.with_suffix(".tar.manifest.json").read_bytes()
            for p in (source, survivor)
        ]
        if manifests[0] != manifests[1] or hashlib.sha256(manifests[0]).hexdigest() != row["manifest_sha256"]:
            raise IOError(f"Recovery manifests differ for {source}")
    return row, source, survivor, old, keep


if not os.path.ismount(base) or len(plan) != 7 or len({r["remove"] for r in plan}) != 7:
    raise RuntimeError("Unexpected BOS mount or release plan")
if {r["remove"] for r in plan} & {r["keep"] for r in plan}:
    raise ValueError("A survivor is also selected for removal")
with ThreadPoolExecutor(max_workers=4) as pool:
    checks = list(pool.map(check_pair, plan))
verified = time.time()
publish(mapping, {"verified_unix": verified, "pairs": plan})
released = []
for row, source, survivor, old, keep in checks:
    if stamp(source) != old or stamp(survivor) != keep:
        raise IOError(f"Object changed after verification: {source}")
    source.unlink()
    if os.path.lexists(source) or stamp(survivor) != keep:
        raise IOError(f"Release verification failed for {source}")
    released.append(row | {"released_unix": time.time()})
    print(f"Released {source.relative_to(base)}", flush=True)
for _, source, survivor, _, keep in checks:
    if os.path.lexists(source) or stamp(survivor) != keep:
        raise IOError(f"Final state changed for {source}")
result = {"verified_unix": verified, "completed_unix": time.time(),
          "released_bytes": sum(r["bytes"] for r in released),
          "replacement_map": str(mapping), "released": released}
publish(receipt, result)
Path("RELEASE.json").write_text(json.dumps(result, indent=2) + "\n")
