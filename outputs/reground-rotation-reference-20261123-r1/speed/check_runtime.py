"""Read-only verification of the native interpreter, distributions and LIBERO assets."""
import csv
from importlib import metadata
import json
import os
from pathlib import Path
import sys

from packaging.version import Version

import reference as ref


def verify():
    qualified = Path(os.environ["REGROUND_REFERENCE_DIR"])
    runtime = qualified / "runtime"
    ref.require(Path(ref.__file__).resolve() == qualified / "reference.py" and
                Path(sys.executable) == ref.ROOT / "env/bin/python" and
                Path(sys.prefix) == ref.ROOT / "env" and sys.version_info[:3] == (3, 12, 15),
                "Wrong reference module or interpreter")
    sys.path.insert(0, str(runtime))
    from prepare_cache import verify as verify_object
    versions = dict(line.split("==") for line in (runtime / "requirements-linux.txt").read_text().splitlines())
    for package, version in versions.items():
        ref.require(Version(metadata.version(package)) == Version(version), package)
    receipt = ref.read(ref.ROOT / "assets-verified.json")
    assets = Path(receipt["snapshot"])
    installed = Path(sys.prefix, "lib/python3.12/site-packages/libero/libero/assets")
    ref.require(installed.resolve(strict=True) == assets, installed)
    with (runtime / "libero-assets.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    ref.require({p.relative_to(assets).as_posix() for p in assets.rglob("*") if p.is_file()} ==
                {r["path"] for r in rows}, "Asset tree differs from release")
    for row in rows:
        meta = dict(size=int(row["size"]), oid=row["git_oid"])
        if row["lfs_sha256"]:
            meta.update(lfs=True, sha256=row["lfs_sha256"])
        verify_object(assets / row["path"], meta)
    ref.configure()
    return dict(python=sys.executable, upstream=ref.PIN, reference=str(qualified),
                packages=len(versions), assets=len(rows), asset_revision=receipt["revision"])


if __name__ == "__main__":
    print(json.dumps(verify()))
