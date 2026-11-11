"""Import verified public LIBERO Git objects on Baige; no model execution."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


PIN = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
ARCHIVE_SHA256 = "dc7cac1ff01c2030ef9a2c352e8466a1419311aeb7042844eb0e74569f5932e6"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    root = args.project
    run = root / "outputs/libero-gen-history-pilot-20261111-r1"
    target = root / "cache/bpp-58125e6-20261111-r1/deps/LIBERO"
    import_dir = root / "setups/libero-gen-history-pilot-20261111-r1/source-import01"
    receipt = run / "SOURCE-IMPORT01.json"
    assert not receipt.exists() and not import_dir.exists(), "Fresh import identity required"
    value = hashlib.sha256()
    with args.archive.open("rb") as stream:
        while block := stream.read(8 * 2**20):
            value.update(block)
    assert value.hexdigest() == ARCHIVE_SHA256
    import_dir.mkdir()
    with tarfile.open(args.archive) as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            assert not path.is_absolute() and ".." not in path.parts
            assert path.parts[0] == "LIBERO.git" and (member.isdir() or member.isfile())
        archive.extractall(import_dir)
    store = import_dir / "LIBERO.git"
    subprocess.run(["git", "--git-dir", str(store), "fsck", "--connectivity-only"], check=True)
    actual = subprocess.check_output(["git", "--git-dir", str(store), "rev-parse", "HEAD"], text=True).strip()
    assert actual == PIN
    preserved = None
    if target.exists():
        # Previous bounded clone did not complete. Preserve any remnants;
        # do not replace an existing valid checkout or a nonempty directory.
        assert target.is_dir() and not any(target.iterdir()), "Existing source requires inspection"
        backup = run / "empty-source-before-import01"
        assert not backup.exists()
        target.rename(backup)
        preserved = str(backup)
    subprocess.run(["git", "clone", "--no-hardlinks", str(store), str(target)], check=True)
    actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    assert actual == PIN
    assert not subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True)
    subprocess.run(["git", "-C", str(target), "fsck", "--connectivity-only"], check=True)
    subprocess.run(["git", "-C", str(target.parent.parent), "diff", "--quiet", "--", "deps/LIBERO"], check=True)
    result = {"complete": True, "imported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "commit": PIN, "archive_sha256": ARCHIVE_SHA256, "source": str(target),
              "preserved_empty_directory": preserved, "scope": "Git source only; runtime unqualified"}
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
