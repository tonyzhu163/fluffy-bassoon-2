#!/usr/bin/env python3
"""Install missing official RoboTwin objects without overwriting shared assets.

This performs only archive/file operations and is safe on a setup host. Download
objects.zip from TianxingChen/RoboTwin2.0 at the revision below beforehand.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile
import zlib


REVISION = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"
SHA256 = "6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27"
ARCHIVE_BYTES = 3737778549
MANIFEST_NAME = ".fb2-archive-manifest.json"


def crc32_file(path):
    crc = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            crc = zlib.crc32(block, crc)
    return crc & 0xFFFFFFFF


def install(archive, root):
    if archive.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("Official objects archive has the wrong byte count")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != SHA256:
        raise ValueError(f"Official objects archive SHA-256 mismatch: {digest}")
    objects = root / "assets" / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    lock = objects / ".fb2-install.lock"
    lock.mkdir()  # Concurrent installations must not race.
    try:
        entries = []
        installed = preserved = 0
        with zipfile.ZipFile(archive) as bundle:
            members = []
            for member in bundle.infolist():
                relative = Path(member.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe archive path: {member.filename}")
                if member.is_dir():
                    continue
                if relative.parts[0] != "objects" or len(relative.parts) < 2:
                    raise ValueError(f"Unexpected archive root: {member.filename}")
                relative = Path(*relative.parts[1:])
                # Root README/scripts are not model assets and the manifest
                # intentionally inventories object-family files only.
                if len(relative.parts) < 2 or relative.name == ".DS_Store":
                    continue
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"Symlink in archive: {member.filename}")
                destination = objects / relative
                # Audit conflicts before extracting anything. Existing contents
                # may serve other active runs and are never replaced here.
                if destination.exists():
                    if (not destination.is_file()
                            or destination.stat().st_size != member.file_size
                            or crc32_file(destination) != member.CRC):
                        raise FileExistsError(f"Existing asset differs: {destination}")
                members.append((member, relative, destination))
            for member, relative, destination in members:
                if destination.exists():
                    preserved += 1
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    partial = destination.with_name(destination.name + ".fb2-part")
                    with bundle.open(member) as source, partial.open("xb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    if (partial.stat().st_size != member.file_size
                            or crc32_file(partial) != member.CRC):
                        raise ValueError(f"Extracted asset CRC mismatch: {relative}")
                    os.replace(partial, destination)
                    installed += 1
                entries.append({
                    "path": str(relative), "size": member.file_size,
                    "crc32": member.CRC,
                })
        manifest = {
            "repository": "TianxingChen/RoboTwin2.0", "hf_revision": REVISION,
            "archive_sha256": SHA256, "archive_bytes": ARCHIVE_BYTES,
            "files": sorted(entries, key=lambda item: item["path"]),
        }
        path = objects / MANIFEST_NAME
        data = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        if path.exists() and path.read_text() != data:
            raise FileExistsError(f"Existing install manifest differs: {path}")
        if not path.exists():
            temporary = path.with_suffix(".json.part")
            temporary.write_text(data)
            os.replace(temporary, path)
        return {
            "status": "installed_and_crc_verified", "robotwin_root": str(root),
            "archive_path": str(archive), "archive_sha256": SHA256,
            "archive_bytes": ARCHIVE_BYTES, "hf_revision": REVISION,
            "installed_files": installed, "preserved_matching_files": preserved,
            "file_count": len(entries),
            "uncompressed_bytes": sum(item["size"] for item in entries),
            "manifest_path": str(path),
            "manifest_sha256": hashlib.sha256(data.encode()).hexdigest(),
        }
    finally:
        lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = install(args.archive.resolve(), args.robotwin_root.resolve())
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
