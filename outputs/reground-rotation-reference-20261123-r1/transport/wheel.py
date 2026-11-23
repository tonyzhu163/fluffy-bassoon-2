"""Qualify the installed Transformers package or stage only its exact package files."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
from zipfile import ZipFile


ROOT = Path("/mowenjie/fluffy-bassoon-2/outputs/reground-rotation-reference-20261123-r1")
WHEEL = "transformers-5.5.4-py3-none-any.whl"
SHA = "0bd6281b82966fe5a7a16f553ea517a9db1dee6284d7cb224dfd88fc0dd1c167"
SITE = ROOT / "env/lib/python3.12/site-packages"


def space(path):
    fs = os.statvfs(path)
    return {"bytes_available": fs.f_bavail * fs.f_frsize, "inodes_available": fs.f_favail}


def transport(dest):
    start = time.monotonic()
    data = (ROOT / "wheels" / WHEEL).read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA:
        raise ValueError("Wheel digest differs from the qualified runtime lock")
    with ZipFile(io.BytesIO(data)) as archive:
        members = [i for i in archive.infolist() if i.filename.startswith("transformers/") and not i.is_dir()]
        names = {i.filename for i in members}
        dirs = {p for name in names for p in Path(name).parents if p != Path(".")}
        if len(names) != len(members) or "transformers/__init__.py" not in names:
            raise ValueError("Incomplete or duplicated package")
        report = dict(wheel_sha256=SHA, wheel_bytes=len(data), files=len(names), directories=len(dirs),
                      package_bytes=sum(i.file_size for i in members))
        if dest is None:
            installed = {str(p.relative_to(SITE)) for p in (SITE / "transformers").rglob("*")
                         if "__pycache__" not in p.parts and p.is_file()}
            if installed != names:
                raise ValueError(f"Installed package differs: {sorted(installed ^ names)}")
        else:
            if dest.stat().st_dev == ROOT.stat().st_dev or any(dest.iterdir()):
                raise ValueError("Staging requires an empty private directory on container scratch")
            report["before"] = space(dest)
            block = os.statvfs(dest).f_frsize
            needed = sum((i.file_size + block - 1) // block for i in members) + len(dirs)
            if (report["before"]["bytes_available"] < needed * block or
                    report["before"]["inodes_available"] < len(names) + len(dirs)):
                raise OSError("Insufficient scratch space or inodes")
        tree = hashlib.sha256()
        for item in sorted(members, key=lambda i: i.filename):
            content = archive.read(item)
            if dest is None:
                if (SITE / item.filename).read_bytes() != content:
                    raise ValueError(f"Installed bytes differ: {item.filename}")
            else:
                path = dest / item.filename
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as out:
                    out.write(content)
                if path.read_bytes() != content:
                    raise OSError(f"Scratch readback differs: {item.filename}")
            tree.update(item.filename.encode() + b"\0" + hashlib.sha256(content).digest())
        report["package_tree_sha256"] = tree.hexdigest()
        if dest is not None:
            report["after"] = space(dest)
        report["elapsed_s"] = time.monotonic() - start
        print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audit-install", action="store_true")
    group.add_argument("--stage", type=Path, metavar="EMPTY_PRIVATE_DIRECTORY")
    args = parser.parse_args()
    transport(args.stage)
