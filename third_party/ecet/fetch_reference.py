"""Fetch immutable ECET reference files outside this repository; no installation."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("upstream.json").read_text())
    destination = args.destination.expanduser().resolve()
    repository = Path(__file__).resolve().parents[2]
    if destination == repository or repository in destination.parents:
        raise ValueError("Reference destination must be outside the repository")
    destination.mkdir(parents=True, exist_ok=False)
    for relative, expected in manifest["files"].items():
        url = f"https://raw.githubusercontent.com/machinelearningnuremberg/ECET/{manifest['commit']}/{relative}"
        with urlopen(url, timeout=60) as response:
            content = response.read()
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise ValueError(f"Pinned ECET checksum mismatch: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (destination / "fluffy-reference-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified ECET {manifest['commit']} reference files in {destination}")


if __name__ == "__main__":
    main()
