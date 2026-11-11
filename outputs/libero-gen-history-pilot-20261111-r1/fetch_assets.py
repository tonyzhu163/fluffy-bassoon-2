"""Download only the pinned, public pilot assets and verify their LFS digests."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.destination.mkdir(parents=True, exist_ok=True)
    total = sum(f["size"] for a in manifest["assets"].values() for f in a["files"].values())
    if shutil.disk_usage(args.destination).free < total + 20 * 2**30:
        raise RuntimeError("Insufficient storage for selected assets plus 20 GiB reserve")
    receipt = {"complete": False, "files": []}
    for key, asset in manifest["assets"].items():
        if asset["missing"]:
            raise RuntimeError(f"Missing published assets: {asset['missing']}")
        kind, repo_id = key.split("/", 1)
        repo_type = "dataset" if kind == "datasets" else "model"
        for name, metadata in asset["files"].items():
            path = Path(hf_hub_download(repo_id, filename=name, revision=asset["revision"],
                                      repo_type=repo_type, token=False,
                                      local_dir=args.destination / repo_type))
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while block := stream.read(8 * 2**20):
                    digest.update(block)
            if path.stat().st_size != metadata["size"] or digest.hexdigest() != metadata["lfs"]["sha256"]:
                raise RuntimeError(f"Asset verification failed: {name}")
            receipt["files"].append({"path": str(path), "bytes": path.stat().st_size,
                                     "sha256": digest.hexdigest(), "revision": asset["revision"]})
            args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
            print("VERIFIED", name, path.stat().st_size, flush=True)
    receipt["complete"] = True
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
