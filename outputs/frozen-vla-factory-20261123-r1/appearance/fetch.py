"""Fetch immutable public snapshots into an external working directory."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen, urlretrieve


def fetch(root, kind, repo, revision, folder):
    url = f"https://huggingface.co/api/{kind}/{repo}/revision/{revision}"
    with urlopen(url) as r:
        metadata = json.load(r)
    assert metadata["sha"] == revision
    names = [item["rfilename"] for item in metadata["siblings"]]
    prefix = "datasets/" if kind == "datasets" else ""

    def download(name):
        if name in {".gitattributes", "README.md"}:
            return
        if folder == "vlm" and ("/" in name or name.endswith(".safetensors")):
            return
        path = root / folder / name
        if path.is_file():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        for attempt in range(3):
            try:
                urlretrieve(f"https://huggingface.co/{prefix}{repo}/resolve/{revision}/{quote(name)}", tmp)
                tmp.replace(path)
                break
            except OSError:
                if attempt == 2:
                    raise

    with ThreadPoolExecutor(8) as workers:
        list(workers.map(download, names))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", type=Path)
    args = parser.parse_args()
    fetch(args.workdir, "models", "lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d", "policy")
    fetch(args.workdir, "models", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "4cde9d208e19519c36917aa170fffc7e381aa0ad", "vlm")
    fetch(args.workdir, "datasets", "lerobot/libero-assets", "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6", "assets")
