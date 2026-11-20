"""Official file identities for offline loading of the original Google PaliGemma releases."""

import hashlib
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

# From the Hugging Face /api/models/{repo}/revision/{revision}?blobs=true responses.
# Small files use Git blob OIDs, LFS files use SHA256 of the file itself.
SHARED = {
    "model.safetensors.index.json": "242066c3d4e40edee88fc0e11671a5fd89336663",
    "tokenizer.json": "ef6773c135b77b834de1d13c75a4c98ab7a3684ffd602d1831e1f1bf5467c563",
    "tokenizer_config.json": "960dcec0bccc6e594d865eceb63c8078e164cfb2",
}
OPTIONAL = {
    "tokenizer.model": "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6",
    "special_tokens_map.json": "0c18fdd94629ea652e2ef36ed9a422474ad0a8e8",
    "added_tokens.json": "ff9246e420971b10c510b39a0de289132c2bf23f",
    "generation_config.json": "6aae7ddfdfa0b117a9975bd2c4830d6222404131",
}
RELEASES = {
    "9d4a7b496c93811f47247052b061dda435af2fed": (
        "google/paligemma-3b-mix-224",
        "d1d8734c9c3ad0ccfeea4afc270faa356c2ba515",
        (
            "63fd690cdad783794ce4c1d7893ef1f88bc26f4ae94b1925f5d3ad199dbbed87",
            "c24cf160613c9fcf69641b4eace7de94d07aaabba88800815137453a5fc32f38",
            "6222ec75c92d438886eaa3a2c5970c7e7973c213937a799ca6c5106f5fae829a",
        ),
    ),
    "b89393ba09b432265173f688b8a41a2c3c9e3e4c": (
        "google/paligemma-3b-pt-224",
        "35e4f46485b4d07967e7e9935bc3786aad50687c",
        (
            "22f48f9c252403ca55bce4f488b5d0b731c3f42850d3ef718f992e328248a6da",
            "b2b70c8b29f36554dd92a97d002cf0bcb867d3f5dc0a983d587f1598513e4cf0",
            "3e5197dfc42a88f4b1134d484c73b9cda815b630da87aa16ecf9c26a99cddb6a",
        ),
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def blob_id(path):
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


@contextmanager
def verified_snapshot(root):
    root = Path(root).resolve(strict=True)
    config_id = blob_id(root / "config.json")
    if config_id not in RELEASES:
        raise ValueError(f"{root}: config differs from both allowed upstream releases")
    repo, revision, shards = RELEASES[config_id]
    required = {"config.json": config_id, **SHARED}
    required.update({f"model-{i:05d}-of-00003.safetensors": value for i, value in enumerate(shards, 1)})
    required.update({k: v for k, v in OPTIONAL.items() if (root / k).is_file()})
    # Pass only checked files to Transformers. A staged directory may also contain
    # an adapter_config, single-file model or alternate tokenizer implementation.
    with TemporaryDirectory(prefix="crossarm-vlm-") as temp:
        clean = Path(temp)
        for name, upstream in required.items():
            file = root / name
            observed = sha256(file) if len(upstream) == 64 else blob_id(file)
            if observed != upstream:
                raise ValueError(f"{file}: observed {observed}, expected upstream {upstream}")
            (clean / name).symlink_to(file)
        yield clean, {"repo": repo, "revision": revision, "upstream_file_ids": required}
