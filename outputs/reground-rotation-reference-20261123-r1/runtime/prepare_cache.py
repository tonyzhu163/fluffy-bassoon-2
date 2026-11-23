"""Verify official Git objects, then expose only the verified files to offline Hub consumers."""
import argparse
import hashlib
import json
from pathlib import Path

MODEL = 'lerobot/pi05_libero_finetuned'
TOKENIZER = 'google/paligemma-3b-pt-224'


def git_id(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def verify(path, row):
    size = path.stat().st_size
    if size != row['size']:
        raise ValueError(f'{path}: expected {row["size"]} bytes, found {size}')
    if row.get('lfs'):
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if 'sha256' in row and digest != row['sha256']:
            raise ValueError(f'{path}: incorrect LFS payload')
        pointer = f'version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {size}\n'
        actual = git_id(pointer.encode())
    else:
        data = path.read_bytes()
        actual = git_id(data)
        digest = hashlib.sha256(data).hexdigest()
    if actual != row['oid']:
        raise ValueError(f'{path}: Git object {actual} differs from official {row["oid"]}')
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--tokenizer', required=True, type=Path)
    parser.add_argument('--hub', required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('hf-files.json').read_text())
    for name, source in ((MODEL, args.model), (TOKENIZER, args.tokenizer)):
        release = manifest[name]
        revision = release['revision']
        cache = args.hub / ('models--' + name.replace('/', '--'))
        snapshot = cache / 'snapshots' / revision
        snapshot.mkdir(parents=True, exist_ok=True)
        allowed = {row['path'] for row in release['files']}
        if {item.name for item in snapshot.iterdir()} - allowed:
            raise ValueError(f'Unexpected snapshot files under {snapshot}')
        for row in release['files']:
            path = (source / row['path']).resolve(strict=True)
            digest = verify(path, row)
            dest = snapshot / row['path']
            if dest.is_symlink():
                if dest.resolve(strict=True) != path:
                    raise ValueError(f'{dest}: unexpected target')
            else:
                dest.symlink_to(path)
            print(json.dumps(dict(repo=name, rev=revision, file=row['path'],
                                  bytes=row['size'], sha256=digest, git_oid=row['oid'])), flush=True)
        # Some loaders resolve "main" even when the caller passes a revision.
        refs = cache / 'refs'
        refs.mkdir(exist_ok=True)
        ref = refs / 'main'
        if ref.exists() and ref.read_text() != revision:
            raise ValueError(f'{ref}: conflicting revision')
        ref.write_text(revision)


if __name__ == '__main__':
    main()
