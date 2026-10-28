"""Prepare a new immutable native-source copy; metadata/file work only, no Torch."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


LOADER = 'rlinf/models/embodiment/openpi/checkpoint.py'
ORIGINAL = '048b6d48a372775d6c2cff9dd6d95ffc54480999d9d6c6786b546ead3b24298a'
PATCHED = 'f910e2160f1b506eec5953d931fd2fb7fec71d05ac6ab02a9f76d6e8109587bd'


def fingerprint(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if any(part in ('.git', '__pycache__', '.pytest_cache') or
               part.endswith(('.pyc', '.egg-info')) for part in path.relative_to(root).parts):
            continue
        if path.is_symlink():
            data = path.readlink().as_posix().encode()
        elif path.is_file():
            data = path.read_bytes()
        else:
            continue
        result[path.relative_to(root).as_posix()] = hashlib.sha256(data).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip()):
        raise ValueError('Require exact clean reviewed compatibility source')
    before = fingerprint(args.original_root)
    if before[LOADER] != ORIGINAL or args.output_root.exists():
        raise ValueError('Require original verified native loader and a fresh output identity')
    patch = repo/'patches/rlinf-e1a82ebf-flow-noise-keys.patch'
    shutil.copytree(args.original_root, args.output_root, symlinks=True,
        ignore=shutil.ignore_patterns('.git', '__pycache__', '.pytest_cache', '*.pyc', '*.egg-info'))
    subprocess.run(['patch', '--batch', '--forward', '-p1', '-i', str(patch)],
                   cwd=args.output_root, check=True)
    after = fingerprint(args.output_root)
    if set(before) != set(after) or {name for name in before if before[name] != after[name]} != {LOADER}:
        raise ValueError('Only the declared checkpoint loader may differ')
    if after[LOADER] != PATCHED or fingerprint(args.original_root) != before:
        raise ValueError('Patched bytes differ or original source changed')
    receipt = dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_commit=args.source_commit, native_base_commit='e1a82ebf0c1af69da6ab9b1519f5f3692fe33566',
        original_root=str(args.original_root), output_root=str(args.output_root),
        patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(), original_loader_sha256=ORIGINAL,
        patched_loader_sha256=PATCHED, unchanged_files=len(before)-1, files_sha256=after,
        model_or_cuda_loaded=False, original_preserved=True, runtime_qualification_pending=True)
    (args.output_root/'FLOW_NOISE_COMPAT_READY.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'files_sha256'}, indent=2))


if __name__ == '__main__':
    main()
