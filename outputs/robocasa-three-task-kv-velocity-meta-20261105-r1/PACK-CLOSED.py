"""Measure one-object cold storage and exact stage-back of an audited closed wave.

This qualification preserves every original hot file. No model, source cache,
checkpoint, incomplete wave or original artifact path is moved or deleted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import time


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cold', type=Path, required=True)
    args = p.parse_args()
    root = Path('/mowenjie/fluffy-bassoon-2')
    run = root / 'outputs/robocasa-three-task-kv-velocity-meta-20261105-r1'
    if (sys.platform != 'linux' or args.bank.is_symlink()
            or not args.bank.resolve().is_relative_to(run)
            or not any(args.output.resolve().is_relative_to(run / name) for name in ('migrations', 'archives'))
            or not args.cold.resolve().is_relative_to(Path('/mnt/bos/adgfyl/strong-plant-archives') / run.name)):
        raise ValueError('Use owned hot closed evidence, a fresh migration output and the private run archive')
    result = json.loads((args.bank / 'RESULT.json').read_text())
    audit = json.loads((args.bank / 'AUDIT.json').read_text())
    if (result.get('optimizer_updates') != 0 or not result.get('native_parameters_frozen')
            or not audit['passed'] or audit['result_sha256'] != sha(args.bank / 'RESULT.json')):
        raise ValueError('Require an independent complete-wave audit before any cold qualification')
    forward = None
    if os.environ.get('STEERING_STORAGE_ACCOUNTING'):
        import importlib.util
        spec = importlib.util.spec_from_file_location('pack_remaining_storage', Path(__file__).with_name('STORAGE-GUARD.py'))
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        forward = guard.require_capacity()
    if forward is None and shutil.disk_usage(root).free < 378 * 2**30:
        raise ValueError('Retain combined375GiB reserve plus3GiB temporary packing space')
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    entries = sorted(args.bank.rglob('*'))
    if any(path.is_symlink() for path in entries):
        raise ValueError('Do not follow linked or shared data')
    files = [path for path in entries if path.is_file()]
    inventory = [dict(relative=str(path.relative_to(args.bank)), bytes=path.stat().st_size,
                      sha256=sha(path)) for path in files]
    local = args.output / 'wave.tar'
    with tarfile.open(local, 'w') as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(args.bank)), recursive=False)
    digest = sha(local)
    packed_seconds = time.monotonic() - began
    args.cold.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with local.open('rb') as incoming, args.cold.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing, 8 << 20)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    cold_copy_seconds = time.monotonic() - start
    start = time.monotonic()
    if args.cold.stat().st_size != local.stat().st_size or sha(args.cold) != digest:
        raise ValueError('Cold object failed full size/SHA readback; retain all hot copies')
    cold_readback_seconds = time.monotonic() - start
    restored = args.output / 'restored'
    restored.mkdir()
    start = time.monotonic()
    with args.cold.open('rb') as incoming, tarfile.open(fileobj=incoming, mode='r|') as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if not member.isfile() or relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Restore only regular owned archive members')
            destination = restored / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, destination.open('xb') as output:
                shutil.copyfileobj(source, output, 8 << 20)
    restored_files = {str(path.relative_to(restored)) for path in restored.rglob('*') if path.is_file()}
    if restored_files != {row['relative'] for row in inventory}:
        raise ValueError('Stage-back leaf set differs')
    for row in inventory:
        path = restored / row['relative']
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Stage-back bytes differ')
        if sha(args.bank / row['relative']) != row['sha256']:
            raise ValueError('Original closed evidence changed; do not release anything')
    restored_seconds = time.monotonic() - start
    receipt = dict(passed=True, source=str(args.bank), cold=str(args.cold),
        forward_storage=forward,
        archive_sha256=digest, archive_bytes=local.stat().st_size,
        payload_bytes=sum(row['bytes'] for row in inventory), files=inventory,
        pack_and_inventory_seconds=packed_seconds, cold_copy_seconds=cold_copy_seconds,
        cold_full_SHA_readback_seconds=cold_readback_seconds,
        cold_stageback_and_all_member_SHA_seconds=restored_seconds,
        total_seconds=time.monotonic() - began,
        original_hot_evidence_untouched=True, all_stageback_bytes_verified=True,
        independent_source_audit_sha256=sha(args.bank / 'AUDIT.json'),
        hot_eviction_enabled=False, main_benchmark_held=True)
    (args.output / 'RESULT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
