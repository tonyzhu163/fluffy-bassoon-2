"""Pack the exact first30 source updates for existing single-tar query readers.

CPU-only artifact preparation. Closed archive members are checked against their
bound inventories; the older Faucet files receive a new hash/CRC snapshot.
This does not claim tensor-semantic or teacher-label qualification.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time


PATTERN = re.compile(r'^environment/rank-(\d{2})/training-i(\d{3})-round00-chunk(\d{2})\.pt\.gz$')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--spec-sha256', required=True)
    p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cold', type=Path, required=True)
    a = p.parse_args()
    assert sys.platform == 'linux'
    code = Path(__file__).resolve().parents[2]
    assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == a.pin
    assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain', '--untracked-files=no'])
    assert sha(a.spec) == a.spec_sha256
    spec = json.loads(a.spec.read_text())
    task = spec['task']
    assert task in ('OpenDrawer', 'CloseToasterOvenDoor', 'CoffeeSetupMug',
                    'PickPlaceCounterToStove', 'TurnOffStove', 'TurnOnSinkFaucet')
    assert spec['updates'] == 30
    assert a.output.is_relative_to('/tmp/robocasa-six-task-scale-20261113-r1')
    assert a.cold.is_relative_to('/mnt/bos/adgfyl/strong-plant-archives/robocasa-six-task-scale-20261113-r1')
    a.output.mkdir(parents=True, exist_ok=False)
    a.cold.mkdir(parents=True, exist_ok=False)
    started = time.time()
    packed = a.output / 'raw.tar'
    rows, seen, segments = [], set(), []
    with tarfile.open(packed, 'w:', format=tarfile.USTAR_FORMAT) as dest:
        def add(name, raw, segment, expected=None):
            match = PATTERN.fullmatch(name)
            if match is None:
                return
            rank, version, chunk = map(int, match.groups())
            if not segment['first_version'] <= version < segment['stop_version']:
                return
            assert 0 <= rank < 8 and 0 <= version < 30 and 1 <= chunk <= 12
            key = (rank, version, chunk)
            assert key not in seen, key
            digest = hashlib.sha256(raw).hexdigest()
            if expected is not None:
                assert expected['bytes'] == len(raw) and expected['sha256'] == digest
            # Read to EOF so the complete gzip CRC and uncompressed length check.
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                while stream.read(8 << 20):
                    pass
            canonical = f'{task}/{name}'
            entry = tarfile.TarInfo(canonical)
            entry.size, entry.mode, entry.mtime = len(raw), 0o600, 0
            dest.addfile(entry, io.BytesIO(raw))
            rows.append(dict(name=canonical, rank=rank, version=version, chunk=chunk,
                bytes=len(raw), sha256=digest, stage='corrected-first30',
                original_name=name, source_segment=len(segments)))
            seen.add(key)
            if len(rows) % 240 == 0:
                progress = dict(task=task, members=len(rows), expected_members=2880,
                                elapsed_seconds=time.time() - started)
                write(a.output / 'PROGRESS.json', progress)
                print(json.dumps(progress), flush=True)

        for segment in spec['segments']:
            count_before = len(rows)
            if segment['kind'] == 'archive':
                receipt_path = Path(segment['receipt'])
                assert sha(receipt_path) == segment['receipt_sha256']
                receipt = json.loads(receipt_path.read_text())
                assert receipt['verified'] and receipt['all_stageback_members_verified']
                archive = Path(receipt['cold'])
                assert archive.stat().st_size == receipt['archive_bytes']
                expected = {x['path']: x for x in receipt['files']}
                with tarfile.open(archive, 'r:') as source:
                    for member in source:
                        match = PATTERN.fullmatch(member.name)
                        if match is None or not segment['first_version'] <= int(match[2]) < segment['stop_version']:
                            continue
                        assert member.isfile() and member.name in expected
                        add(member.name, source.extractfile(member).read(), segment, expected[member.name])
                reference = dict(segment, archive=str(archive), archive_sha256=receipt['archive_sha256'],
                                 verification='Every selected member rehashed against bound verified archive inventory')
            elif segment['kind'] == 'directory':
                root = Path(segment['root'])
                assert task == 'TurnOnSinkFaucet'
                assert root.is_relative_to('/mowenjie/fluffy-bassoon-2/outputs/robocasa-eight-hour-babysit-20261104-r1/source5/TurnOnSinkFaucet')
                for version in range(segment['first_version'], segment['stop_version']):
                    for rank in range(8):
                        for chunk in range(1, 13):
                            name = f'environment/rank-{rank:02d}/training-i{version:03d}-round00-chunk{chunk:02d}.pt.gz'
                            path = root / name
                            assert path.is_file() and not path.is_symlink()
                            add(name, path.read_bytes(), segment)
                reference = dict(segment, verification='New complete payload hash/CRC snapshot of the closed Faucet source; no prior raw-payload hash claimed')
            else:
                raise ValueError('Unknown source segment kind')
            expected_count = (segment['stop_version'] - segment['first_version']) * 8 * 12
            assert len(rows) - count_before == expected_count
            segments.append(dict(reference, selected_members=expected_count))
    assert seen == {(r, v, c) for r in range(8) for v in range(30) for c in range(1, 13)}
    by_name = {r['name']: r for r in rows}
    with tarfile.open(packed, 'r:') as source:
        for member in source:
            row = by_name[member.name]
            row['offset_data'] = member.offset_data
            assert member.size == row['bytes']
            assert hashlib.sha256(source.extractfile(member).read()).hexdigest() == row['sha256']
    local_hash = sha(packed)
    cold_tar = a.cold / 'raw.tar'
    with packed.open('rb') as source, cold_tar.open('xb') as dest:
        shutil.copyfileobj(source, dest, 8 << 20)
    assert cold_tar.stat().st_size == packed.stat().st_size and sha(cold_tar) == local_hash
    write(a.cold / 'TAR-INDEX.json', sorted(rows, key=lambda x: (x['rank'], x['version'], x['chunk'])))
    write(a.cold / 'SPEC.json', spec)
    result = dict(completed=True, CPU_only=True, task=task, updates=30, raw_members=2880,
        episodes=3840, query_decisions=46080, source_commit=a.pin, spec_sha256=a.spec_sha256,
        archive=str(cold_tar), archive_sha256=local_hash, archive_bytes=packed.stat().st_size,
        index_path=str(a.cold / 'TAR-INDEX.json'), index_sha256=sha(a.cold / 'TAR-INDEX.json'),
        segments=segments, gzip_CRC_passed=True, every_packed_member_readback=True,
        cold_full_SHA_readback=True, tensor_semantics_checked=False, teacher_labels_collected=False,
        source_files_unchanged=True, local_staging_reclaimed=False,
        hostname=socket.gethostname(), elapsed_seconds=time.time() - started)
    write(a.cold / 'PACKED.json', result)
    write(a.output / 'PACKED.json', result)
    # Readers use the verified cold archive. Do not retain a second raw corpus
    # on scratch after its complete member and full-archive readback succeeds.
    packed.unlink()
    result['local_staging_reclaimed'] = True
    write(a.cold / 'PACKED.json', result)
    write(a.output / 'PACKED.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
