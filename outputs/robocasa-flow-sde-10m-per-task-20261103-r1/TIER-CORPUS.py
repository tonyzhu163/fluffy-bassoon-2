"""Copy closed native corpora to BOS; reclaim only after full task audit.

Preparation streams a tar directly to BOS and verifies every member and the
entire object on readback. Reclaim is a separate, hash-bound operation so a
closed source can be copied while its remaining numerical audits finish.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile
import time

TASKS = ['OpenDrawer', 'CloseToasterOvenDoor', 'PickPlaceDrawerToCounter',
         'CoffeeSetupMug', 'TurnOffStove']
BOS = Path('/mnt/bos/adgfyl/strong-plant-archives/robocasa-flow-sde-10m-20261103-r1')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(16 << 20), b''):
            value.update(data)
    return value.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


class HashReader:
    def __init__(self, stream):
        self.stream, self.digest, self.bytes = stream, hashlib.sha256(), 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.bytes += len(data)
        return data


class HashWriter:
    def __init__(self, stream):
        self.stream, self.digest, self.bytes = stream, hashlib.sha256(), 0

    def write(self, data):
        written = self.stream.write(data)
        assert written == len(data)
        self.digest.update(data)
        self.bytes += written
        return written


def native_inventory(root, task_root, source):
    result = {}
    # Audit writers only modify the task's top-level receipts. Native stage
    # directories/accounting are immutable after native closure. Checkpoints
    # retain their separate qualified tiering policy.
    scopes = [source] if source != task_root else [p for p in task_root.iterdir()
        if p.is_dir() and (p.name.startswith(('q1-', 'production-')) or p.name == 'accounting')]
    for scope in scopes:
        assert not scope.is_symlink() and scope.resolve().is_relative_to(task_root.resolve())
        for path in sorted(scope.rglob('*')):
            relative = path.relative_to(root)
            if 'checkpoints' in relative.parts:
                continue
            assert not path.is_symlink(), str(path)
            if path.is_dir():
                continue
            info = path.stat()
            assert stat.S_ISREG(info.st_mode) and path.resolve().is_relative_to(task_root.resolve())
            result[str(relative)] = dict(bytes=info.st_size, mtime_ns=info.st_mtime_ns,
                mode=stat.S_IMODE(info.st_mode))
    assert result
    return result


def audit_inputs(paths, task, source=None, require_full=False):
    rows, refs, known, metadata, snapshots = [], [], {}, set(), {}
    for path in paths:
        data = path.read_bytes()
        value = json.loads(data)
        assert value['completed']
        refs.append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest()))
        snapshots[str(path)] = data.decode('utf-8')
        metadata.add(path)
        rows.extend(row for row in value['audits'] if row['task'] == task
                    and (source is None or row['root'] == str(source)))
    versions = sorted(row['collection_iteration'] for row in rows)
    assert versions and versions == list(range(1, max(versions) + 1)), versions
    if require_full:
        assert versions == list(range(1, 131))
    for row in rows:
        path = Path(row['raw_file_manifest'])
        assert sha(path) == row['raw_file_manifest_sha256']
        value = json.loads(path.read_text())
        assert value['completed']
        metadata.add(path)
        join = Path(row['actor_sampling_environment_join'])
        assert sha(join) == row['join_sha256']
        metadata.add(join)
        for name, item in value['files'].items():
            if name in known:
                assert known[name] == item
            known[name] = item
    return dict(rows=rows, refs=refs, known=known, metadata=metadata,
                snapshots=snapshots, through=max(versions))


def prefix_candidates(root, task_root, known):
    result = []
    for name in known:
        path = Path(name)
        assert path.is_relative_to(task_root)
        relative = path.relative_to(root)
        if not (path.name.endswith('.pt') or path.name.endswith('.pt.gz')):
            continue
        if 'environment' not in relative.parts and 'training-rollouts' not in relative.parts:
            continue
        assert 'sampling' not in relative.parts and 'checkpoints' not in relative.parts
        result.append(str(relative))
    return sorted(result)


def validate_closed(task_root, source, qualification):
    stage = source if qualification else task_root / 'production-r1'
    clock = json.loads((stage / 'clock.json').read_text())
    progress = json.loads((stage / 'progress.json').read_text())
    assert clock['exit_code'] == 0 and time.time() - clock['finished_unix'] >= 60
    if qualification:
        assert source.parent == task_root and source.name.startswith('q1-')
        assert progress['completed_iterations'] == 1
        return None
    assert source == task_root and progress['completed_iterations'] == 130
    after = json.loads((stage / 'after-evaluation.json').read_text())
    assert after['completed'] and after['global_step'] == 130
    for path in task_root.glob('*/manifest.json'):
        if not path.parent.name.startswith(('q1-', 'production-')):
            continue
        prior_clock = json.loads((path.parent / 'clock.json').read_text())
        assert time.time() - prior_clock['finished_unix'] >= 60
    cp = stage / task_root.name / 'checkpoints/global_step_130'
    boundary = json.loads((cp / 'pilot-boundary.json').read_text())
    assert boundary['completed'] and boundary['global_step'] == 130 and len(boundary['workers']) == 18
    assert not (cp / 'COLD-STORAGE.json').exists()
    for item in boundary['workers'] + boundary['actor_checkpoint_files']:
        path = Path(item['path'])
        assert path.resolve().is_relative_to(cp.resolve()) and sha(path) == item['sha256']
    return dict(path=str(cp), boundary_sha256=sha(cp / 'pilot-boundary.json'), hash_verified=True)


def stream_archive(root, source, inventory, known, destination, proof):
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    embedded = str(source.relative_to(root) / '.archive-manifests' / (destination.name + '.json'))
    assert embedded not in inventory
    with destination.open('xb', buffering=16 << 20) as output:
        writer = HashWriter(output)
        with tarfile.open(fileobj=writer, mode='w|', bufsize=16 << 20,
                          format=tarfile.PAX_FORMAT, copybufsize=8 << 20) as archive:
            for name, before in sorted(inventory.items()):
                path = root / name
                member = archive.gettarinfo(str(path), arcname=name)
                member.type, member.linkname, member.size = tarfile.REGTYPE, '', before['bytes']
                member.uid = member.gid = 0
                member.uname = member.gname = ''
                with path.open('rb') as source_stream:
                    reader = HashReader(source_stream)
                    archive.addfile(member, reader)
                item = dict(before, sha256=reader.digest.hexdigest())
                assert reader.bytes == item['bytes']
                if str(path) in known:
                    assert dict(bytes=item['bytes'], sha256=item['sha256']) == known[str(path)]
                files[name] = item
            assert set(known) <= {str(root / name) for name in files}
            manifest = dict(completed=True, root=str(root), source=str(source), files=files, proof=proof)
            manifest_bytes = (json.dumps(manifest, indent=2) + '\n').encode()
            member = tarfile.TarInfo(embedded)
            member.size, member.mode = len(manifest_bytes), 0o644
            archive.addfile(member, io.BytesIO(manifest_bytes))
        output.flush()
        os.fsync(output.fileno())
        expected, size = writer.digest.hexdigest(), writer.bytes
    assert destination.stat().st_size == size
    seen = set()
    with destination.open('rb', buffering=16 << 20) as stored:
        reader = HashReader(stored)
        with tarfile.open(fileobj=reader, mode='r|', bufsize=16 << 20) as archive:
            for member in archive:
                assert member.isfile() and member.name not in seen
                seen.add(member.name)
                stream = archive.extractfile(member)
                digest = hashlib.sha256()
                count = 0
                for data in iter(lambda: stream.read(16 << 20), b''):
                    digest.update(data)
                    count += len(data)
                if member.name == embedded:
                    assert count == len(manifest_bytes) and digest.hexdigest() == hashlib.sha256(manifest_bytes).hexdigest()
                else:
                    assert member.name in files
                    assert count == files[member.name]['bytes'] and digest.hexdigest() == files[member.name]['sha256']
        for data in iter(lambda: reader.read(16 << 20), b''):
            pass
        assert reader.bytes == size and reader.digest.hexdigest() == expected
    assert seen == set(files) | {embedded}
    sidecar = destination.with_name(destination.name + '.manifest.json')
    with sidecar.open('xb') as stream:
        stream.write(manifest_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    assert sha(sidecar) == hashlib.sha256(manifest_bytes).hexdigest()
    info = destination.stat()
    return dict(archive=str(destination), archive_sha256=expected, archive_bytes=size,
        archive_mtime_ns=info.st_mtime_ns, manifest=str(sidecar), manifest_sha256=sha(sidecar),
        embedded_manifest=embedded, files=len(files), source_bytes=sum(r['bytes'] for r in files.values()),
        full_object_readback=True, all_archive_members_hash_compared=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--mode', choices=['prepare', 'reclaim-prefix', 'reclaim'], required=True)
    parser.add_argument('--audits', type=Path, nargs='+', required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--qualification-source', type=Path)
    parser.add_argument('--restore-probe', type=Path)
    parser.add_argument('--prepared-receipt', type=Path)
    parser.add_argument('--prepared-sha256')
    parser.add_argument('--resume-reclaim', action='store_true')
    args = parser.parse_args()
    assert not args.resume_reclaim or args.mode == 'reclaim'
    root, task_root = args.root, args.root / args.task
    assert root.resolve() == Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-flow-sde-10m-per-task-20261103-r1')
    assert args.record.resolve().is_relative_to(root.resolve()) and not args.record.exists()
    assert not args.record.resolve().is_relative_to(task_root.resolve())
    if args.mode == 'prepare':
        source = args.qualification_source or task_root
        checkpoint = validate_closed(task_root, source, args.qualification_source is not None)
        audit = audit_inputs(args.audits, args.task, source if args.qualification_source else None)
        inventory = native_inventory(root, task_root, source)
        assert args.archive and args.archive.resolve().is_relative_to(BOS.resolve())
        proof = dict(task=args.task, audit_through=audit['through'], audit_refs=audit['refs'],
            audit_input_snapshots=audit['snapshots'],
            qualification_only=args.qualification_source is not None, checkpoint=checkpoint)
        archive = stream_archive(root, source, inventory, audit['known'], args.archive, proof)
        assert native_inventory(root, task_root, source) == inventory
        candidates = prefix_candidates(root, task_root, audit['known'])
        archive['audited_environment_actor_candidates'] = len(candidates)
        archive['audited_environment_actor_bytes'] = sum(inventory[name]['bytes'] for name in candidates)
        archive['audited_environment_actor_names_sha256'] = hashlib.sha256(json.dumps(candidates).encode()).hexdigest()
        if args.restore_probe:
            probe = args.restore_probe
            assert args.qualification_source and probe.resolve().parent == (root / 'restore-probes').resolve()
            probe.parent.mkdir(exist_ok=True)
            probe.mkdir()
            subprocess.run(['tar', '--no-same-owner', '-xf', str(args.archive), '-C', str(probe)], check=True)
            manifest = json.loads(Path(archive['manifest']).read_text())
            for name, row in manifest['files'].items():
                assert (probe / name).stat().st_size == row['bytes'] and sha(probe / name) == row['sha256']
            assert not probe.is_symlink()
            shutil.rmtree(probe)
            archive['restore_probe_all_files_verified'] = True
        result = dict(completed=True, mode='prepared', root=str(root), task=args.task,
            source=str(source), qualification_only=args.qualification_source is not None,
            checkpoint=checkpoint, audit_through=audit['through'], audit_refs=audit['refs'],
            source_reclaimed=False, verified_unix=time.time(), **archive)
        atomic_json(args.record, result)
        print(json.dumps(result), flush=True)
        return
    assert args.prepared_receipt and args.prepared_sha256 and not args.qualification_source
    assert sha(args.prepared_receipt) == args.prepared_sha256
    prepared = json.loads(args.prepared_receipt.read_text())
    assert prepared['completed'] and not prepared['qualification_only'] and prepared['task'] == args.task
    assert prepared['source'] == str(task_root) and prepared['full_object_readback'] and prepared['all_archive_members_hash_compared']
    checkpoint = validate_closed(task_root, task_root, False)
    audit = audit_inputs(args.audits, args.task, require_full=args.mode == 'reclaim')
    archive = Path(prepared['archive'])
    assert archive.resolve().is_relative_to(BOS.resolve())
    assert archive.stat().st_size == prepared['archive_bytes'] and archive.stat().st_mtime_ns == prepared['archive_mtime_ns']
    manifest_path = Path(prepared['manifest'])
    assert sha(manifest_path) == prepared['manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    files = manifest['files']
    expected_inventory = {name: {k: row[k] for k in ['bytes', 'mtime_ns', 'mode']} for name, row in files.items()}
    prefix_cold = task_root / 'CORPUS-PREFIX-COLD-STORAGE.json'
    removed = set()
    prefix_removed_count = 0
    interrupted = None
    cold = task_root / 'CORPUS-COLD-STORAGE.json'
    if prefix_cold.exists():
        prefix = json.loads(prefix_cold.read_text())
        assert args.mode == 'reclaim' and prefix['reclaim_complete']
        assert prefix['prepared_sha256'] == args.prepared_sha256
        removed = set(prefix['reclaimed_names'])
        prefix_removed_count = len(removed)
        assert removed <= set(files)
        for name in removed:
            assert 'sampling' not in Path(name).parts and 'checkpoints' not in Path(name).parts
            assert 'environment' in Path(name).parts or 'training-rollouts' in Path(name).parts
    if args.resume_reclaim:
        assert args.mode == 'reclaim' and cold.exists()
        interrupted = json.loads(cold.read_text())
        assert interrupted['completed'] and not interrupted['reclaim_complete']
        assert interrupted['prepared_sha256'] == args.prepared_sha256
        assert interrupted['archive_sha256'] == prepared['archive_sha256']
        assert interrupted['manifest_sha256'] == prepared['manifest_sha256']
        assert interrupted['full_corpus_audit_through'] == 130
        assert interrupted['full_audit_refs'] == audit['refs']
        # Only already-authorized Torch removals may be absent. Every surviving
        # native file, including every JSON/journal, still matches its inventory.
        for name in files:
            if name in removed or (root / name).exists():
                continue
            path = Path(name)
            assert path.name.endswith(('.pt', '.pt.gz')) and 'checkpoints' not in path.parts
            removed.add(name)
        qc = interrupted['qc_archive']
        assert Path(qc['archive']).resolve().is_relative_to(BOS.resolve())
        assert Path(qc['manifest']).resolve().is_relative_to(BOS.resolve())
        assert qc['full_object_readback'] and qc['all_archive_members_hash_compared']
        assert Path(qc['archive']).stat().st_size == qc['archive_bytes']
        assert sha(qc['archive']) == qc['archive_sha256']
        assert sha(qc['manifest']) == qc['manifest_sha256']
        qc_manifest = json.loads(Path(qc['manifest']).read_text())
        expected_metadata = {str(p.relative_to(root)) for p in audit['metadata']}
        assert set(qc_manifest['files']) == expected_metadata
        for name, item in qc_manifest['files'].items():
            assert (root / name).stat().st_size == item['bytes']
            assert sha(root / name) == item['sha256']
    else:
        assert not cold.exists(), 'Existing full-reclaim marker requires explicit recovery'
    assert native_inventory(root, task_root, task_root) == {
        name: row for name, row in expected_inventory.items() if name not in removed}
    for name, row in audit['known'].items():
        relative = str(Path(name).relative_to(root))
        assert relative in files and dict(bytes=files[relative]['bytes'], sha256=files[relative]['sha256']) == row
    if args.mode == 'reclaim-prefix':
        assert audit['through'] < 130 and not prefix_cold.exists()
        names = prefix_candidates(root, task_root, audit['known'])
        assert names
        receipt = dict(completed=True, task=args.task, prepared_receipt=str(args.prepared_receipt),
            prepared_sha256=args.prepared_sha256, archive=prepared['archive'],
            archive_sha256=prepared['archive_sha256'], manifest=prepared['manifest'],
            manifest_sha256=prepared['manifest_sha256'], audit_through=audit['through'],
            audit_refs=audit['refs'], audit_input_snapshots=audit['snapshots'],
            checkpoint=checkpoint, sampling_retained_for_continuing_audit=True,
            reclaim_complete=False, planned_names=names, verified_unix=time.time(),
            restore=['tar', '--no-same-owner', '-xf', prepared['archive'], '-C', str(root)])
        atomic_json(prefix_cold, receipt)
        reclaimed = 0
        for name in names:
            path, row = root / name, files[name]
            assert not path.is_symlink() and path.stat().st_size == row['bytes']
            assert path.stat().st_mtime_ns == row['mtime_ns']
            path.unlink()
            reclaimed += row['bytes']
        receipt.update(reclaim_complete=True, reclaimed_names=names, reclaimed_bytes=reclaimed)
        atomic_json(prefix_cold, receipt)
        atomic_json(args.record, receipt)
        print(json.dumps(dict(completed=True, mode='reclaimed-audited-prefix', task=args.task,
            audit_through=audit['through'], reclaimed_bytes=reclaimed, files=len(names),
            sampling_retained=True)), flush=True)
        return
    # Preserve the completed numerical audit itself in a separately verified
    # small metadata tar before reclaiming native Torch payloads.
    metadata_inventory = {}
    metadata_known = {}
    for path in audit['metadata']:
        assert path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
        info = path.stat()
        metadata_inventory[str(path.relative_to(root))] = dict(bytes=info.st_size,
            mtime_ns=info.st_mtime_ns, mode=stat.S_IMODE(info.st_mode))
        metadata_known[str(path)] = dict(bytes=info.st_size, sha256=sha(path))
    if interrupted is None:
        qc_path = archive.parent / (args.record.stem + '.qc.tar')
        qc = stream_archive(root, task_root, metadata_inventory, metadata_known, qc_path,
                            dict(task=args.task, full_corpus_audit_through=130, refs=audit['refs']))
    else:
        qc = interrupted['qc_archive']
    receipt = dict(completed=True, task=args.task, prepared_receipt=str(args.prepared_receipt),
        prepared_sha256=args.prepared_sha256, archive=prepared['archive'],
        archive_sha256=prepared['archive_sha256'], manifest=prepared['manifest'],
        manifest_sha256=prepared['manifest_sha256'], full_audit_refs=audit['refs'],
        full_corpus_audit_through=130, checkpoint=checkpoint, qc_archive=qc,
        restore=['tar', '--no-same-owner', '-xf', prepared['archive'], '-C', str(root)],
        reclaim_complete=False, verified_unix=time.time())
    if interrupted is not None:
        receipt.update(resumed_interrupted_reclaim=True,
            prior_incomplete_receipt_sha256=sha(cold),
            already_reclaimed_files=len(removed),
            already_reclaimed_bytes=sum(files[name]['bytes'] for name in removed),
            recovery_script_sha256=sha(Path(__file__)))
    atomic_json(cold, receipt)
    reclaimed, count = 0, 0
    for name, row in files.items():
        path = root / name
        if name in removed:
            continue
        if not (path.name.endswith('.pt') or path.name.endswith('.pt.gz')):
            continue
        assert 'checkpoints' not in path.relative_to(root).parts and not path.is_symlink()
        assert path.stat().st_size == row['bytes'] and path.stat().st_mtime_ns == row['mtime_ns']
        path.unlink()
        reclaimed += row['bytes']
        count += 1
    assert Path(checkpoint['path']).is_dir() and not (Path(checkpoint['path']) / 'COLD-STORAGE.json').exists()
    receipt.update(reclaim_complete=True, reclaimed_bytes=reclaimed, reclaimed_files=count,
        previously_reclaimed_prefix_files=prefix_removed_count)
    atomic_json(cold, receipt)
    atomic_json(args.record, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
