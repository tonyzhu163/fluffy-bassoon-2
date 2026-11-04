"""Reconcile completed deletion receipts and retained code/results."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import stat
import time

BASE = Path('/mowenjie/fluffy-bassoon-2/outputs/legacy-experiment-retention-audit-20261104-r2')
OLD = BASE.parent / 'experiment-bundles-cold-archive-20261104-r1'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def identity(path, row):
    s = path.lstat()
    assert (s.st_mode, s.st_size, s.st_mtime_ns, s.st_dev, s.st_ino) == (row[1], row[2], row[3], row[5], row[6]), str(path)


def main():
    primary = read(BASE / 'PRUNE-SUMMARY.json')
    extra = read(BASE / 'ADDITIONAL-TRACE-SUMMARY.json')
    cold = read(BASE / 'COLD-RECOVERY.json')
    assert all(d['finished'] and not d.get('preserved') and not d.get('skipped_sources') for d in (primary, extra))
    assert cold['finished'] and len(cold['completed']) == 9 and not cold['errors']
    audit = read(BASE / 'AUDIT.json')
    plan = read(BASE / 'PLAN.json')
    additional = read(BASE / 'private-additional-trace-plan.json')
    additional_paths = {(entry['source'], row[0]) for entry in additional['sources'] for row in entry['rows']}
    expected = defaultdict(lambda: {'files': 0, 'logical_bytes': 0, 'allocated_payload_bytes': 0, 'code_files': 0})
    samples = []
    code_checks = []
    records = []
    kept_checkpoint_checks = []
    families = defaultdict(lambda: {'audited_bundles': 0, 'newly_deleted_files': 0,
                                    'deleted_allocated_payload_bytes': 0, 'retained_regular_files': 0,
                                    'retained_regular_file_allocation_bytes': 0, 'earlier_full_BOS_archives': 0})
    for entry in audit['sources']:
        source = entry['source']
        family = families[entry['family']]
        family['audited_bundles'] += 1
        if entry['status'] == 'full_bundle_already_in_BOS':
            family['earlier_full_BOS_archives'] += 1
            continue
        inventory = read(Path(entry['inventory']))
        root = Path(source).lstat()
        assert (root.st_dev, root.st_ino) == tuple(entry['root_identity'][:2])
        row_map = {r[0]: r for r in inventory['rows']}
        per_parent = defaultdict(list)
        for row in inventory['rows']:
            if row[7] in ('bulk', 'one_off_code'):
                key = (source, str(Path(row[0]).parent))
                totals = expected[key]
                totals['files'] += 1
                totals['logical_bytes'] += row[2]
                totals['allocated_payload_bytes'] += row[4] * 512
                totals['code_files'] += row[7] == 'one_off_code'
                per_parent[key].append(row[0])
                family['newly_deleted_files'] += 1
                family['deleted_allocated_payload_bytes'] += row[4] * 512
            elif (source, row[0]) in additional_paths:
                family['newly_deleted_files'] += 1
                family['deleted_allocated_payload_bytes'] += row[4] * 512
            elif row[7] == 'keep' and stat.S_ISREG(row[1]):
                family['retained_regular_files'] += 1
                family['retained_regular_file_allocation_bytes'] += row[4] * 512
                if row[0].endswith('recovery.zip'):
                    kept_checkpoint_checks.append((Path(source) / row[0], row))
        for (_, _), names in per_parent.items():
            samples.extend([str(Path(source) / names[0]), str(Path(source) / names[-1])])
        review = read(Path(entry['review']))
        for item in review['code']:
            if row_map[item['relative']][7] == 'keep':
                code_checks.append((Path(source) / item['relative'], item['sha256']))
        for item in review['reports']:
            records.append((Path(source) / item['relative'], row_map[item['relative']]))
    files = list((BASE / 'prune-receipts').glob('*.json'))
    with ThreadPoolExecutor(max_workers=24) as pool:
        receipts = list(pool.map(read, files))
    actual = {(r['source'], r['parent']): r for r in receipts}
    assert len(actual) == len(receipts) == len(expected) == primary['directory_groups']
    for key, wanted in expected.items():
        receipt = actual[key]
        assert not receipt['preserved']
        assert all(receipt[k] == v for k, v in wanted.items()), key
    extra_expected = defaultdict(lambda: {'files': 0, 'logical_bytes': 0, 'allocated_payload_bytes': 0})
    for entry in additional['sources']:
        grouped = defaultdict(list)
        for row in entry['rows']:
            key = (entry['source'], str(Path(row[0]).parent))
            wanted = extra_expected[key]
            wanted['files'] += 1; wanted['logical_bytes'] += row[2]; wanted['allocated_payload_bytes'] += row[4] * 512
            grouped[key].append(row[0])
        for names in grouped.values():
            samples.extend([str(Path(entry['source']) / names[0]), str(Path(entry['source']) / names[-1])])
    extra_actual = {}
    with (BASE / 'private-additional-trace-receipts.jsonl').open() as stream:
        for line in stream:
            receipt = json.loads(line)
            key = receipt['source'], receipt['parent']
            assert key not in extra_actual and not receipt['preserved']
            extra_actual[key] = receipt
    assert len(extra_actual) == len(extra_expected) == extra['groups']
    for key, wanted in extra_expected.items():
        assert all(extra_actual[key][k] == value for k, value in wanted.items())
    def absent(name):
        p = Path(name)
        assert not p.exists() and not p.is_symlink(), name
    with ThreadPoolExecutor(max_workers=24) as pool:
        list(pool.map(absent, set(samples)))
    for path, expected_sha in code_checks:
        assert digest(path) == expected_sha, str(path)
    with ThreadPoolExecutor(max_workers=24) as pool:
        list(pool.map(lambda item: identity(*item), records))
    checkpoint_hashes = []
    for item in read(BASE / 'REVIEW.json')['protected_cited_endpoints_and_evaluation_records']:
        path = Path(item['source']) / item['relative']
        assert path.stat().st_size == item['bytes']
        checkpoint_hashes.append(dict(item, sha256=digest(path)))
    for path, row in kept_checkpoint_checks:
        identity(path, row)
        checkpoint_hashes.append({'path': str(path), 'bytes': row[2], 'sha256': digest(path),
                                  'reason': 'reported RoboMimic result recovery'})
    archived = []
    for result in read(OLD / 'SUMMARY.json')['completed']:
        source = Path(result['source']); archive = Path(result['archive'])
        receipt = {k: v for k, v in result.items() if k != 'family'}
        key = archive.name.removesuffix('.tar.zst')
        assert not source.exists()
        assert read(source.with_name(source.name + '.bos-archive.json')) == receipt
        assert read(archive.with_name(key + '.json')) == receipt
        assert archive.stat().st_size == receipt['archive_bytes']
        assert archive.with_name(archive.name + '.sha256').read_text().split() == [receipt['sha256'], archive.name]
        inventory_sha = digest(OLD / 'receipts' / (key + '.inventory.json'))
        assert inventory_sha == digest(archive.with_name(key + '.inventory.json'))
        archived.append(dict(result, inventory_sha256=inventory_sha))
    for result in cold['completed']:
        root = Path(result['destination'])
        for relative, expected_sha in result['file_sha256'].items():
            assert digest(root / relative) == expected_sha
    total_files = primary['files_deleted'] + extra['files_deleted']
    assert total_files == sum(r['newly_deleted_files'] for r in families.values())
    manifest = {'complete': True, 'checked_shanghai': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'selected_legacy_bundles': len(plan['sources']), 'families': dict(families),
                'newly_deleted_files': total_files,
                'deleted_allocated_payload_bytes': primary['deleted_allocated_payload_bytes'] + extra['deleted_allocated_payload_bytes'],
                'deleted_logical_bytes': primary['deleted_logical_bytes'] + extra['deleted_logical_bytes'],
                'Git_recoverable_one_off_code_copies_removed': primary['recoverable_one_off_code_files_deleted'],
                'retained_regular_files': sum(r['retained_regular_files'] for r in families.values()),
                'retained_regular_file_allocation_bytes': sum(r['retained_regular_file_allocation_bytes'] for r in families.values()),
                'primary_retained_record_identity_checks': primary['retained_records_checked'],
                'retained_unique_code_SHA256_checks': len(code_checks), 'canonical_result_records_rechecked': len(records),
                'deleted_path_samples_checked_absent': len(set(samples)),
                'protected_checkpoint_and_evaluation_records': checkpoint_hashes,
                'earlier_full_BOS_archives': archived, 'cold_compact_records_recovered': sum(r['records_recovered'] for r in cold['completed']),
                'cold_compact_recovered_bytes': cold['recovered_bytes'],
                'all_directory_receipts_reconciled': True, 'all_recovered_record_hashes_rechecked': True,
                'all_earlier_archive_pointers_sizes_sidecars_inventory_hashes_reconciled': True,
                'filesystem_before_direct_pruning_free_bytes': primary['filesystem_free_before'],
                'filesystem_now': shutil.disk_usage('/mowenjie')._asdict(),
                'accounting_note': 'Payload allocation sums can count hardlinks; concurrent jobs change filesystem free space. No active-folder census was refreshed.',
                'active_job_folders_refreshed': False, 'shared_models_and_environments_moved': False, 'GPU_jobs_launched': 0}
    save(BASE / 'MANIFEST.json', manifest)
    for entry in audit['sources']:
        source = Path(entry['source'])
        marker = source.with_name(source.name + '.retention.json')
        save(marker, {'retention_audit': str(BASE), 'manifest': str(BASE / 'MANIFEST.json'),
                      'source': str(source), 'state': 'closed legacy bulk retired; compact records retained',
                      'full_BOS_archive': entry.get('archive'), 'reusing_run_identity_for_new_work_prohibited': True})
    manifest['filesystem_now'] = shutil.disk_usage('/mowenjie')._asdict()
    manifest['checked_shanghai'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    save(BASE / 'MANIFEST.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('earlier_full_BOS_archives', 'protected_checkpoint_and_evaluation_records')}, indent=2))


if __name__ == '__main__':
    main()
