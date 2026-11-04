"""Reconcile only this archive campaign; never inspect active run folders."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

RUN = 'experiment-bundles-cold-archive-20261104-r1'
BASE = Path('/mowenjie/fluffy-bassoon-2/outputs') / RUN
BOS = Path('/mnt/bos/adgfyl/strong-plant-archives') / RUN


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    summary = read(BASE / 'SUMMARY.json')
    assert summary['finished'] and not summary['preserved'], summary['preserved']
    plan = read(BASE / 'PLAN.json')
    requested = {r['source']: family for family, rows in plan['families'].items() for r in rows}
    assert len(requested) == summary['requested_sources'] == summary['completed_sources']
    assert set(requested) == {r['source'] for r in summary['completed']}
    families = {}
    rows = []
    for result in summary['completed']:
        source = Path(result['source'])
        archive = Path(result['archive'])
        family = result['family']
        assert family == requested[str(source)]
        assert not source.exists() and not source.is_symlink()
        assert not source.with_name(source.name + '.archiving-' + RUN).exists()
        assert all(result[k] for k in ('source_reclaimed', 'source_compare', 'exact_member_coverage',
                                      'bos_readback_sha256_verified'))
        receipt = {k: v for k, v in result.items() if k != 'family'}
        key = archive.name.removesuffix('.tar.zst')
        assert archive.parent == BOS
        assert read(source.with_name(source.name + '.bos-archive.json')) == receipt
        assert read(BASE / 'receipts' / (key + '.json')) == receipt
        assert read(BOS / (key + '.json')) == receipt
        assert archive.stat().st_size == result['archive_bytes']
        assert (BOS / (archive.name + '.sha256')).read_text().split() == [result['sha256'], archive.name]
        local_inventory = BASE / 'receipts' / (key + '.inventory.json')
        inventory_sha = digest(local_inventory)
        assert inventory_sha == digest(BOS / local_inventory.name)
        row = dict(result, inventory_sha256=inventory_sha, inventory_path=str(local_inventory))
        rows.append(row)
        totals = families.setdefault(family, {'bundles': 0, 'source_allocated_bytes': 0, 'archive_bytes': 0})
        totals['bundles'] += 1
        totals['source_allocated_bytes'] += result['source_allocated_bytes']
        totals['archive_bytes'] += result['archive_bytes']
    stage = Path('/tmp') / RUN
    assert not list(stage.iterdir()), 'Archive staging remains; inspect own staging only'
    fs = shutil.disk_usage('/mowenjie')._asdict()
    manifest = dict(
        complete=True, run_id=RUN, cluster='Baige',
        checked_shanghai=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        bundle_count=len(rows), families=families, archives=rows,
        source_allocated_bytes=sum(r['source_allocated_bytes'] for r in rows),
        archive_bytes=sum(r['archive_bytes'] for r in rows),
        filesystem_before=summary['filesystem_before'], filesystem_after=fs,
        net_free_increase_bytes=fs['free'] - summary['filesystem_before']['free'],
        accounting_note='Summed source allocation can count hardlinks more than once; it is archived payload. Concurrent writes affect filesystem free space.',
        all_receipts_pointers_sizes_sidecars_and_inventory_hashes_reconciled=True,
        all_sources_compared_and_full_BOS_SHA256_readback_verified_before_reclaim=True,
        running_job_folders_scanned=False, GPU_jobs_launched=0,
    )
    payload = json.dumps(manifest, indent=2) + '\n'
    (BASE / 'MANIFEST.json').write_text(payload)
    (BOS / 'MANIFEST.json').write_text(payload)
    assert digest(BASE / 'MANIFEST.json') == digest(BOS / 'MANIFEST.json')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'archives'}, indent=2))


if __name__ == '__main__':
    main()
