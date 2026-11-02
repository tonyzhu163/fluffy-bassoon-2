"""Reconcile only verified moves; retain per-source recovery instructions."""
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path('/mowenjie')
BASE = ROOT / 'fluffy-bassoon-2/outputs/baige-storage-tiering-20261103-r1'
BOS = Path('/mnt/bos/adgfyl/strong-plant-archives/20261103-rl-checkpoint-space-r1')


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


if __name__ == '__main__':
    first = json.loads((BASE / 'PHASE-BOUNDARY.json').read_text())
    third_root = BASE.parent / 'baige-storage-tiering-20261103-r3'
    fourth_root = BASE.parent / 'baige-storage-tiering-20261103-r4'
    fifth_root = BASE.parent / 'baige-storage-tiering-20261103-r5'
    third = json.loads((third_root / 'SUMMARY.json').read_text())
    fourth = json.loads((fourth_root / 'SUMMARY.json').read_text())
    assert third.get('finished_at'), 'Archival still running'
    assert fourth.get('finished_at'), 'Verified reclaim still running'
    if fifth_root.exists():
        assert json.loads((fifth_root/'QUALIFICATION.json').read_text())['qualified']
        assert json.loads((fifth_root/'MIGRATION.json').read_text())['complete']
        assert json.loads((fifth_root/'COVERAGE.json').read_text())['archive_member_set_equals_inventory']
        assert json.loads((fifth_root/'actual/RESULT.json').read_text())['all_passed']
    completed = list(first['completed'])
    preserved = list(first['preserved'])
    reconciled = []
    producer_errors = {x['source']:x for x in third['preserved']}
    for rel in json.loads((third_root/'PLAN.json').read_text())['sources']:
        record = third_root/(rel.replace('/', '__')+'.json')
        item = json.loads(record.read_text()) if record.exists() else None
        if item and item.get('source_moved'):
            completed.append(item)
            if rel in producer_errors:
                receipt = fourth_root/(rel.replace('/', '__')+'.json')
                assert receipt.exists() and json.loads(receipt.read_text())['complete']
                reconciled.append(dict(producer_errors[rel], reclaim_receipt=str(receipt)))
        else:
            assert rel in producer_errors, 'Unaccounted producer source'
            preserved.append(producer_errors[rel])
    expected = {str(ROOT / p) for p in json.loads((BASE / 'PLAN.json').read_text())['sources']}
    moved = {x['source'] for x in completed}
    held = {str(ROOT / x['source']) for x in preserved}
    assert len(moved) == len(completed) and not (moved & held)
    assert expected == moved | held, 'Unaccounted candidate'
    for source in held:
        assert Path(source).exists() or Path(source).is_symlink(), 'Preserved source missing'
    for item in completed:
        source = Path(item['source'])
        assert item['source_moved'] and item['bos_readback_verified'] and item['archive_source_compare']
        assert not source.exists() and not source.is_symlink(), source
        obj = Path(item['bos_archive'])
        assert obj.stat().st_size == item['archive_bytes']
        assert obj.with_name(obj.name+'.sha256').read_text().split()[0] == item['sha256']
        pointer = json.loads(source.with_name(source.name+'.bos-archive.json').read_text())
        assert pointer['source_moved'] and pointer['sha256'] == item['sha256']
        for stage_id in ('baige-storage-tiering-20261103-r1','baige-storage-tiering-20261103-r3'):
            quarantine=source.with_name(source.name+'.tiering-'+stage_id)
            assert not quarantine.exists() and not quarantine.is_symlink(), 'Reclaim incomplete'
            stage = Path('/tmp')/stage_id/obj.name
            if stage.exists():
                assert stage.is_file() and not stage.is_symlink() and stage.stat().st_size==item['archive_bytes']
                stage.unlink()
    protected = json.loads((BASE/'PLAN.json').read_text())['protected'] + ['/mowenjie/.uv-python']
    assert all(Path(p).exists() for p in protected), 'Protected root missing'
    assert not list(ROOT.glob('*.tiering-baige-storage-tiering-*'))
    final = shutil.disk_usage(ROOT)
    result = {
        'run_id': BASE.name, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'cluster': 'Baige', 'filesystem_bytes': final.total,
        'before_free_bytes': first['before_free'], 'after_free_bytes': final.free,
        'net_free_increase_bytes': final.free-first['before_free'],
        'free_space_note': 'Observed filesystem change; concurrent active work can also change usage.',
        'completed_sources': len(completed), 'preserved_sources': preserved,
        'archive_bytes': sum(x['archive_bytes'] for x in completed),
        'archived_entries': sum(x['entries'] for x in completed),
        'source_inventory_allocated_bytes': sum(x['source_allocated_bytes'] for x in completed),
        'allocation_note': 'Inventory allocations can double-count hardlinks; use df change for actual capacity.',
        'reconciled_serial_to_parallel_reclaim_handoffs': reconciled,
        'reclaim_helper_notices': fourth['failures'],
        'object_directory': str(BOS), 'protected_roots_verified_present': protected,
        'verification': 'Every completed archive passed GNU tar content comparison, complete BOS SHA256 readback, unchanged-source checks, and final pointer/object-size reconciliation.',
        'completed': completed,
        'phases': ['r1 first six verified moves', 'r2 stopped before packing; no payload moved', 'r3 bounded parallel continuation with qualified concurrent metadata inventory', 'r4 qualified parallel removal of only inventoried, already verified private quarantine members', 'r5 final source compared by eight exact nonrecursive GNU tar member partitions, with positive/negative qualification and complete archive/inventory member coverage']}
    save(BASE/'MANIFEST.json', result)
    save(BOS/'MANIFEST.json', result)
    script_hashes={}
    for script in BASE.glob('*.py'):
        script_hashes[script.name]=hashlib.sha256(script.read_bytes()).hexdigest()
        shutil.copyfile(script, BOS/script.name)
    save(BOS/'SCRIPT-HASHES.json', script_hashes)
    save(BASE/'SCRIPT-HASHES.json', script_hashes)
    receipts=BOS/'audit-receipts'
    receipts.mkdir(exist_ok=True)
    for phase in (BASE,BASE.parent/'baige-storage-tiering-20261103-r2',third_root,fourth_root,fifth_root):
        for name in ('PLAN.json','SUMMARY.json','MIGRATION.json','PHASE-CLOSEOUT.json',
                     'PHASE-BOUNDARY.json','INVENTORY-QUALIFICATION.json','QUALIFICATION.json',
                     'PROTECTION.json','INTERPRETER-DEPENDENCIES.json','COVERAGE.json'):
            f=phase/name
            if f.exists():
                shutil.copyfile(f,receipts/(phase.name+'__'+name))
    if fifth_root.exists():
        shutil.copyfile(fifth_root/'actual/RESULT.json',receipts/(fifth_root.name+'__COMPARISON.json'))
    save(BOS/'FIRST-PHASE.json', first)
    print(json.dumps({k:v for k,v in result.items() if k!='completed'}, indent=2))
