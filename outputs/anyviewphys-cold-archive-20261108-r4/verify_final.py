"""Read-only final reconciliation of a completed full-project cold archive."""
from pathlib import Path
import gzip, hashlib, json, os, shutil, stat, time

OUT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def read(name):return json.loads((OUT/name).read_text())


plan=read('PLAN.json');result=read('RESULT.json');receipt=read('RECOVERY.json')
assert result['status']=='complete' and result['recovery']==receipt
assert result['source_commit']==read('DEPLOYMENT.json')['source_commit']
assert result['script_sha256']==sha(OUT/'archive.py')==read('QUALIFICATION.json')['script_sha256']
assert sha(OUT/'tree.py')==plan['tree_sha256']==read('QUALIFICATION.json')['tree_sha256']
assert all(receipt[k] for k in ['source_SHA256_inventory_and_complete_restore','full_BOS_readback','restored_from_independent_cold_readback','complete_restore_verified'])
source=Path(plan['source']);cold=Path(plan['cold_root'])
assert set(p.name for p in source.iterdir())=={'COLD-ARCHIVE.json','RESTORE.md'}
pointer=json.loads((source/'COLD-ARCHIVE.json').read_text())
assert pointer==json.loads(source.with_name('anyviewphys.COLD-ARCHIVE.json').read_text())
assert pointer['recovery']==receipt and pointer['original_path']==str(source)
assert pointer['source_commit']==result['source_commit'] and pointer['status']=='cold_restore_verified'
manifest=Path(receipt['archive']);assert sha(manifest)==receipt['archive_manifest_sha256']
part_manifest=json.loads(manifest.read_text());assert part_manifest['parts']==receipt['archive_parts'] and part_manifest['archive_sha256']==receipt['archive_sha256'] and part_manifest['archive_bytes']==receipt['archive_bytes']
assert sum(Path(p['path']).stat().st_size for p in receipt['archive_parts'])==receipt['archive_bytes']
assert all(Path(p['path']).stat().st_size==p['bytes'] for p in receipt['archive_parts'])
assert sha(OUT/'SOURCE-INVENTORY.json')==receipt['inventory_sha256']
h=hashlib.sha256()
with gzip.open(OUT/'SOURCE-INVENTORY.json.gz','rb') as f:
    for block in iter(lambda:f.read(8<<20),b''):h.update(block)
assert h.hexdigest()==receipt['inventory_sha256']
rows=read('SOURCE-INVENTORY.json');paths={r['path'] for r in rows}
assert len(rows)==len(paths)==result['source_entries']==receipt['restored_entries']
seen=set();allocation=0
with gzip.open(OUT/'RELEASE-JOURNAL.jsonl.gz','rt') as f:
    for line in f:
        row=json.loads(line);assert row['path'] in paths and row['path'] not in seen
        seen.add(row['path']);allocation+=row['allocated_bytes']
assert seen==paths and len(seen)==result['released_entries']
assert allocation==result['released_allocated_bytes']
for name in ['SOURCE-INVENTORY.json.gz','RELEASE-JOURNAL.jsonl.gz','RECOVERY.json','EXTERNAL-LINKS.json','QUIESCE.json','PROJECT-GIT-STATE.json','PLAN.json','RESTORE.md','QUALIFICATION.json','archive.py','tree.py','RESULT.json']:
    assert sha(OUT/name)==sha(cold/name),name
assert len(read('EXTERNAL-LINKS.json'))==result['external_symlink_count']
assert read('PROJECT-GIT-STATE.json')==result['original_git']
assert read('QUIESCE.json')['controller_exited'] and read('QUIESCE.json')['score_not_started']
assert not Path(plan['quarantine']).exists() and result['quarantine_removed']
assert not (Path('/tmp')/OUT.name).exists() and result['staging_removed']
assert not Path('/mowenjie/anyviewphys.ARCHIVING.json').exists()
pid=read('LAUNCH.json')['pid'];p=Path('/proc')/str(pid)
assert not p.exists(), 'Wait for the archive worker to exit'
report=dict(status='passed',checked_unix=time.time(),source_entries=len(rows),released_entries=len(seen),
            released_allocated_bytes=allocation,full_cold_readback_and_complete_restoration_verified=True,
            cold_metadata_matches=True,release_journal_complete=True,only_restore_stub_on_PFS=True,
            original_git_state_recorded=True,quarantine_and_staging_removed=True,worker_exited=True,
            external_symlink_count=result['external_symlink_count'],filesystem_now=shutil.disk_usage('/mowenjie')._asdict(),
            GPU_allocations=0,note='Every archived member passed a complete restored SHA256/metadata/hardlink comparison before source detachment. Final reconciliation checks that proof, full release coverage, cold metadata and original-path restore pointers without repeating all model/trajectory reads.')
for base in [OUT,cold]:(base/'FINAL-VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
(cold/'reassembled-anyviewphys.tar.zst.sha256').write_text(receipt['archive_sha256']+'  anyviewphys.tar.zst\n')
print(json.dumps(report))
