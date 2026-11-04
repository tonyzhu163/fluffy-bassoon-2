"""Reconcile completed cold moves and retained dependencies without GPU work."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

BASE=Path('/mowenjie/fluffy-bassoon-2/outputs/storage-audit-baige-20261104-r1')
def read(p): return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

stages=[('closed_runs',BASE),('stale_environments_and_projects',BASE/'workspace-r2'),('retired_models_and_caches',BASE/'legacy-models')]
all_rows=[];categories=[]
for category, folder in stages:
    d=read(folder/'SUMMARY.json');assert d['finished'],folder
    if category=='closed_runs':
        allowed=set(read(BASE/'ADDITIONAL-PROTECTION.json')['protected'])
        assert {str(Path('/mowenjie/fluffy-bassoon-2')/r['source']) for r in d['preserved']}==allowed
        assert all(Path(p).is_dir() for p in allowed)
    else: assert not d['preserved'],d['preserved']
    for row in d['completed']:
        source=Path(row['source']);archive=Path(row['archive'])
        assert row['source_reclaimed'] and row['source_compare'] and row['exact_member_coverage'] and row['bos_readback_sha256_verified']
        assert not source.exists()
        pointer=read(source.with_name(source.name+'.bos-archive.json'))
        assert pointer==row
        assert archive.stat().st_size==row['archive_bytes']
        assert read(archive.with_name(archive.name.removesuffix('.tar.zst')+'.json'))==row
        assert archive.with_name(archive.name+'.sha256').read_text().split()[0]==row['sha256']
        key=archive.name.removesuffix('.tar.zst');inv=folder/(key+'.inventory.json')
        remote=archive.with_name(key+'.inventory.json')
        assert sha(inv)==sha(remote)
        row['inventory_sha256']=sha(inv)
        row['inventory_path']=str(inv)
        row['category']=category
        all_rows.append(row)
    categories.append({'category':category,'archives':len(d['completed']),
        'source_allocated_bytes':sum(r['source_allocated_bytes'] for r in d['completed']),
        'archive_bytes':sum(r['archive_bytes'] for r in d['completed'])})

dedup=read(BASE/'FLORA-DEDUPLICATION.json')
assert dedup['complete'] and len(dedup['shards'])==11
for row in dedup['shards']:
    a,b=Path(row['legacy']).stat(),Path(row['canonical']).stat()
    assert a.st_ino==b.st_ino==row['shared_inode'] and a.st_dev==b.st_dev==row['shared_device']
    assert a.st_size==b.st_size==row['bytes']

protected=read(BASE/'PROTECTED-RUNTIME-CHECK.json')
for row in protected:
    p=Path(row['path']);assert p.exists() and str(p.resolve())==row['resolved']

progress=read(BASE.parent/'robocasa-ppo-grpo-meta-stage1-20261104-r1/ppo/attempt-01/progress.json')
assert progress['completed_iterations']>=11 and not progress['operator_checkpoint_exit']
for pid in (2093731,2094358,2095191,2097828,2096094,2096854):
    p=Path('/proc')/str(pid)/'cmdline'
    assert not p.exists() or not p.read_bytes(),('own worker still active',pid)

fs=shutil.disk_usage('/mowenjie')._asdict();before=read(BASE/'SPACE-BEFORE.json')
result={'completed':True,'checked_shanghai':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'cluster':'Baige','categories':categories,'archives':all_rows,
        'archive_count':len(all_rows),'archive_bytes':sum(r['archive_bytes'] for r in all_rows),
        'archived_source_allocated_bytes':sum(r['source_allocated_bytes'] for r in all_rows),
        'accounting_note':'Source directory allocated-byte totals can count links shared with retained environments; they are archived payload totals, not an exact filesystem space-gain attribution. Concurrent PPO writes also affect df.',
        'deduplicated_frozen_model_bytes':read(BASE/'FLORA-DEDUPLICATION.json')['reclaimed_allocated_bytes'],
        'filesystem_before_bulk_archival':before,'filesystem_after':fs,'net_free_increase_since_bulk_start_bytes':fs['free']-before['free'],
        'protected_interpreters':protected,'active_PPO_job':'job-bylh99ylx9ie','active_PPO_progress':progress,
        'all_archives_source_compared_before_reclaim':True,'all_archives_full_BOS_SHA256_readback_before_reclaim':True,
        'all_receipts_pointers_sizes_sidecars_and_inventory_hashes_reconciled':True,
        'source_corpus_retained':read(BASE/'ADDITIONAL-PROTECTION.json')['protected'],
        'GPU_hours_used':0,'GPU_jobs_launched':0,'own_archive_and_cache_workers_finished':True}
(BASE/'MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n')
mirror=Path('/mnt/bos/adgfyl/strong-plant-archives')/BASE.name/'MANIFEST.json'
mirror.write_text(json.dumps(result,indent=2)+'\n')
assert sha(mirror)==sha(BASE/'MANIFEST.json')
print(json.dumps({k:v for k,v in result.items() if k in ('archive_count','archive_bytes','archived_source_allocated_bytes','deduplicated_frozen_model_bytes','filesystem_after','checked_shanghai')},indent=2))
