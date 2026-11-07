from pathlib import Path
import importlib.util,json,hashlib,os,stat,time,shutil
OUT=Path('/mowenjie/fluffy-bassoon-2/outputs/outputs-retention-20261107-r1')
ROOT=OUT.parents[1]
spec=importlib.util.spec_from_file_location('retention',OUT/'archive_outputs.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
plan=json.loads((OUT/'PLAN.json').read_text());result=json.loads((OUT/'RESULT.json').read_text())
assert result['status']=='complete' and not result['preserved_failures'] and len(result['completed'])==11
expected=json.loads((OUT/'SCIENTIFIC-IDENTITY-EXPECTED.json').read_text());all_inventory={};checks=[];total_files=total_kept=0
for item in plan['selected']:
 key=item['relative_path'].replace('/','__');receipt=json.loads((OUT/(key+'.json')).read_text());invpath=OUT/(key+'.inventory.json');rows=json.loads(invpath.read_text());cold=Path(receipt['archive'])
 assert receipt['status']=='complete' and receipt['payload_released'] and receipt['source_compare'] and receipt['full_BOS_readback'] and receipt['complete_restore_verified']
 assert receipt['source_inventory_sha256']==mod.sha(invpath) and receipt['restored_entries']==receipt['source_entries']==len(rows)
 assert cold.stat().st_size==receipt['archive_bytes']
 assert mod.sha(cold.with_name(cold.name+'.inventory.json'))==receipt['source_inventory_sha256']
 cold_receipt=json.loads(cold.with_name(cold.name+'.receipt.json').read_text());assert cold_receipt==receipt
 pointer=json.loads(Path(receipt['source']).with_name(Path(receipt['source']).name+'.outputs-retention-20261107-r1.json').read_text());assert pointer==receipt
 hot=json.loads((Path(receipt['source'])/'COLD-ARCHIVE-outputs-retention-20261107-r1.json').read_text());assert hot==receipt
 released=json.loads((OUT/(key+'.released.json')).read_text());bulk={r['path']:r for r in rows if mod.bulk(r,plan['preserve_numerical_directories'])}
 assert set(bulk)=={r['path'] for r in released};assert len(released)==receipt['released_files']
 assert sum(r['released_allocated_bytes'] for r in released)==receipt['released_allocated_bytes']
 assert all(not os.path.lexists(ROOT/r['path']) for r in released)
 kept=0
 for row in rows:
  path=ROOT/row['path'];all_inventory[str(path)]=row
  if stat.S_ISREG(row['stamp'][0]) and row['path'] not in bulk:
   assert path.is_file() and mod.sha(path)==row['sha256'];kept+=1
  elif stat.S_ISLNK(row['stamp'][0]):assert path.is_symlink() and os.readlink(path)==row['target']
  elif stat.S_ISDIR(row['stamp'][0]):assert path.is_dir() and not path.is_symlink()
 assert kept==receipt['compact_files_kept'];total_files+=len(released);total_kept+=kept
 checks.append({'source':receipt['source'],'released_files':len(released),'retained_regular_files':kept,'released_allocated_bytes':receipt['released_allocated_bytes'],'archive_bytes':receipt['archive_bytes'],'archive_sha256':receipt['archive_sha256'],'full_restore_verified':True,'receipt_inventory_and_pointer_identity':True})
identity=[]
for row in expected:
 assert row['path'] in all_inventory and all_inventory[row['path']]['sha256']==row['sha256'],row
 identity.append({'label':row['label'],'path':row['path'],'sha256':row['sha256'],'matches_published_manifest':True})
assert sum(r['released_allocated_bytes'] for r in checks)==result['released_allocated_bytes']
for rel in plan['deferred']:
 assert (ROOT/rel).is_dir() and not (ROOT/rel).is_symlink()
protected=[p for p in plan['excluded_hot_paths'] if p.startswith('outputs/')]
assert all((ROOT/p).is_dir() for p in protected)
assert not (Path('/tmp')/plan['run_id']).exists()
assert not (Path('/proc')/str(json.loads((OUT/'LAUNCH.json').read_text())['pid'])).exists()
report={'status':'passed','checked_unix':time.time(),'archives_checked':len(checks),'released_files':total_files,'retained_regular_files':total_kept,'released_allocated_bytes':result['released_allocated_bytes'],'archive_bytes':result['archive_bytes'],'published_actor_identities':identity,'archives':checks,'deferred_and_protected_roots_present':True,'staging_removed':True,'archive_worker_exited':True,'GPU_allocations':0,'filesystem_now':shutil.disk_usage(ROOT)._asdict(),'note':'Every archive passed full independent BOS checksum readback and complete restored bytes/metadata/hardlink identity before release. This final reconciliation checks those receipts, member inventory identity, payload absence, retained file hashes, pointers and eight published actor identities; it does not repeat all bulk reads.'}
mod.save(OUT/'FINAL-VERIFICATION.json',report);mod.save(Path(plan['cold_root'])/'FINAL-VERIFICATION.json',report)
print(json.dumps(report))
