"""Full AnyViewPhys cold preservation, complete restore proof, and hot eviction."""
from pathlib import Path
import argparse, collections, gzip, hashlib, importlib.util, json, os, shutil
import stat, subprocess, tempfile, threading, time
import tree

HERE = Path(__file__).resolve().parent
STATE = {}; LAST = 0; LOCK=threading.RLock()
HELPER = HERE.parent/'outputs-retention-20261107-r1/archive_outputs.py'
assert tree.sha(HELPER) == '3d58eaa4a18a82e719bf1ce28c8527baa6b43b1e9c2811402b449d6d61acffe4'
spec=importlib.util.spec_from_file_location('readers',HELPER); shared=importlib.util.module_from_spec(spec); spec.loader.exec_module(shared)


def emit(phase=None, force=False, **data):
    global LAST
    with LOCK:
        if phase: STATE.update(phase=phase, phase_started_unix=time.time())
        STATE.update(data, checked_unix=time.time(), pid=os.getpid())
        if phase or force or time.monotonic()-LAST > 5:
            tree.save(HERE/'STATE.json',STATE); LAST=time.monotonic()
            if phase: print(json.dumps(STATE),flush=True)


def gate():
    x=json.loads((HERE/'SCHEDULER-CURRENT.json').read_text())
    assert all(x[k] for k in ['complete','all_clusters','all_queues','no_anyview_consumers'])
    assert 0 <= time.time()-x['observed_unix'] <= 25*60, 'Refresh scheduler attestation before eviction'


def command(args, phase, measured=None):
    emit(phase)
    with (HERE/'tar.log').open('ab') as log:
        p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
        tree.save(HERE/'CHILD.json',dict(pid=p.pid,args=args,started_unix=time.time()))
        while p.poll() is None:
            emit(child_pid=p.pid, archive_bytes=measured.stat().st_size if measured and measured.exists() else 0)
            time.sleep(5)
        assert p.returncode == 0, (phase,p.returncode)


def transfer(source,destination,phase):
    emit(phase, bytes_done=0, bytes_total=source.stat().st_size)
    h=hashlib.sha256();count=0
    with source.open('rb') as inp,destination.open('xb') as out:
        while block:=inp.read(16<<20):
            out.write(block);h.update(block);count+=len(block);emit(bytes_done=count)
        out.flush()
        try:os.fsync(out.fileno())
        except OSError as e:
            import errno
            if e.errno not in (errno.EINVAL,errno.ENOTSUP):raise
    assert destination.stat().st_size == count == source.stat().st_size
    emit(force=True,bytes_done=count)
    return h.hexdigest(),count


def pack(source,anchor,stage,cold):
    if source == Path('/mowenjie/anyviewphys') and json.loads((HERE/'PLAN.json').read_text()).get('reuse_verified_cold_and_restored_stage'):
        plan=json.loads((HERE/'PLAN.json').read_text());prior=Path(plan['prior_stage']);origin=Path(plan['prior_output']);oldstate=json.loads((origin/'STATE.json').read_text())
        assert not Path('/proc/'+str(plan['prior_worker_pid'])).exists()
        assert oldstate['phase']=='failed_preserved' and oldstate['verified_cold_bytes']==plan['packed_archive_bytes']
        manifest=Path(plan['prior_cold_manifest']);data=manifest.read_bytes();payload=json.loads(data);proof=json.loads((origin/'PARTS-PROGRESS.json').read_text())
        assert proof['bytes_verified']==payload['archive_bytes']==plan['packed_archive_bytes']
        assert proof['verified_parts']==payload['parts'] and len(payload['parts'])==8
        assert all(Path(p['path']).stat().st_size==p['bytes'] for p in payload['parts'])
        assert (prior/'readback.tar.zst').stat().st_size==payload['archive_bytes'] and (prior/'restored/anyviewphys').is_dir()
        assert not any(stage.iterdir());stage.rmdir();prior.rename(stage)
        target=cold.with_name(cold.name+'.parts.json');target.write_bytes(data);assert target.read_bytes()==data
        emit('resuming_complete_cold_archive_and_extracted_copy',verified_cold_bytes=payload['archive_bytes'],archive_bytes=payload['archive_bytes'])
        return dict(archive=str(target),archive_manifest_sha256=tree.sha(target),archive_parts=payload['parts'],archive_sha256=payload['archive_sha256'],archive_bytes=payload['archive_bytes'],source_compare=False,source_SHA256_inventory_and_complete_restore=True,full_BOS_readback=True,restored_from_independent_cold_readback=True,prior_archive_run=origin.name,prior_full_part_readback_proof=str(origin/'PARTS-PROGRESS.json')),stage/'restored'
    local=stage/'original.tar.zst';readback=stage/'readback.tar.zst'
    flags=['--acls','--xattrs','--xattrs-include=*']
    if source == Path('/mowenjie/anyviewphys'):
        plan=json.loads((HERE/'PLAN.json').read_text());prior=Path(plan['resume_archive'])
        assert not Path('/proc/'+str(plan['prior_worker_pid'])).exists(), 'Prior archive worker must exit before migration'
        assert prior.is_file() and prior.stat().st_size==plan['packed_archive_bytes']
        emit('reusing_complete_packed_archive',archive_bytes=prior.stat().st_size)
        prior.rename(local)
        if not any(prior.parent.iterdir()):prior.parent.rmdir()
    else:
        command(['tar','--format=posix','--sort=name',*flags,'--numeric-owner','--sparse','-I','zstd -1 -T4','-cf',str(local),'-C',str(anchor),source.name], 'packing_full_project',local)
    # Full source SHA256 inventory, complete restored-member comparison and final
    # unchanged-source check establish content equivalence; avoid a redundant
    # serial GNU tar -d pass over PFS before uploading the already-complete archive.
    cap=json.loads((HERE/'PLAN.json').read_text())['archive_part_bytes'] if source==Path('/mowenjie/anyviewphys') else 128
    size=local.stat().st_size;whole=hashlib.sha256();back=hashlib.sha256();parts=[];offset=0
    with local.open('rb') as inp,readback.open('xb') as restored_stream:
        while offset<size:
            count=min(cap,size-offset);part=cold.with_name(cold.name+f'.part-{len(parts):04d}');h=hashlib.sha256();done=0
            emit('uploading_archive_parts_to_BOS',bytes_done=offset,bytes_total=size,part_index=len(parts),verified_cold_bytes=offset)
            with part.open('xb') as out:
                while done<count:
                    block=inp.read(min(16<<20,count-done));assert block
                    out.write(block);whole.update(block);h.update(block);done+=len(block);emit(bytes_done=offset+done)
                out.flush()
                try:os.fsync(out.fileno())
                except OSError as error:
                    import errno
                    if error.errno not in (errno.EINVAL,errno.ENOTSUP):raise
            assert part.stat().st_size==count
            digest=h.hexdigest();check=hashlib.sha256();read=0
            emit('independent_BOS_part_readback',bytes_done=offset,bytes_total=size,part_index=len(parts))
            with part.open('rb') as remote:
                while block:=remote.read(16<<20):
                    restored_stream.write(block);check.update(block);back.update(block);read+=len(block);emit(bytes_done=offset+read)
            assert read==count and check.hexdigest()==digest
            parts.append(dict(path=str(part),bytes=count,sha256=digest,offset=offset));offset+=count
            tree.save(HERE/'PARTS-PROGRESS.json',dict(verified_parts=parts,bytes_verified=offset,archive_bytes=size))
            emit(force=True,verified_cold_bytes=offset)
        restored_stream.flush();os.fsync(restored_stream.fileno())
    assert offset==size==readback.stat().st_size and whole.hexdigest()==back.hexdigest()
    payload=dict(format='split_tar_zstd_v1',archive_bytes=size,archive_sha256=whole.hexdigest(),parts=parts)
    manifest=cold.with_name(cold.name+'.parts.json');manifest.write_text(json.dumps(payload,indent=2)+'\n');assert json.loads(manifest.read_text())==payload
    if source!=Path('/mowenjie/anyviewphys'):assert len(parts)>1, 'Fixture must cover split boundaries'
    local.unlink();restored=stage/'restored';restored.mkdir()
    command(['tar',*flags,'--same-owner','-I','zstd','-xf',str(readback),'-C',str(restored)],'restoring_cold_readback')
    return dict(archive=str(manifest),archive_manifest_sha256=tree.sha(manifest),archive_parts=parts,archive_sha256=whole.hexdigest(),archive_bytes=size,source_compare=False,source_SHA256_inventory_and_complete_restore=True,
                full_BOS_readback=True,restored_from_independent_cold_readback=True),restored


def progress(phase):
    def update(n,size):emit(entries_done=n,hashed_bytes=size)
    emit(phase,entries_done=0,hashed_bytes=0)
    return update


def release(quarantine,source_name,rows,admission=lambda:None):
    import concurrent.futures
    groups=collections.defaultdict(list);directories=collections.defaultdict(list)
    for row in rows:
        if stat.S_ISDIR(row['stamp'][0]):directories[len(Path(row['path']).parts)].append(row)
        else:groups[tuple(row['stamp'][5:7])].append(row)
    counter=[0,0];lock=threading.RLock()
    with (HERE/'RELEASE-JOURNAL.jsonl').open('x') as journal:
        def finish(row,p,charge,is_directory):
            with lock:
                if counter[0]%4096==0:admission();journal.flush();os.fsync(journal.fileno())
                journal.write(json.dumps(dict(path=row['path'],allocated_bytes=charge,kind='directory' if is_directory else 'file'))+'\n')
            if is_directory:p.rmdir()
            else:p.unlink()
            with lock:
                counter[0]+=1;counter[1]+=charge
                emit(released_entries=counter[0],released_allocated_bytes=counter[1])
        def files(group):
            for index,row in enumerate(group):
                p=quarantine/Path(row['path']).relative_to(source_name);s=tree.stamp(p);a=row['stamp'];expected=a.copy();expected[7]-=index
                assert s==expected,('Changed source before unlink',row['path'])
                if index==0:assert p.lstat().st_ctime_ns==row['ctime_ns']
                finish(row,p,s[8] if s[7]==1 else 0,False)
        def directory(row):
            p=quarantine/Path(row['path']).relative_to(source_name);s=tree.stamp(p);a=row['stamp']
            assert s[0]==a[0] and s[3:7]==a[3:7]
            finish(row,p,a[8],True)
        def parallel(pool,fn,items):
            todo=collections.deque(items);pending=set()
            while todo or pending:
                while todo and len(pending)<32:pending.add(pool.submit(fn,todo.popleft()))
                done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:future.result()
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            parallel(pool,files,groups.values())
            for depth in sorted(directories,reverse=True):parallel(pool,directory,directories[depth])
        journal.flush();os.fsync(journal.fileno())
    assert not quarantine.exists()
    return counter[0],counter[1]


def qualify():
    assert os.uname().sysname=='Linux'
    with tempfile.TemporaryDirectory(prefix='anyview-archive-fixture-') as tmp:
        anchor=Path(tmp);source=anchor/'anyviewphys';source.mkdir();(source/'.hidden').write_bytes(b'fixture'*1000)
        os.link(source/'.hidden',source/'hardlink');(source/'dangling').symlink_to('/missing/external-target')
        (source/'sub').mkdir();(source/'sub/tool').write_text('echo fixture\n');os.chmod(source/'sub/tool',0o755)
        os.utime(source/'.hidden',ns=(1790000000123456789,1790000000123456789))
        os.setxattr(source/'sub/tool','user.archive-fixture',b'preserve')
        expected=tree.hash_rows(tree.scan(source,anchor),anchor);stage=anchor/'stage';stage.mkdir()
        receipt,restored=pack(source,anchor,stage,anchor/'cold.tar.zst')
        subprocess.run(['sync','-f',str(restored)],check=True)
        actual=tree.hash_rows(tree.scan(restored/source.name,restored),restored,restored=True,metadata_events=[]);tree.verify(expected,actual)
        assert os.getxattr(restored/source.name/'sub/tool','user.archive-fixture')==b'preserve'
        quarantine=anchor/'quarantine';source.rename(quarantine);count,physical=release(quarantine,source.name,expected)
        assert count==len(expected)
        seen=set();total=0
        for row in expected:
            key=tuple(row['stamp'][5:7])
            if key not in seen:total+=row['stamp'][8];seen.add(key)
        assert physical==total
        report=dict(status='passed',script_sha256=tree.sha(Path(__file__)),tree_sha256=tree.sha(HERE/'tree.py'),
                    complete_restore=True,strict_source_stability=True,full_restored_SHA_and_logical_metadata=True,syncfs_before_restored_inventory=True,multi_object_recovery=True,forced_fixture_part_bytes=128,hardlinks=True,dangling_symlink=True,nanosecond_mtime=True,
                    executable_mode=True,owner_identity=True,user_xattr=True,exact_tree_release=True,checked_unix=time.time())
    (HERE/'RELEASE-JOURNAL.jsonl').unlink();tree.save(HERE/'QUALIFICATION.json',report);emit('qualified',**report)


def run():
    plan=json.loads((HERE/'PLAN.json').read_text());source=Path(plan['source']);anchor=source.parent
    assert source==Path('/mowenjie/anyviewphys') and source.resolve()==source and not source.is_symlink()
    assert tree.sha(Path(__file__))==os.environ['TASK_SCRIPT_SHA256'] and tree.sha(HERE/'tree.py')==plan['tree_sha256']
    assert tree.sha(HERE/'PLAN.json')==os.environ['TASK_PLAN_SHA256']
    q=json.loads((HERE/'QUALIFICATION.json').read_text());assert q['status']=='passed' and q['script_sha256']==tree.sha(Path(__file__)) and q['tree_sha256']==tree.sha(HERE/'tree.py')
    assert json.loads((HERE/'QUIESCE.json').read_text())['controller_exited'];assert not shared.readers(source),shared.readers(source)
    cold=Path(plan['cold_root']);assert cold.name==HERE.name and not cold.exists() and os.path.ismount('/mnt/bos/adgfyl')
    stage=Path('/tmp')/HERE.name;assert not stage.exists();stage.mkdir();cold.mkdir()
    probe=cold/'WRITE-PROBE.txt';probe.write_bytes(b'private full-project recovery\n');assert probe.read_bytes()==b'private full-project recovery\n'
    before=shutil.disk_usage(anchor)._asdict();rows=json.loads((HERE/'METADATA.json').read_text())
    assert rows[0]['path']=='anyviewphys'
    summary=json.loads((HERE/'METADATA-SUMMARY.json').read_text())
    assert summary['inventory_sha256']==tree.sha(HERE/'METADATA.json')
    assert tree.sha(HERE/'SOURCE-INVENTORY.json')==plan['source_inventory_sha256']
    hashed=json.loads((HERE/'SOURCE-INVENTORY.json').read_text());assert tree.same_metadata(hashed,rows)
    assert shutil.disk_usage('/tmp').free>2.25*summary['unique_regular_logical_bytes']+64*2**30
    STATE.update(source_commit=os.environ['TASK_SOURCE'],filesystem_before=before,total_entries=len(rows),
                 source_allocated_bytes=summary['allocated_bytes'],source_unique_logical_bytes=summary['unique_regular_logical_bytes'])
    rows=hashed;emit('reusing_complete_source_SHA256_inventory',entries_done=len(rows),hashed_bytes=summary['unique_regular_logical_bytes'])
    external=[]
    for row in rows:
        if 'target' not in row:continue
        target=Path(row['target']);target=target if target.is_absolute() else (anchor/row['path']).parent/target
        normalized=Path(os.path.normpath(target))
        if not normalized.is_relative_to(source):external.append(dict(path=row['path'],target=row['target'],normalized_external_target=str(normalized)))
    tree.save(HERE/'EXTERNAL-LINKS.json',external)
    receipt,restored=pack(source,anchor,stage,cold/'anyviewphys.tar.zst')
    assert (stage/'readback.tar.zst').stat().st_size+summary['unique_regular_logical_bytes']<plan['maximum_staging_bytes']
    subprocess.run(['sync','-f',str(restored)],check=True)
    storage_events=[]
    actual=tree.hash_rows(tree.scan(restored/'anyviewphys',restored),restored,progress('verifying_every_restored_member'),restored=True,metadata_events=storage_events)
    tree.save(HERE/'RESTORED-STORAGE-METADATA.json',dict(policy='Archive logical attributes, inode/link identity and full SHA256 remain strict; filesystem-local allocation and ctime drift are recorded for this private restored scratch copy. Source metadata/ctime guard remains strict.',events=storage_events))
    tree.verify(rows,actual);receipt.update(complete_restore_verified=True,restored_entries=len(actual),inventory_sha256=tree.sha(HERE/'SOURCE-INVENTORY.json'))
    tree.save(HERE/'RECOVERY.json',receipt)
    emit('checking_source_stability_before_eviction')
    current=tree.scan(source,anchor);same=tree.same_metadata(rows,current)
    if not same:
        a={r['path']:{k:v for k,v in r.items() if k!='sha256'} for r in rows};b={r['path']:r for r in current};changed=[dict(path=p,before=a.get(p),after=b.get(p)) for p in sorted(set(a)|set(b)) if a.get(p)!=b.get(p)]
        tree.save(HERE/'SOURCE-STABILITY-DIFFERENCES.json',dict(changed_count=len(changed),first_changes=changed[:100]));raise AssertionError('Source changed; see SOURCE-STABILITY-DIFFERENCES.json')
    gate();assert not shared.readers(source),shared.readers(source)
    quarantine=Path(plan['quarantine']);assert not quarantine.exists();source.rename(quarantine)
    assert not shared.readers(quarantine),shared.readers(quarantine)
    pointer=dict(run=HERE.name,status='cold_restore_verified',original_path=str(source),recovery=receipt,cold_root=str(cold),
                 inventory=str(cold/'SOURCE-INVENTORY.json.gz'),restore_instructions=str(cold/'RESTORE.md'),source_commit=os.environ['TASK_SOURCE'])
    source.mkdir();tree.save(source/'COLD-ARCHIVE.json',pointer);(source/'RESTORE.md').write_bytes((HERE/'RESTORE.md').read_bytes())
    tree.save(source.with_name('anyviewphys.COLD-ARCHIVE.json'),pointer)
    # Preserve all recovery metadata cold before releasing the verified, detached source.
    with (HERE/'SOURCE-INVENTORY.json').open('rb') as inp,gzip.open(HERE/'SOURCE-INVENTORY.json.gz','wb',compresslevel=1) as out:shutil.copyfileobj(inp,out)
    for name in ['SOURCE-INVENTORY.json.gz','RECOVERY.json','EXTERNAL-LINKS.json','QUIESCE.json','PROJECT-GIT-STATE.json','PLAN.json','RESTORE.md','QUALIFICATION.json','archive.py','tree.py']:
        data=(HERE/name).read_bytes();(cold/name).write_bytes(data);assert tree.sha(cold/name)==hashlib.sha256(data).hexdigest()
    def admission():
        gate();assert not shared.readers(quarantine),shared.readers(quarantine)
    emit('releasing_verified_quarantined_project',released_entries=0,released_allocated_bytes=0)
    count,physical=release(quarantine,source.name,rows,admission)
    assert count==len(rows) and set(p.name for p in source.iterdir())=={'COLD-ARCHIVE.json','RESTORE.md'}
    assert json.loads((source/'COLD-ARCHIVE.json').read_text())==pointer
    with (HERE/'RELEASE-JOURNAL.jsonl').open('rb') as inp,gzip.open(HERE/'RELEASE-JOURNAL.jsonl.gz','wb',compresslevel=1) as out:shutil.copyfileobj(inp,out)
    shutil.copyfile(HERE/'RELEASE-JOURNAL.jsonl.gz',cold/'RELEASE-JOURNAL.jsonl.gz')
    assert tree.sha(HERE/'RELEASE-JOURNAL.jsonl.gz')==tree.sha(cold/'RELEASE-JOURNAL.jsonl.gz')
    shutil.rmtree(stage);fence=Path('/mowenjie/anyviewphys.ARCHIVING.json')
    assert json.loads(fence.read_text())['run']==HERE.name;fence.unlink()
    result=dict(status='complete',source_commit=os.environ['TASK_SOURCE'],script_sha256=tree.sha(Path(__file__)),
                source_entries=len(rows),released_entries=count,released_allocated_bytes=physical,
                recovery=receipt,external_symlink_count=len(external),original_git=json.loads((HERE/'PROJECT-GIT-STATE.json').read_text()),
                filesystem_before=before,filesystem_after=shutil.disk_usage(anchor)._asdict(),
                quarantine_removed=not quarantine.exists(),staging_removed=not stage.exists(),
                original_path_is_restore_stub=True,completed_unix=time.time(),GPU_allocations=0)
    tree.save(HERE/'RESULT.json',result);tree.save(cold/'RESULT.json',result);emit('complete',**result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--qualify',action='store_true');args=parser.parse_args()
    try:qualify() if args.qualify else run()
    except Exception as error:emit('failed_preserved',error=repr(error));raise
