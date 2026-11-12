"""Resume the identical frozen KV panel using local raw data and verified BOS archives.

PFS is read only for code, fixed weights, assets and canonical anchors. Completed
metadata and raw archives go directly to the existing private BOS bucket.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time

RUN = 'robocasa-kv-comparison-20261112-r1'
PROJECT = Path('/mowenjie/fluffy-bassoon-2')
COLD = Path('/mnt/bos/adgfyl/strong-plant-archives') / RUN
REMOTE = COLD / 'local-migration03/metadata'
HERE = Path(__file__).resolve().parent
SCI_PIN = '87984718c9199797d31a158c47225cc695623afe'
VENDOR_PIN = '8f9bfa7661928ef8c17494b04f4d7bf3125634e1'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def copy_exact(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha(target) == sha(source), target
        return
    with source.open('rb') as a, target.open('xb') as b:
        shutil.copyfileobj(a, b, 8 << 20)
        b.flush()
        os.fsync(b.fileno())
    assert sha(target) == sha(source), target


def archive(bank, local, audit_result):
    relative = bank.relative_to(local)
    assert re.fullmatch(r'bank-\d{3}', bank.name)
    audit_result(bank / 'RESULT.json', bank / 'AUDIT.json')
    audit = json.loads((bank / 'AUDIT.json').read_text())
    assert audit['passed'] and audit['result_sha256'] == sha(bank / 'RESULT.json')
    files = sorted(p for p in bank.rglob('*') if p.is_file())
    assert not any(p.is_symlink() for p in bank.rglob('*'))
    inventory = [dict(relative=str(p.relative_to(bank)), bytes=p.stat().st_size, sha256=sha(p)) for p in files]
    work = local / 'archives' / relative
    work.mkdir(parents=True, exist_ok=False)
    tar = work / 'wave.tar'
    with tarfile.open(tar, 'w') as f:
        for p in files:
            f.add(p, arcname=str(p.relative_to(bank)), recursive=False)
    cold = COLD / 'local-migration03/packed' / local.name / relative.with_suffix('.tar')
    copy_exact(tar, cold)
    restored = work / 'restored'
    restored.mkdir()
    with tarfile.open(cold, 'r|') as f:
        for member in f:
            rel = PurePosixPath(member.name)
            assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
            out = restored / member.name
            out.parent.mkdir(parents=True, exist_ok=True)
            with f.extractfile(member) as a, out.open('xb') as b:
                shutil.copyfileobj(a, b, 8 << 20)
    assert {str(p.relative_to(restored)) for p in restored.rglob('*') if p.is_file()} == {x['relative'] for x in inventory}
    for item in inventory:
        assert sha(restored / item['relative']) == sha(bank / item['relative']) == item['sha256']
    manifest = REMOTE / 'archives' / relative / 'packed/RESULT.json'
    saved = dict(passed=True, source=str(bank), cold=str(cold), archive_sha256=sha(tar),
        archive_bytes=tar.stat().st_size, payload_bytes=sum(x['bytes'] for x in inventory),
        files=inventory, all_stageback_bytes_verified=True,
        independent_source_audit_sha256=sha(bank / 'AUDIT.json'),
        original_hot_evidence_untouched=True, storage_migration='local-scratch03')
    write(manifest, saved)
    assert json.loads(manifest.read_text()) == saved
    marker = dict(verified=True, format='audited_packed_wave_v1', source=str(bank),
        manifest=str(manifest), archive_sha256=saved['archive_sha256'], cold=str(cold),
        result_sha256=sha(bank / 'RESULT.json'), raw_stageback_required=True,
        original_metadata_paths_and_bytes_preserved=True,
        raw_reader_admission='Restore archive members to the recorded local source path before raw audit',
        storage_migration='local-scratch03')
    target = REMOTE / relative
    # RESULT is the publication marker: consumers cannot see a closed bank
    # before its audit, complete manifest and cold readback are durable.
    for name in ('AUDIT.json', 'FITTED-DEPLOYMENT.json'):
        if (bank / name).exists():
            copy_exact(bank / name, target / name)
    write(target / 'PACKED-ARCHIVE.json', marker)
    copy_exact(bank / 'RESULT.json', target / 'RESULT.json')
    write(bank / 'PACKED-ARCHIVE.json', marker)
    # Release only verified duplicates; metadata and every raw byte survive BOS.
    for child in list(bank.iterdir()):
        if child.name not in ('RESULT.json', 'AUDIT.json', 'FITTED-DEPLOYMENT.json', 'PACKED-ARCHIVE.json'):
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    shutil.rmtree(work)


def watcher(local, done, errors):
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('local_audit', HERE.parent / 'robocasa-eight-hour-babysit-20261104-r1/AUDIT-ROLLOUT.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pending = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        while True:
            for bank, future in list(pending.items()):
                if future.done():
                    try:
                        future.result()
                    except BaseException as e:
                        errors.append(dict(bank=str(bank), error=repr(e)))
                    del pending[bank]
            banks = list(local.glob('readouts/*/*/*/*/*/bank-[0-9][0-9][0-9]/RESULT.json'))
            for p in banks:
                if not errors and not (p.parent / 'PACKED-ARCHIVE.json').exists() and p.parent not in pending:
                    pending[p.parent] = pool.submit(archive, p.parent, local, module.audit_result)
            hot = 0
            for p in local.rglob('*'):
                try:
                    if p.is_file():
                        hot += p.stat().st_size
                except FileNotFoundError:
                    pass
            state = dict(checked_unix=time.time(), packed_archive_enabled=not errors,
                current_hot_bytes=hot, pending_archives=len(pending), errors=errors,
                local_free_bytes=shutil.disk_usage(local).free, published_banks=sum((p.parent/'PACKED-ARCHIVE.json').exists() for p in banks))
            write(local / 'monitor/HOT-STORAGE.json', state)
            write(REMOTE / 'jobs' / local.name / 'HOT-STORAGE.json', state)
            if done.is_set() and not pending:
                return
            time.sleep(10)


def main():
    pin, method, attempt, minutes = sys.argv[1:]
    assert sys.platform == 'linux' and socket.gethostname().startswith('job-')
    assert method in ('dpt', 'dit', 'ic_iql') and re.fullmatch(r'attempt\d+', attempt)
    assert 10 < int(minutes) <= 360
    code = HERE.parent.parent
    def head(p):
        return subprocess.check_output(['git', '-C', str(p), 'rev-parse', 'HEAD'], text=True).strip()
    assert head(code) == pin and not subprocess.check_output(['git','-C',str(code),'status','--porcelain','--untracked-files=no'])
    science, vendor = PROJECT/'sources/steering-packed-87984718', PROJECT/'sources/critic-selected-a1229239-r1'
    assert head(science) == SCI_PIN and head(vendor) == VENDOR_PIN
    assert (REMOTE / 'MIGRATION.json').exists()
    local = Path('/tmp') / RUN / f'{method}-{attempt}'
    local.mkdir(parents=True, exist_ok=False)
    assert local.stat().st_dev != PROJECT.stat().st_dev
    assert shutil.disk_usage(local).free >= 112*2**30, '96 GiB local floor plus16 GiB hot/packing buffer'
    env = dict(os.environ, PYTHONPATH=str(science/'src')+':'+str(vendor), RLINF_PATH=str(vendor),
        OPENPI_DATA_HOME=str(PROJECT/'setups/robocasa365-native-20261101-r1/cache/.cache/openpi'),
        ROBOCASA_ASSETS_PATH=str(PROJECT/'setups/robocasa365-native-20261101-r1/assets'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NCCL_P2P_DISABLE='1',
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', FLUFFY_GPU_EXECUTION='baige',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1',
        STEERING_CODE=str(science), STEERING_COMMIT=SCI_PIN, STEERING_STORAGE_ACCOUNTING='local03',
        STEERING_WAVE_ALLOWANCE_SECONDS='600', KV_LOCAL_ROOT=str(local))
    if 'CUDA_VISIBLE_DEVICES' not in env:
        uuid = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'], text=True).strip()
        assert uuid.startswith('GPU-') and '\n' not in uuid and uuid == env['NVIDIA_VISIBLE_DEVICES']
        env['CUDA_VISIBLE_DEVICES'] = '0'
    job = REMOTE / 'jobs' / local.name
    job.mkdir(parents=True, exist_ok=False)
    write(job/'STORAGE.json', dict(local=str(local), local_free_bytes=shutil.disk_usage(local).free,
        PFS_raw_growth_bytes=0, archive_root=str(COLD), metadata_root=str(REMOTE), code_commit=pin))
    assert json.loads((job/'STORAGE.json').read_text())['PFS_raw_growth_bytes'] == 0
    probe = job/'GPU-PROBE.json'
    subprocess.run([sys.executable,'-B',str(code/'scripts/[train]gpu_probe.py'),'--expected-count','1',
        '--purpose','evaluation','--baige-container','--output',str(probe)], env=env, check=True)
    fit = PROJECT/'outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/finalized-fits-20261106-r1'
    index = json.loads((fit/'INDEX.json').read_text())
    for name, item in index['models']['kv_bridge/'+method]['artifacts'].items():
        assert sha(Path(item['path'])) == item['sha256']
    source = REMOTE/'readouts/kv_bridge'/method
    target = local/'readouts/kv_bridge'/method
    shutil.copytree(source, target)
    assert all((p.parent/'PACKED-ARCHIVE.json').exists() for p in target.glob('*/*/*/bank-*/RESULT.json'))
    done, errors = threading.Event(), []
    thread = threading.Thread(target=watcher, args=(local,done,errors), daemon=False)
    thread.start()
    while not (local/'monitor/HOT-STORAGE.json').exists():
        if not thread.is_alive():
            raise RuntimeError('Local archive watcher did not start')
        time.sleep(1)
    started = time.time()
    # Leave ten minutes for final independent audits and complete cold stageback.
    deadline = started + int(minutes)*60 - 600
    rc = 0
    try:
        with (job/'console.log').open('x') as log:
            for split in ('ID','OOD'):
                out = target/split
                command = [sys.executable,'-B',str(HERE/'LOCAL-READOUT.py'),'--code',str(science),
                    '--commit',SCI_PIN,'--fit',str(fit/'kv_bridge'/method),'--output',str(out),
                    '--probe',str(probe),'--interface','kv_bridge','--objective',method,
                    '--split',split,'--deadline',str(deadline),'--attempt',attempt]
                child = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                    timeout=max(1,deadline-time.time()+30))
                rc = child.returncode
                if rc or not (out/'RESULT.json').exists():
                    rc = rc or 3
                    break
    except BaseException:
        rc = 1
        raise
    finally:
        done.set()
        thread.join()
        if errors:
            rc = rc or 4
        for split in ('ID','OOD'):
            out = target/split
            for name in ('BINDING.json','selected-candidate.pt'):
                if (out/name).exists():
                    copy_exact(out/name, REMOTE/'readouts/kv_bridge'/method/split/name)
            if not errors and (out/'RESULT.json').exists():
                copy_exact(out/'RESULT.json', REMOTE/'readouts/kv_bridge'/method/split/'RESULT.json')
        write(job/'EXIT.json', dict(exit_code=rc,started_unix=started,ended_unix=time.time(),GPUs=1,
            archive_errors=errors,all_closed_banks_published=all((p.parent/'PACKED-ARCHIVE.json').exists()
                for p in target.glob('*/*/*/bank-*/RESULT.json'))))
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
