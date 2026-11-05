"""CPU audits and verified cold archival of closed waves;20-minute fit snapshots."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time


ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-three-task-kv-velocity-meta-20261105-r1')
COLD = Path('/mnt/bos/adgfyl/strong-plant-archives/robocasa-three-task-kv-velocity-meta-20261105-r1')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp');tmp.write_text(json.dumps(value, indent=2)+'\n');tmp.replace(path)


def archive_closed_bank(bank, audit_result):
    bank = Path(bank)
    if bank.is_symlink():
        return
    relative = bank.relative_to(ROOT)
    if relative.parts[0] not in ('readouts', 'qualification-02', 'qualification-fitted-pfs03', 'qualification-fitted-reset-pfs04'):
        raise ValueError('Archive only owned closed native waves in this run')
    result = json.loads((bank / 'RESULT.json').read_text())
    if result.get('optimizer_updates') != 0 or not result.get('native_parameters_frozen'):
        raise ValueError('Require the completed deployment-only bank receipt')
    audit_path = bank / 'AUDIT.json'
    if not audit_path.exists():
        audit_result(bank / 'RESULT.json', audit_path)
    audit = json.loads(audit_path.read_text())
    if not audit['passed'] or audit['result_sha256'] != sha(bank / 'RESULT.json'):
        raise ValueError('Require an independent audit bound to this exact closed wave')
    entries = sorted(bank.rglob('*'))
    if any(p.is_symlink() for p in entries):
        raise ValueError('Do not move linked/shared dependencies')
    files = [p for p in entries if p.is_file()]
    target = COLD / relative
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for path in files:
        name = path.relative_to(bank);destination = target / name
        digest = sha(path);size = path.stat().st_size
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.stat().st_size != size or sha(destination) != digest:
                raise ValueError('Existing cold artifact differs; diagnose without replacing it')
        else:
            with path.open('rb') as incoming, destination.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, 8 << 20);outgoing.flush();os.fsync(outgoing.fileno())
            if destination.stat().st_size != size or sha(destination) != digest:
                raise ValueError('Cold byte/size verification failed; retain the hot source')
        records.append(dict(relative=str(name), bytes=size, sha256=digest))
    # RESULT is a closure marker: the writer never reopens that wave. Verify
    # the complete unchanged leaf set and bytes again before replacing its path.
    if {p.relative_to(bank) for p in bank.rglob('*') if p.is_file()} != {Path(x['relative']) for x in records}:
        raise ValueError('The supposedly closed wave acquired another file')
    if not all(sha(bank / x['relative']) == x['sha256'] for x in records):
        raise ValueError('The closed source changed during archival')
    record = dict(source=str(bank), target=str(target), result_sha256=sha(bank / 'RESULT.json'),
        files=records, closure='Complete immutable six-episode wave; parent allocation may continue on other waves',
        source_audit_sha256=sha(audit_path))
    manifest = target / 'ARCHIVE-MANIFEST.json'
    write(manifest, record)
    quarantine = bank.with_name(bank.name + '.verified-hot-copy')
    if quarantine.exists():
        raise ValueError('A previous archival transaction needs diagnosis')
    bank.rename(quarantine)
    try:
        bank.symlink_to(target, target_is_directory=True)
        if not all(sha(bank / x['relative']) == x['sha256'] for x in records):
            raise ValueError('Stable original artifact paths failed cold verification')
    except BaseException:
        if bank.is_symlink():bank.unlink()
        quarantine.rename(bank)
        raise
    shutil.rmtree(quarantine)
    write(ROOT / 'archives' / relative / 'RECEIPT.json', dict(completed=True,
        source=str(bank), target=str(target), files=len(records), bytes=sum(x['bytes'] for x in records),
        manifest_sha256=sha(manifest), stable_original_paths_verified=True,
        raw_draw_command_truth_and_all_video_frames_audited=True, finished_unix=time.time()))


def snapshot(job):
    def read(path):
        try:return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):return None
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    ledger = read(ROOT.parent / 'robocasa-eight-hour-babysit-20261104-r1/META-ALLOCATION-LEDGER.json')
    charged = ledger['preparation_GPU_hours']
    for row in ledger['allocations']:
        end = datetime.datetime.fromisoformat(row['released_shanghai']) if row.get('released_shanghai') else now
        charged += row['GPUs'] * (end-datetime.datetime.fromisoformat(row['created_shanghai'])).total_seconds()/3600
    methods = {}
    for interface in ('kv_bridge', 'velocity_bridge'):
        for method in ('ad', 'dpt', 'dit', 'ic_iql'):
            fit = ROOT / 'fits' / interface / method
            workflow = ROOT / 'workflows' / interface / method
            logs = list(workflow.glob('train-*.log'))
            latest_log = max(logs, key=lambda path:path.stat().st_mtime) if logs else None
            tail = ''
            if latest_log:
                with latest_log.open('rb') as stream:
                    stream.seek(max(0, latest_log.stat().st_size-16384))
                    tail = stream.read().decode(errors='replace')
            methods[interface+'/'+method] = dict(status=read(fit/'training-status.json'),
                checkpoint_mtime=(fit/'latest.pt').stat().st_mtime if (fit/'latest.pt').exists() else None,
                latest_worker_log=str(latest_log) if latest_log else None,
                latest_worker_log_mtime=latest_log.stat().st_mtime if latest_log else None,
                worker_failed='Traceback (most recent call last):' in tail,
                last_worker_error=tail.splitlines()[-1] if 'Traceback (most recent call last):' in tail else None,
                workflow_result=read(workflow/'RESULT.json'),
                readouts={s:read(ROOT/'readouts'/interface/method/s/'PROGRESS.json') for s in ('ID','OOD')})
    write(ROOT/'monitor/LIVE-CHECK.json', dict(cluster='Baige Pro6K',job_id=job,
        checked_shanghai=now.isoformat(),charged_GPU_hours=charged,
        remaining_GPU_hours=ledger['approved_GPU_hours']-charged,methods=methods,
        next_check_shanghai=(now+datetime.timedelta(minutes=20)).isoformat()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--auditor',type=Path,required=True)
    p.add_argument('--job-id',required=True)
    p.add_argument('--once-bank',type=Path)
    args=p.parse_args()
    if sys.platform!='linux':raise ValueError('Use the allocated Linux development CPUs')
    import cv2,torch
    cv2.setNumThreads(1);torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('independent_native_auditor',args.auditor)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if args.once_bank:
        archive_closed_bank(args.once_bank,module.audit_result);return
    began,last=time.time(),0
    spec=importlib.util.spec_from_file_location('paired_readout_reducer',Path(__file__).with_name('REDUCE.py'))
    reducer=importlib.util.module_from_spec(spec);spec.loader.exec_module(reducer)
    with ThreadPoolExecutor(max_workers=4) as pool:
        while time.time()-began < 12*3600:
            banks=[p.parent for p in ROOT.glob('readouts/*/*/*/*/*/bank-*/RESULT.json') if not p.parent.is_symlink()]
            for result in pool.map(lambda bank:archive_closed_bank(bank,module.audit_result),banks):pass
            summaries=[]
            for result_path in ROOT.glob('readouts/*/*/*/RESULT.json'):
                summary=reducer.reduce_group(result_path.parent)
                if summary:summaries.append(summary)
            if len(summaries)==16:
                assert sum(x['episodes'] for x in summaries)==24000
                write(ROOT/'COMPLETE.json',dict(completed=True,models=8,audited_readout_groups=16,
                    episodes=24000,policy_controls=15120000,groups=summaries,
                    completed_unix=time.time()))
                snapshot(args.job_id)
                return
            if time.time()-last>=1200:
                snapshot(args.job_id);last=time.time()
            time.sleep(30)


if __name__=='__main__':main()
