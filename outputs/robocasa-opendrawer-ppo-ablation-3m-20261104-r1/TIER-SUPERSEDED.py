"""Cold-tier only superseded actor payloads from this matched experiment."""
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path('/mowenjie/fluffy-bassoon-2')
RUN=ROOT/'outputs/robocasa-opendrawer-ppo-ablation-3m-20261104-r1'
HELPER=ROOT/'sources/stage1-grpo-013c2f10/outputs/storage-audit-baige-20261104-r1/archive.py'
HASH='ce5df6e4d313fbdc6948a897ee1311c143e8efef00fdcd86df4da82e4888dbb5'
ATTEMPTS={'C':('attempt-01','job-1gheyfqughhs'),'E':('attempt-02','job-t2edw03k1kwn'),'D':('attempt-01','job-jbf0ex8k1s3o')}


def read(p):
    return json.loads(p.read_text())


def tier(item):
    arm,step=item
    attempt,job=ATTEMPTS[arm]
    a=RUN/arm/attempt
    assert read(a/'manifest.json')['hostname']==job+'-master-0'
    cps=a/'OpenDrawer/checkpoints'
    source=cps/f'global_step_{step}'
    assert step in (1,10,20,30) and read(source/'pilot-boundary.json')['global_step']==step
    eligible=[]
    for c in cps.glob('global_step_*'):
        m=c/'pilot-boundary.json'
        if c.is_dir() and m.is_file() and (c/'actor/model_state_dict/full_weights.pt').is_file():
            b=read(m)
            if b['completed'] and all(Path(row['path']).is_file() for row in b['actor_checkpoint_files']+b['workers']):
                eligible.append((b['global_step'],c))
    newest,latest=max(eligible)
    assert newest>step and len(read(latest/'pilot-boundary.json')['workers'])==18
    assert latest!=source and (source/'pilot-boundary.json').is_file()
    bound=read(source/'pilot-boundary.json')
    assert len(bound['workers'])==18 and all(Path(w['path']).is_file() for w in bound['workers'])
    # The only training input to all three jobs is the shared step-zero initializer.
    # The fixed common assessment consumes step 40. Keep their latest recovery boundary hot as well.
    actor=source/'actor'
    receipt=source/'actor.bos-archive.json'
    if receipt.exists() and read(receipt).get('source_reclaimed'):
        return read(receipt)
    assert len(bound['actor_checkpoint_files'])==4
    for f in bound['actor_checkpoint_files']:
        path=Path(f['path'])
        assert path.is_relative_to(actor) and path.stat().st_size==f['bytes']
        assert archive.digest(path)==f['sha256']
    row=archive.archive(str(actor.relative_to(ROOT)))
    assert row['source_reclaimed'] and row['source_compare'] and row['exact_member_coverage'] and row['bos_readback_sha256_verified']
    assert (latest/'actor/model_state_dict/full_weights.pt').is_file()
    archive.save(source/'COLD-STORAGE.json',dict(row, retained_local_marker=str(source/'pilot-boundary.json'),
        retained_worker_sidecars=[w['path'] for w in bound['workers']], newer_verified_recovery=str(latest)))
    return row


if __name__=='__main__':
    assert hashlib.sha256(HELPER.read_bytes()).hexdigest()==HASH
    spec=importlib.util.spec_from_file_location('archive_helper',HELPER)
    archive=importlib.util.module_from_spec(spec);spec.loader.exec_module(archive)
    archive.RUN=RUN.name; archive.ROOT=ROOT; archive.AUDIT=RUN/'checkpoint-tier-receipts'
    archive.STAGE=Path('/tmp')/(RUN.name+'-sealed-checkpoints')
    archive.BOS=Path('/mnt/bos/adgfyl/strong-plant-archives')/RUN.name/'sealed-checkpoints'
    assert os.path.ismount('/mnt/bos/adgfyl') and shutil.disk_usage('/tmp').free>64*(1<<30)
    for folder in (archive.AUDIT,archive.STAGE,archive.BOS): folder.mkdir(parents=True,exist_ok=True)
    probe=archive.BOS/'WRITE-PROBE.txt'
    if not probe.exists():probe.write_text(RUN.name+'\n')
    assert probe.read_text()==RUN.name+'\n'
    items=[(s.split('/')[0],int(s.split('/')[1])) for s in sys.argv[1:]]
    assert items and len(items)==len(set(items))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(tier,item):item for item in items}
        failures=[]
        for task in concurrent.futures.as_completed(pending):
            try:
                result=task.result();print(pending[task],result['source_reclaimed'],flush=True)
            except Exception as e:
                failures.append((pending[task],repr(e)));print(failures[-1],flush=True)
        if failures: raise RuntimeError(failures)
