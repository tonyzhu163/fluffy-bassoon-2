"""Read-only final retention, restoration and accounting check on allocated Linux."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

TASKS = ['OpenDrawer', 'CloseToasterOvenDoor', 'PickPlaceDrawerToCounter',
         'CoffeeSetupMug', 'TurnOffStove']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wait-seconds', type=int, default=1800)
    args = p.parse_args()
    root = args.root.resolve()
    assert root.name == 'robocasa-flow-sde-10m-per-task-20261103-r1'
    assert args.output.parent.resolve() == root and not args.output.exists()
    tier = root / 'tools/475e1c5d-TIER-CHECKPOINTS.py'
    assert sha(tier) == '88a12eaf79e18bfd527f22870085414505eb473737773230b364c3b790bc3b97'
    spec = importlib.util.spec_from_file_location('qualified_tier', tier)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoints, cold_models, references = [], [], {}
    stage_root = root / 'offline-initial-ratio-staging-r3'
    stage = read(stage_root / 'PROGRESS.json')
    assert stage['completed'] and len(stage['receipts']) == 80
    consumed = {(r['task'], r['step']): r for r in stage['receipts']}
    assert len(consumed) == 80 and not list(stage_root.rglob('extracted-full_weights.pt'))
    for task in TASKS:
        native = read(root / task / 'FINAL-READOUT-R1.json')
        assert native['completed'] and not native['admitted_teacher']
        cp = Path(native['checkpoint']['path'])
        assert cp.resolve().is_relative_to(root / task) and not (cp / 'COLD-STORAGE.json').exists()
        boundary = module.verify(cp)
        assert boundary['global_step'] == 130
        assert sha(cp / 'pilot-boundary.json') == native['checkpoint']['boundary_sha256']
        checkpoints.append(dict(task=task, path=str(cp), model_and_all_18_workers_SHA256_rechecked=True,
                                boundary_sha256=sha(cp / 'pilot-boundary.json')))
        for marker in (root / task).glob(f'*/{task}/checkpoints/global_step_*/COLD-STORAGE.json'):
            receipt = read(marker)
            assert receipt['tar_compare'] and receipt['full_BOS_readback']
            archive = Path(receipt['archive'])
            assert archive.stat().st_size == receipt['archive_bytes']
            assert read(Path(str(archive) + '.json')) == receipt
            boundary = read(marker.parent / 'pilot-boundary.json')
            assert boundary['completed'] and len(boundary['workers']) == 18
            step = int(boundary['global_step'])
            ack = consumed[task, step]
            assert ack['private_copy_reclaimed']
            ready_path = stage_root / task / str(step) / 'READY.json'
            ready = read(ready_path)
            assert sha(ready_path) == ack['ready_sha256']
            assert ready['checkpoint']['marker'] == str(marker.parent / 'pilot-boundary.json')
            assert ready['storage']['original_archive_sha256'] == receipt['sha256']
            ack_path = root / 'offline-initial-ratio-gpu-r3' / task / f'CONSUMED-{step}.json'
            restored = read(ack_path)
            assert sha(ack_path) == ack['consumed_sha256'] and restored['strict_load_passed']
            assert restored['model_sha256'] == ready['checkpoint']['sha256']
            cold_models.append(dict(task=task, step=step, archive=str(archive),
                archive_sha256=receipt['sha256'], original_full_object_readback=True,
                actual_GPU_model_restore_SHA256=restored['model_sha256'], receipt_sha256=sha(marker)))
        print(json.dumps(dict(task=task, newest_checkpoint_rechecked=True)), flush=True)
    assert len(cold_models) == 75
    joins = read(root / 'COMPLETE-CORPUS-JOINS-R1.json')
    assert joins['completed'] and joins['unique_cohorts'] == 650
    cohorts = set()
    for ref in joins['audit_references']:
        assert sha(ref['path']) == ref['sha256']
        for row in read(ref['path'])['audits']:
            key = row['task'], row['collection_iteration']
            assert key not in cohorts
            cohorts.add(key)
            assert sha(row['actor_sampling_environment_join']) == row['join_sha256']
            assert sha(row['raw_file_manifest']) == row['raw_file_manifest_sha256']
    assert cohorts == {(task, i) for task in TASKS for i in range(1, 131)}
    tables = read(root / 'result-tables-r1/RESULT.json')
    assert tables['completed'] and tables['likelihood_rows'] == 80 and tables['initial_parity_zero_all_tasks']
    for name, digest in tables['output_sha256'].items():
        assert sha(root / 'result-tables-r1' / name) == digest
    cost = read(root / 'GPU-COST-CLOSEOUT-R1.json')
    assert cost['completed'] and cost['active_GPU_jobs'] == 0
    deadline = time.monotonic() + args.wait_seconds
    while not read(root / 'CLOSED-CORPUS-WATCH-R1.json')['completed']:
        assert not (root / 'CLOSED-CORPUS-WATCH-R1-ERROR.json').exists()
        if time.monotonic() >= deadline:
            raise TimeoutError('Last corpus archive is not yet complete')
        time.sleep(5)
    corpora = []
    for task in TASKS:
        prepared_path = root / f'CORPUS-ARCHIVE-{task}-R1-PREPARED.json'
        reclaim_path = root / f'CORPUS-ARCHIVE-{task}-R1-RECLAIMED.json'
        prepared, reclaimed = read(prepared_path), read(reclaim_path)
        assert prepared['completed'] and prepared['full_object_readback'] and prepared['all_archive_members_hash_compared']
        assert reclaimed['completed'] and reclaimed['reclaim_complete'] and reclaimed['full_corpus_audit_through'] == 130
        assert sha(prepared_path) == reclaimed['prepared_sha256']
        assert prepared['archive_sha256'] == reclaimed['archive_sha256']
        archive = Path(prepared['archive'])
        assert archive.stat().st_size == prepared['archive_bytes']
        assert archive.stat().st_mtime_ns == prepared['archive_mtime_ns']
        assert sha(prepared['manifest']) == prepared['manifest_sha256']
        qc = reclaimed['qc_archive']
        assert qc['full_object_readback'] and qc['all_archive_members_hash_compared']
        assert sha(qc['archive']) == qc['archive_sha256'] and sha(qc['manifest']) == qc['manifest_sha256']
        for ref in reclaimed['full_audit_refs']:
            assert sha(ref['path']) == ref['sha256']
        assert read(root / task / 'CORPUS-COLD-STORAGE.json')['reclaim_complete']
        for pattern in ['q1-*', 'production-*']:
            for stage_dir in (root / task).glob(pattern):
                assert not [p for p in stage_dir.rglob('*') if p.is_file() and
                            'checkpoints' not in p.parts and p.name.endswith(('.pt', '.pt.gz'))]
        corpora.append(dict(task=task, archive=str(archive), archive_bytes=prepared['archive_bytes'],
            archive_sha256=prepared['archive_sha256'], full_object_and_member_readback_recorded=True,
            manifest_SHA256_rechecked=True, QC_archive_SHA256_rechecked=True,
            prepared_receipt_sha256=sha(prepared_path), reclaimed_receipt_sha256=sha(reclaim_path),
            restore=reclaimed['restore'], newest_full_checkpoint_retained=True))
    for name in ['COMPLETE-CORPUS-JOINS-R1.json', 'GPU-COST-CLOSEOUT-R1.json',
                 'CLOSED-CORPUS-WATCH-R1.json', 'offline-initial-ratio-staging-r3/PROGRESS.json',
                 'result-tables-r1/RESULT.json']:
        references[name] = sha(root / name)
    result = dict(completed=True, checked_unix=time.time(), source_sha256=sha(Path(__file__)),
        references=references, corpora=corpora, newest_full_checkpoints=checkpoints,
        restored_cold_models=cold_models, complete_corpus_join_count=650,
        TRAIN_episodes=83200, TRAIN_decision_samples=998400, valid_TRAIN_controls=49920000,
        all_attempt_policy_controls=joins['all_attempt_policy_controls'], admitted_teachers=0,
        retained_role='reference_source', student_fitting_started=False,
        fixed_B16_checkpoint_reports=80, initial_parity_zero_all_tasks=True,
        accounted_GPU_hours=cost['accounted_GPU_hours'], active_GPU_jobs=0,
        new_controls=0, SGD=0, model_calls=0)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(completed=True, corpora=5, checkpoints=5, restored_cold_models=75,
                          accounted_GPU_hours=cost['accounted_GPU_hours'])), flush=True)


if __name__ == '__main__':
    main()
