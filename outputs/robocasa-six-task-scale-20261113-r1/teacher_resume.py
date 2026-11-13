"""Explicit step-10 to step-30 continuation; --inspect is CPU-only."""
import argparse
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

HERE = Path(__file__).resolve().parent
CODE = HERE.parent.parent
PROJECT = Path('/mowenjie/fluffy-bassoon-2')
PRIOR = 'robocasa-six-source-decision-20261112-r1'
COLD = Path('/mnt/bos/adgfyl/strong-plant-archives')
sys.path.insert(0, str(HERE.parent / PRIOR))
from teacher_local import archive, sha, snapshot, write


def inputs(task):
    spec = json.loads((HERE / 'SOURCES.json').read_text())[task]
    receipts = {}
    for kind in ('checkpoint', 'before'):
        path = COLD / PRIOR / 'local01/teachers' / task / spec[kind + '_attempt'] / 'ARCHIVE.json'
        assert sha(path) == spec[kind + '_receipt_sha256']
        receipt = json.loads(path.read_text())
        assert receipt['verified'] and receipt['all_stageback_members_verified']
        assert receipt['full_archive_SHA_readback'] and receipt['all_original_sources_rehashed']
        assert Path(receipt['cold']).stat().st_size == receipt['archive_bytes']
        assert Path(receipt['source']) == Path('/tmp') / PRIOR / 'teachers' / task / spec[kind + '_attempt']
        if kind == 'checkpoint':
            assert receipt['archive_sha256'] == spec['checkpoint_archive_sha256']
        receipts[kind] = receipt
    return spec, receipts


def selection(task, kind, receipt, *, inspect=False):
    prefix = f'{task}/checkpoints/global_step_10/'
    selected = []
    for item in receipt['files']:
        name = item['path']
        if kind == 'checkpoint':
            use = name == prefix + 'pilot-boundary.json' if inspect else name.startswith(prefix)
        else:
            use = (name.startswith('canonical-eval/') or name == 'before-evaluation.json'
                   or name.startswith('environment/') and name.endswith('/before-i000-round00-episodes.json'))
        if use:
            selected.append(item)
    assert selected
    return selected


def restore(receipt, members, destination):
    """Verify every selected byte against the hash-bound, previously verified inventory."""
    expected = {item['path']: item for item in members}
    assert len(expected) == len(members)
    destination.mkdir(parents=True, exist_ok=False)
    seen = set()
    with tarfile.open(receipt['cold'], 'r:') as packed:
        for member in packed:
            if member.name not in expected:
                continue
            rel = PurePosixPath(member.name)
            assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
            assert member.name not in seen
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with packed.extractfile(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output, 8 << 20)
            item = expected[member.name]
            assert target.stat().st_size == item['bytes'] and sha(target) == item['sha256']
            seen.add(member.name)
            if seen == set(expected):
                break
    assert seen == set(expected)
    return dict(files=len(seen), bytes=sum(x['bytes'] for x in members), selected_members_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--task', choices=tuple(json.loads((HERE / 'SOURCES.json').read_text())), required=True)
    parser.add_argument('--attempt', default='attempt01')
    parser.add_argument('--inspect', type=Path, help='Fresh CPU-only evidence directory; never launches a model')
    args = parser.parse_args()
    assert sys.platform == 'linux' and re.fullmatch(r'attempt\d+', args.attempt)
    assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() == args.pin
    assert not subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain', '--untracked-files=no'])
    if not args.inspect:
        assert socket.gethostname().startswith('job-')
    spec, receipts = inputs(args.task)
    source = Path('/tmp') / HERE.name / 'teachers' / args.task / args.attempt
    remote = COLD / HERE.name / 'teachers' / args.task / args.attempt
    started = time.time()
    selected = {kind: selection(args.task, kind, receipt, inspect=bool(args.inspect))
                for kind, receipt in receipts.items()}
    if args.inspect:
        args.inspect.mkdir(parents=True, exist_ok=False)
        destinations = {kind: args.inspect / kind for kind in receipts}
    else:
        assert not source.exists() and not remote.exists()
        source.parent.mkdir(parents=True, exist_ok=True)
        assert source.parent.stat().st_dev != PROJECT.stat().st_dev
        required = sum(item['bytes'] for items in selected.values() for item in items)
        # Preserve the previously qualified 200 GiB scratch floor plus room for collection/archival.
        assert shutil.disk_usage(source.parent).free >= 300 * 2**30 + required
        remote.mkdir(parents=True)
        probe = remote / '.write-probe'
        probe.write_text('RoboCasa source continuation'); assert probe.read_text() == 'RoboCasa source continuation'
        probe.unlink()
        write(remote / 'LAUNCH.json', dict(source_commit=args.pin, task=args.task, attempt=args.attempt,
            hostname=socket.gethostname(), started_unix=started, resumed_step=10, target_step=30,
            maximum_scheduler_minutes=120, maximum_GPU_hours=16, automatic_retry=False,
            runtime_commit='8f9bfa7661928ef8c17494b04f4d7bf3125634e1'))
        destinations = {kind: Path(receipt['source']) for kind, receipt in receipts.items()}
    restoration = {}
    if destinations['checkpoint'] == destinations['before']:
        members = {item['path']: item for items in selected.values() for item in items}
        restoration['combined'] = restore(receipts['checkpoint'], list(members.values()), destinations['checkpoint'])
    else:
        for kind in receipts:
            restoration[kind] = restore(receipts[kind], selected[kind], destinations[kind])
    checkpoint = destinations['checkpoint'] / args.task / 'checkpoints/global_step_10'
    boundary_path = checkpoint / 'pilot-boundary.json'
    boundary = json.loads(boundary_path.read_text())
    assert boundary['completed'] and boundary['global_step'] == 10
    expected_workers = {f'{role}-{rank:02d}.pt' for role, count in (('actor', 2), ('env', 8), ('rollout', 8))
                        for rank in range(count)}
    assert {Path(item['path']).name for item in boundary['workers']} == expected_workers
    expected_files = {row['path']: row for row in receipts['checkpoint']['files']}
    for item in boundary['actor_checkpoint_files'] + boundary['workers']:
        rel = Path(item['path']).relative_to(receipts['checkpoint']['source'])
        assert rel.parts[:3] == (args.task, 'checkpoints', 'global_step_10')
        assert expected_files[str(rel)]['sha256'] == item['sha256']
        if not args.inspect:
            assert sha(destinations['checkpoint'] / rel) == item['sha256']
    before_root = destinations['before']
    before = json.loads((before_root / 'before-evaluation.json').read_text())
    assert before['completed']
    refs, successes = [], 0
    for rank in range(8):
        path = before_root / 'environment' / f'rank-{rank:02d}/before-i000-round00-episodes.json'
        data = json.loads(path.read_text())
        assert data['completed'] and len(data['rows']) == 8
        assert [row['seed'] for row in data['rows']] == list(range(spec['eval_seed'] + rank * 8, spec['eval_seed'] + (rank + 1) * 8))
        assert all(row['native_counts']['controls'] - row['native_counts']['reset_controls'] == 600 for row in data['rows'])
        successes += sum(row['success_once'] for row in data['rows'])
        refs.append(dict(path=str(path), sha256=sha(path)))
        canonical = before_root / 'canonical-eval' / f'rank-{rank:02d}'
        assert sha(canonical / 'initial.pt') == json.loads((canonical / 'complete.json').read_text())['snapshot_sha256']
    assert successes == spec['before_successes'] and successes / 64 == before['metrics']['success_once']
    evidence = dict(task=args.task, source_commit=args.pin, restoration=restoration, before_successes=successes,
        checkpoint_boundary_sha256=sha(boundary_path), completed_step=10,
        actor_checkpoint_files=len(boundary['actor_checkpoint_files']), worker_sidecars=len(boundary['workers']),
        checkpoint_payload_rehashed=not bool(args.inspect), original_archives_unchanged=True,
        checked_unix=time.time())
    if args.inspect:
        write(args.inspect / 'RESULT.json', dict(evidence, CPU_only=True, GPU_hours=0))
        print(json.dumps(evidence), flush=True)
        return
    reference = source.parent / (args.attempt + '-before-reference.json')
    assert not reference.exists()
    write(reference, dict(completed=True, cases=64, checkpoint=str(checkpoint),
        checkpoint_marker_sha256=sha(boundary_path), metrics=before['metrics'], episode_refs=refs,
        canonical_directory=str(before_root / 'canonical-eval'),
        policy='Original SFT before panel; exact completed step10 state; continuation target30 fixed before training'))
    write(remote / 'RESTORED.json', evidence)
    env = dict(os.environ, TASK_LOCAL_OUTPUT=str(source), ROBOCASA_SIM_PPO_RESUME=str(checkpoint),
        ROBOCASA_SIM_PPO_BEFORE_REFERENCE=str(reference), ROBOCASA_SIM_PPO_BEFORE_SHA256=sha(reference))
    done = threading.Event()
    monitor = threading.Thread(target=snapshot, args=(source, remote, done), daemon=False)
    monitor.start()
    result = None
    try:
        result = subprocess.run(['bash', str(HERE / 'teacher.sh'), args.pin, args.task, args.attempt], env=env)
    finally:
        done.set(); monitor.join()
    assert result is not None and (source / 'EXIT.json').exists()
    shutil.copyfile(reference, source / 'resume-before-reference.json')
    write(source / 'CHRONOLOGY.json', dict(previous_archive=receipts['checkpoint']['cold'],
        previous_archive_sha256=spec['checkpoint_archive_sha256'], resumed_step=10, target_step=30,
        learner_hyperparameters_unchanged=True, extension='Fixed 20 further updates; eval20 and30; complete checkpoints every5',
        prior_corpus_history='Retain original pilot iterations0..9 and continuation iterations10..29'))
    receipt = archive(source, remote)
    for name in ('progress.json', 'MANIFEST.json', 'before-evaluation.json', 'after-evaluation.json',
                 'operator-checkpoint-exit.json', 'EXIT.json', 'CHRONOLOGY.json'):
        path = source / name
        if path.exists():
            write(remote / name, json.loads(path.read_text()))
    write(remote / 'COMPLETE.json', dict(archive_verified=receipt['verified'], learner_exit_code=result.returncode,
        started_unix=started, ended_unix=time.time(), complete_checkpoint_boundaries=receipt['complete_checkpoint_boundaries']))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
