"""Bind complete update10 recovery and original DEV evidence without rerunning SGD."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUN = ROOT / 'outputs/robocasa-eight-hour-babysit-20261104-r1'
SEEDS = dict(PickPlaceCounterToCabinet=(2200000, 620100), TurnOnSinkFaucet=(2800000, 680100))


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def prepare(task):
    assert sys.platform == 'linux'
    parent = RUN / 'source5' / task
    source = parent / 'attempt-01'
    destination = parent / 'attempt-02'
    assert not destination.exists()
    checkpoint = source / task / 'checkpoints/global_step_10'
    marker = checkpoint / 'pilot-boundary.json'
    boundary = read(marker)
    assert boundary['completed'] and boundary['global_step'] == 10
    assert len(boundary['workers']) == 18 and len(boundary['actor_checkpoint_files']) == 3
    checked_bytes = 0
    for item in boundary['workers'] + boundary['actor_checkpoint_files']:
        path = Path(item['path'])
        assert sha(path) == item['sha256']
        checked_bytes += path.stat().st_size
    for rank in range(2):
        update = read(source / 'updates' / f'iteration-010-rank-{rank:02d}.json')
        assert update['completed'] and update['optimizer_steps_after'] == 240 and update['frozen_vlm_unchanged']
    before = read(source / 'before-evaluation.json')
    assert before['completed'] and .05 <= before['metrics']['success_once'] <= .70
    train_seed, dev_seed = SEEDS[task]
    refs = []
    successes = 0
    for rank in range(8):
        path = source / 'environment' / f'rank-{rank:02d}/before-i000-round00-episodes.json'
        data = read(path)
        assert data['completed'] and [row['seed'] for row in data['rows']] == list(range(dev_seed+rank*8, dev_seed+rank*8+8))
        assert all(row['native_counts']['controls']-row['native_counts']['reset_controls'] == 600 for row in data['rows'])
        successes += sum(row['success_once'] for row in data['rows'])
        refs.append(dict(path=str(path), sha256=sha(path)))
        canonical = source / 'canonical-eval' / f'rank-{rank:02d}'
        assert sha(canonical / 'initial.pt') == read(canonical / 'complete.json')['snapshot_sha256']
    assert successes/64 == before['metrics']['success_once']
    seen = set()
    for iteration in range(10):
        for rank in range(8):
            path = source / 'environment' / f'rank-{rank:02d}' / f'training-i{iteration:03d}-round00-episodes.json'
            data = read(path)
            assert data['completed'] and data['iteration'] == iteration
            seeds = list(range(train_seed+iteration*128+rank*16, train_seed+iteration*128+rank*16+16))
            assert [row['seed'] for row in data['rows']] == seeds
            for row in data['rows']:
                assert row['seed'] not in seen and row['native_counts']['controls']-row['native_counts']['reset_controls'] == 600
                seen.add(row['seed'])
    assert seen == set(range(train_seed, train_seed+1280))
    reference = parent / 'RESUME-REFERENCE-01.json'
    write_new(reference, dict(completed=True, cases=64, checkpoint=str(checkpoint),
        checkpoint_marker_sha256=sha(marker), metrics=before['metrics'], episode_refs=refs,
        canonical_directory=str(source / 'canonical-eval'), pending_evaluation_step=10,
        original_before_file=str(source / 'before-evaluation.json'), original_before_sha256=sha(source / 'before-evaluation.json'),
        metrics_policy='Original SFT before any source SGD; checkpoint field identifies the separate update10 recovery',
        source_config_sha256=sha(source / 'resolved-config.json')))
    write_new(parent / 'ENDPOINT-SOURCE.json', dict(task=task, path=str(destination), step=40,
        fixed_endpoint=True, DEV_selection=False,
        chronological_segments=[dict(path=str(source), training_iterations=[1, 10], producing_policy_iterations=[0, 9]),
                                dict(path=str(destination), training_iterations=[11, 40], producing_policy_iterations=[10, 39])],
        original_before_reference=str(reference), original_before_reference_sha256=sha(reference)))
    receipt = dict(prepared=True, task=task, checkpoint_step=10, checkpoint_marker_sha256=sha(marker),
        checked_checkpoint_bytes=checked_bytes, worker_sidecars=18, actor_optimizer_steps=240,
        complete_TRAIN_episodes=1280, complete_TRAIN_controls=768000, target_updates=40,
        remaining_TRAIN_updates=30, target_actor_optimizer_steps=960,
        reference=str(reference), reference_sha256=sha(reference),
        full_source_input=str(source), continuation_output=str(destination),
        resumed_allocation_ceiling_GPU_hours=24, resumed_storage_reservation_GiB=110,
        fresh_eval_processes=True, GPU_hours=0,
        launch_gate='Passed task-specific original64-case reset qualification, admitted storage, per-task30GPUh and total70GPUh')
    write_new(parent / 'RESUME-PREPARED-01.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=tuple(SEEDS), required=True)
    prepare(parser.parse_args().task)
