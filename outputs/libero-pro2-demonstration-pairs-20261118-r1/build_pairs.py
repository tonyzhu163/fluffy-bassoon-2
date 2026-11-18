"""Index one-support demonstration pairs for the released BASE/S20 conditions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

CONDITIONS = ('BASE', 'S20')
DATA_REVISION = '6c4c5524676815a88880a7025badd17fb724f7f4'
SPLITS = Path(__file__).resolve().parents[1] / 'libero-meta-learning-20261117-r1' / 'SPLITS.json'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2) + '\n')


def pointer(row):
    return {'file': row['hdf5_path'], 'group': row['demo']}


def build(manifest, output):
    splits = json.loads(SPLITS.read_text())
    owner = {}
    for split in ('train', 'development', 'test'):
        for suite, ids in splits[split].items():
            for task_id in ids:
                key = (suite, task_id)
                if key in owner:
                    raise ValueError(f'Task split overlap at {key}')
                owner[key] = split
    groups = defaultdict(list)
    stored, kept = Counter(), Counter()
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        task = (row['suite'], int(row['task'].removeprefix('task_')))
        split = owner[task]
        if split == 'test':
            raise ValueError(f'Official test task in training manifest: {task}')
        stored[split] += 1
        if row['keep_without_declared_fanout']:
            kept[split] += 1
            if row['case'] in CONDITIONS:
                groups[task, row['case']].append(row)

    tasks, pairs = [], []
    for task, split in sorted(owner.items()):
        if split == 'test':
            continue
        suite, task_id = task
        by_id = {}
        for cond in CONDITIONS:
            alternatives = defaultdict(list)
            for row in groups[task, cond]:
                reset = row['init_state_id']
                if type(reset) is not int or reset < 0:
                    raise ValueError(f'Invalid reset ID in {pointer(row)}')
                alternatives[reset].append(row)
            by_id[cond] = alternatives
        common = sorted(set(by_id['BASE']) & set(by_id['S20']))
        summary = {'split': split, 'suite': suite, 'task_id': task_id,
                   'eight_task_subset': task_id in splits['development_scaling']['task_subsets']['8'][suite],
                   'kept': {c: len(groups[task, c]) for c in CONDITIONS},
                   'reset_ids': {c: len(by_id[c]) for c in CONDITIONS},
                   'common_reset_ids': common, 'paired_blocks': 0}
        tasks.append(summary)
        if len(common) < 2:
            continue
        # One block per query reset. A tape can be reused across blocks.
        # Numeric demo order is stable and uses no observations or outcomes.
        for i, query_id in enumerate(common):
            support_id = common[i - 1]
            branches = {}
            for cond in CONDITIONS:
                entries = by_id[cond]
                def first_demo(reset):
                    return min(entries[reset], key=lambda r: int(r['demo'].split('_')[-1]))
                support, query = first_demo(support_id), first_demo(query_id)
                if support['fingerprint'] and support['fingerprint'] == query['fingerprint']:
                    raise ValueError(f'Same source fingerprint across resets: {task} {cond}')
                branches[cond] = {'support': pointer(support), 'query': pointer(query)}
            pairs.append({'split': split, 'suite': suite, 'task_id': task_id,
                          'support_reset_id': support_id, 'query_reset_id': query_id,
                          'conditions': branches})
            summary['paired_blocks'] += 1
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'coverage.json', {'dataset_revision': DATA_REVISION,
               'episode_manifest_sha256': sha256(manifest), 'split_sha256': sha256(SPLITS),
               'stored': dict(stored), 'kept': dict(kept), 'tasks': tasks})
    for split in ('train', 'development'):
        with (output / f'{split}.jsonl').open('w') as stream:
            for block in pairs:
                if block['split'] == split:
                    stream.write(json.dumps(block, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    build(args.manifest, args.output)
