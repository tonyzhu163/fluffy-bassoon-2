"""Episode records and disjoint reference ranges for 5 -> 50 trials per task."""
import hashlib
import json
from pathlib import Path
import random
import re


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha256(path):
    with Path(path).open('rb') as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save_rng(directory):
    import numpy as np
    import torch
    path = Path(directory) / 'rng.pt'
    temporary = path.with_suffix('.tmp')
    torch.save({'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}, temporary)
    temporary.replace(path)
    return {'path': str(path.absolute()), 'sha256': sha256(path)}


def restore_rng(record):
    import numpy as np
    import torch
    if sha256(record['path']) != record['sha256']:
        raise ValueError('Evaluation RNG checkpoint checksum mismatch')
    state = torch.load(record['path'], map_location='cpu', weights_only=False)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Evaluation RNG device count differs')
        torch.cuda.set_rng_state_all(state['cuda'])


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    if set(plan) != {str(i) for i in range(10)}:
        raise ValueError('A suite plan must cover all ten tasks')
    for row in plan.values():
        start, stop = row['start'], row['stop']
        if type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= 50:
            raise ValueError('Reference ranges must be within [0,50)')
    return plan


def records_from(directory, backend):
    """Read only completed episodes; a cancelled in-flight episode is not evidence."""
    directory = Path(directory)
    records = []
    if backend == 'lerobot':
        for path in directory.glob('diagnostics/*/episode_*/episode.json'):
            row = json.loads(path.read_text())
            if row['status'] == 'completed':
                if row['init_state_index'] != row['episode_index']:
                    raise ValueError('Reference-state index does not match episode identity')
                records.append(dict(row, trial=row['episode_index'], steps=row['executed_steps'], source=str(path)))
    elif (directory / 'episodes.jsonl').exists():
        path = directory / 'episodes.jsonl'
        for line in path.read_text().splitlines():
            records.append(dict(json.loads(line), source=str(path)))
    elif backend == 'muvla_native':
        logs = list(directory.glob('evaluation/logs/*.txt'))
        if len(logs) > 1:
            raise ValueError('Ambiguous native log identity')
        for path in logs:
            task, trial = None, None
            for line in path.read_text().splitlines():
                match = re.search(r'=== Task (\d+):', line)
                if match:
                    task = int(match[1])
                match = re.match(r'Episode (\d+)/\d+ —', line)
                if match:
                    trial = int(match[1]) - 1
                match = re.search(r'→ Success: (True|False) \| steps: (\d+)', line)
                if match:
                    if task is None or trial is None:
                        raise ValueError('Native episode lacks task/reference identity')
                    records.append({'task_id': task, 'trial': trial, 'success': match[1] == 'True',
                                    'steps': int(match[2]), 'source': str(path)})
                    trial = None
    return records


def merge_records(directories, backend):
    merged = {}
    for directory in dict.fromkeys(map(str, directories)):
        for row in records_from(directory, backend):
            key = (row['task_id'], row['trial'])
            if not all(type(x) is int and 0 <= x < limit for x, limit in zip(key, (10, 50))):
                raise ValueError('Invalid task/reference identity')
            if key in merged:
                raise ValueError(f'Duplicate evaluation episode {key}')
            merged[key] = row
    return merged


def plan_ranges(records, target):
    """Resume the contiguous completed prefix; never fill arbitrary score-selected holes."""
    if target not in (5, 50):
        raise ValueError('Campaign stages are five or fifty reference trials per task')
    plan = {}
    for task in range(10):
        completed = sorted(trial for (tid, trial) in records if tid == task)
        if completed != list(range(len(completed))):
            raise ValueError(f'Task {task} has a gap in completed reference episodes')
        start = min(len(completed), target)
        previous = records.get((task, start - 1), {})
        plan[str(task)] = {'start': start, 'stop': target,
                          'rng_from': previous.get('rng_after'),
                          'rng_continuation': 'saved' if previous.get('rng_after') else
                                              ('legacy_rng_unavailable' if start else 'initial')}
    return plan


def summarize(records, target):
    selected = [row for (task, trial), row in sorted(records.items()) if trial < target]
    per_task = {str(task): {'episodes': sum(r['task_id'] == task for r in selected),
                          'successes': sum(r['success'] for r in selected if r['task_id'] == task)}
                for task in range(10)}
    return {'target_per_task': target, 'target': 10 * target,
            'episodes': len(selected), 'successes': sum(r['success'] for r in selected),
            'complete': all(row['episodes'] == target for row in per_task.values()), 'per_task': per_task}
