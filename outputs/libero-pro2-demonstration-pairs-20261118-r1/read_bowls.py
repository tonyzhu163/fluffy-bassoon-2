"""Check released Goal 4/8 demonstrations and measure a public timing baseline on CPU."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median

import h5py
import numpy as np

from build_pairs import CONDITIONS, sha256, write_json


def lag_of_prefix(commands, positions):
    if len(commands) < 80:
        return None
    displacements = np.diff(positions[:80].astype(np.float64), axis=0)[20:79]
    displacements -= displacements.mean(axis=0)
    scores = {}
    for offset in range(21):
        window = commands[20-offset:79-offset, :3].astype(np.float64)
        window -= window.mean(axis=0)
        denom = np.linalg.norm(window) * np.linalg.norm(displacements)
        if denom > 0:
            scores[offset] = float(np.sum(window * displacements) / denom)
    modes = [key for key in scores if scores[key] == max(scores.values())]
    return modes[0] if len(modes) == 1 else None


def verified_file(local, release_name, inventory):
    expected = inventory[release_name]
    if local.stat().st_size != expected['bytes'] or sha256(local) != expected['sha256']:
        raise ValueError(f'Release checksum mismatch: {release_name}')
    return local


def read_panel(directory):
    manifest = [json.loads(line) for line in (directory/'episodes.jsonl').read_text().splitlines()]
    inventory = {r['path']: r for r in map(json.loads, (directory/'files.jsonl').read_text().splitlines())}
    records, collection = [], []
    for task in (4, 8):
        for condition in CONDITIONS:
            stem = f'{task}_{condition}'
            folder = f'data/libero_goal/task_{task}/{condition}'
            file_name = folder + '/episodes.hdf5'
            path = verified_file(directory / (stem + '.hdf5'), file_name, inventory)
            meta = json.loads(verified_file(directory / (stem + '.json'), folder + '/metadata.json', inventory).read_text())
            rows = [r for r in manifest if r['hdf5_path'] == file_name and r['keep_without_declared_fanout']]
            collection.append({'task': task, 'condition': condition, 'hdf5_sha256': inventory[file_name]['sha256'],
                               'stored_after_copy_filter': len(rows),
                               'generator_successes': meta.get('successful_trajectories'),
                               'generator_failures': meta.get('failures'), 'generator_attempts': meta.get('attempts')})
            with h5py.File(path) as archive:
                for row in rows:
                    demo = archive[row['demo']]
                    steps = row['steps']
                    source = json.loads(demo.attrs['source_metadata'])
                    if int(source['init_state_id']) != row['init_state_id'] or int(source['seed']) != row['seed']:
                        raise ValueError(f"Source identity mismatch in {file_name}/{row['demo']}")
                    for field, tail in {'actions': (7,), 'obs/robot0_eef_pos': (3,), 'rewards': (),
                                        'dones': (), 'obs/agentview_image': (128,128,3),
                                        'obs/robot0_eye_in_hand_image': (128,128,3)}.items():
                        if demo[field].shape != (steps, *tail):
                            raise ValueError(f"Shape mismatch in {file_name}/{row['demo']}/{field}")
                    records.append({'task': task, 'condition': condition, 'demo': row['demo'],
                                    'init_state_id': row['init_state_id'], 'steps': steps,
                                    'success': bool(demo.attrs['success']), 'terminal_reward': float(demo['rewards'][-1]),
                                    'terminal_done': bool(demo['dones'][-1]), 'prompt': source.get('instruction'),
                                    'lag': lag_of_prefix(demo['actions'][:80], demo['obs/robot0_eef_pos'][:80])})
    return records, collection


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[row['task'], row['condition']].append(row)
    result = []
    for task, condition in sorted(groups):
        other = 8 if task == 4 else 4
        anchors = {c: median(r['lag'] for r in groups[other,c] if r['lag'] is not None) for c in CONDITIONS}
        tally = Counter()
        for row in groups[task, condition]:
            lag = row['lag']
            if lag is None:
                tally['abstain'] += 1
                continue
            dist = {c: abs(lag - anchor) for c, anchor in anchors.items()}
            if dist['BASE'] == dist['S20']:
                tally['abstain'] += 1
            else:
                selected = min(dist, key=dist.get)
                tally['correct' if selected == condition else 'incorrect'] += 1
        rows = groups[task, condition]
        result.append({'task': task, 'condition': condition, 'reference_task': other, 'reference_medians': anchors,
                       'lag_histogram': dict(Counter(str(r['lag']) for r in rows)), 'decisions': dict(tally),
                       'n': len(rows), 'successful': sum(r['success'] for r in rows),
                       'terminal_rewards': dict(Counter(str(r['terminal_reward']) for r in rows)),
                       'terminal_dones': dict(Counter(str(r['terminal_done']) for r in rows)),
                       'instructions': list(dict.fromkeys(r['prompt'] for r in rows))})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release_files', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    demos, collection = read_panel(args.release_files)
    write_json(args.output, {'groups': summarize(demos), 'collection': collection, 'demonstrations': demos})
