"""Reopen saved episodes and verify policy actions and frozen panel membership."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REFERENCE = Path(__file__).resolve().parents[1] / 'robotwin-motus-release-baseline-20261109-r1'
sys.path.insert(0, str(REFERENCE))
from evaluate import sha
# The common Wilson interval is exposed by the Motus evidence reader.
import importlib.util
spec = importlib.util.spec_from_file_location('native_readback', REFERENCE/'readback.py')
native_readback = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native_readback)


def audit(root, partial=False):
    def load(name):
        return json.loads((root/name).read_bytes())
    result = load('RESULT.json')
    probe, ready = load('gpu-probe.json'), load('READY.json')
    assert probe['status'] == 'passed' and len(probe['devices']) == 1
    assert ready['strict_weight_load'] and ready['optimizer_updates'] == result['optimizer_updates'] == 0
    assert ready['chunk_size'] == 50 and ready['flow_steps'] == 10 and ready['absolute_joint_actions']
    assert result['model_revision'] == 'e49e2ab6c11f07511573b67261bd129e88d0a416'
    cases = json.loads((REFERENCE/'PHONE100-CASES.json').read_text())['settings']
    renderers = sorted((root/'renderers').glob('*.json'))
    assert renderers
    for path in renderers:
        renderer = json.loads(path.read_bytes())
        assert renderer['actual_render_pci'] == renderer['expected_cuda_pci'] == probe['devices'][0]['pci_bus_id']
    if not partial:
        assert result['status'] == 'complete' and load('EXIT.json')['exit_code'] == 0
        assert set(result['settings']) == set(cases)
        log = (root/'evaluation.log').read_text(errors='replace')
        assert all(s not in log for s in ('OIDN Error', 'unsupported device type', 'invalid handle'))
    summary = {}
    for setting, episodes in result['settings'].items():
        if not partial:
            assert len(episodes) == result['episodes_per_domain']
        actions_checked = targets = wins = 0
        for case, episode in zip(cases[setting], episodes):
            assert all(episode[k] == case[k] for k in ('index', 'seed', 'instruction'))
            directory = root/f"{setting}-{case['index']:03d}-seed-{case['seed']}"
            assert json.loads((directory/'episode.json').read_bytes()) == episode
            with np.load(directory/'initial.npz', allow_pickle=False) as init:
                assert init['instruction'].item() == case['instruction']
                assert init['images'].shape == (3, 240, 320, 3) and init['images'].dtype == np.uint8
                assert init['state'].shape == (14,) and np.isfinite(init['state']).all()
            assert len(episode['actors']) == 2
            assert episode['success'] or episode['targets'] == 400
            assert 0 < episode['targets'] <= 400
            chunks = episode['chunks']
            assert sum(c['delivered'] for c in chunks) == episode['targets']
            for j, chunk in enumerate(chunks):
                arr = np.load(directory/f'actions-{j:02d}.npy', allow_pickle=False)
                assert arr.shape == (50, 14) and np.isfinite(arr).all()
                assert hashlib.sha256(arr.tobytes()).hexdigest() == chunk['actions_sha256']
                assert 1 <= chunk['delivered'] <= 50
            wins += int(episode['success'])
            targets += episode['targets']
            actions_checked += len(chunks)
        if episodes:
            summary[setting] = dict(successes=wins, episodes=len(episodes), targets=targets,
                action_chunks=actions_checked, wilson_95=native_readback.wilson(wins, len(episodes)))
    assert summary
    return dict(partial=partial, source_commit=result['source_commit'], model_revision=result['model_revision'],
                allocated_renderers=len(renderers), result_sha256=sha(root/'RESULT.json'), settings=summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.partial), indent=2))
