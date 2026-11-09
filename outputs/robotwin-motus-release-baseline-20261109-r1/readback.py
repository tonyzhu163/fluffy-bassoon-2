"""Read completed native SFT evidence and calculate binomial uncertainty."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    d = 1 + z*z/total
    c = (p + z*z/(2*total)) / d
    r = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / d
    return [c-r, c+r]


def audit(root):
    load = lambda name: json.loads((root/name).read_bytes())
    result, probe, exit_ = load('RESULT.json'), load('gpu-probe.json'), load('EXIT.json')
    assert result['status'] == 'complete' and exit_['exit_code'] == 0
    assert probe['status'] == 'passed' and len(probe['devices']) == 1
    assert result['optimizer_updates'] == 0 and result['frozen_parameters_unchanged']
    assert result['action_horizon'] == 32 and result['instruction_type'] == 'unseen'
    reference = Path(__file__).parent
    qualified = json.loads((reference/'REFERENCE-CPU-QUALIFICATION.json').read_bytes())
    assert result['manifest'] == qualified['manifest']
    assert load('ASSET-PATHS.json') == json.loads((reference/'CPU-ASSET-PATHS.json').read_bytes())
    assert set(result['settings']) == {'demo_clean', 'demo_randomized'}
    renderers = sorted((root/'renderers').glob('*.json'))
    assert renderers
    for path in renderers:
        renderer = json.loads(path.read_bytes())
        assert renderer['expected_cuda_pci'] == probe['devices'][0]['pci_bus_id']
        assert renderer['actual_render_pci'] == renderer['expected_cuda_pci']
    log = (root/'evaluation.log').read_text(errors='replace')
    assert not any(error in log for error in ('OIDN Error', 'unsupported device type', 'invalid handle'))
    summary = dict(status='passed', source_commit=result['source_commit'],
        result_sha256=sha(root/'RESULT.json'), gpu_probe_sha256=sha(root/'gpu-probe.json'),
        exit_sha256=sha(root/'EXIT.json'), frozen=True, optimizer_updates=0,
        scope='execution smoke only' if result['target_per_setting'] == 2 else 'phone stand100 trials per domain',
        task='place_phone_stand', settings={}, action_files=[], initial_inputs=[],
        allocated_renderer_receipts=len(renderers), denoiser_errors=0)
    for setting, data in result['settings'].items():
        episodes = data['episodes']
        assert len(episodes) == result['target_per_setting'] == data['completed']
        accepted = [e['seed'] for e in episodes]
        rejected = [e['seed'] for e in data['rejected_candidates']]
        assert accepted == sorted(set(accepted))
        assert len(rejected) == len(set(rejected)) and not set(accepted) & set(rejected)
        assert set(accepted + rejected) == set(range(100000, max(accepted)+1))
        for index, episode in enumerate(episodes):
            assert episode['complete'] and episode['expert_feasible']
            assert isinstance(episode['instruction'], str) and episode['instruction']
            assert np.asarray(episode['initial_state']).shape == (14,)
            assert np.isfinite(episode['initial_state']).all()
            assert episode['step_limit'] == 400 and 1 <= episode['delivered_targets'] <= 400
            assert episode['success'] or episode['delivered_targets'] == 400
            assert all(d['requested'] == 32 and 1 <= d['delivered'] <= 32 for d in episode['decisions'])
            assert sum(d['delivered'] for d in episode['decisions']) == episode['delivered_targets']
            folder = root/f'{setting}-{index:03d}-seed-{episode["seed"]}'
            assert json.loads((folder/'episode.json').read_bytes()) == episode
            assert (folder/'initial.jpg').is_file() and (folder/'final.jpg').is_file()
            initial = folder/'initial-input.npz'
            with np.load(initial, allow_pickle=False) as inputs:
                assert set(inputs.files) == {'state', 'prompt', 'cam_high', 'cam_left_wrist', 'cam_right_wrist'}
                assert np.array_equal(inputs['state'], episode['initial_state'])
                assert inputs['prompt'].shape == () and str(inputs['prompt']) == episode['instruction']
                for camera in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
                    value = inputs[camera]
                    assert value.dtype == np.uint8 and value.shape == (3, 240, 320)
            summary['initial_inputs'].append(dict(path=str(initial.relative_to(root)), sha256=sha(initial)))
            actions = sorted(folder.glob('actions-*.npy'))
            assert len(actions) == len(episode['decisions'])
            for j, path in enumerate(actions):
                assert path.name == f'actions-{j:03d}.npy'
                value = np.load(path, allow_pickle=False)
                assert value.shape == (32, 14) and np.isfinite(value).all()
                summary['action_files'].append(dict(path=str(path.relative_to(root)), sha256=sha(path)))
        successes = sum(bool(e['success']) for e in episodes)
        assert successes == data['successes']
        summary['settings'][setting] = dict(successes=successes, episodes=len(episodes),
            success_rate=successes/len(episodes), wilson_95=wilson(successes, len(episodes)),
            rejected_candidates=len(rejected), delivered_targets=sum(e['delivered_targets'] for e in episodes),
            decisions=sum(len(e['decisions']) for e in episodes))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    value = audit(args.root)
    args.output.write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ('action_files', 'initial_inputs')}))
