"""Independent CPU cold readback of the frozen camera/horizon diagnostic."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from fluffy_bassoon.baige_gpu import pci
from fluffy_bassoon.meta_rl.rlinf_native_ppo import write
from fluffy_bassoon.rl.robotwin_execution import chunk_execution
from fluffy_bassoon.rl.robotwin_noise import _observation_hash


def main():
    assert sys.platform == 'linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    root = Path(sys.argv[1]); hashes = {}

    def read(name, tensor=False):
        payload = (root/name).read_bytes()
        hashes[name] = hashlib.sha256(payload).hexdigest()
        return (torch.load(io.BytesIO(gzip.decompress(payload)), map_location='cpu',
                           weights_only=False) if tensor else json.loads(payload))

    result, exit_info, gpu, resources = [read(name) for name in
        ('RESULT.json', 'EXIT.json', 'gpu-probe.json', 'resources.json')]
    assert result['status'] == 'completed_paired_frozen_camera_horizon_probe'
    assert exit_info['exit_code'] == 0 and gpu['status'] == 'passed'
    assert result['optimizer_updates'] == 0 and result['frozen_model_unchanged']
    assert result['total_episodes'] == 32 and resources['sampled_reserve_passed']
    assert len(gpu['devices']) == 1
    device = pci(gpu['devices'][0]['pci_bus_id'])
    episodes, previous, prefixes, initial, physical, requested = {}, {}, {}, {}, {}, 0
    panel_cases, scenes, pixels_by_near = {}, {}, {}
    for name, profile in result['profiles'].items():
        assert profile['owned_workers_closed'] and len(profile['episodes']) == 8
        panel_cases[name] = {row['case'] for row in profile['episodes']}
        assert panel_cases[name] == set(result['cases'])
        for receipt in profile['resets']:
            assert receipt['episode_limit'] == profile['horizon']
            assert pci(receipt['actual_render_pci']) == device
            scene_key = receipt['actual_seed']
            identity = (receipt['scene'], receipt['instruction'])
            if scene_key in scenes: assert identity == scenes[scene_key]
            else: scenes[scene_key] = identity
            requests = receipt['camera_near_requests']; assert len(requests) >= 3
            assert all(np.isclose(row['selected_near'], profile['camera_near']) for row in requests)
            assert all(row['exposed_actual_near'] is None or
                np.isclose(row['exposed_actual_near'], profile['camera_near']) for row in requests)

    for trace in result['traces']:
        name = trace['profile']; profile = result['profiles'][name]
        intent, transitions = [read(trace[key], tensor=True) for key in ('intent', 'transitions')]
        metadata = intent['native_batch']
        assert metadata['batch_size'] == 2 and metadata['fixed_native_batch']
        assert len(transitions) == len(metadata['active_lanes'])
        for row in transitions:
            lane, case = row['native_batch']['row'], row['case_id']
            key = (name, case); history = episodes.setdefault(key, [])
            assert row['native_batch'] == dict(metadata, row=lane)
            assert row['chunk_index'] == len(history) == trace['decision']
            command = np.asarray(intent['commands'][lane])
            assert command.shape == (32, 14) and np.isfinite(command).all()
            assert np.array_equal(command, row['requested_actions'])
            assert np.array_equal(command, row['base_actions'])
            current_hash = _observation_hash(row['observation'])
            assert current_hash == _observation_hash(intent['observations'][lane])
            if key in previous:
                assert current_hash == previous[key] and not history[-1]['complete']
            else:
                state = np.asarray(row['observation']['states']).tobytes()
                identity = (state, row['observation']['task_descriptions'])
                if case in initial: assert identity == initial[case]
                else: initial[case] = identity
                obs = row['observation']
                pixels = np.concatenate([obs['main_images'][0], obs['wrist_images'][0, 0],
                                         obs['wrist_images'][0, 1]], axis=1)
                seed = int(case.rsplit('-', 1)[1]); image_name = f'{name}-case-{seed}-initial.png'
                image_bytes = (root/image_name).read_bytes()
                hashes[image_name] = hashlib.sha256(image_bytes).hexdigest()
                assert np.array_equal(np.asarray(Image.open(io.BytesIO(image_bytes))), pixels)
                pixel_key = (profile['camera_near'], case)
                if pixel_key in pixels_by_near: assert np.array_equal(pixels, pixels_by_near[pixel_key])
                else: pixels_by_near[pixel_key] = pixels
            prefix_key = (profile['camera_near'], case, row['chunk_index'])
            signature = (current_hash, hashlib.sha256(command.tobytes()).hexdigest())
            if prefix_key in prefixes: assert signature == prefixes[prefix_key]
            else: prefixes[prefix_key] = signature
            execution = chunk_execution({'fb2_execution': row['execution']},
                requested_targets=32, timestep_seconds=row['execution']['timestep_seconds'])
            assert execution.duration_seconds == row['physics_duration_seconds']
            assert np.array_equal(execution.valid_target_mask, row['measured_valid_target_mask'])
            assert np.isfinite(row['reward'])
            assert np.asarray(row['audit']['adapter_rewards']).sum() == row['reward']
            history.append(dict(complete=row['terminated'] or row['truncated'],
                                success=bool(row['audit']['success'])))
            previous[key] = _observation_hash(row['next_observation'])
            totals = physical.setdefault(name, dict(requested=0, delivered=0, physics_steps=0))
            totals['requested'] += 32; totals['delivered'] += execution.delivered_targets
            totals['physics_steps'] += execution.physics_steps
            requested += 32
    assert len(episodes) == 32 and all(v[-1]['complete'] for v in episodes.values())
    assert requested == result['charged_requested_targets'] <= 9216
    journal = read('action-accounting/rank-00/camera01.json')
    assert all(row['requested'] == row['executed_or_conservative'] for row in journal['calls'])
    assert sum(row['requested'] for row in journal['calls']) == requested
    successes = {}
    for name, profile in result['profiles'].items():
        successes[name] = 0
        for episode in profile['episodes']:
            key = (name, f"camera-probe/case-{episode['case']}")
            history = episodes[key]
            assert episode['complete'] and len(history) == episode['decisions']
            success = any(row['success'] for row in history)
            assert success == episode['success']
            successes[name] += success
        assert successes[name] == profile['successes']
    for path in root.glob('*-initial.png'):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sum(name.endswith('-initial.png') for name in hashes) == 32
    assert not torch.cuda.is_initialized()
    output = root.parent/'motuscamera01-readback01'; output.mkdir()
    write(output/'RESULT.json', dict(status='passed_independent_cold_trace_readback',
        source_commit=os.environ['TASK_SOURCE'], gpu_seconds=0, model_calls=0,
        optimizer_updates=0, episodes=32, batches=len(result['traces']),
        command_intents_and_consecutive_observations_exact=True,
        same_camera_common_horizon_prefixes_exact=True, render_binding_passed=True,
        physical=physical, successes=successes, file_sha256=hashes,
        learning_gain_claim=False, camera_causality_claim=False))


if __name__ == '__main__': main()
