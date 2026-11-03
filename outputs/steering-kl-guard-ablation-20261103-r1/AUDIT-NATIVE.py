#!/usr/bin/env python3
"""Read-only, allocated-Linux audit of the completed paired diagnostic."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load(path):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    return torch.load(path, map_location='cpu', weights_only=False)


def checked(ref):
    assert sha(ref['path']) == ref['sha256'], ref['path']
    return load(ref['path'])


def episode(item):
    arm, name, e = item
    assert e['completed'] and e['horizon'] == 600 and e['physical_chunks'] == 12
    initial, final = checked(e['initial']), checked(e['final'])
    checkpoint = checked(e['checkpoint'])
    assert checkpoint['episode']['case'] == e['case'] == initial['case']
    ev = initial['evidence']
    assert ev['pairing']['matched'] and ev['pairing']['physical_state_exact'] and ev['pairing']['public_state_exact']
    marker = json.loads(Path(ev['canonical']).read_text())
    canonical_path = Path(ev['canonical']).parent / 'initial.pt'
    assert sha(canonical_path) == marker['snapshot_sha256']
    pairing = compare_native_pair(load(canonical_path), ev['snapshot'])
    assert len(e['chunks']) == 12
    noises, valid, rewards = [], [], []
    for index, ref in enumerate(e['chunks']):
        c = checked(ref)
        assert c['status'] == 'completed' and c['case'] == e['case'] and c['chunk_index'] == index
        assert c['source_commit'] == profile['source_commit'] and c['runtime_commit'] == profile['runtime_commit']
        assert c['deterministic'] and c['zero_steering'] == (name == 'initial')
        assert c['executed_controls'] == 50 and bool(c['physical_done']) == (index == 11)
        dispatched, executed = np.asarray(c['dispatched_commands']), np.asarray(c['executed_commands'])
        assert dispatched.shape == executed.shape == (50, 12)
        assert np.isfinite(executed).all() and np.array_equal(dispatched, executed)
        control, mean = np.asarray(c['control']), np.asarray(c['steering_mean'])
        assert np.isfinite(control).all() and np.isfinite(mean).all()
        assert np.count_nonzero(control) == 0 if name == 'initial' else np.array_equal(control, mean)
        assert c['actor_noise_seed'] is None and c['actor_standard_normal'] is None
        seed = draw_seed(e['case'], index, 'native_gaussian')
        noise = torch.randn((50, 32), generator=torch.Generator().manual_seed(seed), dtype=torch.float32).numpy()
        assert c['native_noise_seed'] == seed and np.array_equal(c['native_noise'], noise)
        noises.append((seed, hashlib.sha256(noise.tobytes()).hexdigest()))
        valid.append(c['valid_ppo']); rewards.append(c['reward'])
    assert valid == [True] * e['valid_ppo_chunks'] + [False] * (12 - e['valid_ppo_chunks'])
    assert sum(rewards) == int(e['success_once'])
    video = json.loads(Path(e['video_manifest']).read_text())
    monitor = json.loads(Path(e['monitoring_manifest']).read_text())
    assert video['completed'] and video['horizon'] == 600 and video['raw_rgb_only'] and not video['info_on_video']
    clock = video['frame_clock']
    assert clock['frame_count'] == 601 and clock['physical_steps'] == 600 and not clock['resampled']
    assert clock['initial_frame_included'] and clock['native_fps'] == clock['encoded_fps']
    assert video['camera_order'] == list(REWARD_CAMERAS) and len(video['videos']) == 3
    assert monitor['case'] == e['case'] and monitor['first_success_step'] == e['first_success_step']
    assert monitor['simulator_success_once'] == e['success_once'] and monitor['simulator_final_success'] == e['success_at_end']
    counts = monitor['native_counts']
    assert counts['controls'] - counts['reset_controls'] == 600 and counts == final['native_counts']
    assert (min(monitor['success_steps']) if monitor['success_steps'] else None) == e['first_success_step']
    videos = []
    for camera, ref in zip(REWARD_CAMERAS, video['videos']):
        assert ref['camera'] == camera and sha(ref['path']) == ref['sha256']
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
            'stream=nb_frames,width,height,avg_frame_rate,duration', '-of', 'json', ref['path']],
            text=True, capture_output=True, check=True, timeout=30)
        stream = json.loads(probe.stdout)['streams'][0]
        assert int(stream['nb_frames']) == 601 and stream['width'] > 0 and stream['height'] > 0
        assert abs(float(Fraction(stream['avg_frame_rate'])) - clock['native_fps']) < 1e-8
        assert abs(float(stream['duration']) - 601 / clock['native_fps']) < 1e-5
        videos.append((ref['path'], ref['sha256'], stream))
    return dict(arm=arm, candidate=name, case=e['case'], success_once=e['success_once'],
        success_at_end=e['success_at_end'], noises=noises, pairing=pairing, videos=videos,
        reset_controls=counts['reset_controls'], references_checked=15)


def main():
    global torch, np, profile, compare_native_pair, draw_seed, REWARD_CAMERAS
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    assert sys.platform == 'linux' and socket.gethostname() == 'aibox-rffe453e3884-5f96c78b88-rtvhb'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    import numpy as np
    torch.set_num_threads(1)
    begin = time.time(); root = args.run
    profile = json.loads((root / 'PROFILE.json').read_text())
    sys.path.insert(0, str(Path(profile['code']) / 'src'))
    from fluffy_bassoon.meta_rl.robocasa_pairing import compare_native_pair
    from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import draw_seed
    from fluffy_bassoon.meta_rl.robocasa_video import REWARD_CAMERAS
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import tensor_hash, checked_checkout
    from scipy.stats import binomtest
    checked_checkout(Path(profile['code']), profile['source_commit'])
    checked_checkout(Path(profile['runtime_path']), profile['runtime_commit'])
    cpu = json.loads(Path(profile['CPU_result']).read_text())
    assert sha(profile['CPU_result']) == profile['CPU_result_sha256']
    source_hashes = {path: sha(path) for path in cpu['source_hashes_before_and_after']}
    assert source_hashes == cpu['source_hashes_before_and_after']
    for path, key in ((Path(profile['model_directory']) / 'model.safetensors', 'model_sha256'),
                      (Path(profile['model_directory']) / 'config.json', 'model_config_sha256'),
                      (Path(profile['norm_path']), 'norm_sha256')):
        assert sha(path) == profile[key]
    result = json.loads((root / 'RESULT.json').read_text()); assert result['completed']
    clock = json.loads((root / 'clock.json').read_text()); released = json.loads((root / 'RELEASED.json').read_text())
    assert clock['exit_code'] == 0 and released['all_children_released'] and released['child_exit_codes'] == [0]*4
    items, qualifications = [], []
    for rank in range(4):
        d = root / 'workers' / f'rank-{rank:02d}'; worker = json.loads((d/'RESULT.json').read_text())
        assert worker['completed'] and not (d/'FAILURE.json').exists()
        arm = worker['arm']; q = json.loads((d/'qualification.json').read_text())
        assert q['completed'] and q['zero_native_exact'] and q['prefix_reused']
        assert q['extra_physical_controls'] == 0 and q['action_delta_rms'] > 0
        if arm == 'kv_all_layers':
            assert q['original_cache_unchanged'] and q['consumed_cache_layers'] == 18 and len(q['cache_targets']) == 36
        qualifications.append(q)
        signature = worker['native_signature']; assert signature['all_parameters_frozen']
        assert all(signature[k] == profile[k] for k in ('model_sha256','model_config_sha256','norm_sha256'))
        for sequence, name in enumerate(('initial','guarded1','unguarded1'),1):
            candidate = cpu['arms'][arm]['initial_artifact'] if name == 'initial' else cpu['arms'][arm]['variants'][name]['artifact']
            original = checked(candidate)
            request = load(d/f'request-{sequence:06d}.pt')
            assert request['candidate_reference'] == candidate and tensor_hash(request['policy']) == tensor_hash(original['policy'])
            w = worker['policies'][name]; bank = checked(w['result_reference'])
            assert w['candidate_weights_unchanged'] and w['actual_policy_controls'] == 75000
            assert len(bank['episodes']) == 125 and bank['candidate_reference'] == candidate
            assert bank['native_signature'] == signature and bank['execution']['maximum_native_batch'] == 48
            assert bank['execution']['initial_lane_barrier']
            items.extend((arm,name,e) for e in bank['episodes'])
    assert len(items) == 1500
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=[]
        for row in pool.map(episode, items):
            rows.append(row)
            if len(rows)%100 == 0: print(json.dumps(dict(audited_episodes=len(rows),elapsed=time.time()-begin)),flush=True)
    banks={}
    for row in rows:
        case=row['case']; key=(case['task'],case['native_seed'])
        assert case['bank']=='kl_guard_diagnostic' and case['split']=='target'
        bank=banks.setdefault((row['arm'],row['candidate']),{});assert key not in bank;bank[key]=row
    comparisons={}
    for arm in ('velocity','kv_all_layers'):
        initial=banks[(arm,'initial')]
        assert len(initial)==250 and len({k[0] for k in initial})==5
        for name in ('guarded1','unguarded1'):
            bank=banks[(arm,name)];assert set(bank)==set(initial)
            for k in initial: assert bank[k]['noises']==initial[k]['noises']
        for name in ('initial','guarded1','unguarded1'):
            bank=banks[(arm,name)]
            for metric in ('success_once','success_at_end'):
                rate=sum(bool(x[metric]) for x in bank.values())/250
                assert abs(result['arms'][arm]['summaries'][name]['macro_ID_'+metric]-rate)<1e-12
        comparisons[arm]={}
        for metric in ('success_once','success_at_end'):
            left,right=banks[(arm,'guarded1')],banks[(arm,'unguarded1')]
            delta=[int(right[k][metric])-int(left[k][metric]) for k in left]
            wins,losses=delta.count(1),delta.count(-1)
            value=result['arms'][arm]['paired_comparisons']['guarded1_to_unguarded1'][metric]
            assert (wins,losses)==(value['wins'],value['losses'])
            assert abs(value['exact_p']-binomtest(wins,wins+losses,.5).pvalue)<1e-12
            comparisons[arm][metric]=value
    escrow=json.loads((root/'accounting/escrow.json').read_text())
    assert not escrow['pending'] and escrow['charged']==dict(train=0,eval=900000)
    calls=[json.loads(p.read_text()) for p in (root/'accounting').glob('rank-*/*.json')]
    assert calls and all(c['resolved'] for c in calls)
    phases={}
    for c in calls: phases[c['phase']]=phases.get(c['phase'],0)+c['executed_or_conservative']
    assert phases['policy_controls']==900000 and phases['native_constructor_reset']==sum(r['reset_controls'] for r in rows)
    # Decode all frames of one camera from each complete bank, in addition to every-file hash/container checks.
    samples=[rows[i]['videos'][0][0] for i in range(0,len(rows),125)]
    for path in samples:
        subprocess.run(['ffmpeg','-v','error','-threads','1','-i',path,'-f','null','-'],check=True,capture_output=True,timeout=90)
    audit=dict(completed=True,hostname=socket.gethostname(),device='CPU; CUDA hidden',elapsed_seconds=time.time()-begin,
        episodes=1500,physical_chunks=18000,policy_controls=900000,reset_controls=sum(r['reset_controls'] for r in rows),
        episode_artifact_references_checked=sum(r['references_checked'] for r in rows),videos=4500,
        video_frames_per_stream=601,all_video_hashes_and_container_frame_metadata_checked=True,
        complete_sample_video_decodes=len(samples),video_index_sha256=digest([r['videos'] for r in rows]),
        case_noise_pairing_verified=True,physical_and_public_resets_exact=True,
        RGB_pairing_diagnostic_only=True,RGB_nonexact_resets=sum(not r['pairing']['images_exact'] for r in rows),
        frozen_model_hashes_verified=True,candidate_hashes_verified=True,source_files_unchanged=True,
        source_hashes=source_hashes,qualifications=qualifications,accounting=escrow,physical_call_phases=phases,
        primary_comparisons=comparisons,process_clock=clock,children_released=released,
        scheduler_release_and_allocation_overhead='Separate console audit required',finished_unix=time.time())
    args.output.write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps({k:audit[k] for k in ('completed','episodes','videos','elapsed_seconds')}),flush=True)


if __name__=='__main__':main()
