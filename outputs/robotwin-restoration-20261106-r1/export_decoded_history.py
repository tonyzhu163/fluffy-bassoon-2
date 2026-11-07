"""Export complete physical source histories after frozen reconstruction readback.

The sole retained policy version supports mixed-quality offline objectives and
static imitation. It is not admitted as an AD learning-history or DPT corpus.
"""
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from fluffy_bassoon.meta_rl.chunk_history import ChunkHistoryLayout, trial_tokens
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, tensor_hash, write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor, write_tensor


ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
FIELDS = ('features', 'bases', 'issued', 'delivered_masks', 'rewards',
          'terminated', 'truncated', 'physics_seconds', 'episode_starts')


def export():
    assert os.uname().sysname == 'Linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    code = ROOT.parents[1]
    assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == os.environ['TASK_SOURCE']
    assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain', '--untracked-files=no'])
    output = BASE/'decodedhistory01'
    output.mkdir()
    started = time.monotonic()
    report = dict(status='checking', source_commit=os.environ['TASK_SOURCE'],
        gpu_seconds=0, model_calls=0, simulator_actions=0, optimizer_updates=0,
        source_corpus_admitted=False, complete_ad_learning_histories=False,
        available_teacher_policy_versions=[0], dpt_query_labels=False)
    try:
        plan = json.loads((ROOT/'RECONSTRUCTION01-CONTRACT.json').read_bytes())
        source, reconstruction = BASE/'cpureconstruction04', BASE/'reconstruction01-signals'
        assert sha(source/'RESULT.json') == plan['input_report_sha256']
        audit_path = BASE/'reconstruction01-readback01/RESULT.json'
        audited = json.loads(audit_path.read_bytes())
        assert audited['status'] == 'passed' and audited['canonical_bijection_rows'] == 6144
        assert audited['input_report_sha256'] == sha(source/'RESULT.json')
        assert audited['result_sha256'] == sha(reconstruction/'RESULT.json')
        bound = json.loads((source/'RESULT.json').read_bytes())
        rebuilt = json.loads((reconstruction/'RESULT.json').read_bytes())
        split = json.loads((ROOT/'STUDENT-CASE-SPLIT.json').read_bytes())
        split_sha = sha(ROOT/'STUDENT-CASE-SPLIT.json')
        assert split_sha == plan['student_case_split_sha256']
        source_config = json.loads((BASE/'motusppo04/state/resolved-native-config.json').read_bytes())
        boundary = source_config['env']['train']
        assert boundary['max_episode_steps'] == boundary['max_steps_per_rollout_epoch'] == 192
        assert not boundary['auto_reset'] and not boundary['ignore_terminations']
        layout = ChunkHistoryLayout(2062, 32, 14)
        episodes, previous_next = {}, {}
        noops = physical = 0
        seen_coordinates = set()
        for ordinal, (binding, result) in enumerate(zip(bound['packets'], rebuilt['chunks'], strict=True)):
            assert binding['ordinal'] == result['ordinal'] == ordinal
            path = source/binding['packet']
            assert path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256']
            packet = read_tensor(path)
            path = reconstruction/result['output']
            assert path.stat().st_size == result['bytes'] and sha(path) == result['sha256']
            value = read_tensor(path)
            assert value['input_packet_sha256'] == binding['sha256']
            assert value['split_sha256'] == packet['split_sha256'] == split_sha
            rank, round_index, chunk = ordinal//24, ordinal//6 % 4, ordinal % 6
            assert (packet['rank'], packet['round_index'], packet['chunk']) == (rank, round_index, chunk)
            assert packet['policy_iteration'] == 0 and not packet['evaluation_data']
            assert tensor_hash(packet['current_observation']) == packet['current_observation_sha256']
            stream = rank, round_index
            if chunk:
                assert previous_next[stream] == packet['current_observation_sha256']
            previous_next[stream] = packet['next_observation_sha256']
            assert torch.equal(value['teacher_actions'], packet['teacher_actions'])
            assert torch.equal(value['decoded_residual'], value['teacher_actions']-value['frozen_base'])
            for lane in range(64):
                coordinate = tuple(packet['canonical']['coordinates'][lane].tolist())
                assert coordinate not in seen_coordinates
                seen_coordinates.add(coordinate)
                key = rank, round_index, lane
                group, replica = divmod(lane, 8)
                partition = split['partition_by_local_group'][str(group)]
                assert packet['student_partition_per_row'][lane] == partition
                scene, serial = int(packet['reset_state_ids'][lane]), int(packet['episode_serials'][lane])
                if chunk == 0:
                    assert key not in episodes
                    episodes[key] = dict(rank=rank, round_index=round_index, lane=lane,
                        local_group=group, replica=replica, partition=partition,
                        scene_seed=scene, episode_serial=serial, ended=False,
                        rows={field:[] for field in FIELDS}, source_chunks=[])
                episode = episodes[key]
                assert (episode['scene_seed'], episode['episode_serial'], episode['partition']) == (scene, serial, partition)
                execution = packet['physical_execution']
                steps, seconds = execution['physics_steps'][lane], execution['duration_seconds'][lane]
                feedback = packet['native_feedback'][lane]
                reward = float(np.asarray(feedback['reward']).item())
                terminated = bool(np.asarray(feedback['terminated']).item())
                raw_truncated = bool(np.asarray(feedback['truncated']).item())
                truncated = raw_truncated or chunk == 5
                assert np.isfinite(reward)
                assert bool(packet['native_terminated'][lane].any()) == terminated
                assert bool(packet['native_truncated'][lane].any()) == truncated
                assert bool(packet['physical_decision_mask'][lane]) == (steps > 0)
                if steps == 0:
                    assert episode['ended'] and bool(packet['post_terminal_noop_mask'][lane])
                    assert not execution['valid_target_mask'][lane].any() and seconds == 0
                    noops += 1
                    continue
                assert steps > 0 and seconds > 0 and not episode['ended']
                assert seconds == steps*execution['timestep_seconds'][lane]
                mask = execution['valid_target_mask'][lane].numpy().copy()
                assert mask.dtype == np.bool_ and mask.any() and not np.any(mask[1:] & ~mask[:-1])
                rows = episode['rows']
                values = dict(features=value['public_features'][lane].numpy().copy(),
                    bases=value['frozen_base'][lane].numpy().copy(),
                    issued=value['teacher_actions'][lane].numpy().copy(), delivered_masks=mask,
                    rewards=reward, terminated=terminated, truncated=truncated,
                    physics_seconds=seconds, episode_starts=not rows['features'])
                for field, item in values.items():
                    rows[field].append(item)
                episode['source_chunks'].append(dict(ordinal=ordinal, lane=lane,
                    input_packet_sha256=binding['sha256'], reconstruction_sha256=result['sha256']))
                episode['ended'] = terminated or truncated
                physical += 1
            del packet, value
            print('EXPORTED_SOURCE_PACKET', ordinal, flush=True)
        assert len(episodes) == 1024 and len(seen_coordinates) == 6144 and physical+noops == 6144
        assert noops == bound['measured_zero_duration_calls'] == bound['post_terminal_noops']
        assert bound['unexpected_zero_duration_calls'] == 0
        assert all(row['ended'] and row['rows']['features'] for row in episodes.values())
        trials, metadata = {key:[] for key in ('TRAIN', 'DEV', 'TEST')}, []
        for rank in range(4):
            for group in range(8):
                partition = split['partition_by_local_group'][str(group)]
                selected = [episodes[rank, round_index, group*8+replica]
                    for round_index in range(4) for replica in range(8)]
                trial = {}
                for field in FIELDS:
                    dtype = np.bool_ if field in ('delivered_masks', 'terminated', 'truncated', 'episode_starts') else np.float32
                    trial[field] = np.asarray([item for episode in selected for item in episode['rows'][field]], dtype=dtype)
                assert trial['episode_starts'].sum() == 32
                trial_tokens(layout, **trial)
                trials[partition].append(trial)
                metadata.append(dict(trial_id=f'{split["task"]}/rank-{rank:02d}/group-{group:02d}',
                    partition=partition, policy_versions=[0], episodes=32, physical_decisions=len(trial['rewards']),
                    episode_order='round then replica; simultaneous replicas are an offline ordering',
                    source_episodes=[{key:episode[key] for key in
                        ('rank', 'round_index', 'lane', 'scene_seed', 'episode_serial', 'source_chunks')} for episode in selected]))
        assert [len(trials[key]) for key in ('TRAIN', 'DEV', 'TEST')] == [24, 4, 4]
        # These candidate statistics use TRAIN only. DEV/TEST coverage is
        # reported afterwards and never changes a radius or clips a label.
        train_features = torch.from_numpy(np.concatenate([row['features'] for row in trials['TRAIN']]))
        train_residuals = np.concatenate([row['issued']-row['bases'] for row in trials['TRAIN']])
        radius = np.maximum(np.abs(train_residuals).max(0)*np.float32(1.1), np.float32(.001))
        statistics = dict(feature_mean=train_features.mean(0),
            feature_std=train_features.std(0, unbiased=False).clamp(min=1e-4),
            residual_radius=torch.from_numpy(radius.copy()), fitted_partition='TRAIN',
            radius_rule='max(1.1*TRAIN max absolute full residual,0.001 native decoded units) per H,A',
            split_sha256=split_sha, no_label_clipping=True)
        write_tensor(output/'train-statistics.pt.gz', statistics)
        files, coverage = {}, {}
        for partition, data in trials.items():
            path = output/f'{partition.lower()}.pt.gz'
            write_tensor(path, dict(format='decoded_chunk_physical_trials_v1', trials=data,
                metadata=[row for row in metadata if row['partition'] == partition]))
            files[path.name] = dict(bytes=path.stat().st_size, sha256=sha(path))
            residuals = np.concatenate([row['issued']-row['bases'] for row in data])
            coverage[partition] = dict(trials=len(data), episodes=sum(int(row['episode_starts'].sum()) for row in data),
                physical_decisions=len(residuals), coordinates_reaching_or_exceeding_train_bounds=int((np.abs(residuals) >= radius).sum()),
                maximum_absolute_residual=float(np.abs(residuals).max()))
        path = output/'train-statistics.pt.gz'
        files[path.name] = dict(bytes=path.stat().st_size, sha256=sha(path))
        assert sum(row['bytes'] for row in files.values()) <= 128*2**20
        for name, binding in files.items():
            path = output/name
            assert sha(path) == binding['sha256']
            read_tensor(path)
        assert not torch.cuda.is_initialized()
        report.update(status='exported_one_version_physical_corpus', files=files, trial_metadata=metadata,
            physical_decisions=physical, post_terminal_noops_excluded=noops, complete_episodes=1024,
            case_group_split_before_statistics=True, split_sha256=split_sha,
            reconstruction_readback_sha256=sha(audit_path), native_batch=64, native_horizon=32,
            feature_dim=2062, action_dim=14, bellman_discount=.99,
            motor_sha256=plan['model_sha256'], norm_stats_sha256=plan['norm_stats_sha256'],
            feature_contract='frozen masked mean 2048D public VLM prefix and normalized 14D current state',
            residual_contract='full issued decoded32x14 command minus same-row original-Gaussian frozen ODE proposal',
            bellman_clock='one positive-duration issued chunk; termination and timeout stop bootstrap',
            wrapper_timeout_rule='raw simulator truncation OR configured H192 request clock; no-op calls remain excluded',
            source_config_sha256=sha(BASE/'motusppo04/state/resolved-native-config.json'),
            full_planned_tails_retained=True, source_corpus_admitted=False,
            eligible_data_uses=['dit', 'ic_iql', 'static_behavior_cloning'], coverage=coverage,
            student_dev_test_were_teacher_train_cases=True, native_source_dev_panel_exported=False,
            normalization_fitted_on_source_train_only=True, cuda_initialized=False,
            required_next='Review actual reward/coverage and fixed TRAIN-bound violations before objective-specific fitting; one version is not chronological AD improvement data.')
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['seconds'] = time.monotonic()-started
        write(output/'RESULT.json', report)
        print(json.dumps({key:value for key,value in report.items() if key != 'trial_metadata'}), flush=True)


if __name__ == '__main__':
    export()
