"""Fresh CPU numerical readback of physical trials, frozen labels and statistics."""
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from fluffy_bassoon.meta_rl.chunk_history import ChunkHistoryLayout, trial_tokens
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor


BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
ROOT = Path(__file__).resolve().parent
FIELDS = {'features', 'bases', 'issued', 'delivered_masks', 'rewards',
          'terminated', 'truncated', 'physics_seconds', 'episode_starts'}


def audit():
    assert os.uname().sysname == 'Linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    code = ROOT.parents[1]
    assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == os.environ['TASK_SOURCE']
    assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain', '--untracked-files=no'])
    output = BASE/'decodedhistory01-readback01'
    output.mkdir()
    started = time.monotonic()
    report = dict(status='checking', source_commit=os.environ['TASK_SOURCE'],
        gpu_seconds=0, model_calls=0, optimizer_updates=0, simulator_actions=0,
        source_corpus_admitted=False, complete_ad_learning_histories=False)
    try:
        source, reconstructed = BASE/'decodedhistory01', BASE/'reconstruction01-signals'
        exported = json.loads((source/'RESULT.json').read_bytes())
        assert exported['status'] == 'exported_one_version_physical_corpus'
        assert exported['source_commit'] == os.environ['TASK_PRODUCER_SOURCE']
        assert exported['available_teacher_policy_versions'] == [0]
        assert not exported['dpt_query_labels'] and not exported['complete_ad_learning_histories']
        native = json.loads((reconstructed/'RESULT.json').read_bytes())
        assert native['status'] == 'passed' and len(native['chunks']) == 96
        physical_path = BASE/'cpureconstruction-preflight01/PHYSICAL-INPUTS.json'
        qualified = json.loads((BASE/'cpureconstruction-preflight01/QUALIFICATION.json').read_bytes())
        assert qualified['status'] == 'passed' and qualified['physical_inputs_sha256'] == sha(physical_path)
        physical = json.loads(physical_path.read_bytes())
        reference = {(row['rank'], row['round_index'], row['lane']):row for row in physical['episodes']}
        assert len(reference) == 1024 and physical['physical_decisions'] == 5958
        all_data, identities, bindings, coverage = {}, set(), {}, {}
        for partition, expected_trials in [('TRAIN', 24), ('DEV', 4), ('TEST', 4)]:
            name = partition.lower()+'.pt.gz'
            binding = exported['files'][name]
            path = source/name
            assert path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256']
            payload = read_tensor(path)
            assert payload['format'] == 'decoded_chunk_physical_trials_v1'
            assert len(payload['trials']) == len(payload['metadata']) == expected_trials
            all_data[partition] = payload['trials']
            for trial, metadata in zip(payload['trials'], payload['metadata'], strict=True):
                assert set(trial) == FIELDS and metadata['partition'] == partition
                assert metadata['trial_id'] not in identities
                identities.add(metadata['trial_id'])
                assert metadata['policy_versions'] == [0] and metadata['episodes'] == 32
                assert len(metadata['source_episodes']) == int(trial['episode_starts'].sum()) == 32
                tokens, _ = trial_tokens(ChunkHistoryLayout(2062, 32, 14), **trial)
                assert np.isfinite(tokens).all()
                assert len(tokens) == metadata['physical_decisions']
                rank, group = None, None
                position = 0
                for number, episode in enumerate(metadata['source_episodes']):
                    if rank is None:
                        rank, group = episode['rank'], episode['lane']//8
                    round_index, replica = divmod(number, 8)
                    assert (episode['rank'], episode['round_index'], episode['lane']) == (rank, round_index, group*8+replica)
                    assert ('TRAIN' if group < 6 else 'DEV' if group == 6 else 'TEST') == partition
                    ref = reference[rank, round_index, group*8+replica]
                    assert ref['student_partition'] == partition
                    assert episode['scene_seed'] == ref['scene_seed'] and episode['episode_serial'] == ref['episode_serial']
                    chunks = episode['source_chunks']
                    length = len(chunks)
                    assert 1 <= length <= 6
                    assert trial['episode_starts'][position] and not trial['episode_starts'][position+1:position+length].any()
                    ended = trial['terminated'][position:position+length] | trial['truncated'][position:position+length]
                    assert ended[-1] and not ended[:-1].any()
                    assert float(trial['rewards'][position:position+length].sum()) == ref['physical_return']
                    for step, chunk in enumerate(chunks):
                        ordinal, lane = rank*24+round_index*6+step, group*8+replica
                        assert (chunk['ordinal'], chunk['lane']) == (ordinal, lane)
                        key = ordinal, lane
                        assert key not in bindings
                        bindings[key] = trial, position+step, partition, chunk
                    position += length
                assert position == len(tokens)
                del tokens
        assert len(identities) == 32 and len(bindings) == 5958
        for ordinal, recorded in enumerate(native['chunks']):
            path = reconstructed/recorded['output']
            assert sha(path) == recorded['sha256']
            packet = read_tensor(path)
            for lane in range(64):
                selected = bindings.get((ordinal, lane))
                if selected is None:
                    continue
                trial, position, partition, chunk = selected
                assert chunk['reconstruction_sha256'] == recorded['sha256']
                assert chunk['input_packet_sha256'] == packet['input_packet_sha256']
                assert packet['student_partition_per_row'][lane] == partition
                for field, key in [('features', 'public_features'), ('bases', 'frozen_base'), ('issued', 'teacher_actions')]:
                    assert np.array_equal(trial[field][position], packet[key][lane].numpy())
                assert np.array_equal(trial['issued'][position]-trial['bases'][position], packet['decoded_residual'][lane].numpy())
            del packet
        name = 'train-statistics.pt.gz'
        binding = exported['files'][name]
        path = source/name
        assert path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256']
        statistics = read_tensor(path)
        assert statistics['fitted_partition'] == 'TRAIN' and statistics['no_label_clipping']
        features = np.concatenate([trial['features'] for trial in all_data['TRAIN']]).astype(np.float64)
        mean, std = features.mean(0), np.maximum(features.std(0), 1e-4)
        assert np.allclose(statistics['feature_mean'].numpy(), mean, rtol=1e-5, atol=1e-6)
        assert np.allclose(statistics['feature_std'].numpy(), std, rtol=1e-5, atol=1e-6)
        residuals = torch.from_numpy(np.concatenate([trial['issued']-trial['bases'] for trial in all_data['TRAIN']]))
        radius = (residuals.abs().amax(0)*1.1).clamp(min=.001)
        assert torch.equal(radius, statistics['residual_radius'])
        for partition, data in all_data.items():
            residuals = np.concatenate([trial['issued']-trial['bases'] for trial in data])
            count = int((np.abs(residuals) >= radius.numpy()).sum())
            assert count == exported['coverage'][partition]['coordinates_reaching_or_exceeding_train_bounds']
            coverage[partition] = count
        assert not torch.cuda.is_initialized()
        report.update(status='passed', export_result_sha256=sha(source/'RESULT.json'),
            partition_files_and_statistics=4, complete_trials=32, complete_episodes=1024,
            physical_decisions=5958, post_terminal_noops_excluded=186,
            all_public_features_bases_and_full_labels_match_frozen_reconstruction=True,
            episode_order_and_physical_returns_match_preflight=True,
            feature_statistics_match_independent_float64_train_reference=True,
            train_radii_match_independent_torch_reference=True,
            coordinates_outside_fixed_train_bounds=coverage,
            all_partition_labels_inside_fixed_train_bounds=not any(coverage.values()),
            source_corpus_admitted=False, cuda_initialized=False,
            eligible_data_uses=exported['eligible_data_uses'], student_fit_or_deployment_qualified=False)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['seconds'] = time.monotonic()-started
        write(output/'RESULT.json', report)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    audit()
