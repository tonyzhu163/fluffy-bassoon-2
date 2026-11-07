"""Cold CPU readback of all frozen proposals and original behavior bindings.

This checks the reconstruction, not learner quality or source-corpus admission.
No model, simulator, optimizer or accelerator is constructed.
"""
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, tensor_hash, write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor


BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
ROOT = Path(__file__).resolve().parent


def audit():
    assert os.uname().sysname == 'Linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    code = ROOT.parents[1]
    assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == os.environ['TASK_SOURCE']
    assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain', '--untracked-files=no'])
    output = BASE/'reconstruction01-readback01'
    output.mkdir()
    started = time.monotonic()
    report = dict(status='checking', audit_source_commit=os.environ['TASK_SOURCE'],
        gpu_seconds=0, model_calls=0, optimizer_updates=0, simulator_actions=0,
        source_corpus_admitted=False, checked_chunks=[])
    try:
        plan = json.loads((ROOT/'RECONSTRUCTION01-CONTRACT.json').read_bytes())
        source, runtime = BASE/'cpureconstruction04', BASE/'reconstruction01-signals'
        assert plan['status'] == 'admitted' and sha(source/'RESULT.json') == plan['input_report_sha256']
        original = json.loads((source/'RESULT.json').read_bytes())
        result = json.loads((runtime/'RESULT.json').read_bytes())
        complete = json.loads((runtime/'COMPLETE.json').read_bytes())
        assert complete['status'] == 'complete_frozen_reconstruction'
        assert complete['result_sha256'] == sha(runtime/'RESULT.json')
        assert json.loads((runtime/'EXIT.json').read_bytes())['exit_code'] == 0
        assert result['status'] == 'passed' and len(result['chunks']) == 96
        assert result['source_commit'] == os.environ['TASK_RECONSTRUCTION_SOURCE']
        assert result['input_report_sha256'] == sha(source/'RESULT.json')
        assert result['model_sha256'] == plan['model_sha256'] and result['norm_stats_sha256'] == plan['norm_stats_sha256']
        assert result['optimizer_updates'] == result['simulator_actions'] == 0
        assert result['issued_command_rows'] == 6144 and result['model_sampler_rows'] == 6208
        assert result['native_sampler_calls'] == 97 and result['prefix_calls'] == result['sample_calls'] == 96
        assert result['native_adapter_max_abs'] == 0 and result['all_teacher_decodes_exact']
        assert result['frozen_parameter_buffer_versions_unchanged'] and result['no_new_noise_draws']
        assert original['status'] == 'inputs_bound' and original['canonical_behavior_bijection']
        assert len(original['packets']) == 96 and len(original['canonical_sources']) == 4
        manifest = json.loads((runtime/'FILES.json').read_bytes())['files']
        for name, binding in manifest.items():
            assert Path(name).name == name and not Path(name).is_absolute()
            path = runtime/name
            assert path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256']
            if path.suffix == '.json':
                json.loads(path.read_bytes())
        resources = json.loads((runtime/'resources.json').read_bytes())
        assert resources['sampled_reserve_passed'] and resources['samples'] > 0
        assert resources['host_limit_bytes'] >= 128*2**30 and len(resources['gpus']) == 1
        assert all(row['minimum_free_bytes'] >= max(8*2**30, .08*row['total_bytes'])
                   for row in resources['gpus'].values())
        index = original['canonical_index']
        path = Path(index['path'])
        assert path == source/'canonical-index.pt.gz' and sha(path) == index['sha256']
        canonical = read_tensor(path)
        assert canonical['sources'] == original['canonical_sources'] and len(canonical['table']) == 4
        used = set()
        for ordinal, (binding, reconstructed) in enumerate(zip(original['packets'], result['chunks'], strict=True)):
            assert binding['ordinal'] == reconstructed['ordinal'] == ordinal
            path = source/binding['packet']
            assert path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256']
            packet = read_tensor(path)
            name = f'reconstruction-{ordinal:03d}.pt.gz'
            assert reconstructed['output'] == name and manifest[name]['sha256'] == reconstructed['sha256']
            value = read_tensor(runtime/name)
            assert value['format'] == 'motus_native_b64_decoded_reconstruction_v1'
            assert value['input_packet_sha256'] == binding['sha256']
            assert value['original_gaussian_sha256'] == tensor_hash(packet['original_gaussian'])
            assert value['native_batch'] == 64 and value['native_rows'] == list(range(64))
            assert value['split_sha256'] == packet['split_sha256'] == plan['student_case_split_sha256']
            assert value['student_partition_per_row'] == packet['student_partition_per_row']
            for key in ('ordinal', 'rank', 'round_index', 'chunk'):
                assert value[key] == reconstructed[key]
            assert (value['rank'], value['round_index'], value['chunk']) == (ordinal//24, ordinal//6 % 4, ordinal % 6)
            assert value['model_sha256'] == plan['model_sha256'] and value['norm_stats_sha256'] == plan['norm_stats_sha256']
            assert value['source_corpus_admitted'] is False and value['no_op_filtering_performed'] is False
            for key, shape in [('frozen_base', (64, 32, 14)), ('teacher_actions', (64, 32, 14)),
                               ('decoded_residual', (64, 32, 14)), ('public_features', (64, 2062))]:
                assert value[key].shape == shape and value[key].dtype == torch.float32
                assert torch.isfinite(value[key]).all()
            assert torch.equal(value['teacher_actions'], packet['teacher_actions'])
            assert torch.equal(value['decoded_residual'], value['teacher_actions']-value['frozen_base'])
            assert tensor_hash(packet['current_observation']) == packet['current_observation_sha256']
            assert tensor_hash(packet['teacher_actions']) == packet['teacher_command_sha256']
            assert torch.equal(packet['recorded_forward_action'].reshape(64, 32, 14), packet['teacher_actions'])
            coordinates = packet['canonical']['coordinates']
            assert coordinates.shape == (64, 3) and coordinates.dtype == torch.long
            assert packet['canonical']['sha256'] == index['sha256']
            for lane, coordinate in enumerate(coordinates.tolist()):
                coordinate = tuple(coordinate)
                assert coordinate not in used
                used.add(coordinate)
                rank, step, column = coordinate
                assert 0 <= rank < 4 and 0 <= step < 6 and 0 <= column < 256
                fields = canonical['table'][rank]
                assert torch.equal(packet['original_gaussian'][lane], fields['chains'][step, column, 0])
                assert torch.equal(packet['teacher_model_action'][lane], fields['chains'][step, column, -1].reshape(-1))
                assert torch.equal(packet['recorded_forward_action'][lane], fields['action'][step, column])
            assert reconstructed['teacher_decode_max_abs'] == 0
            assert reconstructed['preprocessed_state_exact'] and reconstructed['sampler_rng_unchanged']
            report['checked_chunks'].append(dict(ordinal=ordinal, input_sha256=binding['sha256'],
                output_sha256=reconstructed['sha256'], rows=64))
            write(output/'RESULT.json', report)
            print('CHECKED_RECONSTRUCTION', ordinal, flush=True)
            del value, packet
        assert len(used) == 6144 and not torch.cuda.is_initialized()
        report.update(status='passed', issued_command_rows=6144, cold_manifest_files=len(manifest),
            complete_sha256=sha(runtime/'COMPLETE.json'), result_sha256=sha(runtime/'RESULT.json'),
            manifest_sha256=sha(runtime/'FILES.json'), input_report_sha256=sha(source/'RESULT.json'),
            canonical_bijection_rows=len(used), exact_full_decoded_residuals=True,
            full_undelivered_tails_retained=True, original_gaussians_match_canonical=True,
            finite_public_features=True, cuda_initialized=False, resources=resources,
            physical_episode_noop_audit_still_required=True, complete_ad_learning_histories=False)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['seconds'] = time.monotonic()-started
        write(output/'RESULT.json', report)
        print(json.dumps({k:v for k,v in report.items() if k != 'checked_chunks'}), flush=True)


if __name__ == '__main__':
    audit()
