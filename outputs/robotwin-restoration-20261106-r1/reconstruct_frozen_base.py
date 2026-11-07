"""Qualify frozen native B64 proposals on the behavior sampler's original noise.

All issued chunks of the four-rank source collection are retained, including no-ops
for audit. This is reconstruction qualification, not student dataset admission.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import torch

from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, tensor_hash, write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor, write_tensor


ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
MOTOR = Path('/mowenjie/fluffy-bassoon-2/cache/robotwin-ppo-pilot-20261105-r1-restore02/pi05-motus-effc6e1a-r1')


def contract():
    plan = json.loads((ROOT/'RECONSTRUCTION01-CONTRACT.json').read_bytes())
    assert plan['status'] in ('prepared', 'admitted')
    assert plan['closed_conservative_gpu_seconds']+plan['reserved_gpu_seconds'] == plan['all_worst_gpu_seconds'] <= 180000
    return plan


def load_packet(binding, local):
    """One checksummed CPU packet in flight; no full-corpus GPU-time staging."""
    origin, destination = BASE/'cpureconstruction04'/binding['packet'], local/binding['packet']
    assert not destination.exists() and origin.stat().st_size == binding['bytes']
    shutil.copyfile(origin, destination)
    assert destination.stat().st_size == binding['bytes'] and sha(destination) == binding['sha256']
    packet = read_tensor(destination)
    destination.unlink()
    from fluffy_bassoon.meta_rl.background_io import _bytes
    assert _bytes(packet) <= 512*2**20
    return packet


def stage(*, validate_payloads=False):
    plan = contract()
    source = BASE/'cpureconstruction04'
    assert sha(source/'RESULT.json') == plan['input_report_sha256']
    report = json.loads((source/'RESULT.json').read_bytes())
    assert report['status'] == 'inputs_bound' and len(report['packets']) == 96
    assert report['complete_native_b64_collection'] and report['original_gaussians_and_denoising_indices_match_canonical']
    assert report['canonical_behavior_bijection'] and len(report['canonical_sources']) == 4
    assert report['fixed_selection'] == dict(ranks=[0, 1, 2, 3], rounds=[0, 1, 2, 3], batch=64, chunks=6, policy_iteration=0)
    assert report['source_corpus_admitted'] is False and report['cuda_initialized'] is False
    local, output = Path(os.environ['TASK_LOCAL']), Path(os.environ['TASK_OUT'])
    rows, episodes = [], []
    physical_calls = noops = nonzero_noops = physics_steps = 0
    terminal_seen = episode_return = episode_ids = scene_ids = None
    for index, row in enumerate(report['packets']):
        assert row['ordinal'] == index and row['packet'] == f'packet-{index:03d}.pt.gz'
        rank, within_rank = divmod(index, 24)
        round_index, chunk = divmod(within_rank, 6)
        assert (row['rank'], row['round_index'], row['chunk']) == (rank, round_index, chunk)
        assert row['full_teacher_command_and_final_chain_exact'] and row['canonical_gaussian_and_denoising_indices_exact']
        if validate_payloads:
            packet = load_packet(row, local)
            assert tuple(packet['original_gaussian'].shape) == (64, 32, 32)
            assert packet['original_gaussian'].dtype == torch.float32
            assert torch.equal(packet['recorded_forward_action'].reshape(64, 32, 14), packet['teacher_actions'])
            assert tuple(packet['teacher_model_action'].shape) == (64, 1024)
            assert tuple(packet['processed_model_state'].shape) == (64, 32)
            assert tensor_hash(packet['current_observation']) == packet['current_observation_sha256']
            assert tensor_hash(packet['teacher_actions']) == packet['teacher_command_sha256']
            assert packet['split_sha256'] == plan['student_case_split_sha256']
            assert (packet['rank'], packet['round_index'], packet['chunk']) == (rank, round_index, chunk)
            execution = packet['physical_execution']
            steps = torch.as_tensor(execution['physics_steps'])
            seconds = torch.as_tensor(execution['duration_seconds'], dtype=torch.float64)
            ticks = torch.as_tensor(execution['timestep_seconds'], dtype=torch.float64)
            delivered = torch.as_tensor(execution['delivered_targets'])
            completed = torch.as_tensor(execution['completed_targets'])
            partial = torch.as_tensor(execution['partial_targets'])
            mask = execution['valid_target_mask']
            assert steps.shape == seconds.shape == ticks.shape == delivered.shape == (64,)
            assert mask.shape == (64, 32) and mask.dtype == torch.bool
            assert (steps >= 0).all() and torch.isfinite(seconds).all() and (ticks > 0).all()
            assert torch.equal(seconds, steps*ticks)
            assert torch.equal(completed+partial, delivered) and ((partial == 0) | (partial == 1)).all()
            assert torch.equal(mask, torch.arange(32)[None, :] < delivered[:, None])
            physical = steps > 0
            assert torch.equal(physical, delivered > 0)
            assert torch.equal(physical, torch.as_tensor(packet['physical_decision_mask']))
            feedback = packet['native_feedback']
            assert len(feedback) == 64
            reward = torch.tensor([float(torch.as_tensor(row['reward']).item()) for row in feedback], dtype=torch.float64)
            terminated = torch.tensor([bool(torch.as_tensor(row['terminated']).item()) for row in feedback])
            truncated = torch.tensor([bool(torch.as_tensor(row['truncated']).item()) for row in feedback])
            assert torch.isfinite(reward).all()
            assert torch.equal(terminated, packet['native_terminated'].any(-1))
            assert torch.equal(truncated, packet['native_truncated'].any(-1))
            if chunk == 0:
                terminal_seen = torch.zeros(64, dtype=torch.bool)
                episode_return = torch.zeros(64, dtype=torch.float64)
                episode_ids, scene_ids = packet['episode_serials'], packet['reset_state_ids']
            assert torch.equal(packet['episode_serials'], episode_ids)
            assert torch.equal(packet['reset_state_ids'], scene_ids)
            assert not (physical & terminal_seen).any(), 'Physical activity continued in a completed source episode'
            assert not (~physical & ~terminal_seen).any(), 'A zero-duration call is not post-terminal feedback'
            assert torch.equal(torch.as_tensor(packet['post_terminal_noop_mask']), ~physical & terminal_seen)
            episode_return += torch.where(physical, reward, 0.)
            physical_calls += int(physical.sum())
            noops += int((~physical).sum())
            nonzero_noops += int((~physical & (reward != 0)).sum())
            physics_steps += int(steps.sum())
            terminal_seen |= terminated | truncated
            if chunk == 5:
                assert terminal_seen.all(), 'The retained source round contains an incomplete physical episode'
                episodes.extend(dict(rank=rank, round_index=round_index, lane=lane,
                    scene_seed=int(scene_ids[lane]), episode_serial=int(episode_ids[lane]),
                    student_partition=packet['student_partition_per_row'][lane],
                    physical_return=float(episode_return[lane])) for lane in range(64))
            del packet
        rows.append(row)
    assert sum(row['bytes'] for row in rows) <= plan['input_staging_cap_gib']*2**30
    write(output/'INPUTS.json', dict(input_report_sha256=sha(source/'RESULT.json'),
        packets=rows, input_report_exact=True, all_payloads_validated=validate_payloads,
        original_noise_rows=6144, native_batch=64, model_or_simulator_constructed=False))
    if validate_payloads:
        assert len(episodes) == 1024 and physical_calls+noops == 6144
        assert noops == report['measured_zero_duration_calls'] == report['post_terminal_noops']
        assert nonzero_noops == report['nonzero_feedback_noops'] and report['unexpected_zero_duration_calls'] == 0
        write(output/'PHYSICAL-INPUTS.json', dict(status='passed',
            input_report_sha256=sha(source/'RESULT.json'), episodes=episodes,
            physical_decisions=physical_calls, post_terminal_noops=noops,
            nonzero_feedback_noops=nonzero_noops, physics_steps=physics_steps,
            zero_duration_feedback_excluded_from_physical_returns=True,
            raw_native_flags_match_recorded_flags=True, all_1024_episodes_complete=True,
            source_corpus_admitted=False, gpu_seconds=0,
            model_or_simulator_constructed=False, cuda_initialized=torch.cuda.is_initialized()))


def run():
    from collections import Counter
    from rlinf.models import get_model
    from fluffy_bassoon.rl.robotwin_motor import FrozenRoboTwinMotor
    from fluffy_bassoon.rl.robotwin_noise import model_config

    plan = contract()
    assert plan['status'] == 'admitted' and os.uname().nodename.startswith('job-')
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    local, output = Path(os.environ['TASK_LOCAL']), Path(os.environ['TASK_OUT'])
    inputs = json.loads((output/'INPUTS.json').read_bytes())
    norm = MOTOR/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'
    report = dict(status='starting', source_commit=os.environ['TASK_SOURCE'], native_batch=64,
        native_row_order=list(range(64)), chunks=[], optimizer_updates=0, simulator_actions=0,
        teacher_policy_iteration=0, source_corpus_admitted=False, model_sha256=plan['model_sha256'],
        norm_stats_sha256=plan['norm_stats_sha256'], input_report_sha256=inputs['input_report_sha256'])
    write(output/'RESULT.json', report)
    try:
        model = get_model(model_config(os.environ['REPO_PATH'], MOTOR, norm, action_horizon=32, num_steps=5)).cuda()
        motor = FrozenRoboTwinMotor(model)
        assert motor.noise_shape == (32, 32) and motor.feature_dim == 2062
        report['parameter_elements_by_dtype'] = dict(Counter({dtype:sum(p.numel() for p in model.parameters()
            if str(p.dtype) == dtype) for dtype in {str(p.dtype) for p in model.parameters()}}))
        report['torch_flags'] = dict(tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            flash_sdp=torch.backends.cuda.flash_sdp_enabled(), math_sdp=torch.backends.cuda.math_sdp_enabled(),
            memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled())
        torch.cuda.reset_peak_memory_stats()
        for index, binding in enumerate(inputs['packets']):
            assert time.time() < int(os.environ['TASK_DEADLINE'])-60
            started = time.monotonic()
            packet = load_packet(binding, local)
            load_seconds = time.monotonic()-started
            noise = packet['original_gaussian'].to(model.device)
            noise_hash = tensor_hash(noise.cpu())
            before_cpu, before_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
            native = None
            if index == 0:
                with torch.no_grad():
                    native, _ = model.predict_action_batch(packet['current_observation'], mode='eval', noise=noise)
            features, token = motor.prepare_batch(packet['current_observation'])
            actual_state = token.observation.state
            expected_state = packet['processed_model_state'].to(actual_state.device)
            assert torch.equal(actual_state, expected_state), 'Native preprocessing changed the recorded model state'
            with torch.no_grad():
                teacher = model.decode_actions(packet['teacher_model_action'].reshape(64, 32, 32).to(model.device), expected_state)
            target = packet['teacher_actions'].to(teacher.device)
            decode_error = float((teacher-target).abs().max())
            assert decode_error == 0, 'Recorded normalized model actions no longer decode to submitted commands'
            base = motor.propose_batch(token, noise=noise)
            if native is not None:
                report['native_adapter_max_abs'] = float((native-base).abs().max())
                assert report['native_adapter_max_abs'] == 0
            assert torch.equal(before_cpu, torch.get_rng_state()) and torch.equal(before_cuda, torch.cuda.get_rng_state())
            assert tensor_hash(noise.cpu()) == noise_hash
            residual = target-base
            assert base.shape == residual.shape == (64, 32, 14) and features.shape == (64, 2062)
            assert torch.isfinite(base).all() and torch.isfinite(residual).all() and torch.isfinite(features).all()
            result = dict(format='motus_native_b64_decoded_reconstruction_v1',
                frozen_base=base.cpu(), public_features=features.cpu(), teacher_actions=target.cpu(),
                decoded_residual=residual.cpu(), input_packet_sha256=binding['sha256'],
                original_gaussian_sha256=noise_hash, native_batch=64, native_rows=list(range(64)),
                ordinal=index, rank=packet['rank'], round_index=packet['round_index'],
                chunk=packet['chunk'], student_partition_per_row=packet['student_partition_per_row'],
                split_sha256=packet['split_sha256'], source_corpus_admitted=False, no_op_filtering_performed=False,
                model_sha256=plan['model_sha256'], norm_stats_sha256=plan['norm_stats_sha256'])
            destination = output/f'reconstruction-{index:03d}.pt.gz'
            write_tensor(destination, result)
            row = dict(ordinal=index, rank=packet['rank'], round_index=packet['round_index'],
                chunk=packet['chunk'], bytes=destination.stat().st_size, sha256=sha(destination),
                output=destination.name, teacher_decode_max_abs=decode_error,
                preprocessed_state_exact=True, sampler_rng_unchanged=True,
                input_load_seconds=load_seconds,
                full_command_residual_max_abs=float(residual.abs().max()),
                seconds=time.monotonic()-started)
            report['chunks'].append(row)
            write(output/'RESULT.json', report)
            print(json.dumps(row), flush=True)
            del packet, native, target, teacher, base, features, residual, token, actual_state, expected_state, result, noise
        motor.check_frozen()
        free, total = torch.cuda.mem_get_info()
        assert free >= max(8*2**30, .08*total)
        assert motor.prefix_calls == motor.sample_calls == 96 and motor.sampled_rows == 6144
        report.update(status='passed', issued_command_rows=6144, model_sampler_rows=motor.sampled_rows+64,
            native_sampler_calls=motor.sample_calls+1, prefix_calls=motor.prefix_calls, sample_calls=motor.sample_calls,
            frozen_parameter_buffer_versions_unchanged=True, no_new_noise_draws=True,
            input_streaming_maximum_packets=1, decoded_input_packet_cap_mib=512,
            all_teacher_decodes_exact=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), free_bytes_at_end=free,
            source_corpus_admitted=False, complete_student_dataset=False,
            physical_decision_noop_filtering_still_required=True)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write(output/'RESULT.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--stage', action='store_true')
    group.add_argument('--validate-inputs', action='store_true')
    args = parser.parse_args()
    if args.stage or args.validate_inputs:
        stage(validate_payloads=args.validate_inputs)
    else:
        run()
