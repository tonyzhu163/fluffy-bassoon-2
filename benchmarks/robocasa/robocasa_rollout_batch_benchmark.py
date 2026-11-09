#!/usr/bin/env python
"""Finite real native collection benchmark: shared model, concurrent simulator lanes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import imageio.v2 as imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir

from fluffy_bassoon.baige_gpu import verify_selected_cuda
from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic, ResidualPPOConfig
from fluffy_bassoon.robocasa.residual import FrozenRoboCasaMotor, RoboCasaResidualEnvironment, native_environment_factory
from fluffy_bassoon.robocasa.flora_ppo import write_new
from fluffy_bassoon.rlinf.native_ppo import ActionJournal, sha
from fluffy_bassoon.robocasa.ppo import TASKS
from fluffy_bassoon.rl.action_residual import ResidualBound


def combine(rows):
    result = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        result[key] = (torch.cat(values, 0) if torch.is_tensor(values[0]) else
                       np.concatenate(values, 0) if isinstance(values[0], np.ndarray) else
                       sum((list(value) for value in values), []))
    return result


def run(profile):
    if torch.cuda.device_count() != 1:
        raise RuntimeError('One allocated GPU per shared-model benchmark worker')
    uuid = verify_selected_cuda(torch)
    out = Path(profile['output']); out.mkdir(exist_ok=False)
    started = time.monotonic(); stop = threading.Event()
    state = dict(phase='model_loading', completed_episodes=0, policy_controls=0, peak_nvml_mib=0)
    lock = threading.Lock()

    def heartbeat():
        while not stop.is_set():
            try:
                row = subprocess.check_output(['nvidia-smi', '--id='+uuid,
                    '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                    text=True, timeout=5).strip().split(',')
                with lock:
                    state['peak_nvml_mib'] = max(state['peak_nvml_mib'], float(row[0]))
                    state['gpu_utilization_percent'] = float(row[1])
                    current = Path('/sys/fs/cgroup/memory.current')
                    if current.exists():
                        state['peak_container_ram_bytes'] = max(state.get('peak_container_ram_bytes', 0), int(current.read_text()))
            except Exception as error:
                with lock: state['memory_sample_error'] = str(error)
            with lock: value = dict(state, elapsed_seconds=time.monotonic()-started)
            temporary = out/'status.tmp'; temporary.write_text(json.dumps(value)); temporary.replace(out/'status.json')
            stop.wait(2)

    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    environments = []; result = dict(completed=False, batch_size=profile['batch_size'], rank=profile['rank'])
    try:
        code = Path(__file__).resolve().parents[2]
        with initialize_config_dir(config_dir=str(code/'benchmarks/robocasa/configs'), version_base=None):
            cfg = compose(config_name='robocasa365_pi05_eval')
        initializer = profile['initializer']
        cfg.rollout.model.model_path = initializer['model_directory']
        cfg.rollout.model.openpi_data.norm_stats_path = initializer['norm_path']
        from rlinf.models.embodiment.openpi import get_model
        model = get_model(cfg.rollout.model, torch_dtype=torch.bfloat16).to('cuda').eval().requires_grad_(False)
        stats = json.loads(Path(initializer['norm_path']).read_text()); stats = stats.get('norm_stats', stats)
        motor = FrozenRoboCasaMotor(model, stats)
        checkpoint = torch.load(profile['initial_checkpoint'], map_location='cpu', weights_only=False)
        policy = ResidualActorCritic(ResidualPPOConfig(**checkpoint['contract']['ppo'])).to('cuda').eval()
        policy.load_state_dict(checkpoint['policy'], strict=True)
        bound = ResidualBound(**checkpoint['contract']['bound'])
        batch = int(profile['batch_size']); rank = int(profile['rank'])
        count = len(profile['cases'])
        horizons = {TASKS[case['task']][0] for case in profile['cases']}
        if batch not in (1, 2, 4, 8, 16, 32, 48) or count < batch or count % batch or len(horizons) != 1:
            raise ValueError('Require complete equal-horizon cohorts divisible by the lane count')
        horizon = horizons.pop()
        chunks_per_episode = horizon // 50
        expected_controls = count * horizon
        torch.cuda.synchronize(); result['model_load_seconds'] = time.monotonic()-started
        collect_start = time.monotonic(); inference_seconds = reset_seconds = step_seconds = validation_seconds = 0.
        episodes = []; initial_comparison = []; inferred_decisions = 0
        with ThreadPoolExecutor(max_workers=batch) as pool:
            for first in range(0, count, batch):
                if time.time()+profile['closure_seconds'] >= profile['deadline_unix']:
                    raise TimeoutError('Finite collection clock; preserve partial episodes and escrows')
                environments = []
                for lane, case in enumerate(profile['cases'][first:first+batch]):
                    slot = first+lane
                    journal = ActionJournal(profile['accounting_root'], rank=rank, world=8,
                        attempt=f'capacity-r{rank}-case{slot}')
                    environments.append(RoboCasaResidualEnvironment(native_environment_factory(cfg.env.eval), [case],
                        horizon=horizon, output=out/f'case-{slot:02d}', monitoring_root=out/'monitoring',
                        journal=journal, writer_factory=imageio.get_writer,
                        episode_id_prefix=f'capacity-r{rank}-case{slot}-', rank=rank))
                with lock: state['phase'] = 'native_reset'
                tick = time.monotonic(); observations = list(pool.map(lambda env: env.reset(), environments))
                reset_seconds += time.monotonic()-tick
                for chunk_index in range(chunks_per_episode):
                    if time.time()+profile['closure_seconds'] >= profile['deadline_unix']:
                        raise TimeoutError('Native partial episodes retained at finite deadline')
                    seeds = [6_600_000+(first+lane)*100+chunk_index for lane in range(batch)]
                    joined = combine(observations)
                    # First planned episode also checks real batch-vs-single numerical drift;
                    # it adds no physical episode and is outside collection inference timing.
                    if first == 0 and chunk_index == 0:
                        torch.cuda.synchronize(); validation_start = time.monotonic()
                        chunks, features, _ = motor.infer_batch(joined, noise_seeds=seeds)
                        for lane, (obs, seed) in enumerate(zip(observations, seeds)):
                            single, feature, _ = motor.infer(obs, noise_seed=seed)
                            initial_comparison.append(dict(lane=lane,
                                action_max_abs=float(np.max(np.abs(single-chunks[lane]))),
                                feature_max_abs=float(np.max(np.abs(feature-features[lane])))))
                        torch.cuda.synchronize(); validation_seconds += time.monotonic()-validation_start
                    with lock: state['phase'] = 'batched_policy_inference'
                    torch.cuda.synchronize(); tick = time.monotonic()
                    chunks, features, _ = motor.infer_batch(joined, noise_seeds=seeds)
                    controls = []
                    for lane in range(batch):
                        generator = torch.Generator(device='cuda').manual_seed(seeds[lane]+10_000_000)
                        control, _, _ = policy.act(torch.as_tensor(features[lane:lane+1], device='cuda'),
                                                  generator=generator)
                        controls.append(control[0].detach().cpu().numpy())
                    executed = [bound.apply(chunks[lane], controls[lane]) for lane in range(batch)]
                    torch.cuda.synchronize(); inference_seconds += time.monotonic()-tick
                    with lock: state['phase'] = 'parallel_native_H50_execution'
                    tick = time.monotonic()
                    rows = list(pool.map(lambda pair: pair[0].step_chunk(pair[1]), zip(environments, executed)))
                    step_seconds += time.monotonic()-tick; inferred_decisions += batch
                    observations = [row[0] for row in rows]
                    if any(row[1] != (chunk_index == chunks_per_episode - 1) for row in rows):
                        raise ValueError('Every lane must execute its complete registered horizon')
                    with lock: state['policy_controls'] += batch*50
                    if chunk_index == chunks_per_episode - 1: episodes.extend(row[2] for row in rows)
                for env in environments: env.close()
                environments = []
                with lock: state['completed_episodes'] += batch
        torch.cuda.synchronize(); elapsed = time.monotonic()-collect_start-validation_seconds
        if len(episodes)!=count or sum(e['frame_clock']['physical_steps'] for e in episodes)!=expected_controls:
            raise ValueError('Incomplete benchmark cohort')
        journals = [json.loads(path.read_text()) for path in
                    (Path(profile['accounting_root'])/f'rank-{rank:02d}').glob('*.json')]
        if any(not call['resolved'] for row in journals for call in row['calls']):
            raise ValueError('Unresolved physical intents remain charged')
        result.update(completed=True, episodes=count, horizon=horizon, policy_controls=expected_controls,
            actual_all_in_controls=sum(call['executed_or_conservative'] for row in journals for call in row['calls']),
            collection_wall_seconds=elapsed, controls_per_second=expected_controls/elapsed,
            inference_seconds=inference_seconds, reset_seconds=reset_seconds, simulation_video_seconds=step_seconds,
            extra_numerical_reference_seconds=validation_seconds,
            model_decisions=inferred_decisions, batched_forward_calls=count*chunks_per_episode//batch,
            initial_batch_single_comparison=initial_comparison, optimizer_updates=0,
            full_episode_evidence=episodes, torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            policy_checkpoint_sha256=sha(profile['initial_checkpoint']))
    except BaseException as error:
        result.update(error=f'{type(error).__name__}: {error}',
                      OOM=isinstance(error, torch.cuda.OutOfMemoryError) or 'out of memory' in str(error).lower())
        raise
    finally:
        for env in environments:
            try: env.close()
            except Exception as error: result.setdefault('cleanup_errors', []).append(str(error))
        stop.set(); thread.join(timeout=7)
        result.update(total_seconds=time.monotonic()-started, observed_memory=dict(state), gpu_uuid=uuid)
        write_new(out/'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--profile', required=True)
    run(json.loads(Path(parser.parse_args().profile).read_text()))
