"""Bounded original Pi0/task57 DSRL runner; upstream scientific functions stay clean.

Collection, relabeling, PixelSAC updates, evaluation, shared environment and
within-burst evaluation timing are the authors' implementation. This module
adds local receipts, physical cost limits and complete episode-boundary recovery.
"""
from collections import deque
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import runpy
import socket
import subprocess
import sys
import time

import numpy as np

DSRL = '7f48937d4553e95244cd81c79236a3256df80597'
OPENPI = 'a6d2400d2534ce32e7bdf8747709b97aaef8ec04'
LIBERO = '87edbd1b823e464bc66557d213cc45e9f88aa543'
MAX_ACTIONS, MAX_SECONDS, MAX_BYTES = 1000000, 36000, 64 * 1024**3


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


class Prefetch:
    """The upstream two-batch iterator, with its suspended-yield state explicit."""
    def __init__(self, sample, device_put=lambda value: value):
        self.sample, self.device_put = sample, device_put
        self.queue, self.initialized, self.refill = deque(), False, False

    def __next__(self):
        if not self.initialized:
            self.queue.extend(self.device_put(self.sample()) for _ in range(2))
            self.initialized = True
        if self.refill:
            self.queue.append(self.device_put(self.sample()))
        self.refill = True
        return self.queue.popleft()

    def __iter__(self):
        return self

    def state(self, host=lambda value: value):
        return dict(queue=[host(value) for value in self.queue],
                    initialized=self.initialized, refill=self.refill)

    def restore(self, value):
        self.queue = deque(self.device_put(item) for item in value['queue'])
        self.initialized, self.refill = value['initialized'], value['refill']


class CaptureEnv:
    """Record existing calls; reserve a complete episode before any physics."""
    def __init__(self, env, output, *, action_limit=MAX_ACTIONS, bind=None):
        self.env, self.output, self.action_limit = env, Path(output), action_limit
        self.output.mkdir(exist_ok=True)
        self.role, self.updates, self.bind = 'training', 0, bind
        self.active = None
        self.ledger = dict(known_actions=0, uncertain_actions=0, episodes=[], pending=None)

    @property
    def charged(self):
        return self.ledger['known_actions'] + self.ledger['uncertain_actions']

    def __getattr__(self, name):
        return getattr(self.env, name)

    def publish(self):
        atomic_json(self.output/'physical-cost.json', self.ledger)

    def reset(self):
        self.finish()
        if self.charged + 400 > self.action_limit:
            raise RuntimeError('No complete400-action episode remains')
        identity = f"{len(self.ledger['episodes']):06d}-{self.role}-{self.updates:06d}"
        self.ledger['pending'] = dict(episode=identity, role=self.role, maximum_actions=400)
        self.ledger['uncertain_actions'] += 400
        self.publish()
        observation = self.env.reset()
        state = np.asarray(self.env.get_sim_state()).copy()
        path = self.output/(identity+'.pkl.gz')
        self.active = dict(identity=identity, stream=gzip.open(path, 'xb', compresslevel=1),
                           path=path, steps=0, success=False)
        self.record(dict(event='reset', observation=observation, simulator_state=state,
                         simulator_state_sha256=hashlib.sha256(state.tobytes()).hexdigest()))
        return observation

    def record(self, value):
        if self.active is not None:
            pickle.dump(value, self.active['stream'], protocol=5)

    def step(self, action):
        if self.active is None or self.active['steps'] >= 400:
            raise RuntimeError('Reset before stepping a new complete episode')
        outcome = self.env.step(action)
        self.active['steps'] += 1
        self.active['success'] |= bool(outcome[1] == 1)
        self.record(dict(event='step', action=np.asarray(action).copy(), reward=float(outcome[1]),
                         done=bool(outcome[2])))
        if self.bind is not None:
            self.bind()
            self.bind = None
        return outcome

    def finish(self):
        if self.active is None:
            return
        active, self.active = self.active, None
        active['stream'].close()
        row = dict(episode=active['identity'], role=self.role, updates=self.updates,
                   actions=active['steps'], success=active['success'],
                   trace=active['path'].name, trace_sha256=file_hash(active['path']))
        self.ledger['known_actions'] += active['steps']
        self.ledger['uncertain_actions'] -= 400
        self.ledger['pending'] = None
        self.ledger['episodes'].append(row)
        self.publish()


def compact_replay(buffer):
    def trim(value):
        if isinstance(value, dict):
            return {key: trim(item) for key, item in value.items()}
        return np.asarray(value)[:buffer.size].copy()
    return dict(data=trim(buffer.data), attributes={key: getattr(buffer, key) for key in
        ('capacity', 'size', '_traj_counter', '_start', 'traj_bounds', 'streaming_buffer_size')})


def restore_replay(buffer, saved):
    attributes = saved['attributes']
    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        data = np.empty((attributes['capacity'], *value.shape[1:]), value.dtype)
        data[:attributes['size']] = value
        return data
    buffer.data = expand(saved['data'])
    for key, value in attributes.items():
        setattr(buffer, key, value)


def save_complete(output, agent, buffer, iterator, motor, progress, contract):
    import jax
    from flax import serialization
    from .vla_source40_corpus import save_motor_rng
    output = Path(output)
    payload = dict(format='original_dsrl_pi0_complete_boundary_v1', contract=contract,
        progress=progress, agent=serialization.to_bytes(agent._save_dict),
        agent_rng=np.asarray(jax.device_get(agent._rng)), replay=compact_replay(buffer),
        prefetch=iterator.state(jax.device_get), numpy_rng=np.random.get_state(),
        python_rng=random.getstate(), motor_rng=save_motor_rng(motor._rng))
    slot = output/f"recovery-{progress['episodes'] % 2}.pkl"
    temporary = slot.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(payload, stream, protocol=5)
    temporary.replace(slot)
    atomic_json(output/'recovery.json', dict(path=slot.name, sha256=file_hash(slot),
        bytes=slot.stat().st_size, progress=progress, contract=contract,
        boundary='complete collection episode and complete optimizer burst'))


def restore_complete(path, expected_contract, agent, buffer, iterator, motor):
    import jax.numpy as jnp
    from flax import serialization
    from .vla_source40_corpus import restore_motor_rng
    with Path(path).open('rb') as stream:
        value = pickle.load(stream)
    if value['format'] != 'original_dsrl_pi0_complete_boundary_v1' or value['contract'] != expected_contract:
        raise ValueError('Recovery contract differs')
    saved = serialization.from_bytes(agent._save_dict, value['agent'])
    agent._actor, agent._critic = saved['actor'], saved['critic']
    agent._target_critic_params, agent._temp = saved['target_critic_params'], saved['temp']
    agent._rng = jnp.asarray(value['agent_rng'])
    restore_replay(buffer, value['replay'])
    iterator.restore(value['prefetch'])
    motor._rng = restore_motor_rng(value['motor_rng'])
    np.random.set_state(value['numpy_rng'])
    random.setstate(value['python_rng'])
    return value['progress']


class LocalLogger:
    def __init__(self, output):
        self.path = Path(output)/'metrics.jsonl'

    def log(self, values, step):
        clean = {}
        for key, value in values.items():
            if isinstance(value, (int, float, str)):
                clean[key] = value
            elif hasattr(value, 'ndim') and value.ndim == 0:
                clean[key] = float(np.asarray(value))
        with self.path.open('a') as stream:
            stream.write(json.dumps(dict(updates=step, values=clean))+'\n')
        if 'evaluation/success_rate' in clean:
            print(json.dumps(dict(event='evaluation', updates=step, **clean)), flush=True)

    def log_histogram(self, *args, **kwargs):
        pass


def training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer,
                  wandb_logger, *, agent_dp, shard_fn=None, agent_dp_unused=None):
    """Same scientific control flow as upstream, with explicit recovery state."""
    del agent_dp_unused
    import jax
    from examples import train_utils_sim as upstream
    from .vla_libero import attest_process_gpu
    from .vla_motor_replay import parameter_signature
    output = Path(os.environ['DSRL_REPLICATION_OUTPUT'])
    contract = json.loads((output/'contract.json').read_text())
    deadline = float(os.environ['DSRL_REPLICATION_DEADLINE'])
    if env is not eval_env or replay_buffer is not online_replay_buffer:
        raise ValueError('Preserve the authors shared environment/replay')
    capture = CaptureEnv(env, output/'physical', bind=lambda:
        attest_process_gpu(output/'renderer-binding.json', graphics=True))
    iterator = Prefetch(lambda: replay_buffer.sample(variant.batch_size), jax.device_put)
    progress = dict(updates=0, episodes=0, training_primitives=0, last_checkpoint_updates=0)
    before = parameter_signature(agent_dp._model)
    atomic_json(output/'motor-before.json', before)
    original_infer = agent_dp.infer
    first = True

    def infer(observation, **kwargs):
        nonlocal first
        old_rng = agent_dp._rng
        result = original_infer(observation, **kwargs)
        new_rng = agent_dp._rng
        if first:
            agent_dp._rng = old_rng
            repeated = original_infer(observation, **kwargs)
            agent_dp._rng = new_rng
            delta = float(np.abs(np.asarray(result['actions'])-np.asarray(repeated['actions'])).max())
            if not np.isfinite(delta) or delta > 1e-5:
                raise RuntimeError('Original Pi0 same-noise replay failed')
            attest_process_gpu(output/'motor-binding.json', graphics=False)
            atomic_json(output/'qualification.json', dict(same_noise_max_delta=delta,
                model='pi0_libero', horizon=50, execute_steps=20,
                extra_physical_actions=0, original_rng_restored=True))
            first = False
        capture.record(dict(event='infer', observation=observation,
                            noise=np.asarray(kwargs['noise']), actions=np.asarray(result['actions'])))
        return result
    agent_dp.infer = infer
    stop_reason = 'optimizer_target'
    try:
        while progress['updates'] <= variant.max_steps:
            # One collection episode and at most one10-query panel per burst.
            if time.time()+600 >= deadline or capture.charged+4400 > MAX_ACTIONS:
                stop_reason = 'whole_episode_and_evaluation_reservation'
                break
            size = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
            if size+2*1024**3 > MAX_BYTES:
                stop_reason = 'storage_reservation'
                break
            capture.role, capture.updates = 'training', progress['updates']
            trajectory = upstream.collect_traj(variant, agent, capture, progress['updates'], agent_dp)
            capture.finish()
            upstream.add_online_data_to_buffer(variant, trajectory, replay_buffer)
            progress['episodes'] += 1
            progress['training_primitives'] += int(trajectory['env_steps'])
            count = len(trajectory['rewards'])*variant.multi_grad_step
            if len(replay_buffer) > variant.start_online_updates:
                for _ in range(count):
                    i = progress['updates']
                    if i == 0:
                        capture.role, capture.updates = 'query', i
                        upstream.perform_control_eval(agent, capture, i, variant, wandb_logger, agent_dp)
                        capture.finish()
                        agent.perform_eval(variant, i, wandb_logger, replay_buffer, iterator, capture)
                    batch = next(iterator)
                    metrics = agent.update(shard_fn(batch) if shard_fn else batch)
                    progress['updates'] += 1
                    i = progress['updates']
                    if i % variant.log_interval == 0:
                        finite = {key: float(np.asarray(value)) for key, value in metrics.items()
                                  if np.asarray(value).ndim == 0}
                        if not all(np.isfinite(value) for value in finite.values()):
                            raise RuntimeError('Nonfinite SAC update metric')
                        wandb_logger.log({'training/'+key: value for key, value in finite.items()}, i)
                    if i % variant.eval_interval == 0:
                        wandb_logger.log(dict(num_online_samples=len(replay_buffer),
                            num_online_trajs=progress['episodes'], env_steps=progress['training_primitives']), i)
                        capture.role, capture.updates = 'query', i
                        upstream.perform_control_eval(agent, capture, i, variant, wandb_logger, agent_dp)
                        capture.finish()
                        agent.perform_eval(variant, i, wandb_logger, replay_buffer, iterator, capture)
            if progress['updates'] == 0 or progress['updates']-progress['last_checkpoint_updates'] >= 10000:
                progress['last_checkpoint_updates'] = progress['updates']
                save_complete(output, agent, replay_buffer, iterator, agent_dp, progress, contract)
            atomic_json(output/'progress.json', dict(progress, replay_rows=len(replay_buffer),
                physical_actions=capture.charged, time_unix=time.time(), motor_frozen=True))
            print(json.dumps(dict(event='training', **progress, replay_rows=len(replay_buffer),
                                  actions=capture.charged)), flush=True)
        save_complete(output, agent, replay_buffer, iterator, agent_dp, progress, contract)
        atomic_json(output/'result.json', dict(progress, stop_reason=stop_reason,
            completed=progress['updates'] > variant.max_steps, physical_actions=capture.charged,
            replay_rows=len(replay_buffer), one_training_seed=True))
    finally:
        agent_dp.infer = original_infer
        after = parameter_signature(agent_dp._model)
        atomic_json(output/'motor-after.json', dict(before=before, after=after, equal=before == after))
        env.close()
        if before != after:
            raise RuntimeError('Frozen original Pi0 parameters changed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsrl-root', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--checkpoint-receipt', required=True, type=Path)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--deadline-unix', required=True, type=float)
    args = parser.parse_args(argv)
    if sys.platform != 'linux' or not socket.gethostname().startswith('job-'):
        raise RuntimeError('Run only in the allocated Linux job')
    if not 600 < args.deadline_unix-time.time() <= MAX_SECONDS:
        raise ValueError('Require a live bounded allocation deadline')
    for path, pin in ((args.dsrl_root, DSRL), (args.dsrl_root/'openpi', OPENPI),
                      (args.dsrl_root/'LIBERO', LIBERO)):
        if subprocess.check_output(['git','rev-parse','HEAD'], cwd=path, text=True).strip() != pin:
            raise ValueError('Original upstream source pin differs')
        if subprocess.check_output(['git','diff','--name-only'], cwd=path, text=True).strip():
            raise ValueError('Original upstream source has tracked changes')
    repo = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip() != args.source_commit
            or subprocess.check_output(['git','status','--porcelain'], cwd=repo, text=True).strip()):
        raise ValueError('Require exact clean runner source')
    receipt = json.loads(args.checkpoint_receipt.read_text())
    if receipt.get('status') != 'verified' or receipt['checkpoint'] != 's3://openpi-assets/checkpoints/pi0_libero':
        raise ValueError('Require the original checkpoint receipt')
    for row in receipt['files']:
        if file_hash(args.checkpoint/row['path']) != row['sha256']:
            raise ValueError('Original checkpoint file hash differs')
    args.output.mkdir(parents=True, exist_ok=False)
    contract = dict(format='original_pi0_task57_replication_v1', source_commit=args.source_commit,
        dsrl_commit=DSRL, openpi_commit=OPENPI, libero_commit=LIBERO, seed=0,
        task_id=57, motor='pi0_libero', native_horizon=50, query_frequency=20,
        checkpoint_receipt_sha256=file_hash(args.checkpoint_receipt),
        max_actions=MAX_ACTIONS, max_gpu_seconds=MAX_SECONDS, max_bytes=MAX_BYTES,
        departures=['local scalar/media handling', 'explicit equivalent prefetch recovery',
                    'complete-boundary recovery and resource stopping', 'project-local compilation cache'],
        recipe='unmodified upstream collection/relabel/update/evaluation, exact within-burst10k evaluations')
    atomic_json(args.output/'contract.json', contract)
    sys.path.insert(0, str(args.dsrl_root))
    os.environ.update(DSRL_REPLICATION_OUTPUT=str(args.output),
        DSRL_REPLICATION_DEADLINE=str(args.deadline_unix), EXP=str(args.output/'upstream'))
    import jax
    if len(jax.devices('gpu')) != 1:
        raise RuntimeError('Require exactly one allocated JAX GPU')
    from jax.experimental.compilation_cache import compilation_cache
    original_cache = compilation_cache.initialize_cache
    compilation_cache.initialize_cache = lambda _: original_cache(str(args.output/'jax-cache'))
    from examples import train_sim
    train_sim.trajwise_alternating_training_loop = training_loop
    train_sim.WandBLogger = lambda *unused, **kwargs: LocalLogger(args.output)
    original_download = train_sim.download.maybe_download
    train_sim.download.maybe_download = lambda url, **kwargs: args.checkpoint if url == receipt['checkpoint'] else original_download(url, **kwargs)
    sys.argv = ['launch_train_sim', '--algorithm','pixel_sac','--env','libero',
        '--prefix','original_pi0_task57','--wandb_project','local_original_pi0_task57',
        '--batch_size','256','--discount','0.999','--seed','0','--max_steps','500000',
        '--eval_interval','10000','--log_interval','500','--eval_episodes','10',
        '--multi_grad_step','20','--start_online_updates','500','--resize_image','64',
        '--action_magnitude','1.0','--query_freq','20','--hidden_dims','128']
    runpy.run_module('examples.launch_train_sim', run_name='__main__')


if __name__ == '__main__':
    main()
