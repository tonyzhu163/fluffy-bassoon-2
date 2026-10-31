"""Small, metadata-only contracts for the official RLinf action-expert PPO pilot.

The learner, GAE, masks, FSDP and sampler remain in the pinned RLinf checkout.
Runtime imports live in rlinf_native_ppo_runtime, never in this metadata module.
"""
import hashlib
import json
import os
from pathlib import Path
import time

RLINF_PIN = 'c605b9d55ed93a180119b099e6f836abff293b08'
ROBOTWIN_PIN = 'dca9ec682688821c42944eacd7f5dd9bfb15396f'
SFT_REVISION = 'fa8df6ed103db0f5549c122f3a17c00ba6426c98'
SFT_SHARDS = {
    'model-00001-of-00003.safetensors': '193732b06782c0c6adbf94f16b8ecd6141a1a9d61004fd57c75cc2731c59fb07',
    'model-00002-of-00003.safetensors': '23395e59338f279566a63836b3661e9b88369f6472762faa6afe0b4ed56a2e17',
    'model-00003-of-00003.safetensors': 'ef74f46bff64a2c81b63a8fe380648c7d780b660f67e1c9ef9b4c5c874081f41',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def validate_recipe(cfg):
    """Accept execution accumulation changes only; never reduce scientific batch."""
    from omegaconf import OmegaConf
    expected = {
        'runner.max_steps': 2, 'runner.max_epochs': 2, 'runner.only_eval': False,
        'runner.use_training_pipeline': False, 'runner.overlap_env_bootstrap': False,
        'algorithm.adv_type': 'gae', 'algorithm.loss_type': 'actor_critic',
        'algorithm.loss_agg_func': 'token-mean', 'algorithm.update_epoch': 5,
        'algorithm.gamma': .99, 'algorithm.gae_lambda': .95,
        'algorithm.normalize_advantages': True, 'algorithm.clip_ratio_high': .2,
        'algorithm.clip_ratio_low': .2, 'algorithm.clip_ratio_c': 3.,
        'algorithm.value_clip': .2, 'algorithm.huber_delta': 10.,
        'algorithm.kl_beta': 0., 'algorithm.entropy_bonus': 0.,
        'algorithm.reward_type': 'chunk_level', 'algorithm.logprob_type': 'chunk_level',
        'actor.global_batch_size': 2048, 'actor.micro_batch_size': 32,
        'actor.seed': 1234, 'actor.optim.lr': 5e-6, 'actor.optim.value_lr': 1e-4,
        'actor.optim.adam_beta1': .9, 'actor.optim.adam_beta2': .95,
        'actor.optim.adam_eps': 1e-8, 'actor.optim.weight_decay': .01,
        'actor.optim.clip_grad': 1., 'actor.optim.critic_warmup_steps': 0,
        'actor.optim.lr_scheduler': 'constant', 'actor.optim.total_training_steps': 0,
        'actor.optim.lr_warmup_steps': 0,
        'actor.model.num_action_chunks': 50, 'actor.model.action_dim': 14,
        'actor.model.num_steps': 5, 'actor.model.add_value_head': True,
        'actor.model.openpi.task': 'rl', 'actor.model.openpi.train_expert_only': True,
        'actor.model.openpi.detach_critic_input': True,
        'actor.model.openpi.config_name': 'pi05_aloha_robotwin',
        'actor.model.openpi.num_images_in_input': 3,
        'actor.model.openpi.noise_method': 'flow_sde', 'actor.model.openpi.noise_level': .3,
        'actor.fsdp_config.grad_scaler.enabled': False,
        'env.train.total_num_envs': 256, 'env.train.rollout_epoch': 4,
        'env.train.auto_reset': False, 'env.train.ignore_terminations': False,
        'env.eval.total_num_envs': 128, 'env.eval.rollout_epoch': 1,
        'env.eval.use_fixed_reset_state_ids': True,
        'env.eval.auto_reset': True, 'env.eval.ignore_terminations': True,
    }
    for mode in ('train', 'eval'):
        for field in ('max_episode_steps', 'max_steps_per_rollout_epoch', 'task_config.step_lim'):
            expected[f'env.{mode}.{field}'] = 400
        expected[f'env.{mode}.task_config.task_name'] = 'adjust_bottle'
        expected[f'env.{mode}.seed'] = 0
    for key, value in expected.items():
        actual = OmegaConf.select(cfg, key)
        if actual != value:
            raise ValueError(f'Published-batch pilot contract differs: {key}={actual!r}, expected {value!r}')
    world = int(cfg.native_pilot.world_size)
    if world not in (4, 8) or 2048 != 32 * world * (2048 // (32 * world)):
        raise ValueError('Only explicit 4/8-rank execution of the same global2048 batch is admitted')
    placement = cfg.cluster.component_placement
    if len(placement) != 1 or list(placement.values()) != [f'0-{world-1}']:
        raise ValueError('Actor/environment/rollout must share the explicit allocated rank range')
    if cfg.native_pilot.mode not in ('train', 'evaluate_existing'):
        raise ValueError('Use train or explicit evaluate_existing, never only_eval')
    return dict(world_size=world, global_batch_size=2048, micro_batch_size=32,
        accumulation_steps=2048 // (32 * world), source_iterations=2,
        nominal_train_actions=819200, nominal_before_after_actions=102400,
        nominal_all_in_actions=921600, maximum_actions=1000000,
        maximum_aggregate_gpu_seconds=21600)


class ActionJournal:
    """Rank partitions share a strict 1M ceiling, including unresolved RPC intents.

    Evaluation counts full requested chunks conservatively because upstream only
    publishes valid_action_mask for training. Retrying uses a NEW attempt name
    under the SAME accounting root; old unresolved intents remain fully charged.
    """
    def __init__(self, root, *, rank, world, attempt):
        self.root = Path(root) / f'rank-{rank:02d}'
        self.root.mkdir(parents=True, exist_ok=True)
        self.limit = 1000000 // int(world)
        self.attempt = attempt
        self.path = self.root / (attempt + '.json')
        if self.path.exists():
            raise ValueError('Never overwrite an existing attempt journal')
        self.rows = []
        self.charged()  # Reject a changed rank partition before writing anything.
        self._save()

    def _save(self):
        write(self.path, dict(format='rlinf_native_ppo_actions_v1', attempt=self.attempt,
            rank_action_ceiling=self.limit, calls=self.rows))

    def charged(self):
        total = 0
        for path in self.root.glob('*.json'):
            value = json.loads(path.read_text())
            if value['rank_action_ceiling'] != self.limit:
                raise ValueError('Cannot change rank partition while reusing action escrow')
            total += sum(row.get('executed_or_conservative', row['requested']) for row in value['calls'])
        return total

    def begin(self, requested, *, phase):
        if not isinstance(requested, int) or requested <= 0 or self.charged() + requested > self.limit:
            raise RuntimeError('Action escrow cannot reserve another complete vector chunk')
        self.rows.append(dict(requested=requested, phase=phase, started_unix=time.time(), resolved=False))
        self._save()  # Before the simulator call.
        return len(self.rows) - 1

    def finish(self, cursor, executed):
        row = self.rows[cursor]
        if row['resolved'] or not isinstance(executed, int) or not 0 <= executed <= row['requested']:
            raise ValueError('Invalid or duplicate simulator accounting resolution')
        row.update(executed_or_conservative=executed, resolved=True, ended_unix=time.time())
        self._save()


def snapshot_rng():
    import random
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    import random
    import numpy as np
    import torch
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def tensor_hash(value):
    """Initial public observation identity, including shape and dtype."""
    import numpy as np
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif isinstance(item, (torch.Tensor, np.ndarray)):
            array = item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item
            digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def checked_checkout(path, expected):
    import subprocess
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip() != expected
            or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                cwd=path, text=True).strip()):
        raise ValueError(f'Expected exact clean checkout {expected}: {path}')


def prior_gpu_charge(funding):
    seen, charge = set(), 0
    for row in funding.get('prior_attempts', []):
        path = Path(row['clock_path']).resolve()
        if path in seen or sha(path) != row['sha256']:
            raise ValueError('Duplicate or changed prior GPU clock')
        seen.add(path)
        clock = json.loads(path.read_text())
        if clock.get('status') != 'finished' or clock.get('maximum_aggregate_gpu_seconds') != 21600:
            raise ValueError('Prior attempt must be terminal in the same cumulative21600GPU-s component')
        charge += int(clock['charged_gpu_seconds'])
    if not 0 <= charge < 21600:
        raise ValueError('No cumulative GPU allowance remains')
    return charge


def preflight(args):
    """Called in the allocation, before any actor, simulator or Ray cluster."""
    import math
    import socket
    import sys
    if sys.platform != 'linux' or not socket.gethostname().startswith('job-'):
        raise ValueError('Require an actual allocated Baige Linux container')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    clock = dict(format='rlinf_native_ppo_gpu_clock_v1', source=args.source,
        run_id=root.name, started_unix=args.started_unix, status='started',
        maximum_aggregate_gpu_seconds=21600, gpu_count=args.world,
        startup_allowance_seconds=60, interrupted_charge_gpu_seconds=21600,
        accounting_root=str(Path(os.environ['NATIVE_PPO_ACCOUNTING_ROOT']).resolve()))
    write(root / 'worker-time.json', clock)
    funding = json.loads(Path(args.funding).read_text())
    if (funding.get('format') != 'rlinf_native_ppo_pilot_funding_v1'
            or funding.get('approved') is not True or funding.get('source_commit') != args.source
            or funding.get('run_id') != root.name or funding.get('gpu_count') != args.world
            or funding.get('maximum_gpu_seconds') != 21600
            or funding.get('maximum_actions') != 1000000
            or funding.get('additional_storage_bytes') != 64 * 1024 ** 3):
        raise ValueError('Explicit source/run/resource funding differs; no funding is created here')
    prior = prior_gpu_charge(funding)
    accounting = str(Path(os.environ['NATIVE_PPO_ACCOUNTING_ROOT']).resolve())
    if funding.get('action_accounting_root') != accounting:
        raise ValueError('Funding must bind the one cumulative physical-action journal root')
    for row in funding.get('prior_attempts', []):
        previous = json.loads(Path(row['clock_path']).read_text())
        if previous.get('accounting_root') != accounting or previous.get('gpu_count') != args.world:
            raise ValueError('Retries/eval-existing must retain the same action root and rank partition')
    resume = os.environ.get('NATIVE_PPO_RESUME_DIR', '')
    if funding.get('resume_dir', '') != resume:
        raise ValueError('Boundary resume must be explicit in reviewed funding')
    if resume:
        boundary_path = Path(resume) / 'pilot-boundary.json'
        if not prior or sha(boundary_path) != funding.get('resume_boundary_sha256'):
            raise ValueError('Resume boundary requires its immutable receipt SHA and all prior worker costs')
        boundary = json.loads(boundary_path.read_text())
        if boundary.get('completed') is not True or boundary.get('global_step') not in (0, 1, 2):
            raise ValueError('Only complete two-iteration pilot optimizer boundaries are admitted')
        for item in [*boundary['workers'], *boundary['actor_checkpoint_files']]:
            if sha(item['path']) != item['sha256']:
                raise ValueError('Checkpoint/RNG bytes changed before resume')
    checked_checkout(args.code, args.source)
    checked_checkout(os.environ['REPO_PATH'], RLINF_PIN)
    checked_checkout(os.environ['ROBOTWIN_PATH'], ROBOTWIN_PIN)
    qualified = funding['cpu_qualification']
    if sha(qualified['path']) != qualified['sha256']:
        raise ValueError('CPU qualification receipt changed')
    qualification = json.loads(Path(qualified['path']).read_text())
    if qualification.get('passed') is not True or qualification.get('source_commit') != args.source:
        raise ValueError('Require allocated CPU qualification of this exact derivative source')
    model = Path(os.environ['NATIVE_PPO_MODEL'])
    mode = os.environ.get('NATIVE_PPO_MODE', 'train')
    if funding.get('mode', 'train') != mode:
        raise ValueError('Pilot mode differs from its explicit funding')
    if mode == 'train':
        if os.environ['NATIVE_PPO_INITIALIZER_REVISION'] != SFT_REVISION:
            raise ValueError('Training initializer must be released fa8 SFT')
        model_hashes = {name: sha(model / name) for name in SFT_SHARDS}
        if model_hashes != SFT_SHARDS:
            raise ValueError('Released SFT initializer bytes differ')
    elif mode == 'evaluate_existing':
        ready = model / 'READY.json'
        if (os.environ['NATIVE_PPO_INITIALIZER_REVISION'] != '1223d8b9c387eb22c51350900d842d0f3b86cf14'
                or sha(ready) != '52ebf3a2f84cba5317fbc0bd9bed1e5c37c338309bb532acda1e2dd9c2235a39'):
            raise ValueError('Explicit public PPO evaluation requires the separately staged released checkpoint')
        model_hashes = dict(ready_sha256=sha(ready))
        # Reserve the ENTIRE optional panel before model/environment creation;
        # do not spend its first chunks then discover the primary used escrow.
        limit = 1000000 // args.world
        for rank in range(args.world):
            spent = 0
            for path in (Path(accounting) / f'rank-{rank:02d}').glob('*.json'):
                journal = json.loads(path.read_text())
                if journal['rank_action_ceiling'] != limit:
                    raise ValueError('Optional panel cannot change the old rank partition')
                spent += sum(call.get('executed_or_conservative', call['requested']) for call in journal['calls'])
            if limit - spent < 57600 // args.world:
                raise ValueError('No complete optional128-case panel plus native setup/reset allowance remains')
    else:
        raise ValueError('Unknown pilot mode')
    norm = model / 'physical-intelligence/robotwin/norm_stats.json'
    if sha(norm) != 'bff5fcadaba626c28c1c2350d3baa7f1a04befc9f8b73f3e558a09c1b9c6ee52':
        raise ValueError('ALOHA14D norm stats differ')
    seed_paths = [Path(os.environ['REPO_PATH']) / 'rlinf/envs/sim/robotwin/seeds' / name
        for name in ('train_seeds.json', 'eval_seeds.json')]
    seeds = {str(path): sha(path) for path in seed_paths}
    if list(seeds.values()) != ['55b25866308d4e7da486282982ead4d5857b53fd977f659b216045f1950b45e2',
            '194164f7380fd7cad2a8940ca93def01c2be865da265e4af1c463d73b2aa482f']:
        raise ValueError('Official fixed seed inventories changed')
    seconds = math.floor((21600 - prior) / args.world) - 60
    deadline = args.started_unix + seconds
    if args.world not in (4, 8) or deadline <= time.time() + 180:
        raise ValueError('No complete startup/teardown allowance remains')
    clock.update(prior_gpu_charge_seconds=prior, deadline_unix=deadline,
        maximum_current_attempt_gpu_seconds=21600-prior,
        accounting_root=accounting)
    write(root / 'worker-time.json', clock)
    write(root / 'preflight.json', dict(passed=True, source_commit=args.source,
        rlinf_commit=RLINF_PIN, robotwin_commit=ROBOTWIN_PIN,
        funding_path=str(Path(args.funding).resolve()), funding_sha256=sha(args.funding),
        python=sys.executable, initializer_path=str(model), model_hashes=model_hashes,
        norm_stats_sha256=sha(norm), seed_file_hashes=seeds, deadline_unix=deadline,
        mode=mode, source6_pretraining_admission=False))


def finish(output, exit_code):
    import math
    path = Path(output) / 'worker-time.json'
    clock = json.loads(path.read_text())
    charge = int(clock['gpu_count']) * math.ceil(time.time() - clock['started_unix'] + 60)
    clock.update(status='finished', ended_unix=time.time(), exit_code=exit_code,
        charged_gpu_seconds=charge,
        cumulative_gpu_seconds=charge + clock.get('prior_gpu_charge_seconds', 0))
    clock['resource_cap_exceeded'] = clock['cumulative_gpu_seconds'] > 21600
    write(path, clock)


def cpu_qualify(args):
    import socket
    import subprocess
    import sys
    import torch
    from .vla_source40_cpu_qualification import cgroup_resources, require_zero_gpu
    if (sys.platform != 'linux' or not socket.gethostname().startswith('job-')
            or sys.version_info[:2] != (3, 11)
            or Path(sys.prefix).resolve() != Path('/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1').resolve()):
        raise ValueError('Use identified allocated CPU container and retained native Python3.11 runtime')
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    write(root/'cpu-worker-clock.json', dict(status='started', started_unix=args.started_unix,
        maximum_cpu_worker_seconds=1800, charged_gpu_seconds=0, source_commit=args.source))
    attestation = require_zero_gpu(torch)
    attestation['resources'] = cgroup_resources()
    write(root / 'zero-gpu-attestation.json', attestation)
    checked_checkout(args.code, args.source)
    checked_checkout(os.environ['REPO_PATH'], RLINF_PIN)
    checked_checkout(os.environ['ROBOTWIN_PATH'], ROBOTWIN_PIN)
    rc = 1
    try:
        with (root / 'tests.log').open('x') as log:
            result = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_rlinf_native_ppo.py'],
                cwd=args.code, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT,
                timeout=max(1, args.started_unix + 1700 - time.time()), check=False)
        rc = result.returncode
    finally:
        write(root / 'cpu-qualification-summary.json', dict(passed=rc == 0, exit_code=rc,
            source_commit=args.source, runtime=sys.executable, started_unix=args.started_unix,
            ended_unix=time.time(), charged_gpu_seconds=0,
            charged_cpu_worker_seconds=int(time.time()-args.started_unix)+60,
            test_log_path=str(root / 'tests.log'), test_log_sha256=sha(root / 'tests.log')
                if (root / 'tests.log').is_file() else None,
            model_construction=False, simulator_actions=0, optimizer_updates=0))
    if rc:
        raise RuntimeError('Allocated native entry/config CPU qualification failed; retain its log')


def finish_cpu(output, exit_code):
    path = Path(output)/'cpu-worker-clock.json'
    if not path.is_file():
        return
    clock = json.loads(path.read_text())
    clock.update(status='finished', ended_unix=time.time(), exit_code=exit_code,
        charged_cpu_worker_seconds=int(time.time()-clock['started_unix'])+60)
    write(path, clock)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='operation', required=True)
    for name in ('preflight', 'cpu'):
        item = subs.add_parser(name)
        item.add_argument('--code', type=Path, required=True)
        item.add_argument('--source', required=True)
        item.add_argument('--output', type=Path, required=True)
        item.add_argument('--started-unix', type=float, required=True)
        if name == 'preflight':
            item.add_argument('--world', type=int, choices=(4, 8), required=True)
            item.add_argument('--funding', type=Path, required=True)
    for name in ('finish', 'finish-cpu'):
        item = subs.add_parser(name)
        item.add_argument('--output', type=Path, required=True)
        item.add_argument('--exit-code', type=int, required=True)
    args = parser.parse_args()
    if args.operation == 'preflight':
        preflight(args)
    elif args.operation == 'cpu':
        cpu_qualify(args)
    elif args.operation == 'finish':
        finish(args.output, args.exit_code)
    else:
        finish_cpu(args.output, args.exit_code)


if __name__ == '__main__':
    main()
