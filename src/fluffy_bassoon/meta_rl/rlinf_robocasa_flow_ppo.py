"""Reward semantics for native RoboCasa action-expert PPO."""
import numpy as np


class GlobalActionJournal:
    """One locked action ceiling across old and new environment world sizes."""

    def __init__(self, root, *, rank, world, attempt, maximum_actions):
        from pathlib import Path
        from .rlinf_native_ppo import write
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f'rank-{rank:02d}' / f'{attempt}.json'
        self.limit, self.rank, self.world = maximum_actions, rank, world
        self.rows = []
        self._local_charged = 0
        self._file_cache = {}
        if self.path.exists():
            raise ValueError('Never overwrite an existing attempt journal')
        write(self.path, self._value())

    def _value(self):
        return dict(format='rlinf_native_ppo_actions_v1', scope='global_all_world_sizes',
            maximum_all_in_actions=self.limit, rank=self.rank, world=self.world,
            calls=self.rows)

    def charged(self):
        import json
        total = self._local_charged
        for path in self.root.glob('rank-*/*.json'):
            if path == self.path:
                continue
            stat = path.stat()
            key = (stat.st_mtime_ns, stat.st_size)
            cached = self._file_cache.get(path)
            if cached is None or cached[0] != key:
                amount = sum(row.get('executed_or_conservative', row['requested'])
                    for row in json.loads(path.read_text())['calls'])
                self._file_cache[path] = (key, amount)
            else:
                amount = cached[1]
            total += amount
        return total

    def _change(self, mutate):
        import fcntl
        from .rlinf_native_ppo import write
        with (self.root / '.action-escrow.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = mutate()
            write(self.path, self._value())
            return result

    def begin(self, requested, *, phase):
        import time
        def reserve():
            if (not isinstance(requested, int) or requested <= 0
                    or self.limit is not None and self.charged() + requested > self.limit):
                raise RuntimeError('Global action escrow cannot reserve another complete vector chunk')
            self.rows.append(dict(requested=requested, phase=phase,
                started_unix=time.time(), resolved=False))
            self._local_charged += requested
            return len(self.rows) - 1
        return self._change(reserve)

    def finish(self, cursor, executed):
        import time
        def resolve():
            row = self.rows[cursor]
            if row['resolved'] or not isinstance(executed, int) or not 0 <= executed <= row['requested']:
                raise ValueError('Invalid or duplicate simulator accounting resolution')
            self._local_charged += executed - row['requested']
            row.update(executed_or_conservative=executed, resolved=True, ended_unix=time.time())
        return self._change(resolve)


# Seed-range numbers; never renumber. 1-5 follow the 2026-11-02 portfolio,
# later tasks append. PPO tasks: robocasa-flow-sde-10m-per-task TASK-PORTFOLIO.json.
FLOW_TASK_NUMBERS = dict(OpenDrawer=1, PickPlaceCounterToCabinet=2, PickPlaceDrawerToCounter=3,
                         TurnOffStove=4, CoffeeSetupMug=5, PickPlaceCounterToStove=6,
                         CloseToasterOvenDoor=7, TurnOnSinkFaucet=8)
RESET_CONTROL_RESERVE = 128  # counted_native_factory's per-episode reset reserve


def keep_frozen_image_encoder_in_eval(model):
    """Retain the rollout vision path while the action expert trains.

    PyTorch MultiheadAttention selects a different forward path in train mode,
    even under no_grad. Freezing weights alone does not preserve that path.
    """
    image = model.img
    if any(parameter.requires_grad for parameter in image.parameters()):
        raise ValueError('Flow PPO expects a frozen image encoder')
    image.eval()


def flow_logprob_parity(logprobs, old_logprobs, *, action_dim=12, tolerance=1e-3):
    """Audit the unchanged policy's actual chunk ratio before its first SGD.

    RLinf sums all executed chunk x action coordinates for its chunk-level ratio.
    Read detached tensors only; do not recompute or replace rollout targets.
    """
    import torch
    new, old = logprobs.detach().float(), old_logprobs.detach().to(logprobs.device).float()
    if new.shape != old.shape or new.ndim != 3 or new.shape[-1] != action_dim:
        raise ValueError('Flow logprobs must match [batch, chunk, executed action dimensions]')
    delta = new - old
    finite = bool(torch.isfinite(delta).all().item())
    joint = new.sum(dim=(1, 2)) - old.sum(dim=(1, 2))
    maximum = float(joint.abs().max().item())
    return dict(passed=finite and maximum <= tolerance, finite=finite,
        shape=list(new.shape), action_dim=action_dim, tolerance=tolerance,
        max_abs_chunk_log_ratio=maximum,
        max_abs_coordinate_log_ratio=float(delta.abs().max().item()),
        mean_abs_chunk_log_ratio=float(joint.abs().mean().item()))


def flow_value_parity(values, old_values, *, tolerance=1e-3):
    """Before the first SGD, the recomputed critic must equal the rollout's.

    GAE and value clipping use rollout values; the loss trains recomputed
    ones. Native RLinf averages all denoising steps at rollout but recomputes
    only the sampled step (patches/rlinf-e1a82ebf-critic-chosen-step-value.patch).
    """
    import torch
    new, old = values.detach().float().flatten(), old_values.detach().to(values.device).float().flatten()
    if new.shape != old.shape:
        raise ValueError('Recomputed and rollout values must have one entry per sample')
    delta = (new - old).abs()
    finite = bool(torch.isfinite(delta).all().item())
    maximum = float(delta.max().item())
    return dict(passed=finite and maximum <= tolerance, finite=finite, tolerance=tolerance,
        max_abs_value_delta=maximum, mean_abs_value_delta=float(delta.mean().item()))


def flow_seeds(task):
    """Per-task disjoint ranges: TRAIN 2,x00,000+; before/after eval 6x0100."""
    number = FLOW_TASK_NUMBERS[task]
    return 2000000 + 100000 * number, 600100 + 10000 * number


_MISSING = object()


def _get(cfg, path, default=_MISSING):
    for key in path.split('.'):
        if isinstance(cfg, dict) and key not in cfg:
            if default is not _MISSING:
                return default
            raise KeyError(key)
        try:
            cfg = cfg[key]
        except KeyError:
            if default is not _MISSING:
                return default
            raise
    return cfg


def flow_contract(cfg):
    """Derive and check the per-task run arithmetic from one resolved config.

    Supports both GAE PPO (actor_critic, group_size=1, add_value_head=True)
    and critic-free GRPO (actor, group_size>=2, add_value_head=False).
    """
    train, evaluation = _get(cfg, 'env.train'), _get(cfg, 'env.eval')
    tasks, eval_tasks = list(train['task_filter']), list(evaluation['task_filter'])
    if len(tasks) != 1 or tasks != eval_tasks:
        raise ValueError('Flow PPO trains and evaluates exactly one task')
    task = tasks[0]
    horizon, chunk = int(train['max_episode_steps']), int(_get(cfg, 'actor.model.num_action_chunks'))
    prediction_horizon = int(_get(cfg, 'actor.model.openpi.action_horizon', chunk))
    if chunk < 1 or prediction_horizon < chunk:
        raise ValueError('Executed chunk must be positive and fit the predicted action horizon')
    if (int(_get(cfg, 'actor.model.openpi.action_chunk', chunk)) != chunk
            or int(_get(cfg, 'rollout.model.num_action_chunks', chunk)) != chunk
            or int(_get(cfg, 'rollout.model.openpi.action_chunk', chunk)) != chunk):
        raise ValueError('Actor/rollout executed chunks and likelihood slices must agree')
    denoising_steps = int(_get(cfg, 'actor.model.num_steps', 10))
    ignore_last = bool(_get(cfg, 'actor.model.openpi.ignore_last', False))
    if denoising_steps < (2 if ignore_last else 1):
        raise ValueError('Denoising steps must leave a selectable stochastic transition')
    if (int(_get(cfg, 'actor.model.openpi.num_steps', denoising_steps)) != denoising_steps
            or int(_get(cfg, 'rollout.model.num_steps', denoising_steps)) != denoising_steps):
        raise ValueError('Actor/rollout denoising steps must agree')
    if (horizon != int(train['max_steps_per_rollout_epoch']) or horizon != int(evaluation['max_episode_steps'])
            or horizon != int(evaluation['max_steps_per_rollout_epoch']) or horizon % chunk):
        raise ValueError('Train/eval horizons must agree and be a whole number of chunks')
    envs, rounds = int(train['total_num_envs']), int(train['rollout_epoch'])
    episodes = envs * rounds
    adv_type = str(_get(cfg, 'algorithm.adv_type', 'gae'))
    if adv_type == 'gae':
        loss_type = str(_get(cfg, 'algorithm.loss_type', 'actor_critic'))
        group_size = int(_get(cfg, 'algorithm.group_size', 1))
        value_head = bool(_get(cfg, 'actor.model.add_value_head', True))
        if loss_type != 'actor_critic' or group_size != 1 or not value_head:
            raise ValueError('GAE PPO requires loss_type=actor_critic, group_size=1 and add_value_head=True')
    elif adv_type == 'grpo':
        loss_type = str(_get(cfg, 'algorithm.loss_type', 'actor'))
        group_size = int(_get(cfg, 'algorithm.group_size', 0))
        value_head = bool(_get(cfg, 'actor.model.add_value_head', False))
        if loss_type != 'actor' or group_size < 2 or envs % group_size or value_head:
            raise ValueError('GRPO requires loss_type=actor, group_size>=2 dividing train_envs and add_value_head=False')
    else:
        raise ValueError(f'Unsupported advantage estimator: {adv_type!r}')
    if int(_get(train, 'group_size', group_size)) != group_size or int(_get(evaluation, 'group_size', 1)) != 1:
        raise ValueError('Train group_size must match algorithm.group_size and eval group_size must be 1')
    scenes_per_iteration = episodes // group_size
    samples = episodes * (horizon // chunk)
    batch, epochs = int(_get(cfg, 'actor.global_batch_size')), int(_get(cfg, 'algorithm.update_epoch'))
    if envs < 1 or rounds < 1 or batch < 1 or epochs < 1:
        raise ValueError('Episodes, rollout rounds, global batch and epochs must be positive')
    if samples % batch:
        raise ValueError('Chunk samples per iteration must fill whole global minibatches')
    iterations, eval_cases = int(_get(cfg, 'runner.max_epochs')), int(evaluation['total_num_envs'])
    optimizer_steps_per_update = samples // batch * epochs
    total_optimizer_steps = iterations * optimizer_steps_per_update
    warmup = int(_get(cfg, 'actor.optim.critic_warmup_steps', 0))
    if warmup < 0 or warmup >= total_optimizer_steps or (warmup and not value_head):
        raise ValueError('Critic warmup requires a value head and must leave actor optimizer steps')
    if int(evaluation['rollout_epoch']) != 1:
        raise ValueError('Before/after panels are one fixed case bank')
    train_seed, eval_seed = int(train['seed']), int(evaluation['seed'])
    if (train_seed, eval_seed) != flow_seeds(task):
        raise ValueError(f'{task} seeds must be {flow_seeds(task)}, got {(train_seed, eval_seed)}')
    if train_seed + iterations * scenes_per_iteration > eval_seed and eval_seed + eval_cases > train_seed:
        raise ValueError('TRAIN and eval seed ranges overlap')
    interval = int(_get(cfg, 'runner.save_interval'))
    if interval < 1 or iterations % interval:
        raise ValueError('Checkpoint interval must divide the iteration count')
    eval_interval = int(_get(cfg, 'runner.val_check_interval', iterations))
    if eval_interval < 1 or iterations % eval_interval:
        raise ValueError('Evaluation interval must divide the iteration count')
    eval_panels = 1 + iterations // eval_interval
    policy = iterations * episodes * horizon + eval_panels * eval_cases * horizon
    resets = (iterations * episodes + eval_panels * eval_cases) * RESET_CONTROL_RESERVE
    maximum = _get(cfg, 'native_pilot.maximum_actions')
    if maximum is not None and int(maximum) < policy + resets:
        raise ValueError(f'Action ceiling {maximum} is below nominal {policy} plus reset reserve {resets}')
    noise_level = float(_get(cfg, 'actor.model.openpi.noise_level', 0.3))
    if not np.isfinite(noise_level) or noise_level <= 0:
        raise ValueError('Flow-SDE noise_level must be finite and positive')
    return dict(task=task, horizon=horizon, chunk=chunk, chunks_per_episode=horizon // chunk,
                adv_type=adv_type, loss_type=loss_type, group_size=group_size,
                add_value_head=value_head,
                value_after_vlm=bool(_get(cfg, 'actor.model.openpi.value_after_vlm', False)),
                ignore_last=ignore_last, predicted_action_horizon=prediction_horizon,
                denoising_steps=denoising_steps,
                likelihood_coordinates=chunk * int(_get(cfg, 'actor.model.action_dim', 12)),
                noise_level=noise_level, update_epochs=epochs,
                train_envs=envs, rounds=rounds, episodes_per_iteration=episodes,
                scenes_per_iteration=scenes_per_iteration, iterations=iterations,
                eval_cases=eval_cases, eval_interval=eval_interval, eval_panels=eval_panels,
                optimizer_steps_per_update=optimizer_steps_per_update,
                total_optimizer_steps=total_optimizer_steps,
                critic_only_optimizer_steps=warmup,
                actor_optimizer_steps=total_optimizer_steps - warmup,
                global_minibatch_chunks=batch,
                TRAIN_policy_controls=iterations * episodes * horizon,
                gamma_per_decision=float(_get(cfg, 'algorithm.gamma', .99)),
                gae_lambda_per_decision=float(_get(cfg, 'algorithm.gae_lambda', .95)),
                train_seed=train_seed, eval_seed=eval_seed, checkpoint_interval=interval,
                nominal_policy_controls=policy, reset_reserve_controls=resets,
                maximum_actions=None if maximum is None else int(maximum),
                physics_tick_limit=(horizon + RESET_CONTROL_RESERVE) * 128)


def train_case_seeds(contract, *, iteration, round_index, rank, local_envs):
    """TRAIN seeds for any envs-per-rank / rollout-epoch layout (grouped for GRPO)."""
    if not 0 <= round_index < contract['rounds'] or not 0 <= iteration < contract['iterations']:
        raise ValueError('Round or iteration outside the declared schedule')
    group = int(contract.get('group_size', 1))
    if group < 1 or local_envs % group or contract['train_envs'] % group:
        raise ValueError('Local and total train environments must be positive multiples of group_size')
    scenes_per_iteration = contract['episodes_per_iteration'] // group
    scenes_per_round = contract['train_envs'] // group
    local_scenes = local_envs // group
    start = (contract['train_seed'] + iteration * scenes_per_iteration
             + round_index * scenes_per_round + rank * local_scenes)
    return [start + (i // group) for i in range(local_envs)]


def flow_episode_ids(task, seeds, *, group_size=1):
    """Unique episode IDs preserving `task/seed` for PPO and `task/seed/gNN` for GRPO."""
    seeds = list(seeds)
    group = int(group_size)
    if group < 1 or len(seeds) % group:
        raise ValueError('Seed count must be a positive multiple of group_size')
    if group == 1:
        return [f'{task}/{seed}' for seed in seeds]
    for start in range(0, len(seeds), group):
        if len(set(seeds[start:start + group])) != 1:
            raise ValueError(f'GRPO group [{start}:{start + group}] must share one scene seed')
    return [f'{task}/{seed}/g{i % group:02d}' for i, seed in enumerate(seeds)]


def verify_grpo_group_reset(snapshot, group_size):
    """Confirm every GRPO group shares one scene seed, physical state and prompt."""
    seeds = list(snapshot['case']['seeds'])
    physical = list(snapshot['physical'])
    prompts = list(snapshot['task_descriptions'])
    states = np.asarray(snapshot['states'])
    group = int(group_size)
    if group < 2 or len(seeds) % group:
        raise ValueError('GRPO group verification requires group_size>=2 dividing batch size')
    for start in range(0, len(seeds), group):
        end = start + group
        if len(set(seeds[start:end])) != 1:
            raise ValueError(f'GRPO group [{start}:{end}] seeds differ: {seeds[start:end]}')
        if any(item != physical[start] for item in physical[start + 1:end]):
            raise ValueError(f'GRPO group [{start}:{end}] physical fingerprints differ')
        if any(prompt != prompts[start] for prompt in prompts[start + 1:end]):
            raise ValueError(f'GRPO group [{start}:{end}] task descriptions differ')
        if not np.array_equal(states[start:end], np.broadcast_to(states[start:start + 1], states[start:end].shape)):
            raise ValueError(f'GRPO group [{start}:{end}] initial public states differ')


def first_success_reward(success, previously_succeeded, coefficient=1.0):
    """Pay once, including transient success within an action chunk.

    A signed difference of success flags can cancel a successful primitive
    when success is lost later in the same H50 chunk. The episode's first
    success is absorbing for reward even though accounting steps the horizon.
    """
    success = np.asarray(success, dtype=bool)
    previously_succeeded = np.asarray(previously_succeeded, dtype=bool)
    if success.shape != previously_succeeded.shape:
        raise ValueError('Success latch shape differs from native batch')
    return (success & ~previously_succeeded).astype(np.float32) * float(coefficient)
