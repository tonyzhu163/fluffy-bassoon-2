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
        if self.path.exists():
            raise ValueError('Never overwrite an existing attempt journal')
        write(self.path, self._value())

    def _value(self):
        return dict(format='rlinf_native_ppo_actions_v1', scope='global_all_world_sizes',
            maximum_all_in_actions=self.limit, rank=self.rank, world=self.world,
            calls=self.rows)

    def charged(self):
        import json
        return sum(row.get('executed_or_conservative', row['requested'])
            for path in self.root.glob('rank-*/*.json')
            for row in json.loads(path.read_text())['calls'])

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
            return len(self.rows) - 1
        return self._change(reserve)

    def finish(self, cursor, executed):
        import time
        def resolve():
            row = self.rows[cursor]
            if row['resolved'] or not isinstance(executed, int) or not 0 <= executed <= row['requested']:
                raise ValueError('Invalid or duplicate simulator accounting resolution')
            row.update(executed_or_conservative=executed, resolved=True, ended_unix=time.time())
        return self._change(resolve)


# Portfolio task numbers (TASK-PORTFOLIO.json order). Replacements append.
FLOW_TASK_NUMBERS = dict(OpenDrawer=1, PickPlaceCounterToCabinet=2, PickPlaceDrawerToCounter=3,
                         TurnOffStove=4, CoffeeSetupMug=5)
RESET_CONTROL_RESERVE = 128  # counted_native_factory's per-episode reset reserve


def flow_seeds(task):
    """Per-task disjoint ranges: TRAIN 2,x00,000+; before/after eval 6x0100."""
    number = FLOW_TASK_NUMBERS[task]
    return 2000000 + 100000 * number, 600100 + 10000 * number


def _get(cfg, path):
    for key in path.split('.'):
        cfg = cfg[key]
    return cfg


def flow_contract(cfg):
    """Derive and check the per-task run arithmetic from one resolved config.

    Everything the runtime previously hard-coded for OpenDrawer/400/18 comes
    from here, so a changed config cannot silently disagree with the sidecars.
    """
    train, evaluation = _get(cfg, 'env.train'), _get(cfg, 'env.eval')
    tasks, eval_tasks = list(train['task_filter']), list(evaluation['task_filter'])
    if len(tasks) != 1 or tasks != eval_tasks:
        raise ValueError('Flow PPO trains and evaluates exactly one task')
    task = tasks[0]
    horizon, chunk = int(train['max_episode_steps']), int(_get(cfg, 'actor.model.num_action_chunks'))
    if (horizon != int(train['max_steps_per_rollout_epoch']) or horizon != int(evaluation['max_episode_steps'])
            or horizon != int(evaluation['max_steps_per_rollout_epoch']) or horizon % chunk):
        raise ValueError('Train/eval horizons must agree and be a whole number of chunks')
    envs, rounds = int(train['total_num_envs']), int(train['rollout_epoch'])
    episodes = envs * rounds
    samples = episodes * (horizon // chunk)
    batch, epochs = int(_get(cfg, 'actor.global_batch_size')), int(_get(cfg, 'algorithm.update_epoch'))
    if samples % batch:
        raise ValueError('Chunk samples per iteration must fill whole global minibatches')
    iterations, eval_cases = int(_get(cfg, 'runner.max_epochs')), int(evaluation['total_num_envs'])
    if int(evaluation['rollout_epoch']) != 1:
        raise ValueError('Before/after panels are one fixed case bank')
    train_seed, eval_seed = int(train['seed']), int(evaluation['seed'])
    if (train_seed, eval_seed) != flow_seeds(task):
        raise ValueError(f'{task} seeds must be {flow_seeds(task)}, got {(train_seed, eval_seed)}')
    if train_seed + iterations * episodes > eval_seed and eval_seed + eval_cases > train_seed:
        raise ValueError('TRAIN and eval seed ranges overlap')
    interval = int(_get(cfg, 'runner.save_interval'))
    if interval < 1 or iterations % interval:
        raise ValueError('Checkpoint interval must divide the iteration count')
    policy = iterations * episodes * horizon + 2 * eval_cases * horizon
    resets = (iterations * episodes + 2 * eval_cases) * RESET_CONTROL_RESERVE
    maximum = _get(cfg, 'native_pilot.maximum_actions')
    if maximum is not None and int(maximum) < policy + resets:
        raise ValueError(f'Action ceiling {maximum} is below nominal {policy} plus reset reserve {resets}')
    return dict(task=task, horizon=horizon, chunk=chunk, chunks_per_episode=horizon // chunk,
                train_envs=envs, rounds=rounds, episodes_per_iteration=episodes, iterations=iterations,
                eval_cases=eval_cases, optimizer_steps_per_update=samples // batch * epochs,
                train_seed=train_seed, eval_seed=eval_seed, checkpoint_interval=interval,
                nominal_policy_controls=policy, reset_reserve_controls=resets,
                maximum_actions=None if maximum is None else int(maximum),
                physics_tick_limit=(horizon + RESET_CONTROL_RESERVE) * 128)


def train_case_seeds(contract, *, iteration, round_index, rank, local_envs):
    """Unique TRAIN seeds for any envs-per-rank / rollout-epoch layout."""
    if not 0 <= round_index < contract['rounds'] or not 0 <= iteration < contract['iterations']:
        raise ValueError('Round or iteration outside the declared schedule')
    start = (contract['train_seed'] + iteration * contract['episodes_per_iteration']
             + round_index * contract['train_envs'] + rank * local_envs)
    return [start + i for i in range(local_envs)]


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
