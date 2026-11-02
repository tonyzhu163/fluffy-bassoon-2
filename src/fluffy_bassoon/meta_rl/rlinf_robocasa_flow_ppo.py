"""Reward semantics for native RoboCasa action-expert PPO."""
import numpy as np


class GlobalActionJournal:
    """One locked action ceiling across old and new environment world sizes."""

    def __init__(self, root, *, rank, world, attempt, maximum_actions=1000000):
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
            if not isinstance(requested, int) or requested <= 0 or self.charged() + requested > self.limit:
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
