"""Frozen decoded controllers connected to measured RoboTwin physics.

One proposal retains every native motor slot; completed lanes take no physics
steps and acquire no history. The caller escrows returned commands before
dispatch and persists returned transitions, using the same API for train/eval.
"""
import numpy as np

from .production_process import _cpu_tree


class RoboTwinChunkSession:
    def __init__(self, runtime, controller):
        if (runtime.pool.num_workers != controller.native_batch
                or runtime.horizon != controller.layout.chunk_length
                or controller.layout.action_dim != 14):
            raise ValueError('Bind controller shape and fixed native slots to the measured runtime')
        self.runtime, self.controller = runtime, controller
        self.starts = np.zeros(controller.native_batch, np.bool_)
        self.pending, self.failed, self.closed = None, False, False

    def _check(self):
        if self.failed or self.closed:
            raise RuntimeError('Session failed/closed; preserve physical evidence without automatic replay')
        self.runtime._check()
        self.controller._check()

    def reset(self, tasks, seeds, *, case_ids, indices=None, new_trials=False):
        self._check()
        if self.pending is not None or type(new_trials) is not bool:
            raise ValueError('Finish the pending decision and declare trial boundaries explicitly')
        lanes = self.runtime._indices(indices)
        try:
            receipts = self.runtime.reset(tasks, seeds, case_ids=case_ids, indices=lanes)
            if new_trials:
                self.controller.reset_trials(lanes)
            self.starts[lanes] = True
            return receipts
        except BaseException:
            self.failed = True
            raise

    def propose(self, *, deterministic=True, generator=None):
        self._check()
        if self.pending is not None:
            raise RuntimeError('Observe the pending physical decision before another query')
        active = np.asarray(self.runtime._active, np.bool_)
        try:
            bases, features, metadata = self.runtime.infer(fixed_native_batch=True)
            commands = self.controller.propose(features, bases, episode_starts=self.starts & active,
                active=active, deterministic=deterministic, generator=generator)
            self.pending = dict(commands=_cpu_tree(commands), active=active.copy())
            return commands, features, metadata
        except BaseException:
            self.failed = True
            raise

    def step(self, commands):
        self._check()
        if self.pending is None:
            raise RuntimeError('Physical dispatch requires the exact pending controller command')
        try:
            issued = _cpu_tree(commands)
            if (not isinstance(issued, np.ndarray) or issued.dtype != np.float32
                    or not np.array_equal(issued, self.pending['commands'])):
                raise ValueError('Bind the full issued command before physical dispatch')
            transitions = self.runtime.step(issued)
            batch, horizon = self.controller.native_batch, self.controller.layout.chunk_length
            masks = np.zeros((batch, horizon), np.bool_)
            rewards, seconds = np.zeros(batch, np.float32), np.zeros(batch, np.float64)
            terminated, truncated = np.zeros(batch, np.bool_), np.zeros(batch, np.bool_)
            seen = set()
            for transition in transitions:
                native = transition['native_batch']
                lane = native['row']
                if (lane in seen or not 0 <= lane < batch or not self.pending['active'][lane]
                        or native['batch_size'] != batch or native['lanes'] != list(range(batch))
                        or not native['fixed_native_batch']
                        or not np.array_equal(transition['requested_actions'], issued[lane])):
                    raise ValueError('Physical feedback differs from the fixed-slot proposal')
                seen.add(lane)
                masks[lane] = transition['measured_valid_target_mask']
                rewards[lane] = transition['reward']
                seconds[lane] = transition['physics_duration_seconds']
                terminated[lane], truncated[lane] = transition['terminated'], transition['truncated']
            if seen != set(np.flatnonzero(self.pending['active'])):
                raise ValueError('Missing measured feedback; never fabricate a completed decision')
            self.controller.observe(issued, masks, rewards, terminated=terminated,
                truncated=truncated, physics_seconds=seconds)
            self.starts[self.pending['active']] = False
            self.pending = None
            return transitions
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if not self.closed:
            try:
                self.runtime.close()
            finally:
                self.closed = True
