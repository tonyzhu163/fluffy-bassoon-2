"""Reward semantics for native RoboCasa action-expert PPO."""
import numpy as np


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
