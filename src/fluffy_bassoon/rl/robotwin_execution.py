"""Measured native chunk execution, shared by RoboTwin train and evaluation.

The pinned fast controller normally reports requested targets. The passive
accounting patch adds a separate field without changing RLinf's loss masks.
This reader never substitutes requested counts for missing measurements.
"""
from dataclasses import dataclass
from math import isfinite
from numbers import Integral


@dataclass(frozen=True)
class ChunkExecution:
    requested_targets: int
    completed_targets: int
    delivered_targets: int
    partial_targets: int
    physics_steps: int
    timestep_seconds: float

    @property
    def valid_target_mask(self):
        """A terminal partly delivered target belongs to the executed prefix."""
        return tuple(i < self.delivered_targets for i in range(self.requested_targets))

    @property
    def duration_seconds(self):
        return self.physics_steps * self.timestep_seconds


def chunk_execution(info, *, requested_targets, timestep_seconds):
    """Decode measured metadata; native learner masks remain a separate input.

    ``timestep_seconds`` comes from the actual scene, not the model frequency.
    Completed target boundaries and integrated physics steps are not equivalent.
    """
    if not isinstance(info, dict) or not isinstance(info.get('fb2_execution'), dict):
        raise ValueError('Measured fb2_execution metadata required; requested counts are insufficient')
    measured = info['fb2_execution']
    names = ('requested_targets', 'completed_targets', 'delivered_targets',
             'partial_targets', 'physics_steps')
    values = {}
    for name in names:
        value = measured.get(name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f'Require a nonnegative measured integer: {name}')
        values[name] = int(value)
    if (isinstance(requested_targets, bool) or not isinstance(requested_targets, Integral)
            or requested_targets < 1 or values['requested_targets'] != requested_targets):
        raise ValueError('Measured chunk length differs from the submitted chunk')
    if (isinstance(timestep_seconds, bool) or not isinstance(timestep_seconds, (int, float))
            or not isfinite(timestep_seconds) or timestep_seconds <= 0):
        raise ValueError('Require the finite positive native scene timestep')
    completed, delivered, partial = (values[k] for k in names[1:4])
    if partial not in (0, 1) or completed + partial != delivered or delivered > requested_targets:
        raise ValueError('Inconsistent completed/partial/delivered target prefix')
    if (values['physics_steps'] == 0) != (delivered == 0):
        raise ValueError('Delivered target prefix and measured physics steps disagree')
    official = info.get('executed_action_count')
    if official is not None and (isinstance(official, bool) or not isinstance(official, Integral)
                                 or int(official) != delivered):
        raise ValueError('Full-observation native count disagrees with passive measurement')
    return ChunkExecution(**values, timestep_seconds=float(timestep_seconds))
