"""Measured native chunk execution, shared by RoboTwin train and evaluation.

The pinned fast controller normally reports requested targets. The passive
accounting patch adds a separate field without changing RLinf's loss masks.
This reader never substitutes requested counts for missing measurements.
"""
import ast
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from math import isfinite
from numbers import Integral
from pathlib import Path
from types import MethodType


def load_accounted_controller(source, expected_sha256):
    """Compile the qualified job-owned method without importing another task."""
    source = Path(source)
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('Job-owned controller source changed')
    nodes = [node for node in ast.walk(ast.parse(payload))
             if isinstance(node, ast.FunctionDef) and node.name == 'gen_sparse_reward_data']
    if len(nodes) != 1:
        raise ValueError('Require exactly one qualified native controller method')
    return compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec')


def install_accounted_controller(task, controller, *, metadata=False):
    """Bind passive accounting to this task instance, including recreated scenes.

    Native reward/termination and executed_action_count semantics are retained.
    Optional extra fields let RLinf carry physical duration and delivered masks
    through its ordinary info dictionary in both training and evaluation.
    """
    original = type(task).gen_sparse_reward_data
    namespace = dict(original.__globals__)
    exec(controller, namespace)
    native = MethodType(namespace['gen_sparse_reward_data'], task)
    if not metadata:
        task.gen_sparse_reward_data = native
        return

    def measured(instance, chunk_actions, *args, **kwargs):
        result = native(chunk_actions, *args, **kwargs)
        info = result[3]
        execution = chunk_execution(info, requested_targets=len(chunk_actions),
                                    timestep_seconds=float(instance.scene.get_timestep()))
        info['fb2_physics_timestep_seconds'] = execution.timestep_seconds
        info['fb2_physics_duration_seconds'] = execution.duration_seconds
        info['fb2_measured_valid_target_mask'] = execution.valid_target_mask
        # Keep native feedback before RLinf's relative reward conversion and
        # evaluation ignore_terminations/autoreset replace the learner packet.
        info['fb2_native_feedback'] = deepcopy(dict(reward=result[0],
            terminated=result[1],truncated=result[2]))
        return result

    task.gen_sparse_reward_data = MethodType(measured, task)


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


def collated_chunk_executions(info, *, num_envs, requested_targets):
    """Read native vector info, including the terminal packet before auto-reset.

    RLinf collates each field into a list and moves the complete terminal info
    under ``final_info`` on auto-reset. Its official learner mask is unrelated
    to these passive physical measurements and is never changed here.
    """
    if not isinstance(info, dict):
        raise ValueError('Require native vector execution info')
    packet = info.get('final_info', info)
    fields = ('fb2_execution', 'fb2_physics_timestep_seconds',
              'fb2_physics_duration_seconds', 'fb2_measured_valid_target_mask')
    if (not isinstance(packet, dict) or type(num_envs) is not int or num_envs < 1
            or any(not isinstance(packet.get(key), (list, tuple))
                   or len(packet[key]) != num_envs for key in fields)):
        raise ValueError('Missing or incomplete collated physical measurements')
    rows = []
    for index in range(num_envs):
        row = chunk_execution(dict(fb2_execution=packet['fb2_execution'][index]),
            requested_targets=requested_targets,
            timestep_seconds=packet['fb2_physics_timestep_seconds'][index])
        mask = packet['fb2_measured_valid_target_mask'][index]
        if (not isinstance(mask, (list, tuple))
                or any(type(value) is not bool for value in mask)
                or tuple(mask) != row.valid_target_mask
                or packet['fb2_physics_duration_seconds'][index] != row.duration_seconds):
            raise ValueError('Collated mask/duration disagrees with physical measurements')
        rows.append(row)
    return tuple(rows)


def install_wrist_near(camera_class, near):
    """Create both wrist cameras with ``near`` metres before any image is taken.

    Pinned RoboTwin hard-codes a 10 cm near plane, which clips the gripper
    fingertips that Sidney's training demonstrations show. Wrapping
    ``load_camera`` applies the setting when cameras are built, so the first
    reset observation already uses it. The checkout itself is not edited.
    """
    if isinstance(near, bool) or not isinstance(near, (int, float)) or not isfinite(near) or not 0 < near <= .1:
        raise ValueError('Wrist near plane must be in (0, 0.1] metres')
    original = camera_class.load_camera
    installed = getattr(original, '_fb2_wrist_near', None)
    if installed is not None:
        if installed != near:
            raise ValueError('A different wrist near plane is already installed in this process')
        return

    def load_camera(self, scene):
        result = original(self, scene)
        if getattr(self, 'collect_wrist_camera', True):
            for camera in (self.left_camera, self.right_camera):
                camera.set_near(near)
                if abs(camera.near - near) > 1e-9:
                    raise RuntimeError('Renderer did not accept the wrist near plane')
        return result

    load_camera._fb2_wrist_near = near
    camera_class.load_camera = load_camera
