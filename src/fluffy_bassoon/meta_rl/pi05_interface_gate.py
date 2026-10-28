"""Bounded-flow interface qualification, separate from student adaptation.

Private theta may select a separately fitted teacher, but never enters its
public inputs. A teacher must execute through sample_chunk(), not action labels.
Offline flow-matching bounds and paired action distances are diagnostics only;
live success on fresh scenes is the competence gate.
"""

from collections.abc import Mapping
import math
import time

import numpy as np
import torch
from torch import nn

from .bridge_distillation import corrected_action_label, distillation_loss, update_distillation


def slice_history_lanes(data, indices):
    """Select complete lanes, retaining every chronological public history input.

    Histories contain one record per chunk [B,...] and boundary/reward arrays
    [T,B]. Private labels, rewards and boundaries are sliced identically. No
    theta value is added to a record or recurrent input.
    """
    indices = torch.as_tensor(indices)
    if (indices.ndim != 1 or not indices.numel() or indices.dtype == torch.bool
            or indices.is_floating_point() or indices.is_complex()):
        raise ValueError("Require nonempty integer lane indices")
    indices = indices.to(dtype=torch.long, device="cpu")
    lanes = data["rewards"].shape[1]
    if (indices.min() < 0 or indices.max() >= lanes
            or indices.unique().numel() != indices.numel()):
        raise ValueError("Lane indices must be unique and within the complete batch")
    result = dict(data)
    for key in ("rewards", "resets", "attempt_ends", "trial_ends"):
        value = data[key]
        if value.ndim != 2 or value.shape != data["rewards"].shape:
            raise ValueError("Require matching chronological [T,B] arrays")
        result[key] = value.index_select(1, indices.to(value.device)).clone()
    result["records"] = []
    for record in data["records"]:
        if any(not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != lanes
               for value in record.values()):
            raise ValueError("Every cached record tensor must have the lane axis first")
        result["records"].append({key: value.index_select(0, indices.to(value.device)).clone()
                                  for key, value in record.items()})
    # These counts refer to this slice, rather than the original collection batch.
    result["policy_chunks"] = len(result["records"]) * indices.numel()
    result["simulator_steps"] = result["policy_chunks"] * data["supervised_action_steps"]
    return result


@torch.no_grad()
def residual_headroom(histories, *, residual_scale, active_dims=7):
    """Report a sample-wise box-bound loss floor on executed active coordinates.

    The exact infimum over independent residuals in [-scale,scale] is the
    squared excess beyond that box. It is an optimistic lower bound on this
    finite flow-matching dataset, not proof of decoded-action reachability or
    of a useful controller. Native conditional-flow error contributes too.
    """
    if (not math.isfinite(residual_scale) or residual_scale <= 0
            or type(active_dims) is not int or active_dims < 1):
        raise ValueError("Require a positive residual scale and active dimension count")
    if isinstance(histories, Mapping):
        histories = (histories,)
    # Sum, square sum, excess sum, floor sum and saturation count per channel.
    totals = torch.zeros(5, active_dims, dtype=torch.float64)
    maxima = torch.zeros(active_dims, dtype=torch.float64)
    count, batches, chunks = 0, 0, 0
    supervised_steps = set()
    for data in histories:
        steps = data["supervised_action_steps"]
        if type(steps) is not int or steps < 1:
            raise ValueError("Require the declared executed action prefix")
        supervised_steps.add(steps)
        batches += 1
        for record in data["records"]:
            native, target = record["native_velocity"], record["target_velocity"]
            if (native.ndim != 3 or native.shape != target.shape or not native.shape[0]
                    or steps > native.shape[1] or active_dims > native.shape[2]
                    or not native.isfinite().all() or not target.isfinite().all()):
                raise ValueError("Require finite matching [B,H,A] target/native velocities")
            required = (target[:, :steps, :active_dims].to(device="cpu", dtype=torch.float64)
                        - native[:, :steps, :active_dims].to(device="cpu", dtype=torch.float64)).abs()
            excess = (required - residual_scale).clamp_min(0)
            totals += torch.stack((required.sum((0, 1)), required.square().sum((0, 1)),
                excess.sum((0, 1)), excess.square().sum((0, 1)),
                (required > residual_scale).sum((0, 1))))
            maxima = torch.maximum(maxima, required.amax((0, 1)))
            count += required.shape[0] * steps
            chunks += required.shape[0]
    if count == 0:
        raise ValueError("Require at least one flow-matching record")
    def summary(sums, maximum, denominator):
        return dict(target_residual_abs_mean=float(sums[0] / denominator),
                    target_residual_abs_max=float(maximum),
                    zero_residual_flow_matching_loss=float(sums[1] / denominator),
                    residual_excess_abs_mean=float(sums[2] / denominator),
                    bounded_flow_matching_loss_floor=float(sums[3] / denominator),
                    saturation_fraction=float(sums[4] / denominator))
    return dict(protocol="native_bounded_flow_headroom_v1", residual_scale=float(residual_scale),
        active_dims=active_dims, supervised_action_steps=sorted(supervised_steps),
        histories=batches, policy_chunks=chunks, scalar_count=count * active_dims,
        **summary(totals.sum(1), maxima.max(), count * active_dims),
        per_channel=[dict(channel=index, scalar_count=count,
                         **summary(totals[:, index], maxima[index], count))
                     for index in range(active_dims)],
        interpretation="sample-wise optimistic flow-loss bound; live sampler success is required")


class PrivilegedThetaSelector(nn.Module):
    """Select one independently fitted teacher for a homogeneous-theta trial.

    Call select_theta(plan['theta_degrees']) before collect_live_trials(). The
    selector receives private task metadata; the selected policy receives the
    unchanged motor, observation, feedback and recurrent memory only. Evaluate
    same-theta fresh scenes. These teachers do not establish held-out-theta
    adaptation, and the selector is never a student policy or PPO behavior.
    """

    def __init__(self, teachers):
        super().__init__()
        if not teachers:
            raise ValueError("Require at least one separately fitted teacher")
        self._theta_to_key = {}
        heads = {}
        configurations = []
        owned_parameters = set()
        for index, (theta, policy) in enumerate(teachers.items()):
            theta = float(theta)
            if not math.isfinite(theta) or theta in self._theta_to_key:
                raise ValueError("Require unique finite teacher theta values")
            key = f"head_{index}"
            parameters = {id(parameter) for parameter in policy.parameters()}
            if owned_parameters & parameters:
                raise ValueError("Private theta teachers must have independent fitted weights")
            owned_parameters.update(parameters)
            self._theta_to_key[theta] = key
            heads[key] = policy
            configurations.append(policy.config())
        if any(config != configurations[0] for config in configurations[1:]):
            raise ValueError("All teachers must use the same native intervention interface")
        self.teachers = nn.ModuleDict(heads)
        self._selection = None
        self._lane_count = None
        self.eval()

    def config(self):
        return dict(protocol="native_private_theta_interface_gate_v1",
                    theta_degrees=list(self._theta_to_key),
                    selection="private metadata selects independent teacher; never a public input",
                    interface=next(iter(self.teachers.values())).config())

    def select_theta(self, theta_degrees):
        values = [float(theta) for theta in theta_degrees]
        if not values or any(theta != values[0] for theta in values):
            raise ValueError("Privileged teacher trials require homogeneous theta lanes")
        if values[0] not in self._theta_to_key:
            raise ValueError("No separately fitted teacher for this theta")
        self._selection = values[0]
        self._lane_count = len(values)
        return self

    @property
    def selected_policy(self):
        if self._selection is None:
            raise ValueError("Select private trial theta before calling the teacher")
        return self.teachers[self._theta_to_key[self._selection]]

    @property
    def active_dims(self):
        return self.selected_policy.active_dims

    def sample_chunk(self, motor, prepared_observation, feedback, memory=None, **kwargs):
        if self.training or not kwargs.get("deterministic", False):
            raise ValueError("Privileged teacher qualification requires deterministic eval sampling")
        policy = self.selected_policy
        if feedback.shape[0] != self._lane_count:
            raise ValueError("Teacher selection and public feedback lane counts differ")
        return policy.sample_chunk(motor, prepared_observation, feedback, memory, **kwargs)


def fit_privileged_teacher(policy, optimizer, histories, *, updates, sampler=None,
                           max_elapsed_seconds=900., validation_histories=(), on_update=None):
    """Bounded offline fitting for one theta; caller owns allocation/checkpoints.

    Pass only complete lanes assigned to this theta and disjoint validation
    histories. No native model is loaded. on_update(completed,metrics) can save
    resumable policy/optimizer/sampler snapshots at complete update boundaries.
    """
    histories, validation_histories = tuple(histories), tuple(validation_histories)
    if (not histories or type(updates) is not int or updates < 1
            or not math.isfinite(max_elapsed_seconds) or max_elapsed_seconds <= 0):
        raise ValueError("Require histories and positive bounded fitting limits")
    sampler = np.random.default_rng(0) if sampler is None else sampler
    started, rows = time.monotonic(), []
    def validation():
        with torch.no_grad():
            sizes = [len(data["records"]) * data["rewards"].shape[1] for data in validation_histories]
            losses = [float(distillation_loss(policy, data)) for data in validation_histories]
        return float(np.average(losses, weights=sizes)) if losses else None
    initial = validation()
    while len(rows) < updates and time.monotonic() - started < max_elapsed_seconds:
        index = int(sampler.integers(len(histories)))
        metrics = dict(update_distillation(policy, optimizer, histories[index]), history_index=index,
                       elapsed_seconds=time.monotonic() - started)
        rows.append(metrics)
        if on_update is not None:
            on_update(len(rows), metrics)
    return dict(updates_completed=len(rows), target_updates=updates,
                completed=len(rows) == updates, elapsed_seconds=time.monotonic() - started,
                initial_validation_loss=initial, final_validation_loss=validation(), metrics=rows)


@torch.no_grad()
def paired_sample_diagnostics(policy, motor, observation, prepared_observation, feedback,
                               theta_degrees, memory=None, *, rng=None, noise=None):
    """Run actual four-Euler policy sampling and paired native/oracle references.

    References share the initial native noise and current observation. External
    labels are never executed here or supplied to the learned sampler. Return
    learned actions/memory/record plus action distances; live rollouts must still
    compare success on matched private trial plans and fresh scenes.
    """
    if noise is None:
        if rng is None:
            raise ValueError("Provide explicit native noise or its checkpointed generator")
        noise = torch.randn(feedback.shape[0], motor.action_horizon, motor.action_dim,
                            device=feedback.device, dtype=torch.float32, generator=rng)
    native = motor.sample_actions(observation, num_steps=4, noise=noise)
    target = corrected_action_label(motor, observation.state, native, theta_degrees)
    actions, memory, record = policy.sample_chunk(motor, prepared_observation, feedback, memory,
        rng=rng, noise=noise, deterministic=True, num_steps=4)
    decoded = {name: motor.decode_actions(value, observation.state)
               for name, value in (("learned", actions), ("zero", native), ("external_oracle", target))}
    steps = decoded["learned"].shape[1]
    dimensions = policy.active_dims
    metrics = dict(protocol="native_four_euler_paired_interface_v1", num_steps=4,
                   executed_action_steps=steps, active_dims=dimensions,
                   references="same initial native noise and observation; external oracle bypasses bridge")
    for name, value in (("learned", actions), ("zero", native), ("external_oracle", target)):
        model_error = value[:, :steps, :dimensions] - target[:, :steps, :dimensions]
        decoded_error = decoded[name] - decoded["external_oracle"]
        metrics[name] = dict(model_action_mse=float(model_error.square().mean()),
            model_action_abs_max=float(model_error.abs().max()),
            decoded_action_mse=float(decoded_error.square().mean()),
            decoded_xy_mse=float(decoded_error[..., :2].square().mean()),
            decoded_action_abs_max=float(decoded_error.abs().max()))
    if not all(math.isfinite(value) for group in ("learned", "zero", "external_oracle")
               for value in metrics[group].values()):
        raise RuntimeError("Non-finite paired interface action diagnostic")
    return actions, memory, record, metrics
