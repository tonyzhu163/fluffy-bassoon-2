"""Concurrent native lanes sharing one frozen motor, with serial PPO records."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
import torch


def combine_observations(observations):
    result = {}
    for key in observations[0]:
        values = [row[key] for row in observations]
        result[key] = (torch.cat(values, 0) if torch.is_tensor(values[0]) else
                       np.concatenate(values, 0) if isinstance(values[0], np.ndarray) else
                       sum((list(value) for value in values), []))
    return result


class BatchedRoboCasaResidualEnvironment:
    """Complete case groups; shorter native episodes leave without padding.

    ``make_environment(index, case)`` creates an existing single native lane,
    with its own physical-action journal and original case/episode identity.
    The completed output is restored to serial case/chunk order before GAE.
    """
    def __init__(self, make_environment, cases, *, horizon, environments_per_gpu):
        from .robocasa_residual_parallel import native_chunk_offsets
        if not cases or type(environments_per_gpu) is not int or environments_per_gpu not in (2, 4, 8):
            raise ValueError('Use an explicitly qualified batched lane count')
        native_chunk_offsets(cases, horizon)
        self.make_environment, self.cases = make_environment, deepcopy(cases)
        self.horizon, self.width = horizon, environments_per_gpu
        self.cursor, self.active = 0, False
        self.environments, self.completed_states = [], []

    def state_dict(self):
        if self.active:
            raise ValueError('An incomplete native bank cannot become a resumable boundary')
        return dict(cases=deepcopy(self.cases), cursor=self.cursor, horizon=self.horizon,
                    rollout_envs_per_gpu=self.width, completed_states=deepcopy(self.completed_states))

    def load_state_dict(self, state):
        if (self.active or state['cases'] != self.cases or state['horizon'] != self.horizon
                or state.get('rollout_envs_per_gpu', 1) != self.width
                or state['cursor'] not in (0, len(self.cases))):
            raise ValueError('Preserve exact cases and batching on checkpoint recovery')
        self.cursor = state['cursor']
        self.completed_states = deepcopy(state.get('completed_states', []))

    @torch.no_grad()
    def collect(self, policy, bound, motor, *, episodes, generator, deterministic=False,
                verify_zero=False, planned_motor_seeds=None, planned_action_noise=None):
        from .robocasa_residual_parallel import native_chunk_offsets
        if self.active or self.cursor or episodes != len(self.cases):
            raise ValueError('Collect each exact complete worker bank once')
        offsets = native_chunk_offsets(self.cases, self.horizon)
        if (planned_motor_seeds is None or len(planned_motor_seeds) != offsets[-1]
                or (not deterministic and planned_action_noise is None)
                or (planned_action_noise is not None and len(planned_action_noise) != offsets[-1])):
            raise ValueError('Require the complete serial per-case motor and actor draw plan')
        records = [[] for _ in self.cases]
        completed = [None] * episodes
        self.active = True
        try:
            with ThreadPoolExecutor(max_workers=self.width) as pool:
                for first in range(0, episodes, self.width):
                    indices = list(range(first, min(first+self.width, episodes)))
                    self.environments = [self.make_environment(index, self.cases[index]) for index in indices]
                    observations = list(pool.map(lambda env: env.reset(), self.environments))
                    active = list(range(len(indices)))
                    for chunk_index in range(max(offsets[i+1]-offsets[i] for i in indices)):
                        selected = [indices[lane] for lane in active]
                        draws = [offsets[index]+chunk_index for index in selected]
                        seeds = [int(planned_motor_seeds[draw]) for draw in draws]
                        joined = combine_observations([observations[lane] for lane in active])
                        chunks, features, returned_seeds = motor.infer_batch(joined, noise_seeds=seeds)
                        if list(returned_seeds) != seeds:
                            raise ValueError('Batched motor changed the producing draw identities')
                        controls, probabilities, values, executed = [], [], [], []
                        reference = next(policy.parameters())
                        for position, draw in enumerate(draws):
                            noise = {} if planned_action_noise is None else dict(action_noise=planned_action_noise[draw:draw+1])
                            control, probability, value = policy.act(
                                torch.as_tensor(features[position:position+1], device=reference.device,
                                                dtype=reference.dtype),
                                deterministic=deterministic, generator=generator, **noise)
                            control = control[0].cpu().numpy()
                            controls.append(control); probabilities.append(float(probability[0])); values.append(float(value[0]))
                            executed.append(bound.apply(chunks[position], control))
                        verified = verify_zero and 0 in selected
                        if verified:
                            # Compare the SAME native batched kernel, not a singleton
                            # kernel with different BF16 rounding. No extra episode.
                            original, _, _ = motor.infer_batch(joined, noise_seeds=seeds)
                            position = selected.index(0)
                            if np.any(controls[position] != 0) or not np.array_equal(executed[position], original[position]):
                                raise ValueError('Zero residual changed the native batched action')
                        returned = list(pool.map(lambda pair: pair[0].step_chunk(pair[1]),
                            [(self.environments[lane], chunk) for lane, chunk in zip(active, executed)]))
                        remaining = []
                        for position, (lane, index, result) in enumerate(zip(active, selected, returned)):
                            observation, done, info = result
                            expected_end = chunk_index+1 == offsets[index+1]-offsets[index]
                            if bool(done) != expected_end:
                                raise ValueError('Each lane must finish at its exact native horizon')
                            records[index].append(dict(features=np.array(features[position], dtype=np.float32, copy=True),
                                noise_seed=seeds[position], base_chunk=np.array(chunks[position], copy=True),
                                executed_chunk=np.array(executed[position], copy=True), control=controls[position].astype(np.float32),
                                log_prob=probabilities[position], value=values[position], reward=0., done=bool(done),
                                episode_start=chunk_index==0, zero_residual_verified=bool(verified and index==0)))
                            observations[lane] = observation
                            if done: completed[index] = info
                            else: remaining.append(lane)
                        active = remaining
                    if active:
                        raise ValueError('All native lanes must complete before another group')
                    for env in self.environments:
                        self.completed_states.append(env.state_dict())
                        env.close()
                    self.environments = []
            if any(row is None for row in completed) or sum(map(len, records)) != offsets[-1]:
                raise ValueError('Incomplete worker bank cannot enter reward labeling')
            self.cursor, self.active = episodes, False
            return [row for episode in records for row in episode], completed
        except BaseException:
            self.close()  # Keep active=True: no recovery/recollection of a partial bank.
            raise

    def close(self):
        for env in self.environments:
            env.close()
        self.environments = []
