"""Observe the fitted bridge on real readout calls without changing its outputs."""
import hashlib
import threading
from copy import deepcopy


def observed_motor(native, codec, policy):
    import numpy as np
    import torch
    from fluffy_bassoon.meta_rl.flow_history import FlowBridgeMotor
    from fluffy_bassoon.rl.pi05_residual import _add

    class ObservedMotor(FlowBridgeMotor):
        def __init__(self):
            super().__init__(native, codec, policy)
            self.zero_policy = deepcopy(policy).eval().requires_grad_(False)
            if self.zero_policy.kv_adapter is not None:
                for basis in self.zero_policy.kv_adapter.output_bases:
                    basis.weight.zero_()
            else:
                self.zero_policy.velocity_head.weight.zero_()
                self.zero_policy.velocity_head.bias.zero_()
            self.zero_motor = FlowBridgeMotor(native, codec, self.zero_policy)
            self.lock = threading.Lock()
            self.original_step = policy.step

            def step(token, state=None, **kwargs):
                before = 0 if state is None else int(state['length'][0])
                control, logp, after = self.original_step(token, state, **kwargs)
                assert int(after['length'][0]) == before + 1
                public = token[0].detach().float().cpu().numpy()
                feedback = public[-3:]
                if bool(feedback[2]):
                    if self.mode == 'episode' or before == 0:
                        assert before == 0 and not public[policy.config.observation_dim:-1].any()
                    else:
                        assert before % self.chunks_per_episode == 0 and feedback[1] == 1
                    boundary = dict(incoming_history_queries=before,
                        previous_reward=float(feedback[0]), previous_done=bool(feedback[1]),
                        episode_start=True, token_sha256=hashlib.sha256(public.tobytes()).hexdigest())
                    with self.lock:
                        self.boundaries.append(boundary)
                else:
                    assert before > 0 and feedback[1] == 0
                with self.lock:
                    self.queries += 1
                return control, logp, after

            policy.step = step
            if policy.kv_adapter is not None:
                def cache_hook(module, inputs, output):
                    cache, mask, context = inputs
                    assert len(cache) == len(output) == 18
                    changed = []
                    for layer, (old, new) in enumerate(zip(cache, output)):
                        assert all(torch.equal(a[~mask], b[~mask]) for a, b in zip(old, new))
                        if any(not torch.equal(a[mask], b[mask]) for a, b in zip(old, new)):
                            changed.append(layer)
                    self.edited_layers.update(changed)
                    self.cache_calls += 1
                self.cache_hook = policy.kv_adapter.register_forward_hook(cache_hook)

        def start_wave(self, mode, horizon):
            self.mode, self.chunks_per_episode = mode, horizon // 50
            self.boundaries, self.queries = [], 0
            self.cache_calls, self.velocity_calls, self.dispatches = 0, 0, 0
            self.edited_layers, self.command_deltas = set(), []
            self.maximum_velocity_delta = 0.

        @torch.no_grad()
        def execute_batch(self, prepared, *, noise, steering, spec):
            # One real-observation/common-draw numerical comparison per wave.
            # This extra forward never dispatches controls or touches history.
            baseline = None
            if self.dispatches == 0:
                original_prefix = self.model.build_prefix_cache
                try:
                    self.model.build_prefix_cache = lambda *a, **k: prepared['prefix']
                    baseline, _ = self.model.predict_action_batch(prepared['observation'],
                        mode='eval', noise=torch.as_tensor(noise, dtype=torch.float32, device=self.model.device))
                    baseline = baseline.detach().float().cpu().numpy()
                finally:
                    self.model.build_prefix_cache = original_prefix
                zero_commands = self.zero_motor.execute_batch(prepared, noise=noise, steering=steering, spec=spec)
                assert np.array_equal(zero_commands, baseline), 'Actual zero-output bridge differs from original native motor'
                self.native._prepared = prepared
            original_predict = self.model.predict_action_batch
            original_velocity = self.model.velocity_from_suffix
            base = []
            delta = (policy.config.scale * torch.tanh(policy.velocity_head(torch.as_tensor(
                steering, dtype=torch.float32, device=self.model.device)[:, :policy.config.width]))[:, None, :]
                if policy.velocity_head is not None else None)

            def record_base(suffix):
                value = original_velocity(suffix)
                base.append(value)
                return value

            def predict(*args, **kwargs):
                fitted_velocity = self.model.velocity_from_suffix

                def checked_velocity(suffix):
                    value = fitted_velocity(suffix)
                    if delta is not None:
                        expected = _add(base[-1], delta.expand_as(base[-1]).to(base[-1].dtype))
                        assert torch.equal(value, expected)
                        self.maximum_velocity_delta = max(self.maximum_velocity_delta,
                            float((value - base[-1]).abs().max()))
                    self.velocity_calls += 1
                    return value

                self.model.velocity_from_suffix = checked_velocity
                return original_predict(*args, **kwargs)

            self.model.predict_action_batch = predict
            self.model.velocity_from_suffix = record_base
            before = self.velocity_calls
            try:
                commands = super().execute_batch(prepared, noise=noise, steering=steering, spec=spec)
                assert self.velocity_calls - before == 10
                if baseline is not None:
                    difference = float(np.max(np.abs(commands - baseline)))
                    assert difference > 0, 'Fitted bridge must actually alter native commands'
                    self.command_deltas.append(difference)
                self.dispatches += 1
                return commands
            finally:
                self.model.predict_action_batch = original_predict
                self.model.velocity_from_suffix = original_velocity

        def finish_wave(self, *, cases, horizon, output):
            from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import _json
            chunks = horizon // 50
            assert self.queries == cases * 6 * chunks and self.dispatches == 6 * chunks
            assert len(self.boundaries) == cases * 6 and self.velocity_calls == self.dispatches * 10
            lengths = [row['incoming_history_queries'] for row in self.boundaries]
            assert sorted(lengths) == sorted(([0] * 6 if self.mode == 'episode'
                else [index * chunks for index in range(6)]) * cases)
            if policy.config.interface == 'kv_bridge':
                assert self.cache_calls == self.dispatches and self.edited_layers == set(range(18))
            else:
                assert self.maximum_velocity_delta > 0
            result = dict(passed=True, interface=policy.config.interface, mode=self.mode,
                cases=cases, episodes=cases * 6, controller_queries=self.queries,
                native_dispatches=self.dispatches, actual_denoising_steps=self.velocity_calls,
                edited_KV_layers=sorted(self.edited_layers), maximum_velocity_delta=self.maximum_velocity_delta,
                fitted_vs_native_command_max_deltas=self.command_deltas,
                zero_output_bridge_exact_native_command_parity=True,
                first_query_zero_history=True, actual_episode_boundary_checks=self.boundaries,
                SGD=0, extra_native_comparison_forwards=2)
            _json(output, result)
            return result

    return ObservedMotor()
