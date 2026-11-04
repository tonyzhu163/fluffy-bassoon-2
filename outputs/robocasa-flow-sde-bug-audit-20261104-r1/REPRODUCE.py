"""CPU control-flow reproduction against the deployed OpenPI methods.

Execute on allocated Linux. Real sampler/value helpers and unmodified method ASTs
are used with deterministic suffix features; no VLA weights, simulator or GPU.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def load_methods(path, names, namespace):
    tree = ast.parse(path.read_text())
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            found[node.name] = node
    assert set(found) == set(names), found.keys()
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[
        ast.alias(name='annotations')], level=0), *found.values()], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rlinf', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    code = args.rlinf / 'rlinf/models/embodiment/openpi/tasks/rl.py'
    sampler_path = args.rlinf / 'rlinf/models/embodiment/openpi/sampling/rl_sampler.py'
    env_path = args.rlinf / 'rlinf/envs/sim/robocasa365/robocasa365_env.py'
    spec = importlib.util.spec_from_file_location('audit_rl_sampler', sampler_path)
    sampler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sampler)
    mapping = load_methods(env_path, ['map_released_control_scores'], {'np': np})[
        'map_released_control_scores']
    scores = np.arange(72, dtype=np.float32).reshape(6, 12) / 100
    scores[:, 6] = scores[:, 11] = [-1, 0, .499, .5, 1, 2]
    original = scores.copy()
    mapped = mapping(scores, dict(env_action_dim=12, binarize_gripper_control=True,
        binarize_base_mode_control=True))
    assert np.array_equal(scores, original)
    assert np.array_equal(mapped[:, 6], [-1, -1, -1, 1, 1, 1])
    assert np.array_equal(mapped[:, 11], mapped[:, 6])
    continuous = [i for i in range(12) if i not in (6, 11)]
    assert np.array_equal(mapped[:, continuous], scores[:, continuous])
    # Independent float64 scalar formula checks all valid native indices.
    noise_rows = []
    for index in range(10):
        _, std = sampler.sample_mean_var(torch.zeros(1, 50, 32), torch.zeros(1, 50, 32),
            torch.tensor([index]), noise_method='flow_sde', noise_level=.3, num_steps=10)
        t = 1-index/10
        denominator = .1 if index == 0 else 1-t
        reference = .3 * (t/denominator/10)**.5
        actual = float(std[0, 0, 0])
        assert abs(actual-reference) < 3e-7
        noise_rows.append(dict(index=index, native_std=actual, independent_std=reference))
    chosen = 0
    methods = load_methods(code, ['_predict_train', 'default_forward'], dict(
        torch=torch, random=SimpleNamespace(randint=lambda lo, hi: chosen),
        preprocess_observation=lambda observation, **kw: observation, rl_sampler=sampler))

    class FakePolicy:
        device = torch.device('cpu')
        num_steps = 10
        action_horizon = action_chunk = 50
        model_action_dim = 32
        action_env_dim = 12

        def __init__(self, value_after_vlm=False):
            self.rl_cfg = SimpleNamespace(add_value_head=True, value_after_vlm=value_after_vlm,
                value_vlm_mode='mean_token', is_nft=False, joint_logprob=False,
                ignore_last=False, noise_method='flow_sde', noise_level=.3,
                chunk_critic_input=False, detach_critic_input=True)
            self.value_head = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.value_head.weight.fill_(1)

        def build_prefix_cache(self, observation):
            return torch.full((2, 1, 1), .37), torch.ones(2, 1, dtype=torch.bool), None

        _build_prefix_cache_for_actor = build_prefix_cache

        def run_suffix(self, observation, x, t, cache, mask):
            return t[:, None, None].expand(2, 50, 1)

        def velocity_from_suffix(self, suffix):
            return torch.zeros(2, 50, 32)

        def _init_nft_state(self, *args):
            return None

        def _update_nft_state(self, *args):
            pass

        def decode_actions(self, actions, state):
            return actions[:, :, :12]

        def _observation_from_forward_inputs(self, inputs):
            return observation

    for name, method in methods.items():
        setattr(FakePolicy, name, method)
    observation = SimpleNamespace(state=torch.zeros(2, 16),
        tokenized_prompt=torch.zeros(2, 200, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(2, 200, dtype=torch.bool))
    rows = []
    with torch.no_grad():
        policy = FakePolicy()
        for chosen in range(10):
            _, rollout = policy._predict_train(observation, noise=None,
                rng=torch.Generator().manual_seed(111), compute_values=True)
            replay = policy.default_forward(rollout['forward_inputs'], compute_values=True)
            old, new = rollout['prev_values'][:, 0], replay['values']
            row = dict(index=chosen, rollout_value=old.tolist(), replay_value=new.tolist(),
                max_value_gap=float((old-new).abs().max()),
                logprob_max_gap=float((rollout['prev_logprobs']-replay['logprobs']).abs().max()),
                clipped_before_sgd=int(((old-new).abs() > .2).sum()))
            assert row['logprob_max_gap'] == 0
            assert row['max_value_gap'] > .04
            rows.append(row)
        # A prefix critic is a control: both paths evaluate the same features.
        control = FakePolicy(value_after_vlm=True)
        _, rollout = control._predict_train(observation, noise=None,
            rng=torch.Generator().manual_seed(111), compute_values=True)
        replay = control.default_forward(rollout['forward_inputs'], compute_values=True)
        prefix_gap = float((rollout['prev_values'][:, 0]-replay['values']).abs().max())
        assert prefix_gap == 0
    resolved = json.loads(args.config.read_text())
    report = dict(
        scope='Unmodified deployed method control flow with synthetic suffix features; not measured production-model errors',
        device='cpu', torch_version=torch.__version__,
        files={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [code, sampler_path, env_path, args.config]},
        actual_run_openpi=resolved['actor']['model']['openpi'],
        rows=rows, prefix_critic_control_gap=prefix_gap,
        confirmed='Rollout averages ten suffix values; replay uses selected one even at identical weights',
        max_gap=max(r['max_value_gap'] for r in rows),
        samples_clipped_before_sgd=sum(r['clipped_before_sgd'] for r in rows),
        samples=20, denoising_noise_checks=noise_rows,
        action_mapping='PASS: threshold 0.5 at indices 6 and 11; continuous controls and input unchanged')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['confirmed','max_gap','samples_clipped_before_sgd',
        'samples','prefix_critic_control_gap']}))


if __name__ == '__main__':
    main()
