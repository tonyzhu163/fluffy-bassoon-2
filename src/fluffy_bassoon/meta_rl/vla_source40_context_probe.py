"""Offline input-sensitivity probe of a frozen controller on real robot history.

Consumes already executed, hash-bound support/query traces. No simulator, VLA
backbone, optimizer or counterfactual robot outcome is constructed here.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from .vla_history import FrozenHistoryController, file_sha256, load_policy
from .warmstart import atomic_save


VARIANTS = ('genuine', 'episode_reset', 'all_rewards_masked', 'terminal_reward_flipped')
PI05_RECEIPT = 'ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170'


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def checked_json(path, expected):
    path = Path(path)
    if file_sha256(path) != expected:
        raise ValueError(f'JSON checksum differs: {path}')
    value = json.loads(path.read_text())
    if file_sha256(path) != expected:
        raise ValueError(f'JSON changed while reading: {path}')
    return value


def policy_signature(policy):
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        array = value.detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def snapshot(controller):
    def cpu(value):
        if isinstance(value, dict):
            return {key: cpu(item) for key, item in value.items()}
        return value.detach().cpu().clone() if torch.is_tensor(value) else deepcopy(value)
    return dict(state=cpu(controller.state), previous_noise=controller.previous_noise.copy(),
        previous_reward=controller.previous_reward, previous_done=controller.previous_done,
        episode_start=controller.episode_start, pending=controller.pending,
        generator=controller.generator.get_state().cpu().clone())


def restore(controller, saved):
    device = next(controller.policy.parameters()).device
    def target(value):
        if isinstance(value, dict):
            return {key: target(item) for key, item in value.items()}
        return value.to(device).clone() if torch.is_tensor(value) else deepcopy(value)
    controller.state = target(saved['state'])
    for name in ('previous_noise', 'previous_reward', 'previous_done', 'episode_start', 'pending'):
        setattr(controller, name, deepcopy(saved[name]))
    controller.generator = torch.Generator(device=device)
    controller.generator.set_state(saved['generator'])


def state_summary(saved):
    state = saved['state']
    return dict(memory_l2=0. if state is None else float(torch.linalg.vector_norm(state['memory'].float())),
        memory_length=[] if state is None else state['length'].tolist(),
        previous_noise_l2=float(np.linalg.norm(saved['previous_noise'])),
        previous_reward=float(saved['previous_reward']), previous_done=bool(saved['previous_done']),
        episode_start=bool(saved['episode_start']), pending=bool(saved['pending']))


def validate_rows(rows, receipt, observation_dim, noise_dim):
    if not isinstance(rows, list) or not rows or receipt.get('completed') is not True or len(rows) != receipt['chunks']:
        raise ValueError('Require a complete recorded physical episode')
    for index, row in enumerate(rows):
        for name, shape in (('public_features', (observation_dim,)), ('proposed_noise', (noise_dim,)),
                            ('executed_noise', (noise_dim,))):
            value = np.asarray(row[name])
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f'Require finite float32 recorded {name}')
        if (type(row['done']) not in (bool, np.bool_) or bool(row['done']) != (index == len(rows) - 1)
                or not np.isfinite(row['reward']) or row['reward'] not in (-1., 0.)
                or type(row['primitive_steps']) is not int or not 1 <= row['primitive_steps'] <= 10):
            raise ValueError('Require actual chunk rewards, costs and a single final episode boundary')
        if not np.array_equal(np.clip(row['proposed_noise'], -1., 1.), row['executed_noise']):
            raise ValueError('Recorded executed noise differs from the declared fixed clipping map')
    if sum(row['primitive_steps'] for row in rows) != receipt['primitive_steps']:
        raise ValueError('Trace primitive-action costs differ from the episode receipt')


def load_episode(group_directory, receipt, observation_dim, noise_dim):
    directory = Path(group_directory).resolve()
    path = (directory / receipt['trace']).resolve()
    if not path.is_relative_to(directory) or file_sha256(path) != receipt['trace_sha256']:
        raise ValueError('Trace path or checksum differs')
    with gzip.open(path, 'rb') as stream:
        rows = pickle.load(stream)
    if file_sha256(path) != receipt['trace_sha256']:
        raise ValueError('Trace changed while reading')
    validate_rows(rows, receipt, observation_dim, noise_dim)
    return rows


@torch.no_grad()
def probe_context(policy, support, query, *, support_seed, query_seed, tolerance=1e-5,
                  deadline=float('inf')):
    """Replay fixed actions, then query the same recorded first-query feature.

    Each variant consumes identical real observations/actions/boundaries. Only
    declared reward inputs or the query episode reset differ. Query reward and
    later query observations are never consumed.
    """
    if not np.isfinite(tolerance) or not 0 < tolerance <= 1e-5:
        raise ValueError('Require a proposal replay tolerance in (0, 1e-5]')
    signature = policy_signature(policy)
    device = next(policy.parameters()).device
    variants, snapshots = {}, {}
    for variant in VARIANTS:
        controller = FrozenHistoryController(policy, mode='episode' if variant == 'episode_reset' else 'full')
        controller.generator = torch.Generator(device=device).manual_seed(support_seed)
        support_delta = 0.
        rewards = []
        for index, row in enumerate(support):
            if time.monotonic() >= deadline:
                raise RuntimeError('Offline context-probe elapsed-time ceiling reached')
            proposal = controller.act(row['public_features'])
            if variant == 'genuine':
                support_delta = max(support_delta, float(np.abs(proposal - row['proposed_noise']).max()))
            reward = (0. if variant == 'all_rewards_masked' else
                      -1. - float(row['reward']) if variant == 'terminal_reward_flipped' and index == len(support) - 1
                      else float(row['reward']))
            rewards.append(reward)
            # Feed the noise ACTUALLY executed by the robot, never the new proposal.
            controller.observe(executed_noise=row['executed_noise'], reward=reward, done=bool(row['done']))
        support_end = snapshot(controller)
        controller.generator = torch.Generator(device=device).manual_seed(query_seed)
        before = snapshot(controller)
        mean = controller.act(query['public_features'], deterministic=True)
        restore(controller, before)
        proposal = controller.act(query['public_features'])
        after = snapshot(controller)
        clipped = np.clip(proposal, -1., 1.).astype(np.float32)
        std = policy.log_std.clamp(-5, 2).exp().detach().cpu().numpy().copy()
        variants[variant] = dict(mean=mean.tolist(), proposed_noise=proposal.tolist(), executed_noise=clipped.tolist(),
            std=std.tolist(), paired_standard_normal=((proposal - mean) / std).tolist(),
            support_reward_inputs=rewards, support_proposal_replay_max_error=support_delta if variant == 'genuine' else None,
            support_end=state_summary(support_end), query_before=state_summary(before), query_after=state_summary(after),
            query_episode_reset_applied=variant == 'episode_reset',
            effective_prior_memory_l2=0. if variant == 'episode_reset' else state_summary(before)['memory_l2'])
        snapshots[variant] = dict(support_end=support_end, query_before=before, query_after=after)
    genuine = variants['genuine']
    proposal_error = float(np.abs(np.asarray(genuine['proposed_noise']) - query['proposed_noise']).max())
    clipped_error = float(np.abs(np.asarray(genuine['executed_noise']) - query['executed_noise']).max())
    if max(proposal_error, clipped_error, genuine['support_proposal_replay_max_error']) > tolerance:
        raise RuntimeError(f'Genuine support/query policy replay differs: proposal={proposal_error}, clipped={clipped_error}, support={genuine["support_proposal_replay_max_error"]}')
    for name, value in variants.items():
        value['mean_delta_l2'] = float(np.linalg.norm(np.asarray(value['mean']) - genuine['mean']))
        value['proposal_delta_l2'] = float(np.linalg.norm(np.asarray(value['proposed_noise']) - genuine['proposed_noise']))
        value['clipped_delta_l2'] = float(np.linalg.norm(np.asarray(value['executed_noise']) - genuine['executed_noise']))
        value['paired_standard_normal_max_error'] = float(np.abs(np.asarray(value['paired_standard_normal']) - genuine['paired_standard_normal']).max())
        if value['std'] != genuine['std'] or value['paired_standard_normal_max_error'] > 1e-5:
            raise RuntimeError(f'Query sampling noise was not paired across variants: {name}')
    if policy_signature(policy) != signature or any(parameter.grad is not None for parameter in policy.parameters()):
        raise RuntimeError('Frozen probe changed policy weights or populated gradients')
    return dict(variants=variants, genuine_first_query_proposal_max_error=proposal_error,
        genuine_first_query_clipped_max_error=clipped_error, weights_unchanged=True,
        policy_signature=signature, optimizer_updates=0, new_simulator_actions=0), snapshots


def load_inputs(scales_path, scales_sha256, group_index, group_sha256, *, device):
    scales = checked_json(scales_path, scales_sha256)
    if (scales.get('format') != 'openpi_pi05_source40_subset_scales_v1' or scales.get('completed') is not True
            or scales.get('qualification_subset') is not True or scales.get('seed') != 0
            or len(scales['scales']) != 3 or {row['updates'] for row in scales['scales']} != {25, 100, 500}):
        raise ValueError('Require the completed, explicit one-source-goal optimizer-scale diagnostic')
    assembly = checked_json(scales['assembly_path'], scales['assembly_file_sha256'])
    assembly_hash = content_hash({key: value for key, value in assembly.items() if key != 'assembly_sha256'})
    if assembly_hash != assembly.get('assembly_sha256') or assembly_hash != scales['assembly_sha256']:
        raise ValueError('Source assembly content checksum differs')
    scale = next(row for row in scales['scales'] if row['updates'] == 500)
    matrix = checked_json(scale['index_path'], scale['index_sha256'])
    rows = [row for row in matrix['controllers'] if (row['objective'], row['mode'], row['seed']) == ('ad', 'full', 0)]
    if matrix.get('completed') is not True or matrix['contract']['assembly_sha256'] != assembly_hash or len(rows) != 1:
        raise ValueError('Require exactly one completed AD500 full-context export')
    selected = rows[0]
    policy, manifest = load_policy(selected['final_path'], expected_sha256=selected['final_sha256'], device=device)
    training = manifest['contract']
    if (selected['updates'] != 500 or manifest.get('update') != 500 or training.get('objective') != 'ad'
            or training.get('mode') != 'full' or training.get('seed') != 0
            or manifest.get('selection') != 'fixed requested final update'
            or manifest.get('assembly_sha256') != assembly_hash
            or manifest.get('full_recovery_sha256') != selected['full_recovery_sha256']
            or training.get('train_metadata', {}).get('motor_sha256') != PI05_RECEIPT
            or training['train_metadata'].get('feature_contract') != assembly['feature_contract']
            or (policy.config.observation_dim, policy.config.noise_dim) != (4104, 32)):
        raise ValueError('AD500 export provenance or public feature contract differs')
    result = checked_json(group_index, group_sha256)
    directory = Path(group_index).resolve().parent
    group_contract = json.loads((directory / 'contract.json').read_text())
    candidate = group_contract.get('candidates', {}).get('ad_full', {})
    if (result.get('completed') is not True or result.get('weights_unchanged') is not True
            or result.get('budget_matched_primary_eligible') is not True
            or group_contract.get('case') != result['case'] or group_contract.get('methods') != ['ad_full']
            or group_contract.get('support_budgets') != [0, 1]
            or group_contract.get('motor_receipt_sha256') != PI05_RECEIPT
            or group_contract.get('noise_contract') != training['train_metadata'].get('noise_contract')
            or candidate.get('sha256') != selected['final_sha256'] or candidate.get('mode') != 'full'
            or result.get('source_provenance', {}).get('optimizer_update_scale') != 500):
        raise ValueError('Require an eligible, completed robot group of these exact AD500 full weights')
    episodes = [row for row in result['episodes'] if row['support_budget'] == 1 and row['method'] == 'ad_full']
    support = [row for row in episodes if row['role'] == 'support']
    query = [row for row in episodes if row['role'] == 'query']
    if (len(support) != 1 or len(query) != 1 or support[0]['role_episode_ordinal'] != 0
            or query[0]['role_episode_ordinal'] != 0
            or support[0]['episode_id'] == query[0]['episode_id']
            or not support[0].get('initial_state_vector_sha256')
            or support[0]['initial_state_vector_sha256'] != query[0]['initial_state_vector_sha256']):
        raise ValueError('Require one actual support and a separate query on the same physical case')
    for receipt in (*support, *query):
        expected_seed = int(content_hash(dict(case=result['case'], role=receipt['role'], ordinal=0))[:8], 16)
        if receipt['episode_random_seed'] != expected_seed or receipt['initial_state'] != result['case']['initial_state'] or receipt['scene_seed'] != result['case']['scene_seed']:
            raise ValueError('Support/query seed or physical case differs from the declared pairing')
    support_rows = load_episode(directory, support[0], 4104, 32)
    query_rows = load_episode(directory, query[0], 4104, 32)
    provenance = dict(scales_path=str(Path(scales_path).resolve()), scales_sha256=scales_sha256,
        assembly_sha256=assembly_hash, matrix_index_sha256=scale['index_sha256'],
        policy_path=selected['final_path'], policy_sha256=selected['final_sha256'],
        full_recovery_sha256=selected['full_recovery_sha256'], optimizer_update_scale=500,
        group_index=str(Path(group_index).resolve()), group_index_sha256=group_sha256,
        group_contract_sha256=file_sha256(directory / 'contract.json'), case=result['case'],
        support_receipt=support[0], query_receipt=query[0], feature_contract=assembly['feature_contract'],
        actual_recorded_query_success=query[0]['success'], actual_recorded_query_primitive_steps=query[0]['primitive_steps'])
    return policy, support_rows, query_rows[0], support[0]['episode_random_seed'], query[0]['episode_random_seed'], provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scales', 'group-index', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('scales-sha256', 'group-index-sha256', 'source-commit', 'expected-hostname'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-elapsed-seconds', type=float, default=420.)
    args = parser.parse_args(argv)
    if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname:
        raise ValueError('Run the offline model probe only on the explicitly allocated Linux host')
    if not 0 < args.max_elapsed_seconds <= 480 or not args.device.startswith('cuda:'):
        raise ValueError('Require one allocated CUDA device and <=480 seconds of model computation')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean reviewed probe source commit')
    deadline = time.monotonic() + args.max_elapsed_seconds
    args.output.mkdir(parents=True, exist_ok=False)
    policy, support, query, support_seed, query_seed, provenance = load_inputs(
        args.scales, args.scales_sha256, args.group_index, args.group_index_sha256, device=args.device)
    result, snapshots = probe_context(policy, support, query, support_seed=support_seed, query_seed=query_seed, deadline=deadline)
    snapshot_path = args.output / 'context-rng-snapshots.pt'
    snapshot_sha256 = atomic_save(snapshot_path, dict(format='source40_offline_context_snapshots_v1',
        snapshots=snapshots, provenance=provenance, support_seed=support_seed, query_seed=query_seed))
    result.update(format='source40_offline_context_probe_v1', completed=True, source_commit=args.source_commit,
        provenance=provenance, snapshots_path=str(snapshot_path), snapshots_sha256=snapshot_sha256,
        elapsed_seconds=args.max_elapsed_seconds - (deadline - time.monotonic()),
        interpretation='Input sensitivity on fixed recorded history only. No counterfactual closed-loop success or causal reward-benefit claim.')
    (args.output / 'index.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(completed=True, output=str(args.output), replay_error=result['genuine_first_query_proposal_max_error'],
        deltas={key: value['mean_delta_l2'] for key, value in result['variants'].items()}, new_simulator_actions=0)), flush=True)


if __name__ == '__main__':
    main()
