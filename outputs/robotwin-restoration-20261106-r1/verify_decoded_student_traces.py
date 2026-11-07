"""CPU readback of real trained-student support/query commands and feedback.

Reconstruct tokens directly from physical records, without calling the online
controller or session. No motor inference, simulator, SGD or TEST access.
"""
import gc
import gzip
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from evaluate_decoded_students import BASE, ROOT, policies
from fit_decoded_student import write
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha
from fluffy_bassoon.rl.robotwin_execution import chunk_execution
from fluffy_bassoon.rl.robotwin_noise import _observation_hash


def main():
    assert sys.platform == 'linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    root = BASE/'studentonline01-signals'
    output = BASE/'studentonline01-readback01'
    output.mkdir()
    started = time.monotonic()
    report = dict(status='checking',gpu_seconds=0,optimizer_updates=0,test_tensor_loaded=False,
        motor_or_simulator_constructed=False,learning_improvement_claim=False,
        source_commit=os.environ['TASK_SOURCE'],arms={},file_sha256={})
    write(output/'RESULT.json',report)
    try:
        plan = json.loads((ROOT/'STUDENTONLINE01-CONTRACT.json').read_bytes())
        result = json.loads((root/'RESULT.json').read_bytes())
        assert result['status'] == 'complete_trained_student_runtime_qualification'
        assert result['contract_sha256'] == sha(ROOT/'STUDENTONLINE01-CONTRACT.json')
        assert json.loads((root/'EXIT.json').read_bytes())['exit_code'] == 0
        assert result['real_episodes'] == 16 and result['optimizer_updates'] == 0
        assert not result['test_tensor_loaded'] and result['deployment_native_batch'] == 2
        tolerance = plan['cpu_replay_tolerances']
        report['cpu_replay_tolerances'] = tolerance
        report['gpu_result_sha256'] = sha(root/'RESULT.json')
        loaded = policies(plan)
        initial, first_bases = {}, {}

        def read(name):
            path = root/name
            assert path.resolve().parent == root.resolve() and name not in report['file_sha256']
            report['file_sha256'][name] = sha(path)
            with gzip.open(path,'rb') as stream:
                return torch.load(stream,map_location='cpu',weights_only=False)

        for arm in plan['arms']:
            record = result['arms'][arm]
            method = 'static_behavior_cloning' if arm == 'frozen_base' else arm
            policy = loaded[method]
            if arm == 'frozen_base':
                # The following static arm needs its original trained parameters.
                import copy
                policy = copy.deepcopy(policy)
                with torch.no_grad():
                    policy.actor.weight.zero_();policy.actor.bias.zero_()
            policy.eval().requires_grad_(False)
            static = method == 'static_behavior_cloning'
            assert record['mode'] == ('static' if static else 'full')
            assert record['owned_workers_closed'] and record['frozen_weights_and_statistics_unchanged']
            layout = policy.config.layout
            feedback = np.zeros((2,layout.feedback_dim),np.float32)
            state = policy.core.initial_state(2)
            first = np.ones(2,np.bool_)
            transitions_count = targets = physics_steps = 0
            max_command_error = max_normalized_error = max_state_error = 0.
            for episode in range(2):
                ep = record['episodes'][episode]
                assert ep['index'] == episode
                for lane,receipt in enumerate(ep['resets']):
                    assert receipt['actual_seed'] == plan['source_dev_cases'][lane]
                    key = episode,lane
                    if arm == 'frozen_base':initial[key] = receipt['public_observation_sha256']
                    assert initial[key] == receipt['public_observation_sha256']
                active = np.ones(2,np.bool_)
                starts = np.ones(2,np.bool_)
                chunks = np.zeros(2,np.int64)
                success = np.zeros(2,np.bool_)
                previous = [None,None]
                entries = [row for row in result['traces'] if row['arm'] == arm and row['episode'] == episode]
                assert entries
                for entry in entries:
                    intent, rows = read(entry['intent']), read(entry['transitions'])
                    native = intent['native_batch']
                    assert native['batch_size'] == 2 and native['lanes'] == [0,1]
                    assert native['fixed_native_batch'] and native['active_lanes'] == np.flatnonzero(active).tolist()
                    assert np.array_equal(intent['episode_starts'],starts)
                    features, bases, commands = (torch.as_tensor(intent[key],dtype=torch.float32)
                        for key in ('features','bases','commands'))
                    assert bases.shape == commands.shape == (2,32,14)
                    assert features.shape == (2,2062)
                    if not chunks.any():
                        if arm == 'frozen_base':first_bases[episode] = bases.clone()
                        assert torch.equal(bases,first_bases[episode])
                    expected_ids = [f'case-{seed}/attempt-{episode}/chunk-{int(chunks[lane])}'
                        for lane,seed in enumerate(plan['source_dev_cases'])]
                    assert native['decision_ids'] == expected_ids
                    supplied_feedback = np.zeros_like(feedback) if static else feedback.copy()
                    token = torch.cat((features,bases.flatten(1),torch.from_numpy(supplied_feedback),
                        torch.from_numpy(starts[:,None]).float()),-1)
                    reset = torch.from_numpy((first | static) & active)
                    with torch.no_grad():
                        residual,_,after = policy.step(token,state,reset=reset,deterministic=True)
                    mask = torch.from_numpy(active)
                    state = {key:torch.where(mask.reshape(-1,*([1]*(value.ndim-1))),value,state[key])
                             for key,value in after.items()}
                    predicted = bases+torch.where(mask[:,None,None],residual,0.)
                    error = (predicted-commands).abs()
                    normalized = error/policy.residual_radius
                    max_command_error = max(max_command_error,float(error.max()))
                    max_normalized_error = max(max_normalized_error,float(normalized.max()))
                    assert error.max() <= tolerance['command_absolute']
                    assert normalized.max() <= tolerance['command_relative_to_radius']
                    if arm == 'frozen_base':assert torch.equal(commands,bases)
                    assert [row['native_batch']['row'] for row in rows] == np.flatnonzero(active).tolist()
                    for row in rows:
                        lane = row['native_batch']['row']
                        assert row['native_batch'] == dict(native,row=lane)
                        assert row['chunk_index'] == chunks[lane]
                        assert row['case_id'] == f'case-{plan["source_dev_cases"][lane]}/attempt-{episode}'
                        assert np.array_equal(row['requested_actions'],commands[lane].numpy())
                        assert np.array_equal(row['base_actions'],bases[lane].numpy())
                        assert np.array_equal(row['public_features'],features[lane].numpy())
                        obs = _observation_hash(row['observation'])
                        assert obs == (initial[episode,lane] if previous[lane] is None else previous[lane])
                        previous[lane] = _observation_hash(row['next_observation'])
                        execution = chunk_execution({'fb2_execution':row['execution']},requested_targets=32,
                            timestep_seconds=row['execution']['timestep_seconds'])
                        assert execution.duration_seconds == row['physics_duration_seconds'] > 0
                        assert execution.valid_target_mask == tuple(row['measured_valid_target_mask'])
                        feedback[lane] = np.concatenate((row['requested_actions'].reshape(-1),
                            row['measured_valid_target_mask'].astype(np.float32),
                            np.array([row['reward'],row['terminated'],row['truncated'],
                                      row['physics_duration_seconds']],np.float32)))
                        chunks[lane] += 1;first[lane] = starts[lane] = False
                        success[lane] |= bool(row['audit']['success'])
                        active[lane] = not (row['terminated'] or row['truncated'])
                        transitions_count += 1;targets += execution.delivered_targets
                        physics_steps += execution.physics_steps
                assert not active.any()
                for lane,value in enumerate(ep['lanes']):
                    assert value['case'] == plan['source_dev_cases'][lane] and value['complete']
                    assert value['role'] == ('support' if episode == 0 else 'query')
                    assert value['success'] == bool(success[lane]) and len(value['transitions']) == chunks[lane]
                boundary = read(f'{arm}-episode-{episode}-controller.pt.gz')
                assert boundary['mode'] == record['mode']
                assert boundary['policy_export'] == plan['policy_exports'][method]
                for lane,encoder in enumerate(boundary['encoders']):
                    assert not encoder['pending'] and not encoder['first'] and encoder['previous_terminal']
                    assert np.array_equal(encoder['previous'],feedback[lane])
                assert set(state) == set(boundary['state'])
                for key,value in state.items():
                    saved = boundary['state'][key]
                    if value.is_floating_point():
                        max_state_error = max(max_state_error,float((value-saved).abs().max()))
                        assert torch.allclose(value,saved,atol=tolerance['state_absolute'],rtol=tolerance['state_relative'])
                    else:assert torch.equal(value,saved)
            report['arms'][arm] = dict(episodes=4,physical_decisions=transitions_count,
                delivered_targets=targets,physics_steps=physics_steps,max_command_absolute_error=max_command_error,
                max_command_error_relative_to_radius=max_normalized_error,max_memory_absolute_error=max_state_error,
                physical_observation_chain_exact=True,feedback_exact=True,support_memory_replayed=True)
            del policy,state;gc.collect()
        assert sum(row['physical_decisions'] for row in report['arms'].values()) <= 16*6
        assert result['native_motor_calls'] == len(result['traces'])
        assert not torch.cuda.is_initialized()
        report.update(status='passed',all16_real_episodes_replayed=True,
            paired_initial_observations_and_base_proposals_exact=True,
            controller_tokens_reconstructed_without_online_controller=True)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report['seconds'] = time.monotonic()-started
        write(output/'RESULT.json',report)


if __name__ == '__main__':main()
