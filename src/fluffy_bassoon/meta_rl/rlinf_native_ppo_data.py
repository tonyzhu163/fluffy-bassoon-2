"""Versioned native-PPO retention contract and small CPU history reconstruction.

Actual behavior actions are AD targets. DPT query labels are intentionally
absent until a preselected final teacher is evaluated and queried offline.
Gaussian draws of an updated action expert are not frozen-SFT steering labels.
"""
FORMAT = 'rlinf_native_ppo_history_capture_v1'

FIELD_PATHS = {
    'raw_current_rgb_state_instruction': 'traces/*.pt.gz:current_observation',
    'raw_next_rgb_state_instruction': 'traces/*.pt.gz:outputs[0][-1]',
    'raw_final_before_autoreset': 'traces/*.pt.gz:outputs[4][-1].final_observation when present; otherwise outputs[0][-1]',
    'initial_source_state_rng': 'initial-states/source-*.pt:initial_observation, reset_state_ids, reset_rng, sampler_generator, sampler_cursor',
    'task_case_episode_time': 'traces:task_key,rank,global_env_indices,reset_state_ids,episode_serials,round_index,chunk_index,primitive_time_before',
    'native14D_motor_action': 'traces:proposed_motor_actions plus outputs[4][-1].valid_action_mask; inactive padded rows are preserved',
    'actual_duration': 'traces:executed_or_conservative_duration (measured_valid_mask marks actual versus conservative)',
    'raw_reward_term_trunc': 'traces:outputs[1],outputs[2],outputs[3]',
    'raw_untrimmed_bootstrap': 'training-rollouts/*.pt:rollout_batch.prev_values,dones,terminations,truncations (final rows retained)',
    'processed_current_inputs': 'training-rollouts:rollout_batch.forward_inputs (raw replay inputs plus tokenized prompt)',
    'policy_statistics': 'training-rollouts:rollout_batch.prev_logprobs,prev_values,advantages,returns,loss_mask,loss_mask_sum,versions',
    'flow_chain32D': 'training-rollouts:rollout_batch.forward_inputs.chains,denoise_inds,model_action',
    'actual_sde_parameters_rng': 'sampling/*.pt.gz:actual_gaussian_parameters,rng_before,rng_after,forward_sampling_fields',
    'actual_processed_model_state': 'sampling/*.pt.gz:processed_model_state (the actual normalized/padded Observation.state passed into native train prediction)',
    'producer_mapping': 'training-rollouts:collection_iteration=i,producing_policy_iteration=i-1; checkpoint global_step_i is AFTER update i',
    'chronology': 'pre-SGD [chunk_index, round_index*local_env_count+env_index,...]; traces retain rank/round/chunk/env identity',
    'roles': 'traces/sampling:evaluation_data; training-rollouts never contains evaluate() panels',
    'actor_config_and_units': 'resolved-native-config.json plus preflight.json norm/config/source identities',
    'final_teacher_label': 'not generated during collection; later immutable label generation must bind final checkpoint/config/norm and query sampling seeds',
}


def shape_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype))
    if isinstance(value, dict):
        return {key: shape_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [shape_tree(item) for item in value]
    return dict(type=type(value).__name__)


def collection_contract():
    return dict(format=FORMAT, field_paths=FIELD_PATHS,
        sampling_unchanged=True, optimizer_unchanged=True,
        ad_direct_action_or_bridge_ready=True,
        frozen_sft_noise_steering_labels_ready=False, dpt_final_teacher_labels_ready=False,
        teacher_quality='candidate only; robot competence measured separately, never inferred from checkpoint existence',
        raw_state_units='native RoboTwin public qpos coordinates, preserved without logger conversion',
        action_units='native decoded RoboTwin ALOHA14 commands; exact coordinate/delta/normalization semantics bound by actor config, norm stats and author decode_actions code',
        model_action_units='OpenPI normalized padded32D flow state; distinct from decoded14D robot commands',
        noise_compatibility='Native PPO changes action-expert weights. Its initial Gaussian is not a steering label for the unchanged SFT motor. Direct action/bridge objectives are supported; frozen-motor noise objectives require an explicit inverse/differentiable compatibility mapping later.',
        dpt_later='Retain behavior trajectories verbatim. Query a preselected final/source teacher on recorded query observations with checkpoint/config/norm hashes and fixed sampling seeds, without new environment interactions. These are teacher action labels, distinct from behavior actions.',
        reset_limit='Public initial observation+scene reset ID+sampler/global RNG are saved. No unsupported native simulator full-physics snapshot or mid-episode exact resume is claimed.')


def record_gaussian_parameters(original, sink, clone):
    """Observe actual returned sampler parameters without a second evaluation."""
    def recorded(*args, **kwargs):
        mean, std = original(*args, **kwargs)
        active = sink()
        if active is not None:
            active.append(dict(index=clone(args[2]), noise_method=kwargs['noise_method'],
                noise_level=float(kwargs['noise_level']), num_steps=int(kwargs['num_steps']),
                mean=clone(mean), std=clone(std)))
        return mean, std
    return recorded


def reconstruct_rows(traces, teacher_reference):
    """Reconstruct lossless CPU action histories; teacher labels remain deferred.

    This small reader is exercised by the synthetic contract fixture. Its
    explicit raw masks retain post-termination padded records, while context
    indices include only decisions that physically executed actions.
    """
    import torch
    required_teacher = ('checkpoint_path', 'checkpoint_sha256', 'actor_config_sha256',
        'norm_stats_sha256', 'sampling_seed_rule')
    if any(key not in teacher_reference for key in required_teacher):
        raise ValueError('DPT query reference must bind checkpoint/config/norm and sampling seeds')
    def item(value, index):
        if isinstance(value, torch.Tensor):
            return value[index].clone()
        if isinstance(value, dict):
            return {key: item(child, index) for key, child in value.items()}
        if isinstance(value, (tuple, list)):
            if value and all(isinstance(child, str) for child in value):
                return value[index]
            return type(value)(item(child, index) for child in value)
        return value
    rows, contexts = [], {}
    ordered = sorted((trace for trace in traces if not trace['evaluation_data']),
        key=lambda trace: (trace['collection_iteration'], trace['round_index'], trace['chunk_index'], trace['rank']))
    for trace in ordered:
        outputs = trace['outputs']
        mask = outputs[4][-1].get('valid_action_mask')
        if mask is None or not trace['measured_valid_mask']:
            raise ValueError('Training history must retain actual action-duration masks')
        for index in trace['local_env_indices']:
            stream = (trace['task_key'], trace['rank'], index)
            history = contexts.setdefault(stream, [])
            duration = int(mask[index].sum())
            row = dict(task_key=trace['task_key'], rank=trace['rank'], env_index=index,
                collection_iteration=trace['collection_iteration'],
                producing_policy_iteration=trace['producing_policy_iteration'],
                round_index=trace['round_index'], chunk_index=trace['chunk_index'],
                episode_serial=int(trace['episode_serials'][index]),
                reset_state_id=int(trace['reset_state_ids'][index]),
                primitive_time_before=int(trace['primitive_time_before'][index]),
                current_observation=item(trace['current_observation'], index),
                next_observation=item(outputs[0][-1], index),
                final_observation=item(outputs[4][-1].get('final_observation', outputs[0][-1]), index),
                submitted_actions=trace['proposed_motor_actions'][index].clone(),
                executed_actions=trace['proposed_motor_actions'][index][mask[index].bool()].clone(),
                valid_action_mask=mask[index].clone(), duration=duration,
                raw_rewards=outputs[1][index].clone(), terminations=outputs[2][index].clone(),
                truncations=outputs[3][index].clone(), history_row_indices=list(history),
                dpt_query_observation=item(trace['current_observation'], index),
                dpt_label_teacher=dict(teacher_reference), dpt_teacher_action=None,
                ad_target_kind='executed14D_behavior_action', frozen_sft_noise_target=None)
            rows.append(row)
            if duration:
                history.append(len(rows)-1)
    return rows
