"""One center-bowl teacher: replay the successful grasp, then align and descend.

This privileged simulator-state servo generates a demonstration. It is not
the learned policy or the future history reader. Actual actions remain OSC deltas.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from teacher import BOWLS, execute, new_env, object_position, restore, sha, write_json


PREFIX_CONTROLS = 133
ALIGN_CONTROLS = 80
DESCEND_CONTROLS = 160
ALIGN_XY_TOLERANCE = 0.005
HIGH_Z_TOLERANCE = 0.005
ALIGN_STEP = 0.005       # 0.10 m/s command bound at 20 Hz.
DESCEND_XY_STEP = 0.0025 # 0.05 m/s horizontal correction bound.
DESCEND_Z_STEP = 0.0015  # 0.03 m/s vertical command bound.
ROTATION_STEP = 0.025   # 0.50 rad/s command bound.


def bounded(vector, maximum):
    return vector * min(1.0, maximum / max(float(np.linalg.norm(vector)), 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--teacher-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.source / 'behavior_prompting/train_network'
    config = args.output / 'libero_config'
    config.mkdir()
    import yaml
    (config / 'config.yaml').write_text(yaml.safe_dump({
        'benchmark_root': str(args.source / 'deps/LIBERO/libero/libero'),
        'assets': str(args.source / 'deps/LIBERO/libero/libero/assets'),
        'bddl_files': str(base / 'env/libero/bddl_files'),
        'init_states': str(base / 'env/libero/init_files'),
        'datasets': str(args.assets / 'dataset/demonstration_data')}))
    os.environ['LIBERO_CONFIG_PATH'] = str(config)
    from behavior_prompting.train_network.utils.libero_util import discover_and_register_benchmarks
    from robosuite.utils.transform_utils import mat2quat, quat2axisangle
    discover_and_register_benchmarks()

    previous = args.teacher_root / 'layout00-0'
    prior_meta = json.loads((previous / 'result.json').read_text())
    trajectory = previous / 'trajectory.npz'
    assert hashlib.sha256(trajectory.read_bytes()).hexdigest() == prior_meta['trajectory_sha256']
    with np.load(trajectory) as prior:
        actions = prior['actions'][:PREFIX_CONTROLS].copy()
        expected_prefix_state = prior['states_post'][PREFIX_CONTROLS - 1].copy()
    native_meta = json.loads((args.teacher_root / 'native-0/result.json').read_text())
    native_path = args.teacher_root / 'native-0/trajectory.npz'
    assert hashlib.sha256(native_path.read_bytes()).hexdigest() == native_meta['trajectory_sha256']
    with np.load(native_path) as native_source:
        reference_z_offset = float(native_source['body_positions'][-1, 0, 2] - native_source['body_positions'][-1, 2, 2])
    assert native_meta['native_final'] == [1, 0]

    state = np.load(args.geometry / 'layout-00/state.npy')
    spec = json.loads((args.geometry / 'layout-00/geometry.json').read_text())
    assert sha(state) == spec['state_sha256'] and spec['candidate_object_ids'] == BOWLS
    metadata = {
        'kind': 'repaired_center_teacher', 'layout': 0, 'candidate': 0,
        'source_teacher_root': str(args.teacher_root),
        'prefix_trajectory_sha256': prior_meta['trajectory_sha256'],
        'native_height_reference_trajectory_sha256': native_meta['trajectory_sha256'],
        'prefix_actions_sha256': sha(actions), 'prefix_controls': PREFIX_CONTROLS,
        'phase_budgets': {'align': ALIGN_CONTROLS, 'descend': DESCEND_CONTROLS},
        'servo_parameters': {'align_xy_tolerance_m': ALIGN_XY_TOLERANCE,
            'high_z_tolerance_m': HIGH_Z_TOLERANCE, 'align_step_m': ALIGN_STEP,
            'descend_xy_step_m': DESCEND_XY_STEP, 'descend_z_step_m': DESCEND_Z_STEP,
            'rotation_step_rad': ROTATION_STEP, 'minimum_lift_during_alignment_m': 0.06},
        'placement_body_z_above_plate_m': reference_z_offset,
        'servo_parameter_role': 'finite motion budgets, convergence precision and lift guard; no model score threshold',
        'privileged_teacher': True, 'policy_loaded': False, 'uses_hidden_rule': False,
        'stopping_contract': 'first selected native On, same task endpoint as BPP; retain any bounded failure',
        'phase_per_control': [], 'requested_goal_pos': [], 'unclipped_normalized_actions': []}
    env = new_env(base, 0)
    phase = {'name': 'prefix', 'count': 0}
    try:
        initial, _ = restore(env, state, 7)
        hashes = {key: sha(value) for key, value in initial.items()}
        assert hashes == spec['observation_hashes']
        metadata['geometry_observation_hashes_equal'] = True
        metadata['initial_observation_hashes'] = hashes
        initial_bowl_z = float(object_position(env, BOWLS[0])[2])

        def action_fn(index):
            c = env.robots[0].controller
            c.update(force=True)
            if index < PREFIX_CONTROLS:
                metadata['phase_per_control'].append('prefix')
                metadata['requested_goal_pos'].append(None)
                metadata['unclipped_normalized_actions'].append(None)
                return actions[index]
            bowl = object_position(env, BOWLS[0])
            plate = object_position(env, 'plate_1')
            if phase['name'] == 'prefix':
                phase.update(name='align', count=0, high_z=float(c.ee_pos[2]), orientation=c.ee_ori_mat.copy())
                metadata['prefix_state_equal'] = bool(np.array_equal(env.get_sim_state(), expected_prefix_state))
                metadata['prefix_actual_state_sha256'] = sha(env.get_sim_state())
                metadata['prefix_expected_state_sha256'] = sha(expected_prefix_state)
                metadata['alignment_start'] = {'bowl': bowl.tolist(), 'plate': plate.tolist(),
                    'controller_eef': c.ee_pos.tolist(), 'controller_orientation': c.ee_ori_mat.tolist()}
            xy_error = plate[:2] - bowl[:2]
            if (phase['name'] == 'align' and np.linalg.norm(xy_error) <= ALIGN_XY_TOLERANCE
                    and abs(c.ee_pos[2] - phase['high_z']) <= HIGH_Z_TOLERANCE):
                phase.update(name='descend', count=0)
                metadata['descent_start_control_index'] = index
            if phase['name'] == 'align':
                delta_pos = bounded(np.r_[xy_error, phase['high_z'] - c.ee_pos[2]], ALIGN_STEP)
            else:
                delta_pos = np.r_[bounded(xy_error, DESCEND_XY_STEP),
                    np.clip(plate[2] + reference_z_offset - bowl[2], -DESCEND_Z_STEP, DESCEND_Z_STEP)]
            delta_rot = bounded(quat2axisangle(mat2quat(phase['orientation'] @ c.ee_ori_mat.T)), ROTATION_STEP)
            delta = np.r_[delta_pos, delta_rot]
            scaled = ((delta - (c.output_max + c.output_min) / 2)
                      / ((c.output_max - c.output_min) / (c.input_max - c.input_min))
                      + (c.input_max + c.input_min) / 2)
            phase['count'] += 1
            metadata['phase_per_control'].append(phase['name'])
            metadata['requested_goal_pos'].append((c.ee_pos + delta_pos).tolist())
            metadata['unclipped_normalized_actions'].append(scaled.tolist())
            return np.r_[np.clip(scaled, c.input_min, c.input_max), 1.0]

        def stop_fn(index, goals):
            if goals[0]:
                return 'selected_native_on'
            if index < PREFIX_CONTROLS:
                return None
            if phase['name'] == 'align' and object_position(env, BOWLS[0])[2] < initial_bowl_z + 0.06:
                return 'lost_lift_during_alignment'
            if phase['name'] == 'align' and phase['count'] >= ALIGN_CONTROLS:
                return 'alignment_budget_exhausted'
            if phase['name'] == 'descend' and phase['count'] >= DESCEND_CONTROLS:
                return 'descent_budget_exhausted'
            return None

        _, receipt = execute(env, args.output / 'layout00-0', 0, action_fn,
                             PREFIX_CONTROLS + ALIGN_CONTROLS + DESCEND_CONTROLS,
                             metadata, stop_fn=stop_fn)
        write_json(args.output / 'result.json', {'complete': True, 'policy_loaded': False, 'records': [receipt]})
    finally:
        env.close()


if __name__ == '__main__':
    main()
