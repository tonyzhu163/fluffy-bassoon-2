"""Four simulator episodes: native demo replay and fixed geometric retargeting.

No BPP weights are loaded. All saved actions are actually executed OSC deltas.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

CUSTOM = Path(__file__).resolve().parent.parent / 'libero-gen-custom-layouts-20261111-r1'
sys.path.insert(0, str(CUSTOM))
from geometry import BOWLS, SPLIT, TASKS, object_position, sha

PUBLIC = ['agentview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos',
          'robot0_eef_quat', 'robot0_gripper_qpos']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def native(env):
    return [int(env.env._eval_predicate(['on', bowl, 'plate_1'])) for bowl in BOWLS]


def observation(env):
    env.env._update_observables(force=True)
    raw = env.env._get_observations()
    return env.prepare_observation(raw), raw['robot0_joint_pos'].copy()


def controller_receipt(env):
    c = env.robots[0].controller
    assert c.use_delta and c.impedance_mode == 'fixed' and c.use_ori
    return {name: np.asarray(getattr(c, name)).tolist() for name in
            ['input_min', 'input_max', 'output_min', 'output_max', 'kp', 'kd']} | {
                'control_delta': True, 'control_freq': env.env.control_freq,
                'orientation_composition': 'R_goal = R_delta @ R_current',
                'controller_class': type(c).__name__}


def new_env(base, candidate):
    from behavior_prompting.train_network.env.libero.env_wrapper import ControlEnvForRunner
    env = ControlEnvForRunner(
        bddl_file_name=str(base / 'env/libero/bddl_files' / SPLIT / (TASKS[candidate] + '.bddl')),
        camera_heights=128, camera_widths=128, render_size=None,
        has_renderer=False, has_offscreen_renderer=True, control_delta=True,
        render_gpu_device_id=0, ignore_done=True, horizon=1000)
    env.env.hard_reset = False
    return env


def resolve_source_xml(xml, asset_root):
    """Rebase known author-local paths only; leave every model attribute intact."""
    import importlib.util
    import xml.etree.ElementTree as ET
    robot_root = Path(importlib.util.find_spec('robosuite').origin).parent
    tree = ET.fromstring(xml)
    mappings = []
    for element in tree.iter():
        old = element.get('file')
        if old is None:
            continue
        target = Path(old)
        if '/robosuite/' in old:
            target = robot_root / old.rsplit('/robosuite/', 1)[1]
        elif '/chiliocosm/assets/' in old:
            target = asset_root / old.split('/chiliocosm/assets/', 1)[1]
        elif '/libero/libero/assets/' in old:
            target = asset_root / old.split('/libero/libero/assets/', 1)[1]
        assert target.is_file(), f'Unresolved source asset: {old} -> {target}'
        element.set('file', str(target))
        mappings.append({'source': old, 'resolved': str(target),
                         'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    converted = ET.tostring(tree, encoding='unicode')
    return converted, {'original_xml_sha256': hashlib.sha256(xml.encode()).hexdigest(),
                       'resolved_xml_sha256': hashlib.sha256(converted.encode()).hexdigest(),
                       'file_references': mappings, 'all_resolved': True,
                       'only_model_change': 'file attributes rebased to installed official assets'}


def restore(env, state, seed, model_xml=None):
    env.seed(int(seed))
    env.env.reset()  # Deliberately bypass the upstream exception-swallowing reset.
    if model_xml is not None:
        env.seed(int(seed))
        env.reset_from_xml_string(model_xml)
        env.sim.reset()
    env.regenerate_obs_from_state(state)
    c = env.robots[0].controller
    c.update(force=True)
    c.goal_pos = c.ee_pos.copy()
    c.goal_ori = c.ee_ori_mat.copy()
    return observation(env)


def execute(env, folder, candidate, action_fn, count, metadata, stop_fn=None):
    """Record each real control; an optional teacher callback can stop the schedule."""
    folder.mkdir()
    (folder / 'model.xml').write_text(env.sim.model.get_xml())
    arrays = {k: [] for k in ['actions', 'states_pre', 'states_post', 'native_trace',
                              'goal_pos', 'goal_ori', 'body_positions', 'joint_pos_pre', 'joint_pos_post']}
    arrays.update({prefix + key: [] for prefix in ['pre_', 'post_'] for key in PUBLIC})
    metadata['controller'] = controller_receipt(env)
    metadata['initial_positions'] = {name: object_position(env, name).tolist() for name in BOWLS + ['plate_1']}
    metadata['initial_state_sha256'] = sha(env.get_sim_state())
    metadata['initial_native'] = native(env)
    for index in range(count):
        before, joints_pre = observation(env)
        before = {key: np.array(value, copy=True) for key, value in before.items()}
        arrays['states_pre'].append(env.get_sim_state().copy())
        action = np.asarray(action_fn(index), dtype=np.float64)
        assert action.shape == (7,) and np.isfinite(action).all()
        obs, reward, done, info = env.step(action)
        after, joints_post = observation(env)
        arrays['actions'].append(action.copy())
        arrays['states_post'].append(env.get_sim_state().copy())
        arrays['native_trace'].append(native(env))
        arrays['goal_pos'].append(env.robots[0].controller.goal_pos.copy())
        arrays['goal_ori'].append(env.robots[0].controller.goal_ori.copy())
        arrays['body_positions'].append([object_position(env, name) for name in BOWLS + ['plate_1']])
        arrays['joint_pos_pre'].append(joints_pre)
        arrays['joint_pos_post'].append(joints_post)
        for key in PUBLIC:
            arrays['pre_' + key].append(np.array(before[key], copy=True))
            arrays['post_' + key].append(np.array(after[key], copy=True))
        if (index + 1) % 50 == 0:
            print('TEACHER_PROGRESS', folder.name, index + 1, arrays['native_trace'][-1], flush=True)
        if stop_fn is not None:
            reason = stop_fn(index, arrays['native_trace'][-1])
            if reason:
                metadata['termination_reason'] = reason
                break
    arrays = {key: np.stack(value) for key, value in arrays.items()}
    count = len(arrays['actions'])
    path = folder / 'trajectory.npz'
    np.savez_compressed(path, **arrays)
    metadata.update({'candidate': candidate, 'count': count,
                     'selected_physical_index': candidate,
                     'native_any': arrays['native_trace'].max(axis=0).astype(int).tolist(),
                     'native_final': arrays['native_trace'][-1].astype(int).tolist(),
                     'trajectory_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'trajectory_bytes': path.stat().st_size,
                     'array_hashes': {k: sha(v) for k, v in arrays.items()},
                     'image_convention': 'ControlEnv prepare_observation: upright RGB; flip vertically for source HDF5',
                     'stopping': ('fixed complete source schedule, independent of native outcome' if stop_fn is None
                                  else 'documented teacher servo: selected native goal or bounded phase failure')})
    metadata['selected_native_success'] = metadata['native_any'][candidate]
    write_json(folder / 'result.json', metadata)
    # Small video makes physical selection and placement inspectable without transferring the NPZ.
    import imageio
    with imageio.get_writer(str(folder / 'video.mp4'), fps=20, codec='libx264', quality=7) as writer:
        for i in range(count):
            writer.append_data(np.concatenate([arrays['post_' + key][i] for key in PUBLIC[:2]], axis=1))
    print('TEACHER_EPISODE', folder.name, count, metadata['native_any'], flush=True)
    return arrays, metadata


def alignment(source, actual):
    from robosuite.utils.transform_utils import quat2axisangle
    result = {}
    for prefix in ['pre', 'post']:
        actual_ori = np.stack([quat2axisangle(q) for q in actual[prefix + '_robot0_eef_quat']])
        fields = {'ee_pos': actual[prefix + '_robot0_eef_pos'], 'ee_ori': actual_ori,
                  'gripper_states': actual[prefix + '_robot0_gripper_qpos'],
                  'joint_states': actual['joint_pos_' + prefix]}
        result[prefix] = {name: {'rmse': float(np.sqrt(np.mean((source['obs/' + name][()] - values) ** 2))),
                                 'max_abs': float(np.max(np.abs(source['obs/' + name][()] - values)))}
                          for name, values in fields.items()}
        # Pixel equality can be affected by driver/rendering differences; report rather than enforce it.
        for old, new in [('agentview_rgb', 'agentview_image'), ('eye_in_hand_rgb', 'robot0_eye_in_hand_image')]:
            saved = source['obs/' + old][()].astype(np.float32)
            live = actual[prefix + '_' + new][:, ::-1].astype(np.float32)
            result[prefix][old] = {'mean_abs_pixel_difference': float(np.abs(saved - live).mean())}
        result[prefix]['state_rmse'] = float(np.sqrt(np.mean((source['states'][()] - actual['states_' + prefix]) ** 2)))
    return result


def retarget_schedule(source, source_meta, env, actions):
    """Fixed spatial warp, with source motion defining the carry blend."""
    c = env.robots[0].controller
    selected = source_meta['candidate']
    source_bowl = np.asarray(source_meta['initial_positions'][BOWLS[selected]])
    source_plate = np.asarray(source_meta['initial_positions']['plate_1'])
    delta_bowl = object_position(env, BOWLS[selected]) - source_bowl
    delta_plate = object_position(env, 'plate_1') - source_plate
    # Table heights match; retain source vertical grasp/carry/place trajectory.
    delta_bowl[2] = delta_plate[2] = 0
    close = int(np.flatnonzero(actions[:, -1] > 0)[0])
    reopened = np.flatnonzero((np.arange(len(actions)) > close) & (actions[:, -1] < 0))
    reopen = int(reopened[0]) if len(reopened) else len(actions)
    line = source_plate[:2] - source_bowl[:2]
    assert np.dot(line, line) > 1e-8
    raw_progress = (source['body_positions'][:, selected, :2] - source_bowl[:2]) @ line / np.dot(line, line)
    achieved = float(np.max(raw_progress[close:max(close + 1, reopen)]))
    assert achieved > 1e-6, 'Successful source had no forward carry displacement'
    progress = np.clip(raw_progress / achieved, 0, 1)
    # Do not reverse the warp when the bowl settles after placement.
    progress = np.maximum.accumulate(progress)
    weights, positions = [], []
    for i, goal in enumerate(source['goal_pos']):
        if i < close:
            approach = i / max(close, 1)
            weights.append([approach, 0.0])
            displacement = approach * delta_bowl
        else:
            carry = float(progress[i]) if i < reopen else 1.0
            weights.append([1.0 - carry, carry])
            displacement = (1.0 - carry) * delta_bowl + carry * delta_plate
        positions.append(goal + displacement)
    positions = np.stack(positions)
    from robosuite.utils.transform_utils import mat2quat, quat2axisangle
    unclipped = []
    def action_fn(step):
        i = step // 2
        c.update(force=True)
        delta = np.concatenate((positions[i] - c.ee_pos,
                                quat2axisangle(mat2quat(source['goal_ori'][i] @ c.ee_ori_mat.T))))
        # Invert the controller's exact affine input/output scaling, then bound inputs.
        scaled = (delta - (c.output_max + c.output_min) / 2) / ((c.output_max - c.output_min) / (c.input_max - c.input_min)) + (c.input_max + c.input_min) / 2
        unclipped.append(scaled.tolist())
        return np.concatenate((np.clip(scaled, c.input_min, c.input_max), actions[i, -1:]))
    meta = {'method': 'source-controller-goals translated with bowl/plate anchors',
            'source_index_per_control': np.repeat(np.arange(len(actions)), 2).tolist(),
            'controls_per_source_target': 2, 'first_close_index': close, 'first_reopen_index': reopen,
            'bowl_displacement': delta_bowl.tolist(), 'plate_displacement': delta_plate.tolist(),
            'carry_progress_normalizer': achieved, 'raw_source_carry_progress': raw_progress.tolist(),
            'anchor_weights': weights, 'requested_goal_pos': positions.tolist(),
            'requested_goal_ori': source['goal_ori'].tolist(),
            'unclipped_normalized_actions': unclipped,
            'phase_rule': 'linear zero-to-bowl shift before first close; monotone source bowl progress normalized by its achieved pre-release destination during carry; full plate shift after reopen',
            'uses_hidden_rule': False, 'uses_reward_to_change_actions': False}
    return action_fn, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.source / 'behavior_prompting/train_network'
    config = args.output / 'libero_config'
    config.mkdir()
    import yaml
    (config / 'config.yaml').write_text(yaml.safe_dump({'benchmark_root': str(args.source / 'deps/LIBERO/libero/libero'),
        'assets': str(args.source / 'deps/LIBERO/libero/libero/assets'),
        'bddl_files': str(base / 'env/libero/bddl_files'), 'init_states': str(base / 'env/libero/init_files'),
        'datasets': str(args.assets / 'dataset/demonstration_data')}))
    os.environ['LIBERO_CONFIG_PATH'] = str(config)
    from behavior_prompting.train_network.utils.libero_util import discover_and_register_benchmarks
    discover_and_register_benchmarks()
    import h5py
    records = []
    for candidate in [0, 1]:
        file = args.assets / 'dataset/demonstration_data' / SPLIT / (TASKS[candidate] + '_demo.hdf5')
        env = new_env(base, candidate)
        try:
            with h5py.File(file, 'r') as data:
                demo = data['data/demo_0']
                actions = demo['actions'][()]
                state = demo['states'][0]
                init = np.asarray(demo.attrs['init_state'])
                seed = int(demo.attrs.get('seed', 7))
                if seed < 0:
                    seed = 7
                xml = demo.attrs['model_file']
                if isinstance(xml, bytes):
                    xml = xml.decode()
                resolved_xml, path_receipt = resolve_source_xml(xml, args.source / 'deps/LIBERO/libero/libero/assets')
                write_json(args.output / f'source-paths-{candidate}.json', path_receipt)
                restore(env, state, seed, resolved_xml)
                metadata = {'kind': 'native_source_replay', 'source_file': str(file), 'source_demo': 'demo_0',
                            'source_seed': seed, 'candidate': candidate,
                            'source_state0_sha256': sha(state), 'source_init_attr_equals_state0': bool(np.array_equal(init, state)),
                            'source_model_xml_sha256': hashlib.sha256(xml.encode()).hexdigest()}
                source, source_meta = execute(env, args.output / f'native-{candidate}', candidate,
                                              lambda index: actions[index], len(actions), metadata)
                source_meta['alignment'] = alignment(demo, source)
                write_json(args.output / f'native-{candidate}/result.json', source_meta)
                records.append(source_meta)
        finally:
            env.close()
        if not source_meta['native_final'][candidate]:
            print('RETARGET_SKIPPED_UNSUCCESSFUL_SOURCE', candidate, flush=True)
            continue
        env = new_env(base, candidate)
        try:
            state_file = args.geometry / 'layout-00/state.npy'
            spec = json.loads((args.geometry / 'layout-00/geometry.json').read_text())
            state = np.load(state_file)
            assert sha(state) == spec['state_sha256']
            assert spec['candidate_object_ids'] == BOWLS
            initial_obs, _ = restore(env, state, 7)
            initial_hashes = {key: sha(value) for key, value in initial_obs.items()}
            assert initial_hashes == spec['observation_hashes'], 'Custom initial model/render differs from geometry receipt'
            action_fn, metadata = retarget_schedule(source, source_meta, env, actions)
            metadata.update({'kind': 'retargeted_teacher', 'layout': 0, 'source_demo': 'demo_0',
                             'source_file': str(file), 'geometry_state_sha256': sha(state),
                             'initial_observation_hashes': initial_hashes,
                             'geometry_observation_hashes_equal': initial_hashes == spec['observation_hashes']})
            _, receipt = execute(env, args.output / f'layout00-{candidate}', candidate, action_fn, 2 * len(actions), metadata)
            records.append(receipt)
        finally:
            env.close()
    write_json(args.output / 'result.json', {'complete': True, 'policy_loaded': False, 'records': records})


if __name__ == '__main__':
    main()
