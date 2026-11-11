"""Write a real successful teacher execution in BPP's native HDF5 format.

Run in the qualified Linux environment. The official dataset loader still
does image resizing, rotation conversion, action chunking and normalization.
"""
import argparse
import hashlib
import json
from pathlib import Path


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-hdf5', type=Path, required=True)
    parser.add_argument('--trajectory', type=Path)
    parser.add_argument('--model-xml', type=Path)
    parser.add_argument('--alignment', choices=['pre', 'post'])
    parser.add_argument('--selected-bowl', type=int, choices=[0, 1])
    parser.add_argument('--stop-at-first-native-success', action='store_true',
                        help='Use the first native goal endpoint, matching policy episode stopping')
    parser.add_argument('--copy-reference', action='store_true',
                        help='Copy original demo0 unchanged to qualify the prompt loading path')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Use a fresh prompt output identity'
    if not args.copy_reference:
        assert all(x is not None for x in [args.trajectory, args.model_xml,
                                          args.alignment, args.selected_bowl])
    import h5py
    import numpy as np
    from robosuite.utils.transform_utils import quat2axisangle

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.source_hdf5, 'r') as source, h5py.File(args.output, 'x') as out:
        data = out.create_group('data')
        # The official task/environment loader reads this metadata. No hidden
        # rule or privileged state is added to the policy's observation fields.
        data.attrs['env_args'] = source['data'].attrs['env_args']
        if args.copy_reference:
            source.copy(source['data/demo_0'], data, name='demo_0')
            count = len(data['demo_0/actions'])
            receipt = {'kind': 'unchanged_native_demo0_reference', 'controls': count}
        else:
            with np.load(args.trajectory, allow_pickle=False) as trajectory:
                actions = trajectory['actions']
                full_count = count = len(actions)
                native = trajectory['native_trace']
                assert actions.shape == (count, 7) and np.isfinite(actions).all()
                assert native.shape == (count, 2) and np.isin(native, [0, 1]).all()
                full_final_native = native[-1].astype(int).tolist()
                if args.stop_at_first_native_success:
                    successes = np.flatnonzero(native[:, args.selected_bowl])
                    assert len(successes), 'No intended native goal occurred'
                    count = int(successes[0]) + 1
                    actions, native = actions[:count], native[:count]
                expected = [int(i == args.selected_bowl) for i in range(2)]
                assert count > 0 and native[-1].tolist() == expected, 'Teacher must demonstrate the intended distinct goal'
                # ControlEnv prepare_observation returns upright images. HDF5
                # stores them upside down; the official loader flips once.
                prefix = args.alignment + '_'
                pos = trajectory[prefix + 'robot0_eef_pos'][:count]
                quat = trajectory[prefix + 'robot0_eef_quat'][:count]
                gripper = trajectory[prefix + 'robot0_gripper_qpos'][:count]
                assert pos.shape == (count, 3) and quat.shape == (count, 4)
                assert gripper.shape == (count, 2)
                ori = np.stack([quat2axisangle(q.copy()) for q in quat])
                demo = data.create_group('demo_0')
                demo.attrs['num_samples'] = count
                demo.attrs['model_file'] = args.model_xml.read_text()
                obs = demo.create_group('obs')
                for destination, values in [('ee_pos', pos), ('ee_ori', ori),
                                            ('ee_states', np.concatenate([pos, ori], axis=-1)),
                                            ('gripper_states', gripper)]:
                    assert np.isfinite(values).all()
                    obs.create_dataset(destination, data=values)
                for public_key, dataset_key in [('agentview_image', 'agentview_rgb'),
                                                 ('robot0_eye_in_hand_image', 'eye_in_hand_rgb')]:
                    images = trajectory[prefix + public_key][:count]
                    assert images.shape == (count, 128, 128, 3) and images.dtype == np.uint8
                    obs.create_dataset(dataset_key, data=images[:, ::-1].copy(), compression='gzip')
                demo.create_dataset('actions', data=actions)
                demo.create_dataset('states', data=trajectory['states_pre'][:count])
                demo.create_dataset('rewards', data=native[:, args.selected_bowl].astype(np.uint8))
                demo.create_dataset('dones', data=native[:, args.selected_bowl].astype(np.uint8))
                receipt = {'kind': 'executed_retargeted_demo', 'controls': count,
                           'full_source_controls': full_count,
                           'full_source_final_native': full_final_native,
                           'endpoint': 'first_native_success' if args.stop_at_first_native_success else 'full_trajectory',
                           'trajectory_sha256': file_sha(args.trajectory),
                           'model_xml_sha256': file_sha(args.model_xml),
                           'observation_alignment': args.alignment,
                           'action_representation': 'executed_delta',
                           'selected_physical_bowl_index': args.selected_bowl,
                           'final_native': native[-1].astype(int).tolist(),
                           'image_storage': 'undo_ControlEnv_vertical_flip; official_loader_flips_once',
                           'orientation_storage': 'robosuite_quat2axisangle; no_controller_frame_correction'}
        data.attrs['total'] = count
    receipt.update(source_hdf5_sha256=file_sha(args.source_hdf5),
                   output_hdf5_sha256=file_sha(args.output))
    args.output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('PROMPT_PACK', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
