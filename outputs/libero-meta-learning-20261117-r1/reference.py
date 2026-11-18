"""Check the two nominal development resets with the original JAX policy."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import numpy as np
from openpi.policies import policy_config
from openpi.training import config
from openpi_client import image_tools

from job import ROOT
from simulator import Simulator

CHECKPOINT = Path('/mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1')


def observation(raw, prompt):
    rgb = [image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(raw[key][::-1, ::-1]), 224, 224))
        for key in ('agentview_image', 'robot0_eye_in_hand_image')]
    quat = raw['robot0_eef_quat_xyzw']
    w = np.clip(quat[3], -1., 1.)
    den = np.sqrt(1. - w * w)
    rotation = quat[:3] * 2. * np.arccos(w) / den if den else np.zeros(3)
    state = np.concatenate((raw['robot0_eef_pos'], rotation, raw['robot0_gripper_qpos']))
    return {'observation/image': rgb[0], 'observation/wrist_image': rgb[1],
            'observation/state': state, 'prompt': prompt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    assert len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu'
    policy = policy_config.create_trained_policy(config.get_config('pi05_libero'), str(CHECKPOINT))
    prior = ROOT / 'outputs' / Path(__file__).parent.name / 'devqual03'
    for rank, suite in enumerate(('libero_spatial', 'libero_goal')):
        case = dict(suite=suite, task_id=1, init_state_id=0, seed=171117+rank, variant='BASE')
        simulator = Simulator()
        try:
            spec, first = simulator.call('create', case)
            expected = json.loads((prior/f'rank{rank}/episode-0/result.json').read_text())
            initial = first['environment_reproducibility']['initial_state_fingerprint']
            assert expected['initial_state'] == initial and expected['prompt'] == first['language_instruction']
            raw = first['observation']
            for key, arr in raw.items():
                digest = hashlib.sha256(str((arr.shape, arr.dtype.str)).encode() + arr.tobytes()).hexdigest()
                assert expected['initial_hashes'][key] == digest
            rng = np.random.default_rng(case['seed'] + 817117)
            commands = []
            start = time.monotonic()
            done = False
            while not done:
                item = observation(raw, first['language_instruction'])
                noise = rng.standard_normal((10, 32)).astype(np.float32)
                chunk = policy.infer(item, noise=noise)['actions']
                assert chunk.shape == (10, 7) and np.isfinite(chunk).all()
                for action in chunk[:5]:
                    step = simulator.call('step', action)
                    commands.append(action.copy())
                    raw = step['observation']
                    done = step['terminated'] or step['truncated']
                    if done:
                        break
            assert len(commands) <= spec['max_steps']
            np.save(args.output/f'{suite}-actions.npy', commands)
            result = dict(case=case, controls=len(commands), success=bool(step['success']),
                          matched_reset=True, seconds=time.monotonic()-start)
            (args.output/f'{suite}.json').write_text(json.dumps(result) + '\n')
            print(json.dumps(result), flush=True)
        finally:
            simulator.close()


if __name__ == '__main__':
    main()
