"""Run matched development resets or collect fitting trials with the original JAX policy."""
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


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, allow_nan=False) + '\n')
    temporary.replace(path)


def episode(policy, case, output, *, expected=None, paired_prompt=None):
    output.mkdir(parents=True)
    simulator = Simulator()
    try:
        spec, first = simulator.call('create', case)
        initial = first['environment_reproducibility']['initial_state_fingerprint']
        raw = first['observation']
        released_prompt = first['language_instruction']
        prompt = released_prompt if paired_prompt is None else paired_prompt
        if expected is not None:
            assert expected['initial_state'] == initial and expected['prompt'] == prompt
            for key, arr in raw.items():
                digest = hashlib.sha256(str((arr.shape, arr.dtype.str)).encode() + arr.tobytes()).hexdigest()
                assert expected['initial_hashes'][key] == digest
        rng = np.random.default_rng(case['seed'] + 817117)
        states, images, noises, proposals, commands, masks, rewards, dones = [], [], [], [], [], [], [], []
        start = time.monotonic()
        controls = 0
        done = False
        while not done:
            item = observation(raw, prompt)
            noise = rng.standard_normal((10, 32)).astype(np.float32)
            chunk = policy.infer(item, noise=noise)['actions']
            assert chunk.shape == (10, 7) and np.isfinite(chunk).all()
            issued = np.zeros((10, 7), dtype=np.float32)
            mask = np.zeros((10, 7), dtype=bool)
            reward = 0.
            for k, action in enumerate(chunk[:5]):
                step = simulator.call('step', action)
                issued[k] = action
                mask[k] = True
                controls += 1
                raw = step['observation']
                reward += step['reward']
                done = step['terminated'] or step['truncated']
                if done:
                    break
            images.append(np.stack((item['observation/image'], item['observation/wrist_image'])))
            states.append(item['observation/state'].astype(np.float32))
            noises.append(noise[None])
            proposals.append(chunk)
            commands.append(issued)
            masks.append(mask)
            rewards.append(reward)
            dones.append(done)
        assert controls <= spec['max_steps']
        artifact = output/'episode.npz'
        np.savez_compressed(artifact, images=images, states=states, noise=noises, proposed_actions=proposals,
                            issued_actions=commands, action_masks=masks, rewards=np.asarray(rewards, np.float32),
                            dones=dones)
        result = dict(case=case, prompt=prompt, released_prompt=released_prompt, controls=controls,
                      initial_state=initial, success=bool(step['success']), matched_reset=expected is not None,
                      seconds=time.monotonic()-start, npz_bytes=artifact.stat().st_size)
        write(output/'result.json', result)
        print(json.dumps(result), flush=True)
        return result
    finally:
        simulator.close()


def main():
    from collect_pilot import fitting_trials

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    args.output.mkdir()
    assert len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu'
    policy = policy_config.create_trained_policy(config.get_config('pi05_libero'), str(CHECKPOINT))
    if not args.collect:
        prior = ROOT / 'outputs' / Path(__file__).parent.name / 'devqual03'
        for rank, suite in enumerate(('libero_spatial', 'libero_goal')):
            case = dict(suite=suite, task_id=1, init_state_id=0, seed=171117+rank, variant='BASE')
            expected = json.loads((prior/f'rank{rank}/episode-0/result.json').read_text())
            row = episode(policy, case, args.output/suite, expected=expected)
            write(args.output/f'{suite}.json', row)
            with np.load(args.output/suite/'episode.npz') as data:
                selected = data['action_masks'].all(axis=-1)
                np.save(args.output/f'{suite}-actions.npy', data['issued_actions'][selected])
        return

    trials = fitting_trials()
    write(args.output/'design.json', dict(trials=trials, split='fitting', source=str(CHECKPOINT),
          runtime='original JAX OpenPI', executed_before_replan=5, flow_steps=10,
          prompt='BASE wording in both timing arms', features='derive offline from saved public observations'))
    prompts = {}
    results = []
    for trial_id, trial in enumerate(trials):
        for attempt, case in enumerate(trial, start=1):
            key = case['suite'], case['task_id']
            paired_prompt = prompts[key] if case['variant'] == 'S20' else None
            destination = args.output/f'trial-{trial_id:03d}'/f'attempt-{attempt}'
            row = episode(policy, case, destination, paired_prompt=paired_prompt)
            if case['variant'] == 'BASE':
                if key in prompts:
                    assert prompts[key] == row['prompt']
                prompts[key] = row['prompt']
            results.append(dict(trial=trial_id, attempt=attempt, **row))
            write(args.output/'progress.json', results)


if __name__ == '__main__':
    main()
