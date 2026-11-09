"""Teacher-forced replay of released Motus pi0.5 on RoboTwin place_phone_stand expert demos.

Feeds recorded expert frames (images, joint_action/vector state, instruction) through
the exact policy loader used by the native baseline and compares predicted 32-step
chunks with the recorded future joint_action/vector targets. No simulator, no training.
"""
import argparse
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import zipfile

DEFAULT_PROMPT = (
    'The whole scene is in a realistic, industrial art style with three views: '
    'a fixed rear camera, a movable left arm camera, and a movable right arm camera. '
    'The aloha robot is currently performing the following task: ')
H = 32
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIP = [6, 13]
# Variants share the same per-query flow noise.
VARIANTS = ('base', 'rgb_swap', 'unseen', 'prefix', 'wrist_swap')


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.writing')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_episode(archive, index):
    import cv2
    import h5py
    import numpy as np
    name = f'aloha-agilex_clean_50/data/episode{index}.hdf5'
    with h5py.File(io.BytesIO(archive.read(name)), 'r') as f:
        state = np.asarray(f['joint_action/vector'][()], dtype=np.float64)
        frames = {}
        for cam in ('head_camera', 'left_camera', 'right_camera'):
            raw = f[f'observation/{cam}/rgb'][()]
            # RoboTwin's own pi0/pi05 converter decodes with cv2 and applies no
            # colour conversion; reproduce that exactly as the 'base' order.
            frames[cam] = [cv2.imdecode(np.frombuffer(bytes(b).rstrip(b'\0') or bytes(b), np.uint8),
                                        cv2.IMREAD_COLOR) for b in raw]
    instr_names = [n for n in archive.namelist()
                   if re.search(rf'instructions/episode{index}\.json$', n)]
    assert len(instr_names) == 1, instr_names
    instr = json.loads(archive.read(instr_names[0]))
    return state, frames, instr


def observation(state, frames, t, prompt, swap_rgb=False, swap_wrists=False):
    import numpy as np
    def chw(img):
        img = img[..., ::-1] if swap_rgb else img
        return np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
    left, right = ('right_camera', 'left_camera') if swap_wrists else ('left_camera', 'right_camera')
    return dict(state=state[t].copy(), prompt=prompt, images=dict(
        cam_high=chw(frames['head_camera'][t]),
        cam_left_wrist=chw(frames[left][t]),
        cam_right_wrist=chw(frames[right][t])))


def target(state, t, shift):
    import numpy as np
    idx = np.minimum(np.arange(t + shift, t + shift + H), len(state) - 1)
    return state[idx]


def errors(pred, state, t):
    """Mean abs error for arm joints (rad) and grippers at horizons 1/8/32, two alignments."""
    import numpy as np
    out = {}
    for shift in (0, 1):
        gt = target(state, t, shift)
        err = np.abs(pred - gt)
        for h in (1, 8, 32):
            out[f's{shift}_arm_h{h}'] = float(err[:h][:, ARM].mean())
            out[f's{shift}_grip_h{h}'] = float(err[:h][:, GRIP].mean())
    return out


def summarize(rows):
    import numpy as np
    keys = [k for k in rows[0] if k[:2] in ('s0', 's1')]
    return {k: dict(mean=float(np.mean([r[k] for r in rows])),
                    median=float(np.median([r[k] for r in rows]))) for k in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robotwin-motus-release-baseline-20261109-r1'))
    from evaluate import configuration, sha, MODEL_SHA, NORM_SHA  # identical loader config
    model = Path(args.model)
    archive = zipfile.ZipFile(args.archive)
    state0, frames0, instr0 = load_episode(archive, 0)
    probe = dict(state_shape=list(state0.shape), head_shape=list(frames0['head_camera'][0].shape),
                 head_channel_means=[float(c) for c in frames0['head_camera'][0].reshape(-1, 3).mean(0)],
                 instruction_keys=sorted(instr0), seen0=instr0['seen'][0], unseen0=instr0['unseen'][0],
                 gripper_range=[float(state0[:, GRIP].min()), float(state0[:, GRIP].max())])
    if args.check_only:
        save(args.output / 'CPU-CHECK.json', dict(status='passed', probe=probe))
        return
    import torch
    assert torch.cuda.device_count() == 1
    assert sha(model / 'model.safetensors') == MODEL_SHA
    assert sha(model / 'assets/pi0.5_clean_randomize_joint_training/norm_stats.json') == NORM_SHA
    from openpi.policies.policy_config import create_trained_policy
    policy = create_trained_policy(configuration(model), model, pytorch_device='cuda:0',
                                   sample_kwargs={'num_steps': 10})
    original = getattr(policy._model.sample_actions, '_torchdynamo_orig_callable', None)
    if original is not None:
        policy._model.sample_actions = original
        policy._sample_actions = original
    policy._model.requires_grad_(False)
    assert not policy._model.training
    versions = [(p, p._version) for p in policy._model.parameters()]

    rows = {v: [] for v in VARIANTS}
    hold_rows = []
    preds = {v: [] for v in VARIANTS}
    meta = []
    started = time.time()
    for ep in range(args.episodes):
        state, frames, instr = load_episode(archive, ep)
        seen, unseen = instr['seen'][0], instr['unseen'][0]
        for t in range(0, len(state) - 1, args.stride):
            noise = np.random.RandomState(ep * 10000 + t).randn(H, 32).astype(np.float32)
            obs = dict(base=observation(state, frames, t, seen),
                       rgb_swap=observation(state, frames, t, seen, swap_rgb=True),
                       unseen=observation(state, frames, t, unseen),
                       prefix=observation(state, frames, t, DEFAULT_PROMPT + seen),
                       wrist_swap=observation(state, frames, t, seen, swap_wrists=True))
            for v in VARIANTS:
                with torch.inference_mode():
                    pred = np.asarray(policy.infer(obs[v], noise=noise)['actions'], dtype=np.float64)
                assert pred.shape == (H, 14) and np.isfinite(pred).all()
                rows[v].append(dict(episode=ep, t=t, **errors(pred, state, t)))
                preds[v].append(pred.astype(np.float32))
            hold = np.repeat(state[t][None], H, 0)
            hold_rows.append(dict(episode=ep, t=t, **errors(hold, state, t)))
            meta.append((ep, t))
        save(args.output / 'PROGRESS.json', dict(episodes_done=ep + 1, queries=len(meta),
                                                  elapsed_s=time.time() - started))
    assert all(p._version == v and p.grad is None for p, v in versions)
    np.savez_compressed(args.output / 'predictions.npz', meta=np.asarray(meta),
                        **{v: np.stack(preds[v]) for v in VARIANTS})
    result = dict(status='complete', probe=probe, episodes=args.episodes, stride=args.stride,
                  queries=len(meta), sampler='OpenPI ODE K10, fixed per-query noise shared by variants',
                  frozen_parameters_unchanged=True, elapsed_s=time.time() - started,
                  summary={v: summarize(rows[v]) for v in VARIANTS},
                  hold_baseline=summarize(hold_rows), rows=rows, hold_rows=hold_rows)
    save(args.output / 'RESULT.json', result)


if __name__ == '__main__':
    main()
