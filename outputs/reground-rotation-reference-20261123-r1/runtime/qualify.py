"""One reset and step on the reference body, optionally issuing a real pi05 action."""
import argparse
import contextlib
import hashlib
import io
import json
import os
import socket
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--cpu-policy', action='store_true', help='Also qualify the native model on CPU')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    root = Path(os.environ['REGROUND_RUNTIME_ROOT'])
    from reground.envs.libero_env import TaskEnv
    from reground.policies.vla import VLA
    from packaging.requirements import Requirement

    packages = {}
    for line in (root / 'upstream/requirements.txt').read_text().splitlines():
        req = Requirement(line)
        packages[req.name] = version(req.name)
        if packages[req.name] not in req.specifier:
            raise RuntimeError(f'Installed {req.name}=={packages[req.name]} violates {line}')
    assert torch.version.cuda == '12.8'
    tensor_sum = torch.arange(4, device=args.device).sum().item()
    assert tensor_sum == 6
    report = dict(host=socket.gethostname(), time_utc=datetime.now(timezone.utc).isoformat(),
                  device=args.device, torch_cuda=torch.version.cuda, torch_sum=tensor_sum,
                  cuda_available=torch.cuda.is_available(), packages=packages, task='spatial_0',
                  shift='rot_z/90', init_id=0, seed=0)
    env = TaskEnv(report['task'], shift=report['shift'], cameras=True)
    try:
        reset = env.reset(init_id=0, seed=0)
        from OpenGL.GL import GL_RENDERER, GL_VENDOR, glGetString
        report.update(renderer=glGetString(GL_RENDERER).decode(), gl_vendor=glGetString(GL_VENDOR).decode(),
                      assets=str(Path(os.environ['LIBERO_CONFIG_PATH'], 'config.yaml')),
                      image_shapes={k: list(v.shape) for k, v in reset['obs']['pixels'].items()})
        assert all(shape == [256, 256, 3] for shape in report['image_shapes'].values())
        assert len(report['image_shapes']) == 2
        action = np.array([0.1, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        if args.device == 'cuda':
            assert 'NVIDIA' in report['gl_vendor']
        if args.device == 'cuda' or args.cpu_policy:
            rev = '7152f13e34c6c115d808aea0e5d08aca74eac14b'
            capture = io.StringIO()
            try:
                with contextlib.redirect_stdout(capture):
                    vla = VLA('pi05', args.device, revision=rev)
            finally:
                print(capture.getvalue(), flush=True)
            # LeRobot 0.6.1 can swallow weight-loading errors. Require its strict success branch.
            if 'All keys loaded successfully!' not in capture.getvalue():
                raise RuntimeError('LeRobot did not confirm that every pretrained key loaded')
            from huggingface_hub import HfApi
            assert HfApi().model_info('lerobot/pi05_libero_finetuned').sha == vla.revision == rev
            actions = vla.predict_chunk([reset['obs']], [env.description], [0], 0)
            assert actions.shape == (1, 50, 7) and np.isfinite(actions).all()
            action = actions[0, 0]
            report.update(policy=type(vla.policy).__qualname__, wrapper=type(vla).__module__ + '.VLA',
                          revision=vla.revision, strict_load=True, noise_seed=0, chunk_idx=0,
                          chunk_shape=list(actions.shape), chunk_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                          first_action=action.tolist())
            if args.device == 'cuda':
                report['peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
        step = env.step(action, render=True)
        assert env.t == 1 and reset['x'].shape == step['x'].shape == (25,)
        assert np.isfinite(step['x']).all() and np.isfinite(reset['x']).all()
        report.update(steps=env.t, state_dim=step['x'].size,
                      state_l2_change=float(np.linalg.norm(step['x'] - reset['x'])))
    finally:
        env.close()
    report['elapsed_seconds'] = time.monotonic() - started
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
