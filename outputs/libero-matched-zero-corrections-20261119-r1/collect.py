"""Repeat frozen VLA practice at identical resets and fork both query destinations and cadences."""
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'libero-first-attempt-20261119-r1'))
from experiment import pilot, simulator

GOALS = (4, 8)


def pooled(public):
    return np.stack([public[k].reshape(28, 8, 28, 8, 3).mean(axis=(1, 3)).round().astype(np.uint8)
                     for k in ('observation/image', 'observation/wrist_image')])


def execute(motor, pipe, raw, prompt, seed, cadence, path):
    path.mkdir()
    rng = np.random.default_rng(seed)
    issued, states, native = [], [], []
    view = pilot.observation(raw, prompt)
    frames = [pooled(view)]
    while len(issued) < 300:
        noise = rng.standard_normal((10, 32)).astype(np.float32)
        commands = motor.infer(view, noise=noise)['actions'][:cadence]
        assert commands.shape == (cadence, 7) and np.isfinite(commands).all()
        raw, state, hits = pilot.call(pipe, 'chunk', commands)
        view = pilot.observation(raw, prompt)
        issued.extend(commands)
        states.extend(state)
        native.extend(hits)
        if len(issued) % 10 == 0:
            frames.append(pooled(view))
    assert np.shape(frames) == (31, 2, 28, 28, 3)
    assert np.shape(native) == (300, 2) and np.shape(states) == (300, 9)
    np.savez_compressed(path/'tape.npz', frames=frames, actions=issued, states=states, native=native)


def collect(motor, output, poses):
    with simulator(GOALS) as pipe:
        for pose in poses:
            block = output / f'p{pose}'
            block.mkdir(parents=True)
            identity = None
            for slot, goal in enumerate(GOALS):
                runs = [(f'practice{draw}', 5, 19194000+100*pose+draw) for draw in range(4)]
                runs += [(f'query{cad}', cad, 19198000+pose) for cad in (5, 10)]
                for label, cad, seed in runs:
                    raw, physics, prompts = pilot.call(pipe, 'reset', (pose, 'S02', 8))
                    current = dict(physics=physics, public={k:hashlib.sha256(v.tobytes()).hexdigest()
                                                           for k,v in raw.items()}, prompts=prompts)
                    if identity is None:
                        identity = current
                        pilot.write(block/'initial.json', current)
                    assert current == identity
                    path = block / f'g{goal}-{label}'
                    execute(motor, pipe, raw, prompts[slot], seed, cad, path)
                    pilot.write(path/'result.json', dict(pose=pose, goal=goal, role=label, cadence=cad, seed=seed))
            print('completed',pose,flush=True)


def main(out):
    devices = pilot.jax.devices()
    assert len(devices) == 1 and devices[0].platform == 'gpu'
    for repo, pin in [(pilot.SRC,'215abfb217dbac7d5f1273282331b9b1866c0479'),
                      (pilot.PRO,'e5500c690c1498eca1b7050350654964190cdf70')]:
        assert subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip() == pin
    motor = pilot.policy_config.create_trained_policy(pilot.config.get_config('pi05_libero'), str(pilot.CKPT))
    collect(motor, out/'fitting', range(32,36))
    collect(motor, out/'evaluation', range(44,48))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
