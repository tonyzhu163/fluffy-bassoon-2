"""Run the established common-world collector with a frozen source reachability model."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

scene_dir = Path(__file__).resolve().parents[1] / 'libero-scene-probes-20261119-r1'
sys.path.insert(0, str(scene_dir))
import experiment
from selector import Reachability


def main(output, source):
    pilot = experiment.pilot
    devices = pilot.jax.devices()
    assert len(devices) == 1 and devices[0].platform == 'gpu'
    for repo, pin in [(pilot.SRC, '215abfb217dbac7d5f1273282331b9b1866c0479'),
                      (pilot.PRO, 'e5500c690c1498eca1b7050350654964190cdf70')]:
        assert subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip() == pin
    frozen = source / 'frozen.npz'
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == json.loads((source / 'fit.json').read_text())['sha256']
    motor = pilot.policy_config.create_trained_policy(pilot.config.get_config('pi05_libero'), str(pilot.CKPT))
    experiment.LAYOUTS = {k: experiment.LAYOUTS[k] for k in ('base', 'cabinet')}
    experiment.collect(motor, output, range(36, 44), (4, 0, 1, 5), Reachability(frozen))


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:]))
