"""Collect twelve previously unused nominal openings with the frozen bottle policy."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'libero-first-attempt-20261119-r1'))
from experiment import pilot, simulator, episode

def main(out):
    assert len(pilot.jax.devices())==1 and pilot.jax.devices()[0].platform=='gpu'
    out.mkdir()
    motor=pilot.policy_config.create_trained_policy(pilot.config.get_config('pi05_libero'),str(pilot.CKPT))
    with simulator((2,9)) as pipe:
        for pose in range(12,24):
            episode(motor,pipe,out/f'pose{pose}',pose,'BASE',1,'switch',bottle=True)

if __name__=='__main__':
    main(Path(sys.argv[1]))
