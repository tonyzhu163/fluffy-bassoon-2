"""Evaluate the closed-loop frozen motor on current images under the actual rack event."""
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'libero-first-attempt-20261119-r1'))
from experiment import pilot, simulator, episode


class RackWorld:
    def __init__(self, pipe):
        self.pipe=pipe

    def send(self,msg):
        op,args=msg
        self.pipe.send((op,(*args,9) if op=='reset' else args))

    def recv(self):
        return self.pipe.recv()


def main(out,prior):
    assert len(pilot.jax.devices())==1 and pilot.jax.devices()[0].platform=='gpu'
    motor=pilot.policy_config.create_trained_policy(pilot.config.get_config('pi05_libero'),str(pilot.CKPT))
    out.mkdir()
    with simulator((2,9)) as pipe:
        for pose in range(12,24):
            with np.load(prior/f'pose{pose}'/'public.npz') as a:
                opening=a['issued'][:24]
            for variant in ('BASE','D02'):
                path=out/f'p{pose}-{variant}'
                rec=episode(motor,RackWorld(pipe),path,pose,variant,1,'switch',bottle=True,prefix=opening)
                with np.load(path/'public.npz') as b, np.load(prior/f'pose{pose}'/'public.npz') as a:
                    np.testing.assert_array_equal(b['issued'][:24],a['issued'][:24])
                    np.testing.assert_array_equal(b['state_after'][:24],a['state_after'][:24])
                assert rec['controls']==300


if __name__=='__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
