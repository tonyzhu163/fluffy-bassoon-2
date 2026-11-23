"""Verify all six intended frozen-policy openings using native software-rendered RGB."""
import json
import os
from pathlib import Path
import sys

import numpy as np

from cpu import save
from live import METHODS, SEEDS
from native import Decoder, JOINT, OSC, Pair
from ur5e_motor import public_digest


def main(out):
    c = json.loads((Path(__file__).parent/"evidence/calibration.json").read_text())
    entries = []
    for layout in (0,2,3):
        for suite,base_seed in SEEDS.items():
            seed = base_seed + 100*layout
            pair = Pair(suite, layout, seed, cameras=True)
            try:
                actions = {}
                for method in METHODS:
                    name = OSC if method == OSC else JOINT
                    obs = pair.start(name)
                    q = obs["robot0_joint_pos"]
                    a = Decoder(pair.chain,q,method,c).action(q,np.array([0.2,0.1,-0.1,0.05,0,-0.01,-1]))
                    _, reward = pair.step(name,a)
                    actions[method] = dict(controller=name,native_action=a,reward=reward)
                entries.append(dict(suite=suite,layout=layout,seed=seed,prompt=pair.prompt,
                                horizon=pair.horizon,state=pair.full_hash,
                                public=public_digest(pair.first[OSC]),actions=actions))
            finally:
                pair.close()
    save(out/"panel.json",dict(cases=entries,affinity=sorted(os.sched_getaffinity(0))))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
