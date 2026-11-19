"""Supplementary branch replay of a prior-lag rule versus a fixed correction."""
import json
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'libero-first-attempt-20261119-r1'))
from design import lag24, use_pacing


def main(result, archive):
    data=json.loads(result.read_text())
    policy=json.loads((archive/'frozen-policy.json').read_text())['learned']['model']
    rows=[]
    for trial in data['evaluation']:
        pose,timing=trial['pose'],trial['timing']
        # Opposite member of the pre-existing cabinet/rack pose pair, same timing.
        block=archive/'bottle'/f's{pose%6}-{timing}'
        if pose<6:
            path=block/'query-g0'
        else:
            path=next(p for p in block.glob('support-g0-*') if json.loads((p/'result.json').read_text())['paced'])
        rec=json.loads((path/'result.json').read_text())
        assert rec['scene']!=pose and rec['goal']==0 and rec['variant']==timing
        with np.load(path/'public.npz') as tape:
            lag=lag24(tape['issued'],tape['state_after'])
        choose_correction=use_pacing(policy,lag)
        rows.append(dict(pose=pose,source_pose=rec['scene'],timing=timing,lag=lag,
                         corrected=trial['hit'],original=trial['baseline'],use_correction=choose_correction,
                         retained=trial['hit'] if choose_correction else trial['baseline'],
                         reversed=trial['baseline'] if choose_correction else trial['hit']))
    return dict(prior_lag_model=policy,rows=rows)


if __name__=='__main__':
    print(json.dumps(main(Path(sys.argv[1]),Path(sys.argv[2])),indent=2))
