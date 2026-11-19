"""Verify and summarize the native, current-image comparator from physical controls."""
import json
from pathlib import Path
import sys
import numpy as np


def summarize(root):
    rows=[]
    for pose in range(12,24):
        pair=[]
        for g in ('BASE','D02'):
            path=root/f'p{pose}-{g}'
            record=json.loads((path/'result.json').read_text())
            native=np.load(path/'evaluator-predicates.npy')
            assert native.shape==(300,2) and record['scene']==pose and record['variant']==g and record['goal']==1
            first=[int(np.flatnonzero(col)[0])+1 if col.any() else None for col in native.T]
            assert first==record['first']
            with np.load(path/'public.npz') as t:
                pair.append([t['issued'][:24],t['state_after'][:24]])
                assert t['issued'].shape==(300,7)
            rows.append(dict(pose=pose,regime=g,hit=first[1] is not None,first=first[1],final=bool(native[-1,1])))
        for a,b in zip(*pair):
            np.testing.assert_array_equal(a,b)
    return dict(rows=rows,hit=sum(r['hit'] for r in rows),final=sum(r['final'] for r in rows))


if __name__=='__main__':
    print(json.dumps(summarize(Path(sys.argv[1])),indent=2))
