"""Locate first observable differences and bound the value of knowing timing at reset."""
from pathlib import Path
import sys,json
import numpy as np
from pilot import MODES


def split(root):
    rows=[]
    for base in sorted(root.glob('*-BASE')):
        slow=base.with_name(base.name[:-4]+'S20')
        outputs=[]
        differences={}
        for mode in ('support',*MODES):
            blobs=[p/mode for p in (base,slow)]
            records=[json.loads((p/'result.json').read_text()) for p in blobs]
            assert records[0]['identity']==records[1]['identity']
            if mode!='support':
                outputs.append([int(v['success']) for v in records])
            with np.load(blobs[0]/'public.npz') as a,np.load(blobs[1]/'public.npz') as b:
                first={}
                for key in ('issued','positions','gripper','rewards'):
                    m=min(len(a[key]),len(b[key]))
                    diff=(a[key][:m]!=b[key][:m]).reshape(m,-1).any(axis=1)
                    first[key]=int(np.flatnonzero(diff)[0])+1 if diff.any() else None
                first['reset_revisited_7']=[np.array_equal(x['positions'][6],x['initial_position']) for x in (a,b)]
                differences[mode]=first
        scores=np.array(outputs)
        blind=int(scores.sum(axis=1).max())
        aware=int(scores.max(axis=0).sum())
        rows.append(dict(key=base.name[:-5],modes=outputs,visible_state_oracle=blind,
                         timing_and_state_oracle=aware,first_differences=differences))
    assert len(rows)==16
    return dict(state_only=sum(r['visible_state_oracle'] for r in rows),
                state_and_timing=sum(r['timing_and_state_oracle'] for r in rows),rows=rows)


if __name__=='__main__':
    print(json.dumps({k:split(Path(sys.argv[1])/k) for k in ('fitting','development')},indent=2))
