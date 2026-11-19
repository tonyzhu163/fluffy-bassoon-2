"""Describe dependence on support assignment across all nonzero cyclic pose offsets."""
import json
from pathlib import Path
import sys
from itertools import product
from refit import read, score, REGIMES


def summary(panel,model):
    data=read(panel/'native.jsonl')
    tree=model['tree']
    poses=list(range(12,24))
    counts=[]
    for offset in range(1,len(poses)):
        pairs=list(zip(poses,poses[offset:]+poses[:offset]))
        rows=score(data,pairs,tree)
        fixed=score(data,pairs,model['constant'])
        counts.append(dict(offset=offset,hit=sum(r['hit'] for r in rows),final=sum(r['final'] for r in rows),
                           gains=sum(r['hit'] and not f['hit'] for r,f in zip(rows,fixed)),
                           losses=sum(f['hit'] and not r['hit'] for r,f in zip(rows,fixed))))
    probe={g:sum(data[p,g,tree[0]]['hit'] for p in poses) for g in REGIMES}
    by_pose=[]
    for p in poses:
        base=data[p,'BASE','suffix']
        move=data[p,'D02','suffix']
        by_pose.append(dict(pose=p,same_physics=base['physics']==move['physics'],same_prefix=base['prefix']==move['prefix'],
                            lift=move['lift'],motion=move['first_motion'],native_displacement_m=move['max_xy']))
    return dict(offsets=counts,probe_hits=probe,paired_reset=by_pose)


if __name__=='__main__':
    print(json.dumps(summary(Path(sys.argv[1]),json.loads(Path(sys.argv[2]).read_text())),indent=2))
