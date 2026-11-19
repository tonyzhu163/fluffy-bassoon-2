"""Compare the best attainable action for identical public starts with and without execution evidence."""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze import trials
from vision import read_split


def ceilings(rows):
    totals = {}
    for endpoint in ('native','final'):
        for evidence in ('reset_reward','execution_reward','hidden'):
            groups = defaultdict(list)
            for r in rows:
                key = (r['first'],r['second'],r['query'],r['keys'][0][1],r['bits'])
                if evidence == 'execution_reward':
                    # With identical resets, the realized practice draws identify the two
                    # complete physical tapes. Hidden goal can still be ambiguous.
                    key += (r['keys'][0][2],r['keys'][1][2])
                elif evidence == 'hidden':
                    key += (r['hidden'],)
                groups[key].append(r[endpoint])
            totals[endpoint+'_'+evidence] = int(sum(np.stack(group).sum(axis=0).max() for group in groups.values()))
    return dict(assignments=len(rows), **totals)


def source_plot(ax, practice, queries, title):
    poses = sorted({p for p,g,j in practice})
    data = []
    for p in poses:
        counts = [sum(practice[p,g,j]['native'][g] for j in range(4))/4 for g in (0,1)]
        data.append(counts+[float(queries[p,g,c]['native'][g]) for g in (0,1) for c in (5,10)])
    matrix = np.array(data)
    ax.imshow(matrix, cmap='Blues',vmin=0,vmax=1,aspect='auto')
    for i,row in enumerate(matrix):
        for j,x in enumerate(row):
            label = f'{int(4*x)}/4' if j<2 else ('hit' if x else 'miss')
            ax.text(j,i,label,ha='center',va='center',color='white' if x>.6 else '#17354B',fontsize=10)
    ax.axvline(1.5,color='white',lw=4)
    ax.set_yticks(range(4),[f'Pose {p}' for p in poses])
    ax.set_xticks(range(6),['Cabinet','Plate','Cabinet\n5','Cabinet\n10','Plate\n5','Plate\n10'])
    ax.tick_params(length=0)
    ax.set_title(title,loc='left',fontsize=12,fontweight='bold')
    ax.set_xlabel('Practice hits / draws             Query hit by replanning interval',fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)


if __name__ == '__main__':
    root, out = map(Path,sys.argv[1:])
    result = {}
    fig, axes = plt.subplots(1,2,figsize=(11,3.0),layout='constrained')
    for name,ax in zip(('fitting','evaluation'),axes):
        practice, queries = read_split(root/name)
        rows = list(trials(practice,queries))
        result[name] = {k:ceilings(v) for k,v in [('all',rows),('zero',[r for r in rows if r['bits']==(0,0)])]}
        source_plot(ax,practice,queries, 'Fitting poses' if name=='fitting' else 'Fresh poses')
    (out/'ceilings.json').write_text(json.dumps(result,indent=2)+'\n')
    fig.savefig(out/'motor-coverage.png',dpi=180,bbox_inches='tight')
    print(result)
