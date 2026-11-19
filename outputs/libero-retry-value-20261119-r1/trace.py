"""Check the physical effect of the override and plot the first opposite-outcome placement."""
from pathlib import Path
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(out):
    root=out.parent/'timing01'
    practices=json.loads((root/'timing.json').read_text())['practices']
    checks=[]
    traces={}
    for row in practices:
        pose,goal,mode=row['pose'],row['goal'],row['mode']
        prefix=root/'episodes'/f'p{pose}-g{goal}'
        with np.load(f'{prefix}-{mode}.npz') as t:
            traces[pose,goal,mode]=(t['states'],t['hits'][:,goal])
            if row['start'] is None:
                continue
            n=row['start']
            with np.load(f'{prefix}-normal.npz') as ref:
                checks.append(dict(pose=pose,goal=goal,mode=mode,start=n,
                                   intended_close=int((t['issued'][n:n+10,-1]>0).sum()),
                                   already_succeeded=bool(t['hits'][:n,goal].any()),
                                   changed_motion=not np.array_equal(t['states'][n:],ref['states'][n:])))
    starts={(r['pose'],r['goal'],r['mode']):r['start'] for r in checks}
    for pose,goal in sorted({k[:2] for k in traces}):
        if (pose,goal,'early') in starts and (pose,goal,'late') in starts and all(
            traces[pose,goal,m][1].any()==(m!='early') for m in ('normal','early','late')):
            break
    else:
        raise ValueError('No triggered opposite-outcome placement')
    fig,axes=plt.subplots(3,1,sharex=True,figsize=(7.3,4.4))
    colors=['#4d5861','#ae4b36','#00756d']
    for ax,m,color,label in zip(axes,('normal','early','late'),colors,('Normal','Opening after lift','Opening on descent')):
        states,hits=traces[pose,goal,m]
        width=np.abs(states[:,-2]-states[:,-1])*1000
        ax.plot(np.arange(1,301),width,color=color,lw=1.7)
        if m!='normal':
            n=starts[pose,goal,m]
            ax.axvspan(n+.5,n+10.5,color=color,alpha=.2,lw=0)
        ax.fill_between(np.arange(1,301),0,1,where=hits,step='mid',color='#79ab62',alpha=.45,
                        transform=ax.get_xaxis_transform(),lw=0)
        ax.set_title(label,loc='left',fontsize=10,pad=2)
        ax.spines[['top','right']].set_visible(False)
        ax.set_ylim(0,90)
    axes[1].set_ylabel('Measured jaw separation (mm)')
    axes[-1].set_xlabel('Physical control')
    fig.suptitle('One bowl-to-plate attempt under three release conditions',fontsize=13)
    fig.tight_layout()
    fig.savefig(out/'release-timing.png',dpi=190,bbox_inches='tight')
    report=dict(executions=checks,selected=dict(pose=pose,goal=goal,
                first_success={m:int(np.flatnonzero(h)[0]+1) if h.any() else None
                               for (p,g,m),(_,h) in traces.items() if (p,g)==(pose,goal)}))
    (out/'trace.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main(Path(sys.argv[1]))
