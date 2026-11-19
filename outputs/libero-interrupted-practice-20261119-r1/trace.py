"""Illustrate an observed practice interruption using the sensor and issued command."""
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(live,paired,out):
    info=json.loads((paired/'closed.json').read_text())
    triggered=[r for r in info['interrupted_pairs'] if r['start'] is not None]
    # Prefer the first live recovery to show what reactive control changes.
    r=next((r for r in triggered if r['closed'] and not r['open']),triggered[0])
    p,g,start=(r[k] for k in ('pose','goal','start'))
    fig,axes=plt.subplots(2,1,figsize=(8.4,4.3),sharex=True,layout='constrained')
    for path,label,color,style in [
        (live/f'p{p}-g{g}-sham/tape.npz','Uninterrupted','#4E5560',':'),
        (paired/f'branches/live-p{p}-g{g}-d0-1.npz','Interrupted, saved commands','#A45B17','--'),
        (live/f'p{p}-g{g}-interrupt/tape.npz','Interrupted, live motor','#146953','-')]:
        with np.load(path) as data:
            gap=100*np.abs(np.diff(data['states'][:,-2:],axis=1)).ravel()
            ax=axes[0]
            ax.plot(np.arange(1,301),gap,label=label,color=color,linestyle=style,lw=1.6)
            axes[1].plot(np.arange(1,301),data['issued'][:,-1],color=color,linestyle=style,lw=1.4)
    for ax in axes:
        ax.axvspan(start+1,start+10,color='#BA758C',alpha=.16)
        ax.grid(axis='y',alpha=.20)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_title(f'Bottle to {("cabinet","rack")[g]}, pose {p} · shaded interval opens the gripper',loc='left',fontsize=12)
    axes[0].set_ylabel('Sensed jaw gap (cm)')
    axes[0].legend(loc='upper right',frameon=False,fontsize=8)
    axes[1].set_ylabel('Issued gripper command')
    axes[1].set_yticks([-1,1],['Open','Close'])
    axes[1].set_xlabel('Physical control step')
    fig.savefig(out/'paired-response.png',dpi=185,bbox_inches='tight')
    (out/'trace-case.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main(*map(Path,sys.argv[1:]))
