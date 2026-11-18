"""Render the prespecified equal-budget transfer comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(evidence, output):
    budgets=np.array(evidence['budgets'])
    curves={k:np.array(v)*100 for k,v in evidence['curves'].items()}
    fig,(ax,diff)=plt.subplots(2,1,figsize=(7.4,5.6),sharex=True,gridspec_kw={'height_ratios':[2.2,1]},layout='constrained')
    ax.plot(budgets,curves['learned'],label='Bowl-trained policy',color='#04675c',lw=2.6)
    ax.plot(budgets,curves['bowl_fixed'],label='Bowl-trained fixed strategy',color='#ad4d18',ls='--',lw=1.9)
    ax.plot(budgets,curves['balanced_always_pace'],label='Always pace, balanced first target',color='#394e78',ls=':',lw=2.2)
    ax.set_ylim(-3,103)
    ax.set_ylabel('Retry success (%)')
    ax.set_title('Bottle placement on six fresh pose pairs',loc='left',fontweight='bold')
    ax.legend(loc='upper left',frameon=False,fontsize=9)
    diff.axhline(0,color='#505760',lw=1)
    diff.plot(budgets,curves['learned']-curves['bowl_fixed'],color='#ad4d18',label='Versus fitted fixed')
    diff.plot(budgets,curves['learned']-curves['balanced_always_pace'],color='#394e78',ls=':',lw=2,label='Versus balanced pacing')
    diff.set_ylabel('Difference (pp)')
    diff.set_xlabel('Controls available to each attempt')
    diff.legend(loc='best',frameon=False,fontsize=8)
    for a in (ax,diff):
        a.spines[['top','right']].set_visible(False)
        a.grid(axis='y',alpha=.18)
        a.set_xlim(budgets[0],budgets[-1])
    fig.savefig(output,dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('evidence',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    render(json.loads(args.evidence.read_text()),args.output)
