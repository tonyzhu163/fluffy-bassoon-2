"""Show when the two native conditions first become distinguishable."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(panel,output):
    fig,ax=plt.subplots(figsize=(6.4,3.5),layout='constrained')
    steps=np.arange(21)
    for g,color in [('BASE','#21665f'),('S20','#af541d')]:
        curves=[]
        for path in sorted(panel.glob('*-'+g)):
            with np.load(path/'default/public.npz') as t:
                motion=np.linalg.norm(t['positions'][:20]-t['initial_position'],axis=1)
                curves.append(np.r_[0.,100*motion])
        lo,med,hi=np.percentile(curves,[25,50,75],axis=0)
        ax.fill_between(steps,lo,hi,color=color,alpha=.14)
        ax.plot(steps,med,marker='.',label=g,color=color)
    ax.axvline(5,ls='--',lw=.9,color='#666666')
    ax.text(5,.96,'First cadence decision ',ha='right',va='top',transform=ax.get_xaxis_transform(),fontsize=8)
    ax.set(xlim=(0,20),ylim=(0,None),xlabel='Physical controls since reset',ylabel='Observed wrist displacement (cm)')
    ax.set_xticks([0,5,7,10,15,20])
    ax.legend(frameon=False,loc='upper center')
    ax.spines[['right','top']].set_visible(False)
    fig.savefig(output,dpi=180)
    plt.close(fig)


if __name__=='__main__':
    render(*map(Path,sys.argv[1:]))
