"""Show the four public initial views from a common pose and the measured pair results."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(root,results,output):
    names=('Native layout','Cabinet moved','Plate moved','Rack moved')
    fig=plt.figure(figsize=(10,6.1),layout='constrained')
    grid=fig.add_gridspec(2,4,height_ratios=[1,1.17])
    for i,(layout,title) in enumerate(zip(('base','cabinet','plate','rack'),names)):
        ax=fig.add_subplot(grid[0,i])
        with np.load(root/'fresh'/'p32'/layout/'views.npz') as x:
            ax.imshow(x['agent'])
        ax.axis('off')
        ax.set_title(title,fontsize=11)
    ax=fig.add_subplot(grid[1,:])
    names={4:'bowl → cabinet',8:'bowl → plate',2:'bottle → cabinet',9:'bottle → rack',0:'open drawer',1:'bowl → stove'}
    pairs=results['pairs']
    y=np.arange(len(pairs))
    for k,label,marker,color in [('selected','Scene model','o','#006c62'),('fit_constant','Fitting-set constant / balanced unseen','x','#b34f1c'),('block_oracle','Best choice for each start','|','#32455b')]:
        ax.scatter([p['totals'][k] for p in pairs],y,marker=marker,c=color,s=62,label=label,zorder=3,linewidths=1.8)
    ax.set_yticks(y,labels=[' / '.join(names[g] for g in p['goals']) for p in pairs],fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-.5,32.5)
    ax.set_xticks(np.arange(0,33,4))
    ax.set_xlabel('Native retry successes per 32 goal assignments')
    ax.grid(axis='x',alpha=.2)
    ax.spines[['right','top']].set_visible(False)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.24),ncol=3,frameon=False,fontsize=8)
    fig.savefig(output,dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('episodes',type=Path)
    p.add_argument('evidence',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    render(a.episodes,json.loads(a.evidence.read_text()),a.output)
