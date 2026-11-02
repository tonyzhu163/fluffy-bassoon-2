"""Scientific plot of the completed, fixed video replay; CPU only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from REPLAY import OUT
r=json.loads((OUT/'RESULT.json').read_text());assert r['completed']
fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
colors=['#5576a5','#d27a3e'];v=r['cohorts']['TRAIN/OpenDrawer'];a=v['rerun_original'];b=v['candidate']
x=np.arange(2);w=.35
for k,(name,d) in enumerate([('Original prompt',a),('New rubric',b)]):
    vals=[d['Score5_false_positives'],d['Score5_true_positives']]
    bars=axes[0].bar(x+(k-.5)*w,vals,w,label=name,color=colors[k]);axes[0].bar_label(bars,padding=3)
axes[0].set_xticks(x,['Failures given 5\n(out of 183)','Successes given 5\n(out of 201)'])
axes[0].set_ylim(0,220);axes[0].set_title('OpenDrawer TRAIN: full-success claims');axes[0].set_ylabel('Episodes');axes[0].legend(frameon=False)
labels=['TRAIN drawer','Calibration drawer','Into cabinet','Onto counter','Stove off','Mug setup']
for k,key in enumerate(['rerun_original','candidate']):
    axes[1].plot(np.arange(6),[v[key]['AUROC'] for v in r['cohorts'].values()],marker='o',color=colors[k],label=['Original prompt','New rubric'][k])
axes[1].set_xticks(np.arange(6),labels,rotation=45,ha='right');axes[1].set_ylim(.5,1.01);axes[1].set_ylabel('AUROC');axes[1].set_title('Reward ranking by retained video bank');axes[1].grid(axis='y',alpha=.2)
f=r['focus_original_failed_Score5'];hist=f['candidate_histogram']
bars=axes[2].bar(np.arange(1,6),[hist.get(str(i),0) for i in range(1,6)],color=colors[1]);axes[2].bar_label(bars,padding=3)
axes[2].set_xticks(np.arange(1,6));axes[2].set_ylim(0,62);axes[2].set_xlabel('New rubric grade');axes[2].set_ylabel('Failed episodes');axes[2].set_title('58 failures originally graded 5')
fig.suptitle('Frozen prompt comparison on 634 existing three-view videos',fontsize=14)
p=OUT/'comparison-final.png';assert not p.exists();fig.savefig(p,dpi=180);plt.close(fig)
print(p)
