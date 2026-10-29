"""Plot audited saved outcomes only; no model or checkpoint execution."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter

root = Path(__file__).resolve().parents[2]
check = root / 'checks/20261029-0835'
out = root / 'figures'
out.mkdir(exist_ok=True)
labels = {'ctm': 'CTM T1 · cached100', 'rl2': 'RL²-PPO · GRU64', 'ecet': 'ECET', 'amago': 'AMAGO1 · 10 replay passes'}
colors = {'ctm':'#1876b9', 'rl2':'#87584b', 'ecet':'#22a135', 'amago':'#d13235'}
audit = json.loads((check / 'completion-audit.json').read_text())
series = {}
csv_rows = []
for method in labels:
    rows = [json.loads(s) for s in (check / f'{method}-metrics.jsonl').read_text().splitlines() if s]
    points = [r for r in rows if 'evaluation' in r and not r.get('final')]
    final = rows[-1]
    assert len(points) == 17 and final.get('final')
    # The final normal evaluation exactly repeats round80 outcomes; charge
    # all final paired evaluations at its endpoint, and draw no duplicate point.
    assert points[-1]['evaluation']['train_eval']['normal'] == final['evaluation']['train_eval']['normal']
    points[-1] = final
    series[method] = points
    for p in points:
        for split in ('train_eval','test'):
            v=p['evaluation'][split]['normal']
            csv_rows.append(dict(method=method, split=split, iteration=p['iteration'], environment_steps=p['environment_steps'], active_gpu_hours=p['gpu_hours'], training_gpu_hours=p['training_gpu_hours'], success=v['success'], first5_success=sum(v['success_by_episode'][:5])/5))
with (out/'ml10-final-curves-20261029-0835.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)

plt.rcParams.update({'font.family':'serif','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
specs = [('gpu_hours','Active GPU Hours','gpu-hours'),('training_gpu_hours','Training GPU Hours','training-gpu-hours'),('environment_steps','Training Environment Steps','steps')]
for split,title in [('train_eval','ML10 Train'),('test','ML10 Held-out Families')]:
    for xkey,xlabel,suffix in specs:
        fig,ax=plt.subplots(figsize=(7.5,4.8),layout='constrained')
        ax.set_facecolor('#f1f3f6');ax.grid(color='white',linewidth=1)
        for method,points in series.items():
            x=[p[xkey] for p in points]
            y=[p['evaluation'][split]['normal']['success'] for p in points]
            ax.plot(x,y,lw=2,color=colors[method],label=labels[method])
            ax.scatter(x[-1:],y[-1:],s=28,color=colors[method],zorder=3)
        ax.set(xlabel=xlabel,ylabel='Success',ylim=(0,1),xlim=(0,None),title=title)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        if suffix=='steps': ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1e6:g}M'))
        ax.legend(loc='upper right',frameon=True,fontsize=9)
        fig.suptitle('Seed 1000 · 10M training steps per method · one RPBZZZ6 per method',fontsize=9,color='#444')
        foot = 'Active time includes collection, optimization and all evaluation; no uncertainty bands.' if suffix=='gpu-hours' else 'Observed evaluations only; different learners, model sizes and gradient horizons.'
        fig.text(.5,-.025,foot,ha='center',fontsize=8,color='#444')
        name=f'ml10-{split}-{suffix}-20261029-0835'
        for extension in ('png','pdf'):fig.savefig(out/f'{name}.{extension}',dpi=200,bbox_inches='tight')
        plt.close(fig)

# Keep competence within training's first-five-episode clock range visible,
# without replacing the prespecified K25 headline.
fig,axes=plt.subplots(1,2,figsize=(11.0,4.2),sharey=True,layout='constrained')
for ax,split,title in zip(axes,('train_eval','test'),('Training families · first 5 episodes','Held-out families · first 5 episodes')):
    ax.set_facecolor('#f1f3f6');ax.grid(color='white',linewidth=1)
    for method,points in series.items():
        x=[p['gpu_hours'] for p in points]
        y=[sum(p['evaluation'][split]['normal']['success_by_episode'][:5])/5 for p in points]
        ax.plot(x,y,lw=2,color=colors[method],label=labels[method])
        ax.scatter(x[-1:],y[-1:],s=20,color=colors[method])
    ax.set(title=title,xlabel='Active GPU Hours',ylim=(0,1),xlim=(0,None))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
axes[0].set_ylabel('Success');axes[1].legend(loc='upper right',fontsize=8)
fig.suptitle('Companion only · first K5 outcomes from original K25 evaluations · full evaluation time charged',fontsize=10)
for extension in ('png','pdf'):fig.savefig(out/f'ml10-first5-gpu-hours-20261029-0835.{extension}',dpi=200,bbox_inches='tight')
plt.close(fig)
print('Saved six headline/companion figures, first-five companion, PDFs and observed curve CSV.')
