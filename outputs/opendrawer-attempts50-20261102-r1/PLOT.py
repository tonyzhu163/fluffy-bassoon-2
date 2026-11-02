"""Plot verified actual-attempt success on allocated Linux CPUs."""
import argparse,csv,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
assert sys.platform=='linux'
p=argparse.ArgumentParser();p.add_argument('results',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
results=[json.loads(Path(r).read_text()) for r in a.results]
for r in results:assert r['completed'] and r['actual_attempts_only'] and r['separate_queries']==False
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True,constrained_layout=True)
rows=[];colors={'ad_full':'#1960a5','dpt_full':'#bb4a29'}
for axis,cohort in zip(axes,('ID','OOD')):
 for r in results:
  table=r['cohorts'][cohort];assert table['cases']==50
  x=np.arange(1,6);s=np.array([table['attempt_successes'][str(k)] for k in x]);n=50;rate=s/n;z=1.95996398454
  center=(rate+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
  axis.errorbar(x,rate*100,yerr=np.vstack((rate-(center-half),(center+half)-rate))*100,label=r['method'].split('_')[0].upper(),color=colors[r['method']],marker='o',capsize=3)
  for k,v,lo,hi in zip(x,s,center-half,center+half):rows.append([r['method'],cohort,int(k),int(v),n,float(v/n),float(lo),float(hi)])
 axis.set(title='ID · OpenDrawer' if cohort=='ID' else 'OOD · CloseFridge',xlabel='Actual attempt number',xticks=x,ylim=(0,100))
 axis.grid(alpha=.2);axis.legend(frameon=False)
axes[0].set_ylabel('Success rate (%)')
fig.suptitle('Frozen weights; memory retained within each case',fontsize=12)
fig.savefig(out/'attempt-success.png',dpi=200);fig.savefig(out/'attempt-success.svg')
with (out/'attempt-success.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['method','cohort','actual_attempt','successes','cases','success_rate','wilson95_low','wilson95_high']);w.writerows(rows)
(out/'CAPTION.md').write_text('Each point is success on that actual attempt, with50 paired cases per task and95% Wilson intervals. Memory contains previous actual attempts only; environment resets to the same case between attempts and memory clears between cases. Weights frozen. Attempt-specific motor noise is fixed and matched between models, but differs across attempts. This is not cumulative ever-success. Later histories extend beyond the64-token training window. Single training source and fit seed; OOD concerns student training only.\n')
