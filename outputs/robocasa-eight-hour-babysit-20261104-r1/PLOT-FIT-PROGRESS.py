"""Freeze complete metric lines and plot within-method actor validation traces on CPU."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def main():
    assert sys.platform == 'linux'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True)
    a = p.parse_args()
    assert a.tag.replace('-', '').replace('_', '').isalnum()
    root = Path('/mowenjie/fluffy-bassoon-2')
    run = root/'outputs/robocasa-eight-hour-babysit-20261104-r1'
    fits = root/'outputs/robocasa-three-task-meta-baseline-20261104-r2/joint/fits'
    output = run/'figures'/a.tag
    output.mkdir(parents=True, exist_ok=False)
    os.environ['MPLCONFIGDIR'] = str(run/'plot-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    inputs = []
    checked = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    for ax, method in zip(axes.flat, ('ad', 'dpt', 'dit', 'ic_iql')):
        source = fits/method/'metrics.jsonl'
        raw = source.read_bytes()
        lines = raw.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b'\n'):
            lines.pop()
        frozen = b''.join(lines)
        target = output/(method+'-metrics.jsonl')
        target.write_bytes(frozen)
        rows = [json.loads(line) for line in lines]
        actor = [row for row in rows if row.get('phase','actor')=='actor'
            and ('validation_loss' in row or 'validation_nll' in row)]
        def loss(row):
            return row['validation_loss'] if 'validation_loss' in row else row['validation_nll']
        assert actor and all(math.isfinite(loss(row)) for row in actor)
        status = json.loads((fits/method/'training-status.json').read_text())
        (output/(method+'-status.json')).write_text(json.dumps(status,indent=2)+'\n')
        best = min(actor,key=loss)
        x,y = [row['update'] for row in actor], [loss(row) for row in actor]
        ax.plot(x,y,color='#147dba',linewidth=1.5)
        ax.scatter([best['update']],[loss(best)],color='#df812e',s=25,zorder=4)
        stop = 'converged' if status['converged'] else 'training'
        ax.set_title(method.upper().replace('_','-')+': '+stop+'; latest '+str(actor[-1]['update']))
        ax.set_xlabel('Actor updates'); ax.set_ylabel('Held-out actor loss')
        ax.grid(alpha=.2)
        inputs.append(dict(method=method,source=str(source),frozen_metrics=str(target),
            sha256=hashlib.sha256(frozen).hexdigest(),complete_lines=len(lines),
            observed_best=best,latest_actor=actor[-1],status=status))
    fig.suptitle('Three-task joint fits: within-method validation progress',fontsize=15,y=.98)
    fig.tight_layout(rect=(0,.09,1,.94))
    fig.text(.5,.025,'Different objectives have different loss scales; do not rank methods by these curves.\n'
        'Orange marks the best observed validation point. Robot success requires the separate audited readouts.',
        ha='center',fontsize=10,color='#505860')
    images=[]
    for suffix in ('png','svg'):
        path=output/('training-progress.'+suffix)
        fig.savefig(path,dpi=160,facecolor='white')
        images.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    plt.close(fig)
    manifest=dict(checked_shanghai=checked,inputs=inputs,images=images,CPU_only=True,GPU_hours=0,
        source_learner_commit='d71cb532f6cf4933a4a5e236e1b085bce74ce969',
        meaning='Snapshot of complete metric lines; active histories continue after this figure.')
    (output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),checked_shanghai=checked)))


if __name__=='__main__':
    main()
